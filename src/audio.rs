use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::process::{self, TaskContext};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interval {
    pub start: f64,
    pub end: f64,
}

pub fn duration(path: &Path) -> Result<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(path)
        .output()
        .context("ffprobe is unavailable")?;
    ensure!(
        output.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: f64 = String::from_utf8(output.stdout)?
        .trim()
        .parse()
        .context("invalid audio duration")?;
    ensure!(
        value.is_finite() && value > 0.0,
        "invalid audio duration {value}"
    );
    Ok(value)
}

pub fn normalize(input: &Path, output: &Path) -> Result<()> {
    normalize_observed(input, output, None)
}

fn progress_ms(line: &str) -> Result<Option<u64>> {
    let Some(value) = line.strip_prefix("out_time_us=") else {
        return Ok(None);
    };
    let micros: u64 = value.parse().context("invalid ffmpeg out_time_us")?;
    Ok(Some(micros / 1000))
}

pub fn normalize_observed(
    input: &Path,
    output: &Path,
    context: Option<&TaskContext>,
) -> Result<()> {
    if let Some(context) = context {
        context.report_operation("normalize audio")?;
    }
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-progress",
            "pipe:1",
            "-i",
        ])
        .arg(input)
        .args(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
        .arg(output);
    process::run_observed_lines(
        &mut command,
        &output.with_extension("ffmpeg.stderr"),
        Duration::from_secs(24 * 60 * 60),
        context,
        |line| {
            if let Some(ms) = progress_ms(line)?
                && let Some(context) = context
            {
                context.report_progress_ms(ms)?;
            }
            Ok(())
        },
    )
    .with_context(|| format!("ffmpeg failed to decode {}", input.display()))
}

pub fn chunks(duration: f64, max_duration: f64, overlap: f64) -> Result<Vec<Interval>> {
    ensure!(
        duration.is_finite()
            && duration > 0.0
            && max_duration.is_finite()
            && overlap.is_finite()
            && max_duration > overlap
            && overlap >= 0.0,
        "invalid chunk settings"
    );
    let mut result = Vec::new();
    let mut start = 0.0;
    while start < duration {
        let end = (start + max_duration).min(duration);
        result.push(Interval { start, end });
        if end >= duration {
            break;
        }
        start = end - overlap;
    }
    Ok(result)
}

pub fn extract(wav: &Path, interval: &Interval, directory: &Path, index: usize) -> Result<PathBuf> {
    extract_observed(wav, interval, directory, index, None)
}

pub fn extract_observed(
    wav: &Path,
    interval: &Interval,
    directory: &Path,
    index: usize,
    context: Option<&TaskContext>,
) -> Result<PathBuf> {
    if interval.end <= interval.start {
        bail!("empty audio chunk");
    }
    if let Some(context) = context {
        context.check_cancelled()?;
        context.report_operation("extract audio chunk")?;
    }
    let path = directory.join(format!("{index:05}.wav"));
    let partial = directory.join(format!("{index:05}.partial.wav"));
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
        ])
        .arg(interval.start.to_string())
        .args(["-t", &(interval.end - interval.start).to_string(), "-i"])
        .arg(wav)
        .args(["-c:a", "pcm_s16le"])
        .arg(&partial);
    process::run_observed(
        &mut command,
        &directory.join(format!("{index:05}.ffmpeg.stdout")),
        &directory.join(format!("{index:05}.ffmpeg.stderr")),
        Duration::from_secs(60 * 60),
        context,
    )
    .with_context(|| format!("cannot extract chunk {index}"))?;
    let mut header = [0; 44];
    std::fs::File::open(&partial)
        .with_context(|| format!("cannot open chunk {index}"))?
        .read_exact(&mut header)
        .with_context(|| format!("cannot read chunk {index}"))?;
    ensure!(
        &header[..4] == b"RIFF" && &header[8..12] == b"WAVE",
        "invalid WAV chunk {index}"
    );
    if let Some(context) = context {
        context.check_cancelled()?;
    }
    std::fs::rename(&partial, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_overlap_keeps_edges() {
        let intervals = chunks(70.0, 28.0, 0.8).unwrap();
        assert_eq!(intervals.first().unwrap().start, 0.0);
        assert_eq!(intervals.last().unwrap().end, 70.0);
        for pair in intervals.windows(2) {
            assert!((pair[0].end - pair[1].start - 0.8).abs() < 1e-6);
        }
    }

    #[test]
    fn quiet_intervals_are_not_discarded() {
        let intervals = chunks(65.0, 28.0, 0.8).unwrap();
        assert!(
            intervals
                .windows(2)
                .all(|pair| pair[0].end >= pair[1].start)
        );
        assert_eq!(intervals.last().unwrap().end, 65.0);
    }

    #[test]
    fn ffmpeg_progress_is_measured_in_milliseconds() {
        assert_eq!(progress_ms("out_time_us=1234567").unwrap(), Some(1234));
        assert_eq!(progress_ms("out_time_us=0").unwrap(), Some(0));
        assert_eq!(progress_ms("progress=continue").unwrap(), None);
        assert!(progress_ms("out_time_us=not-a-number").is_err());
    }

    #[test]
    fn normalization_reports_real_progress_without_stdout_noise() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("source.wav");
        let output = dir.path().join("normalized.wav");
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=8000",
                "-t",
                "1",
            ])
            .arg(&input)
            .status()
            .unwrap();
        assert!(status.success());
        let values = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let measured = values.clone();
        let context = TaskContext::new(
            None,
            Some(std::sync::Arc::new(move |ms| {
                measured.lock().unwrap().push(ms);
                Ok(())
            })),
            None,
        );
        normalize_observed(&input, &output, Some(&context)).unwrap();
        assert!(duration(&output).unwrap() >= 0.9);
        assert!(values.lock().unwrap().iter().any(|ms| *ms >= 900));
        assert!(!output.with_extension("stdout").exists());
    }
}
