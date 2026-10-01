use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

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
    let status = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
        .arg(input)
        .args(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
        .arg(output)
        .status()
        .context("ffmpeg is unavailable")?;
    ensure!(
        status.success(),
        "ffmpeg failed to decode {}",
        input.display()
    );
    Ok(())
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
    if interval.end <= interval.start {
        bail!("empty audio chunk");
    }
    let path = directory.join(format!("{index:05}.wav"));
    let partial = directory.join(format!("{index:05}.partial.wav"));
    let status = Command::new("ffmpeg")
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
        .arg(&partial)
        .status()
        .context("ffmpeg is unavailable")?;
    ensure!(status.success(), "cannot extract chunk {index}");
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
}
