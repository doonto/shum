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

pub fn speech_intervals(wav: &Path, length: f64) -> Result<Vec<Interval>> {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(wav)
        .args(["-af", "silencedetect=noise=-38dB:d=0.35", "-f", "null", "-"])
        .output()
        .context("ffmpeg is unavailable")?;
    ensure!(
        output.status.success(),
        "silence detection failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut silences = Vec::new();
    let mut begin = None;
    for line in String::from_utf8_lossy(&output.stderr).lines() {
        if let Some(value) = line.split("silence_start: ").nth(1) {
            begin = Some(
                value
                    .trim()
                    .parse::<f64>()
                    .context("invalid silence start")?,
            );
        }
        if let Some(value) = line.split("silence_end: ").nth(1) {
            let end = value
                .split_whitespace()
                .next()
                .context("invalid silence end")?
                .parse::<f64>()?;
            if let Some(start) = begin.take() {
                silences.push(Interval { start, end });
            } else {
                silences.push(Interval { start: 0.0, end });
            }
        }
    }
    if let Some(start) = begin {
        silences.push(Interval { start, end: length });
    }
    let mut speech = Vec::new();
    let mut cursor = 0.0_f64;
    for silence in silences {
        let start = silence.start.clamp(cursor, length);
        if start - cursor >= 0.15 {
            speech.push(Interval {
                start: cursor,
                end: start,
            });
        }
        cursor = silence.end.clamp(cursor, length);
    }
    if length - cursor >= 0.15 {
        speech.push(Interval {
            start: cursor,
            end: length,
        });
    }
    Ok(speech)
}

pub fn chunks(speech: &[Interval], max_duration: f64, overlap: f64) -> Result<Vec<Interval>> {
    ensure!(
        max_duration > overlap && overlap >= 0.0,
        "invalid chunk settings"
    );
    let mut result = Vec::new();
    for region in speech {
        ensure!(
            region.start >= 0.0 && region.end >= region.start,
            "invalid speech interval"
        );
        let mut start = region.start;
        while start < region.end - 0.001 {
            let end = (start + max_duration).min(region.end);
            result.push(Interval { start, end });
            if end >= region.end {
                break;
            }
            start = end - overlap;
        }
    }
    Ok(result)
}

pub fn split_on_turns(speech: &[Interval], turns: &[Interval]) -> Vec<Interval> {
    let mut result = Vec::new();
    for region in speech {
        let mut boundaries = vec![region.start, region.end];
        for turn in turns {
            if turn.start > region.start && turn.start < region.end {
                boundaries.push(turn.start);
            }
            if turn.end > region.start && turn.end < region.end {
                boundaries.push(turn.end);
            }
        }
        boundaries.sort_by(f64::total_cmp);
        boundaries.dedup_by(|a, b| (*a - *b).abs() < 0.02);
        for pair in boundaries.windows(2) {
            if pair[1] > pair[0] {
                result.push(Interval {
                    start: pair[0],
                    end: pair[1],
                });
            }
        }
    }
    result
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
        let intervals = chunks(
            &[Interval {
                start: 5.0,
                end: 70.0,
            }],
            28.0,
            0.8,
        )
        .unwrap();
        assert_eq!(intervals.first().unwrap().start, 5.0);
        assert_eq!(intervals.last().unwrap().end, 70.0);
        for pair in intervals.windows(2) {
            assert!((pair[0].end - pair[1].start - 0.8).abs() < 1e-6);
        }
    }

    #[test]
    fn turn_boundaries_split_speech() {
        let result = split_on_turns(
            &[Interval {
                start: 0.0,
                end: 12.0,
            }],
            &[
                Interval {
                    start: 0.0,
                    end: 5.0,
                },
                Interval {
                    start: 5.0,
                    end: 12.0,
                },
            ],
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].end, 5.0);
        assert_eq!(result[1].start, 5.0);
    }
}
