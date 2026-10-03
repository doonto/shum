use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use wait_timeout::ChildExt;

use crate::audio::Interval;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub max_scan_seconds: f64,
    pub max_sample_frames: u64,
    pub max_decode_seconds: u64,
    pub near_zero_threshold: f64,
    pub full_scale_threshold: f64,
    pub low_rms_dbfs: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_scan_seconds: 1800.0,
            max_sample_frames: 120_000_000,
            max_decode_seconds: 60,
            near_zero_threshold: 0.001,
            full_scale_threshold: 0.999,
            low_rms_dbfs: -50.0,
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_scan_seconds.is_finite()
                && self.max_scan_seconds > 0.0
                && self.max_scan_seconds <= 7200.0,
            "max_scan_seconds must be in (0, 7200]"
        );
        ensure!(
            (1..=500_000_000).contains(&self.max_sample_frames),
            "max_sample_frames must be in [1, 500000000]"
        );
        ensure!(
            (1..=3600).contains(&self.max_decode_seconds),
            "max_decode_seconds must be in [1, 3600]"
        );
        ensure!(
            self.near_zero_threshold.is_finite() && (0.0..1.0).contains(&self.near_zero_threshold),
            "near_zero_threshold must be in [0, 1)"
        );
        ensure!(
            self.full_scale_threshold.is_finite()
                && (0.0..=1.0).contains(&self.full_scale_threshold)
                && self.full_scale_threshold > self.near_zero_threshold,
            "full_scale_threshold must be in (near_zero_threshold, 1]"
        );
        ensure!(
            self.low_rms_dbfs.is_finite() && (-120.0..=0.0).contains(&self.low_rms_dbfs),
            "low_rms_dbfs must be in [-120, 0]"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub sample_count: u64,
    pub peak_dbfs: Option<f64>,
    pub rms_dbfs: Option<f64>,
    pub zero_signal: bool,
    pub near_zero_fraction: Option<f64>,
    pub full_scale_fraction: Option<f64>,
    pub low_rms: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Window {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub start_frame: u64,
    pub end_frame: u64,
    pub scanned_frames: u64,
    pub partial_coverage: bool,
    pub metrics: Metrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub sample_rate: u32,
    pub channel_count: u16,
    pub available: bool,
    pub unavailable_reason: Option<String>,
    pub total_frames: u64,
    // False when a decoded stream was stopped at the configured scan limit.
    pub total_frames_known: bool,
    pub scanned_frames: u64,
    pub partial_coverage: bool,
    pub metrics: Metrics,
    pub channels: Vec<Metrics>,
    pub windows: Vec<Window>,
    // Only for two-channel PCM; neither value identifies speech or a preferred channel.
    pub stereo_correlation: Option<f64>,
    pub possible_phase_cancellation: Option<bool>,
}

#[derive(Default)]
struct Accumulator {
    count: u64,
    peak: f64,
    sum_squares: f64,
    near_zero: u64,
    full_scale: u64,
}

impl Accumulator {
    fn push(&mut self, sample: f64, settings: &Settings) {
        let absolute = sample.abs();
        self.count += 1;
        self.peak = self.peak.max(absolute);
        self.sum_squares += sample * sample;
        self.near_zero += u64::from(absolute <= settings.near_zero_threshold);
        self.full_scale += u64::from(absolute >= settings.full_scale_threshold);
    }

    fn metrics(&self, settings: &Settings) -> Metrics {
        let rms = (self.count > 0).then(|| (self.sum_squares / self.count as f64).sqrt());
        Metrics {
            sample_count: self.count,
            peak_dbfs: dbfs(self.peak),
            rms_dbfs: rms.and_then(dbfs),
            zero_signal: self.peak == 0.0,
            near_zero_fraction: (self.count > 0).then(|| self.near_zero as f64 / self.count as f64),
            full_scale_fraction: (self.count > 0)
                .then(|| self.full_scale as f64 / self.count as f64),
            low_rms: rms.map(|value| value <= 10_f64.powf(settings.low_rms_dbfs / 20.0)),
        }
    }
}

fn dbfs(value: f64) -> Option<f64> {
    (value > 0.0).then(|| 20.0 * value.log10())
}

struct Pcm {
    sample_rate: u32,
    channel_count: u16,
    data_start: u64,
    data_bytes: u64,
}

fn read_pcm(reader: &mut BufReader<File>) -> Result<Pcm> {
    let file_len = reader.get_ref().metadata()?.len();
    ensure!(file_len >= 12, "truncated RIFF header");
    let mut header = [0_u8; 12];
    reader.read_exact(&mut header)?;
    ensure!(
        &header[..4] == b"RIFF" && &header[8..] == b"WAVE",
        "expected RIFF WAVE"
    );
    let riff_end = u32::from_le_bytes(header[4..8].try_into()?) as u64 + 8;
    ensure!(
        riff_end == file_len,
        "RIFF length does not match file length"
    );

    let mut offset = 12_u64;
    let mut format = None;
    let mut data = None;
    while offset < riff_end {
        ensure!(riff_end - offset >= 8, "truncated WAV chunk header");
        reader.seek(SeekFrom::Start(offset))?;
        let mut chunk = [0_u8; 8];
        reader.read_exact(&mut chunk)?;
        let size = u32::from_le_bytes(chunk[4..8].try_into()?) as u64;
        let start = offset + 8;
        let end = start + size;
        ensure!(end <= riff_end, "truncated WAV chunk");
        match &chunk[..4] {
            b"fmt " => {
                ensure!(format.is_none(), "duplicate WAV fmt chunk");
                ensure!(size >= 16, "truncated WAV fmt chunk");
                let mut fmt = [0_u8; 16];
                reader.read_exact(&mut fmt)?;
                let encoding = u16::from_le_bytes(fmt[0..2].try_into()?);
                let channels = u16::from_le_bytes(fmt[2..4].try_into()?);
                let rate = u32::from_le_bytes(fmt[4..8].try_into()?);
                let byte_rate = u32::from_le_bytes(fmt[8..12].try_into()?);
                let block_align = u16::from_le_bytes(fmt[12..14].try_into()?);
                let bits = u16::from_le_bytes(fmt[14..16].try_into()?);
                ensure!(encoding == 1 && bits == 16, "only PCM16 WAV is supported");
                ensure!(
                    (1..=8).contains(&channels),
                    "WAV channel count must be in [1, 8]"
                );
                ensure!(rate > 0 && rate <= 384_000, "invalid WAV sample rate");
                ensure!(
                    block_align == channels * 2 && byte_rate == rate * u32::from(block_align),
                    "invalid WAV block alignment or byte rate"
                );
                format = Some((rate, channels, u64::from(block_align)));
            }
            b"data" => {
                ensure!(data.is_none(), "duplicate WAV data chunk");
                data = Some((start, size));
            }
            _ => {}
        }
        offset = end + (size & 1);
        ensure!(offset <= riff_end, "missing WAV chunk padding");
        if size & 1 != 0 {
            reader.seek(SeekFrom::Start(end))?;
            let mut padding = [0];
            reader.read_exact(&mut padding)?;
            ensure!(padding[0] == 0, "invalid WAV chunk padding");
        }
    }
    let (sample_rate, channel_count, block_align) = format.context("missing WAV fmt chunk")?;
    let (data_start, data_bytes) = data.context("missing WAV data chunk")?;
    ensure!(data_bytes % block_align == 0, "partial PCM sample frame");
    Ok(Pcm {
        sample_rate,
        channel_count,
        data_start,
        data_bytes,
    })
}

fn requested_frame(seconds: f64, sample_rate: u32) -> u64 {
    let position = seconds * f64::from(sample_rate);
    let nearest = position.round();
    let aligned = if (position - nearest).abs() <= 8.0 * f64::EPSILON * position.abs().max(1.0) {
        nearest
    } else {
        position
    };
    aligned.ceil() as u64
}

pub fn analyze(wav: &Path, chunks: &[Interval], settings: &Settings) -> Result<Report> {
    settings.validate()?;
    ensure!(chunks.len() <= 100_000, "too many diagnostic windows");
    for chunk in chunks {
        ensure!(
            chunk.start.is_finite()
                && chunk.end.is_finite()
                && chunk.start >= 0.0
                && chunk.end > chunk.start,
            "invalid diagnostic window"
        );
    }
    let file = File::open(wav).with_context(|| format!("cannot open WAV {}", wav.display()))?;
    let mut reader = BufReader::new(file);
    let pcm = read_pcm(&mut reader).with_context(|| format!("invalid WAV {}", wav.display()))?;
    let channels = usize::from(pcm.channel_count);
    let total_frames = pcm.data_bytes / (channels as u64 * 2);
    let seconds_limit = (settings.max_scan_seconds * f64::from(pcm.sample_rate)).floor() as u64;
    let scan_frames = total_frames
        .min(seconds_limit)
        .min(settings.max_sample_frames);

    let mut windows: Vec<Window> = chunks
        .iter()
        .map(|chunk| {
            let start = requested_frame(chunk.start, pcm.sample_rate).min(total_frames);
            let requested_end = requested_frame(chunk.end, pcm.sample_rate);
            let end = requested_end.min(total_frames);
            Window {
                start_seconds: chunk.start,
                end_seconds: chunk.end,
                start_frame: start,
                end_frame: end,
                scanned_frames: end.min(scan_frames).saturating_sub(start.min(scan_frames)),
                partial_coverage: end > scan_frames || requested_end > total_frames,
                metrics: Metrics {
                    sample_count: 0,
                    peak_dbfs: None,
                    rms_dbfs: None,
                    zero_signal: true,
                    near_zero_fraction: None,
                    full_scale_fraction: None,
                    low_rms: None,
                },
            }
        })
        .collect();
    let mut window_acc: Vec<Accumulator> =
        (0..windows.len()).map(|_| Accumulator::default()).collect();
    let mut starts: Vec<usize> = (0..windows.len()).collect();
    starts.sort_unstable_by_key(|&i| windows[i].start_frame);
    let mut ends = BinaryHeap::new();
    for &index in &starts {
        let window = &windows[index];
        if window.start_frame >= scan_frames || window.end_frame <= window.start_frame {
            continue;
        }
        while ends
            .peek()
            .is_some_and(|&Reverse(end)| end <= window.start_frame)
        {
            ends.pop();
        }
        ends.push(Reverse(window.end_frame.min(scan_frames)));
        ensure!(
            ends.len() <= 16,
            "more than 16 simultaneous diagnostic windows"
        );
    }
    let mut next = 0;
    let mut active = Vec::new();
    let mut next_end = u64::MAX;
    let mut overall = Accumulator::default();
    let mut channel_acc: Vec<Accumulator> = (0..channels).map(|_| Accumulator::default()).collect();
    let mut cross = 0.0;
    let mut downmix_squares = 0.0;
    reader.seek(SeekFrom::Start(pcm.data_start))?;
    let mut buffer = [0_u8; 65536];
    let frame_bytes = channels * 2;
    let frames_per_block = buffer.len() / frame_bytes;
    let mut frame = 0_u64;
    while frame < scan_frames {
        let count = (scan_frames - frame).min(frames_per_block as u64) as usize;
        reader.read_exact(&mut buffer[..count * frame_bytes])?;
        for bytes in buffer[..count * frame_bytes].chunks_exact(frame_bytes) {
            while next < starts.len() && windows[starts[next]].start_frame <= frame {
                let index = starts[next];
                if windows[index].end_frame > frame {
                    next_end = next_end.min(windows[index].end_frame);
                    active.push(index);
                }
                next += 1;
            }
            if frame >= next_end {
                active.retain(|&i| windows[i].end_frame > frame);
                next_end = active
                    .iter()
                    .map(|&i| windows[i].end_frame)
                    .min()
                    .unwrap_or(u64::MAX);
            }
            let mut pair = [0.0; 2];
            for (channel, sample_bytes) in bytes.as_chunks::<2>().0.iter().enumerate() {
                let sample = f64::from(i16::from_le_bytes(*sample_bytes)) / 32768.0;
                overall.push(sample, settings);
                channel_acc[channel].push(sample, settings);
                for &i in &active {
                    window_acc[i].push(sample, settings);
                }
                if channels == 2 {
                    pair[channel] = sample;
                }
            }
            if channels == 2 {
                cross += pair[0] * pair[1];
                downmix_squares += ((pair[0] + pair[1]) / 2.0).powi(2);
            }
            frame += 1;
        }
    }
    let stereo_correlation =
        if channels == 2 && channel_acc[0].sum_squares > 0.0 && channel_acc[1].sum_squares > 0.0 {
            Some(
                (cross / (channel_acc[0].sum_squares * channel_acc[1].sum_squares).sqrt())
                    .clamp(-1.0, 1.0),
            )
        } else {
            None
        };
    let possible_phase_cancellation = stereo_correlation.map(|correlation| {
        let channel_rms = (overall.sum_squares / overall.count as f64).sqrt();
        let downmix_rms = (downmix_squares / scan_frames as f64).sqrt();
        correlation <= -0.9 && downmix_rms <= channel_rms * 0.25
    });
    for (window, acc) in windows.iter_mut().zip(&window_acc) {
        window.metrics = acc.metrics(settings);
    }
    Ok(Report {
        sample_rate: pcm.sample_rate,
        channel_count: pcm.channel_count,
        available: true,
        unavailable_reason: None,
        total_frames,
        total_frames_known: true,
        scanned_frames: scan_frames,
        partial_coverage: scan_frames < total_frames,
        metrics: overall.metrics(settings),
        channels: channel_acc
            .iter()
            .map(|acc| acc.metrics(settings))
            .collect(),
        windows,
        stereo_correlation,
        possible_phase_cancellation,
    })
}

struct SourceProcess {
    child: Arc<Mutex<Child>>,
    done: Option<mpsc::Sender<()>>,
    watchdog: Option<thread::JoinHandle<()>>,
    expired: Arc<AtomicBool>,
}

impl SourceProcess {
    fn new(child: Child, seconds: u64) -> Self {
        let child = Arc::new(Mutex::new(child));
        let expired = Arc::new(AtomicBool::new(false));
        let (done, receiver) = mpsc::channel();
        let owned_child = Arc::clone(&child);
        let timed_out = Arc::clone(&expired);
        let watchdog = thread::spawn(move || {
            if let Err(mpsc::RecvTimeoutError::Timeout) =
                receiver.recv_timeout(Duration::from_secs(seconds))
            {
                timed_out.store(true, Ordering::SeqCst);
                if let Ok(mut process) = owned_child.lock() {
                    let _ = process.kill();
                }
            }
        });
        Self {
            child,
            done: Some(done),
            watchdog: Some(watchdog),
            expired,
        }
    }

    fn finish(&mut self) -> Result<()> {
        let status = loop {
            if let Some(status) = self
                .child
                .lock()
                .expect("ffmpeg process lock poisoned")
                .wait_timeout(Duration::from_millis(100))?
            {
                break status;
            }
        };
        self.done.take();
        if let Some(watchdog) = self.watchdog.take() {
            watchdog.join().expect("ffmpeg watchdog panicked");
        }
        ensure!(
            !self.expired.load(Ordering::SeqCst),
            "ffmpeg decode deadline exceeded"
        );
        ensure!(
            status.success(),
            "ffmpeg failed to decode original stereo audio"
        );
        Ok(())
    }
}

impl Drop for SourceProcess {
    fn drop(&mut self) {
        self.done.take();
        if let Ok(mut process) = self.child.lock() {
            let _ = process.kill();
            let _ = process.wait();
        }
        if let Some(watchdog) = self.watchdog.take() {
            let _ = watchdog.join();
        }
    }
}

// The original channel count is preserved in interleaved PCM16; the source
// length is unknown when a bounded scan stops before decoder EOF.
pub fn analyze_channels(source: &Path, settings: &Settings) -> Result<Report> {
    settings.validate()?;
    let mut probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
        ])
        .arg(source)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("ffprobe is unavailable")?;
    match probe.wait_timeout(Duration::from_secs(15)) {
        Ok(Some(_)) => {}
        Ok(None) => {
            let _ = probe.kill();
            let _ = probe.wait();
            anyhow::bail!("ffprobe deadline exceeded");
        }
        Err(error) => {
            let _ = probe.kill();
            let _ = probe.wait();
            return Err(error).context("cannot wait for ffprobe");
        }
    }
    let probe_output = probe.wait_with_output()?;
    ensure!(
        probe_output.status.success(),
        "ffprobe failed to inspect original audio"
    );
    let value: serde_json::Value =
        serde_json::from_slice(&probe_output.stdout).context("invalid ffprobe output")?;
    let stream = value["streams"]
        .as_array()
        .and_then(|streams| streams.first())
        .context("source has no audio stream")?;
    let channel_count: u16 = stream["channels"]
        .as_u64()
        .context("source channel count is unavailable")?
        .try_into()
        .context("invalid source channel count")?;
    ensure!(channel_count > 0, "invalid source channel count");
    let sample_rate: u32 = stream["sample_rate"]
        .as_str()
        .context("source sample rate is unavailable")?
        .parse()
        .context("invalid source sample rate")?;
    ensure!(
        (1..=384_000).contains(&sample_rate),
        "invalid source sample rate"
    );
    if channel_count > 32 {
        return Ok(Report {
            sample_rate,
            channel_count,
            available: false,
            unavailable_reason: Some("more than 32 source channels are unsupported".to_owned()),
            total_frames: 0,
            total_frames_known: false,
            scanned_frames: 0,
            partial_coverage: true,
            metrics: Metrics {
                sample_count: 0,
                peak_dbfs: None,
                rms_dbfs: None,
                zero_signal: false,
                near_zero_fraction: None,
                full_scale_fraction: None,
                low_rms: None,
            },
            channels: Vec::new(),
            windows: Vec::new(),
            stereo_correlation: None,
            possible_phase_cancellation: None,
        });
    }

    let mut child = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(source)
        .args(["-map", "0:a:0", "-vn", "-ar"])
        .arg(sample_rate.to_string())
        .args(["-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("ffmpeg is unavailable")?;
    let stdout = child.stdout.take().context("ffmpeg stdout unavailable")?;
    let mut process = SourceProcess::new(child, settings.max_decode_seconds);
    let mut reader = BufReader::new(stdout);
    let limit = ((settings.max_scan_seconds * f64::from(sample_rate)).floor() as u64)
        .min(settings.max_sample_frames);
    let mut overall = Accumulator::default();
    let mut channels: Vec<Accumulator> =
        (0..channel_count).map(|_| Accumulator::default()).collect();
    let mut cross = 0.0;
    let mut downmix_squares = 0.0;
    let mut scanned_frames = 0_u64;
    let frame_bytes = usize::from(channel_count) * 2;
    let mut pending = Vec::with_capacity(65536 + frame_bytes);
    let mut buffer = [0_u8; 65536];
    let mut partial_coverage = false;
    loop {
        let count = reader
            .read(&mut buffer)
            .context("cannot read ffmpeg PCM stream")?;
        if count == 0 {
            ensure!(pending.is_empty(), "truncated decoded PCM sample frame");
            break;
        }
        pending.extend_from_slice(&buffer[..count]);
        let used = pending.len() / frame_bytes * frame_bytes;
        for bytes in pending[..used].chunks_exact(frame_bytes) {
            if scanned_frames == limit {
                partial_coverage = true;
                break;
            }
            let mut stereo = [0.0; 2];
            for (index, sample_bytes) in bytes.as_chunks::<2>().0.iter().enumerate() {
                let sample = f64::from(i16::from_le_bytes(*sample_bytes)) / 32768.0;
                overall.push(sample, settings);
                channels[index].push(sample, settings);
                if channel_count == 2 {
                    stereo[index] = sample;
                }
            }
            if channel_count == 2 {
                cross += stereo[0] * stereo[1];
                downmix_squares += ((stereo[0] + stereo[1]) / 2.0).powi(2);
            }
            scanned_frames += 1;
        }
        if partial_coverage {
            break;
        }
        pending.drain(..used);
    }
    if !partial_coverage {
        process.finish()?;
    } else {
        ensure!(
            !process.expired.load(Ordering::SeqCst),
            "ffmpeg decode deadline exceeded"
        );
    }
    ensure!(
        scanned_frames > 0,
        "original audio contains no decoded samples"
    );
    let correlation = if channel_count == 2
        && channels[0].sum_squares > 0.0
        && channels[1].sum_squares > 0.0
    {
        Some((cross / (channels[0].sum_squares * channels[1].sum_squares).sqrt()).clamp(-1.0, 1.0))
    } else {
        None
    };
    let possible_phase_cancellation = correlation.map(|value| {
        let channel_rms = (overall.sum_squares / overall.count as f64).sqrt();
        let downmix_rms = (downmix_squares / scanned_frames as f64).sqrt();
        value <= -0.9 && downmix_rms <= channel_rms * 0.25
    });
    Ok(Report {
        sample_rate,
        channel_count,
        available: true,
        unavailable_reason: None,
        total_frames: scanned_frames,
        total_frames_known: !partial_coverage,
        scanned_frames,
        partial_coverage,
        metrics: overall.metrics(settings),
        channels: channels.iter().map(|acc| acc.metrics(settings)).collect(),
        windows: Vec::new(),
        stereo_correlation: correlation,
        possible_phase_cancellation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn wav(channels: u16, samples: &[i16]) -> Vec<u8> {
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(36 + samples.len() as u32 * 2).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&16_000_u32.to_le_bytes());
        bytes.extend_from_slice(&(16_000_u32 * u32::from(channels) * 2).to_le_bytes());
        bytes.extend_from_slice(&(channels * 2).to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(samples.len() as u32 * 2).to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    fn measure(bytes: &[u8], chunks: &[Interval], settings: &Settings) -> Result<Report> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("audio.wav");
        fs::write(&path, bytes)?;
        analyze(&path, chunks, settings)
    }

    #[test]
    fn sine_peak_and_rms() {
        let samples: Vec<i16> = (0..16_000)
            .map(|i| ((i as f64 * std::f64::consts::TAU / 16.0).sin() * 16_384.0).round() as i16)
            .collect();
        let result = measure(&wav(1, &samples), &[], &Settings::default()).unwrap();
        assert!((result.metrics.peak_dbfs.unwrap() + 6.0206).abs() < 0.001);
        assert!((result.metrics.rms_dbfs.unwrap() + 9.0309).abs() < 0.001);
        assert!(!result.partial_coverage);
    }

    #[test]
    fn zeros_and_full_scale() {
        let zero = measure(&wav(1, &[0, 0]), &[], &Settings::default()).unwrap();
        assert!(zero.metrics.zero_signal);
        assert_eq!(zero.metrics.peak_dbfs, None);
        assert_eq!(zero.metrics.rms_dbfs, None);
        assert_eq!(zero.metrics.near_zero_fraction, Some(1.0));
        let full = measure(&wav(1, &[-32768, 32767]), &[], &Settings::default()).unwrap();
        assert_eq!(full.metrics.peak_dbfs, Some(0.0));
        assert_eq!(full.metrics.full_scale_fraction, Some(1.0));
    }

    #[test]
    fn overlapping_windows_and_scan_tail() {
        let settings = Settings {
            max_sample_frames: 3,
            ..Settings::default()
        };
        let windows = [
            Interval {
                start: 0.0,
                end: 2.0 / 16_000.0,
            },
            Interval {
                start: 1.0 / 16_000.0,
                end: 4.0 / 16_000.0,
            },
            Interval {
                start: 4.0 / 16_000.0,
                end: 5.0 / 16_000.0,
            },
        ];
        let result = measure(&wav(1, &[0, 16_384, 32767, 0, -32768]), &windows, &settings).unwrap();
        assert_eq!(result.scanned_frames, 3);
        assert!(result.partial_coverage);
        assert_eq!(result.windows[0].metrics.sample_count, 2);
        assert_eq!(result.windows[1].metrics.sample_count, 2);
        assert_eq!(result.windows[1].metrics.full_scale_fraction, Some(0.5));
        assert!(result.windows[1].partial_coverage);
        assert_eq!(result.windows[2].scanned_frames, 0);
        assert!(result.windows[2].partial_coverage);
    }

    #[test]
    fn many_disjoint_windows_are_scanned_without_a_window_count_failure() {
        let count = 100_000;
        let chunks: Vec<Interval> = (0..count)
            .map(|frame| Interval {
                start: frame as f64 / 16_000.0,
                end: (frame + 1) as f64 / 16_000.0,
            })
            .collect();
        let mut samples = vec![0; count];
        samples[count - 1] = -32768;
        let result = measure(&wav(1, &samples), &chunks, &Settings::default()).unwrap();
        assert_eq!(result.windows.len(), count);
        assert!(
            result
                .windows
                .iter()
                .all(|window| window.scanned_frames == 1)
        );
        assert_eq!(result.windows[count - 1].metrics.peak_dbfs, Some(0.0));
        assert_eq!(result.metrics.sample_count, count as u64);
    }

    #[test]
    fn pathological_overlap_is_rejected_before_scanning() {
        let chunks: Vec<Interval> = (0..17)
            .map(|_| Interval {
                start: 0.0,
                end: 1.0,
            })
            .collect();
        assert!(
            measure(&wav(1, &[0; 32]), &chunks, &Settings::default())
                .unwrap_err()
                .to_string()
                .contains("simultaneous diagnostic windows")
        );
    }

    #[test]
    fn final_frame_and_odd_chunk_padding() {
        let mut bytes = wav(1, &[0, 0, 0, -32768]);
        bytes.splice(12..12, [b'J', b'U', b'N', b'K', 1, 0, 0, 0, 42, 0]);
        let riff_size = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&riff_size.to_le_bytes());
        let result = measure(
            &bytes,
            &[Interval {
                start: 3.0 / 16_000.0,
                end: 4.0 / 16_000.0,
            }],
            &Settings::default(),
        )
        .unwrap();
        assert_eq!(result.scanned_frames, 4);
        assert!(!result.partial_coverage);
        assert_eq!(result.windows[0].scanned_frames, 1);
        assert_eq!(result.windows[0].metrics.peak_dbfs, Some(0.0));
        assert!(!result.windows[0].partial_coverage);
    }

    #[test]
    fn stereo_antiphase_reports_cancellation_not_speech() {
        let result = measure(
            &wav(2, &[16_384, -16_384, -16_384, 16_384]),
            &[],
            &Settings::default(),
        )
        .unwrap();
        assert_eq!(result.stereo_correlation, Some(-1.0));
        assert_eq!(result.possible_phase_cancellation, Some(true));
        assert_eq!(result.metrics.sample_count, 4);
        assert_eq!(result.channels.len(), 2);
    }

    #[test]
    fn decoded_stereo_distinguishes_complete_and_limited_scans() {
        if !Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
            || !Command::new("ffprobe")
                .arg("-version")
                .output()
                .is_ok_and(|o| o.status.success())
        {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("stereo.wav");
        fs::write(&source, wav(2, &[16_384, -16_384, -16_384, 16_384, 0, 0])).unwrap();
        let full = analyze_channels(&source, &Settings::default()).unwrap();
        assert_eq!(full.scanned_frames, 3);
        assert!(full.total_frames_known);
        assert!(!full.partial_coverage);
        assert_eq!(full.possible_phase_cancellation, Some(true));
        let limited = analyze_channels(
            &source,
            &Settings {
                max_sample_frames: 2,
                ..Settings::default()
            },
        )
        .unwrap();
        assert_eq!(limited.scanned_frames, 2);
        assert!(!limited.total_frames_known);
        assert!(limited.partial_coverage);
        assert_eq!(limited.metrics.sample_count, 4);
        let mono = dir.path().join("mono.wav");
        fs::write(&mono, wav(1, &[16_384, -16_384])).unwrap();
        let mono_report = analyze_channels(&mono, &Settings::default()).unwrap();
        assert!(mono_report.available);
        assert_eq!(mono_report.channel_count, 1);
        assert_eq!(mono_report.channels.len(), 1);
        assert_eq!(mono_report.metrics.sample_count, 2);
        assert_eq!(mono_report.stereo_correlation, None);
        assert_eq!(mono_report.possible_phase_cancellation, None);
        let surround = dir.path().join("surround.wav");
        fs::write(&surround, wav(3, &[16_384, 0, -16_384])).unwrap();
        let surround_report = analyze_channels(&surround, &Settings::default()).unwrap();
        assert!(surround_report.available);
        assert_eq!(surround_report.channel_count, 3);
        assert_eq!(surround_report.channels.len(), 3);
        assert_eq!(surround_report.metrics.sample_count, 3);
        assert_eq!(surround_report.stereo_correlation, None);
        assert_eq!(surround_report.possible_phase_cancellation, None);
        let too_many = dir.path().join("many-channels.wav");
        fs::write(&too_many, wav(33, &[0; 33])).unwrap();
        let unavailable = analyze_channels(&too_many, &Settings::default()).unwrap();
        assert!(!unavailable.available);
        assert_eq!(unavailable.channel_count, 33);
        assert!(unavailable.unavailable_reason.is_some());
        assert!(unavailable.partial_coverage);
        assert_eq!(unavailable.metrics.peak_dbfs, None);
    }

    #[test]
    fn mono_mp3_is_a_valid_original_source() {
        if !Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
            || !Command::new("ffprobe")
                .arg("-version")
                .output()
                .is_ok_and(|o| o.status.success())
        {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let wav_path = dir.path().join("mono.wav");
        let mp3_path = dir.path().join("mono.mp3");
        let samples: Vec<i16> = (0..16_000)
            .map(|i| ((i as f64 * std::f64::consts::TAU / 16.0).sin() * 16_384.0) as i16)
            .collect();
        fs::write(&wav_path, wav(1, &samples)).unwrap();
        let status = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
            .arg(&wav_path)
            .args(["-c:a", "libmp3lame", "-q:a", "4"])
            .arg(&mp3_path)
            .status()
            .unwrap();
        assert!(status.success());
        let report = analyze_channels(&mp3_path, &Settings::default()).unwrap();
        assert!(report.available);
        assert_eq!(report.unavailable_reason, None);
        assert_eq!(report.channel_count, 1);
        assert_eq!(report.channels.len(), 1);
        assert_eq!(report.metrics.sample_count, report.scanned_frames);
        assert!(report.metrics.rms_dbfs.is_some());
        assert!(!report.partial_coverage);
        assert_eq!(report.stereo_correlation, None);
        assert_eq!(report.possible_phase_cancellation, None);
    }

    #[test]
    fn malformed_wav_and_settings_are_errors() {
        let mut truncated = wav(1, &[1, 2]);
        truncated.pop();
        assert!(measure(&truncated, &[], &Settings::default()).is_err());
        let limited = Settings {
            max_sample_frames: 2,
            ..Settings::default()
        };
        let mut truncated_after_limit = wav(1, &[1; 64]);
        truncated_after_limit.pop();
        assert!(measure(&truncated_after_limit, &[], &limited).is_err());
        let partial = measure(&wav(1, &[1; 64]), &[], &limited).unwrap();
        assert!(partial.partial_coverage);
        assert!(partial.total_frames_known);
        assert_eq!(partial.scanned_frames, 2);
        let mut invalid = wav(1, &[1, 2]);
        invalid[32] = 4; // Wrong block alignment.
        assert!(measure(&invalid, &[], &Settings::default()).is_err());
        let mut missing_padding = wav(1, &[1]);
        missing_padding.extend_from_slice(b"JUNK\x01\0\0\0\x42");
        let riff_size = (missing_padding.len() - 8) as u32;
        missing_padding[4..8].copy_from_slice(&riff_size.to_le_bytes());
        assert!(measure(&missing_padding, &[], &Settings::default()).is_err());
        let mut nonzero_padding = wav(1, &[1]);
        nonzero_padding.extend_from_slice(b"JUNK\x01\0\0\0\x42\xff");
        let riff_size = (nonzero_padding.len() - 8) as u32;
        nonzero_padding[4..8].copy_from_slice(&riff_size.to_le_bytes());
        assert!(measure(&nonzero_padding, &[], &Settings::default()).is_err());
        let settings = Settings {
            near_zero_threshold: f64::NAN,
            ..Settings::default()
        };
        assert!(measure(&wav(1, &[1]), &[], &settings).is_err());
        assert_eq!(Settings::default().max_decode_seconds, 60);
        assert!(
            Settings {
                max_decode_seconds: 0,
                ..Settings::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Settings {
                max_decode_seconds: 3601,
                ..Settings::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Settings {
                max_decode_seconds: 3600,
                ..Settings::default()
            }
            .validate()
            .is_ok()
        );
    }
}
