use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    audio::{self, Interval},
    config::{Adapter, Config},
    process, resources,
    workflow::{self, Outcome},
};

static WORKER_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, PartialEq)]
enum WorkerOutcome {
    Complete,
    OutOfMemory,
    Cancelled,
}

#[derive(Clone, Copy)]
struct WorkerMode<'a> {
    concurrent: bool,
    offload: bool,
    worker_number: usize,
    worker_count: usize,
    deadline: Option<Instant>,
    benchmark: Option<&'a BenchmarkTelemetry>,
    execution: Option<&'a crate::execution::Handle>,
    retry: Option<&'a crate::execution::RetryWork>,
    task: Option<&'a process::TaskContext>,
}

fn task_context(
    execution: &crate::execution::Handle,
    step: &str,
    total_ms: Option<u64>,
) -> process::TaskContext {
    let cancel = execution.clone();
    let progress = execution.clone();
    let operation = execution.clone();
    let measurements = execution.clone();
    let cancel_step = step.to_owned();
    let progress_step = step.to_owned();
    let operation_step = step.to_owned();
    let measured_step = step.to_owned();
    process::TaskContext::new(
        Some(Arc::new(move || cancel.cancelled(&cancel_step))),
        Some(Arc::new(move |ms| {
            progress.numeric_progress(
                &progress_step,
                ms.min(total_ms.unwrap_or(ms)),
                total_ms,
                "audio ms",
            )
        })),
        Some(Arc::new(move |message| {
            operation.operation(&operation_step, message)
        })),
    )
    .with_measurements(Arc::new(move |phase, done, total| {
        measurements.operation(&measured_step, phase)?;
        measurements.numeric_progress(&measured_step, done, Some(total), "batches")
    }))
}

fn elapsed_seconds(started: Instant) -> u64 {
    let elapsed = started.elapsed();
    elapsed
        .as_secs()
        .saturating_add(u64::from(elapsed.subsec_nanos() != 0))
}

fn auto_asr_task_context(
    execution: &crate::execution::Handle,
    step: &str,
    attempt_id: &str,
    started: Instant,
) -> process::TaskContext {
    let mut task = task_context(execution, step, None);
    let handle = execution.clone();
    let id = attempt_id.to_owned();
    task.cancel = Some(Arc::new(move || {
        handle.record_runtime(&id, elapsed_seconds(started))?;
        handle.attempt_cancelled(&id)
    }));
    task
}

fn plan_oom_restart(
    execution: &crate::execution::Handle,
    role: &str,
    missing: &[usize],
    reason: &str,
    available_mib: u64,
) -> Result<String> {
    execution.plan_restart(
        &format!("asr:{role}"),
        missing,
        crate::execution::RetryParameters {
            threads: None,
            max_in_flight: Some(1),
        },
        reason,
        vec![serde_json::json!({
            "result": "oom-kill",
            "role": role,
            "available_mib": available_mib,
            "missing_fragments": missing,
        })],
    )
}

fn cancellation_outcome(execution: &crate::execution::Handle) -> Result<Outcome> {
    if process::interrupted() || execution.snapshot()?.interrupted {
        execution.interrupt()?;
        Ok(Outcome::Interrupted)
    } else {
        Ok(Outcome::Blocked)
    }
}

fn block_required_descendants(
    job: &workflow::Project,
    source: &str,
    include_source: bool,
    reason: &str,
) -> Result<()> {
    let execution = job.execution();
    let snapshot = execution.snapshot()?;
    for step in workflow::descendants_for_roles(source, &job.status.asr_roles)? {
        if step == source && !include_source {
            continue;
        }
        let stage = snapshot
            .stages
            .get(&step)
            .context("missing descendant stage")?;
        if !matches!(
            stage.status,
            workflow::State::Ready | workflow::State::Skipped | workflow::State::Cancelled
        ) {
            execution.fail_stage(&step, workflow::State::Blocked, reason)?;
        }
    }
    Ok(())
}

fn required_stage_outcome(job: &workflow::Project, source: &str, reason: &str) -> Result<Outcome> {
    let outcome = cancellation_outcome(&job.execution())?;
    if outcome == Outcome::Blocked {
        block_required_descendants(job, source, false, reason)?;
    }
    Ok(outcome)
}

fn retry_task_context(
    execution: &crate::execution::Handle,
    step: &str,
    attempt_id: &str,
) -> process::TaskContext {
    let mut task = task_context(execution, step, None);
    let handle = execution.clone();
    let id = attempt_id.to_owned();
    task.cancel = Some(Arc::new(move || handle.attempt_cancelled(&id)));
    task
}

fn cancelled_worker_or_error(
    task: &process::TaskContext,
    error: anyhow::Error,
) -> Result<WorkerOutcome> {
    if error.is::<process::TaskCancelled>() {
        return Ok(WorkerOutcome::Cancelled);
    }
    match task.check_cancelled() {
        Ok(()) => Err(error),
        Err(cancel) if cancel.is::<process::TaskCancelled>() => Ok(WorkerOutcome::Cancelled),
        Err(cancel) => Err(cancel),
    }
}

fn run_pending_asr_retries(
    job: &mut workflow::Project,
    config: &Config,
    roles: &[String],
    clips: &Path,
    output: &Path,
    chunks: &[Interval],
) -> Result<()> {
    job.apply_controls()?;
    let execution = job.execution();
    execution.snapshot()?;
    for retry in execution.pending_retries()? {
        let role = retry
            .step
            .strip_prefix("asr:")
            .with_context(|| format!("unsupported retry step {}", retry.step))?;
        ensure!(
            roles.iter().any(|selected| selected == role),
            "retry role {role} is outside the selected ASR roles"
        );
        let engine = config.asr_engine(role)?;
        let mut retry_config = config.clone();
        let adapter = retry_config
            .asr
            .get_mut(role)
            .with_context(|| format!("missing adapter for retry role {role}"))?;
        if let Some(threads) = retry.parameters.threads {
            adapter.threads = threads;
        }
        if let Some(max_in_flight) = retry.parameters.max_in_flight {
            adapter.max_in_flight = max_in_flight;
        }
        execution.begin_retry(&retry.attempt_id)?;
        let task = retry_task_context(&execution, &retry.step, &retry.attempt_id);
        let run = run_asr(
            &retry_config,
            role,
            engine,
            clips,
            output,
            chunks,
            WorkerMode {
                concurrent: false,
                offload: false,
                worker_number: 0,
                worker_count: 1,
                deadline: None,
                benchmark: None,
                execution: Some(&execution),
                retry: Some(&retry),
                task: Some(&task),
            },
        );
        match run {
            Ok(WorkerOutcome::Complete) => {
                execution.finish_retry(&retry.attempt_id)?;
                job.select_retry(&retry.attempt_id)?;
            }
            Ok(WorkerOutcome::OutOfMemory) => {
                execution.fail_stage(
                    &retry.step,
                    workflow::State::Failed,
                    "retry worker ran out of memory",
                )?;
                bail!("retry {} ran out of memory", retry.attempt_id);
            }
            Ok(WorkerOutcome::Cancelled) => {
                execution.fail_stage(&retry.step, workflow::State::Cancelled, "retry cancelled")?;
                return Err(process::TaskCancelled.into());
            }
            Err(error) => {
                let state = if error.is::<process::TaskCancelled>() {
                    workflow::State::Cancelled
                } else {
                    workflow::State::Failed
                };
                execution.fail_stage(&retry.step, state, &format!("{error:#}"))?;
                return Err(error);
            }
        }
    }
    Ok(())
}

fn run_pending_checks(job: &mut workflow::Project, config: &Config, output: &Path) -> Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    job.apply_controls()?;
    let execution = job.execution();
    let checks = execution.pending_checks()?;
    if !checks.is_empty() {
        let directory = output.join(".execution").join("checks");
        if let Err(error) = fs::DirBuilder::new().mode(0o700).create(&directory)
            && error.kind() != std::io::ErrorKind::AlreadyExists
        {
            return Err(error).context("cannot create owned check directory");
        }
        ensure!(
            directory.is_dir() && !directory.is_symlink(),
            "owned check directory must not be a symlink"
        );
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
    }
    for check in checks {
        ensure!(
            check
                .fragments
                .iter()
                .all(|index| *index < config.quality.max_chunks),
            "quality.max_chunks does not cover all planned check fragments"
        );
        ensure!(
            !check.attempt_id.is_empty()
                && check
                    .attempt_id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
            "invalid check attempt ID"
        );
        let task = retry_task_context(&execution, &check.step, &check.attempt_id);
        task.check_cancelled()?;
        execution.begin_check(&check.attempt_id)?;
        let result = (|| -> Result<()> {
            let report = crate::quality::inspect_inputs(
                output,
                &config.quality,
                false,
                Some(&check.fragments),
                Some(&check.input_folders),
                &|index| {
                    task.check_cancelled()?;
                    execution.check_unit_done(&check.attempt_id, index)
                },
            )?;
            let path = output
                .join(".execution")
                .join("checks")
                .join(format!("{}.json", check.attempt_id));
            let partial = path.with_extension("json.partial");
            ensure!(
                !path.is_symlink() && !partial.is_symlink(),
                "check report paths must not be symlinks"
            );
            write_json_with_temporary(&path, &partial, &report)?;
            execution.finish_check(&check.attempt_id, &path)
        })();
        if let Err(error) = result {
            let state = if error.is::<process::TaskCancelled>() {
                workflow::State::Cancelled
            } else {
                workflow::State::Failed
            };
            execution.fail_stage(&check.step, state, &format!("{error:#}"))?;
            return Err(error);
        }
    }
    Ok(())
}

fn checkpoint_with_checks(
    job: &mut workflow::Project,
    config: &Config,
    output: &Path,
    step: &str,
    outcome: Outcome,
) -> Result<Outcome> {
    match run_pending_checks(job, config, output) {
        Ok(()) => {
            let execution = job.execution();
            if execution.cancelled(step)? || process::interrupted() {
                if matches!(step, "audio" | "chunks" | "review") {
                    required_stage_outcome(job, step, &format!("{step} was cancelled"))
                } else {
                    cancellation_outcome(&execution)
                }
            } else {
                Ok(outcome)
            }
        }
        Err(error) if error.is::<process::TaskCancelled>() => {
            cancellation_outcome(&job.execution())
        }
        Err(error) => Err(error),
    }
}

#[derive(Debug, Deserialize)]
struct CorpusFile {
    path: PathBuf,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct CorpusWindow {
    index: usize,
    start: f64,
    end: f64,
    #[serde(alias = "frames")]
    samples: usize,
    #[serde(alias = "path")]
    wav_path: PathBuf,
    #[serde(alias = "sha256")]
    wav_sha256: String,
    pcm_sha256: String,
}

#[derive(Debug, Deserialize)]
struct CorpusManifest {
    #[serde(alias = "schema")]
    schema_version: u32,
    #[serde(default, alias = "input_sha256")]
    source_sha256: Option<String>,
    #[serde(alias = "duration_seconds")]
    audio_seconds: f64,
    #[serde(default)]
    audio_duration_seconds: Option<f64>,
    #[serde(default)]
    wav: Option<CorpusFile>,
    #[serde(default)]
    audio_path: Option<PathBuf>,
    #[serde(default)]
    audio_sha256: Option<String>,
    #[serde(alias = "clips")]
    windows: Vec<CorpusWindow>,
}

fn corpus_path(base: &Path, relative: &Path) -> Result<PathBuf> {
    if !relative.is_absolute() {
        ensure!(
            relative
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_))),
            "corpus path contains traversal or invalid components"
        );
    }
    let path = base.join(relative).canonicalize()?;
    ensure!(path.is_file(), "corpus path is not a file");
    Ok(path)
}

fn pcm_sha256(path: &Path) -> Result<(String, usize)> {
    let wav = fs::read(path)?;
    ensure!(
        wav.len() >= 44 && &wav[..4] == b"RIFF" && &wav[8..12] == b"WAVE",
        "invalid PCM WAV"
    );
    let mut offset = 12;
    let mut format = false;
    let mut pcm = None;
    while offset + 8 <= wav.len() {
        let size = u32::from_le_bytes(wav[offset + 4..offset + 8].try_into()?) as usize;
        let start = offset + 8;
        let end = start.checked_add(size).context("invalid WAV chunk size")?;
        ensure!(end <= wav.len(), "truncated WAV chunk");
        match &wav[offset..offset + 4] {
            b"fmt " => {
                ensure!(size >= 16, "invalid WAV format");
                ensure!(
                    u16::from_le_bytes(wav[start..start + 2].try_into()?) == 1
                        && u16::from_le_bytes(wav[start + 2..start + 4].try_into()?) == 1
                        && u32::from_le_bytes(wav[start + 4..start + 8].try_into()?) == 16000
                        && u16::from_le_bytes(wav[start + 14..start + 16].try_into()?) == 16,
                    "corpus WAV must be mono 16kHz PCM s16le"
                );
                format = true;
            }
            b"data" => {
                ensure!(pcm.is_none(), "multiple WAV data chunks");
                pcm = Some(&wav[start..end]);
            }
            _ => {}
        }
        offset = end + size % 2;
    }
    ensure!(format, "WAV format chunk missing");
    let pcm = pcm.context("WAV data chunk missing")?;
    ensure!(pcm.len() % 2 == 0, "unaligned PCM samples");
    Ok((hex::encode(Sha256::digest(pcm)), pcm.len() / 2))
}

fn verify_corpus(
    manifest_path: &Path,
    input: &Path,
    output: &Path,
    fresh: bool,
) -> Result<(PathBuf, Vec<Interval>)> {
    let manifest_path = manifest_path
        .canonicalize()
        .context("corpus manifest unavailable")?;
    let base = manifest_path
        .parent()
        .context("corpus manifest has no parent")?;
    let manifest: CorpusManifest = read_json(&manifest_path)?;
    ensure!(
        manifest.schema_version == 1 && manifest.audio_seconds == 1800.0,
        "unsupported corpus manifest or duration"
    );
    let audio = match (manifest.wav, manifest.audio_path, manifest.audio_sha256) {
        (Some(wav), None, None) => wav,
        (None, Some(path), Some(sha256)) => CorpusFile { path, sha256 },
        _ => bail!("corpus manifest requires either wav or audio_path and audio_sha256"),
    };
    let wav = corpus_path(base, &audio.path)?;
    ensure!(
        hash_file(&wav)? == audio.sha256,
        "corpus source WAV hash mismatch"
    );
    let actual_audio_seconds = audio::duration(&wav)?;
    ensure!(
        manifest.audio_seconds <= actual_audio_seconds,
        "corpus selected prefix exceeds source WAV duration"
    );
    if let Some(reported_duration) = manifest.audio_duration_seconds {
        ensure!(
            reported_duration.is_finite()
                && (reported_duration - actual_audio_seconds).abs() <= 1.0 / 16000.0,
            "corpus audio_duration_seconds does not match source WAV"
        );
    }
    if let Some(source_sha256) = manifest.source_sha256 {
        ensure!(
            hash_file(input)? == source_sha256,
            "corpus source input hash mismatch"
        );
    } else {
        ensure!(
            input.canonicalize()? == wav,
            "without source_sha256, benchmark input must be the manifest audio_path WAV"
        );
    }
    let expected = audio::chunks(1800.0, 28.0, 0.8)?;
    ensure!(
        manifest.windows.len() == expected.len(),
        "corpus window count mismatch"
    );
    let mut clips: Option<PathBuf> = None;
    for (i, (window, interval)) in manifest.windows.iter().zip(&expected).enumerate() {
        ensure!(
            window.index == i
                && (window.start - interval.start).abs() < 1e-6
                && (window.end - interval.end).abs() < 1e-6,
            "corpus window index or boundaries mismatch"
        );
        let path = corpus_path(base, &window.wav_path)?;
        ensure!(
            path.file_name().is_some_and(
                |name| name == std::ffi::OsStr::new(&format!("{:05}.wav", window.index))
            ),
            "corpus clip filename mismatch"
        );
        ensure!(
            clips
                .as_deref()
                .is_none_or(|dir| path.parent() == Some(dir)),
            "corpus clips must share a directory"
        );
        clips = Some(path.parent().unwrap().to_path_buf());
        ensure!(
            hash_file(&path)? == window.wav_sha256,
            "corpus window {} WAV hash mismatch",
            window.index
        );
        let (pcm_hash, samples) = pcm_sha256(&path)?;
        ensure!(
            pcm_hash == window.pcm_sha256 && samples == window.samples,
            "corpus window {} PCM hash or sample count mismatch",
            window.index
        );
        ensure!(
            (samples as f64 / 16000.0 - (window.end - window.start)).abs() <= 1.0 / 16000.0,
            "corpus window {} PCM duration mismatch",
            window.index
        );
    }
    let clips = clips.context("corpus has no clips")?;
    let output_parent = output.parent().unwrap_or(Path::new(".")).canonicalize()?;
    let destination = output_parent.join(
        output
            .file_name()
            .context("benchmark output must name a directory")?,
    );
    ensure!(
        !manifest_path.starts_with(&destination)
            && !wav.starts_with(&destination)
            && !clips.starts_with(&destination)
            && !destination.starts_with(&clips)
            && !destination.starts_with(&wav),
        "benchmark output must be separate from corpus files"
    );
    ensure!(
        !fresh || !output.exists(),
        "prepared-corpus benchmark output must be a new directory"
    );
    Ok((clips, expected))
}

#[derive(Debug, Serialize)]
struct WindowTiming {
    worker_id: usize,
    index: usize,
    sent_offset_ms: u128,
    received_offset_ms: u128,
    latency_ms: u128,
    completion_order: usize,
}

#[derive(Debug, Serialize)]
struct WorkerTiming {
    worker_id: usize,
    pid: u32,
    start_offset_ms: u128,
    ready_offset_ms: Option<u128>,
    first_result_offset_ms: Option<u128>,
    finish_offset_ms: Option<u128>,
}

#[derive(Default)]
struct BenchmarkEvents {
    windows: Vec<WindowTiming>,
    failed_indices: BTreeSet<usize>,
    workers: BTreeMap<usize, WorkerTiming>,
    pids: BTreeSet<u32>,
    samples: Vec<resources::BenchmarkSample>,
    monitoring_error: Option<String>,
    resource_stop_reason: Option<String>,
}

struct BenchmarkTelemetry {
    started: Instant,
    deadline: Instant,
    stop: AtomicBool,
    events: Mutex<BenchmarkEvents>,
}

impl BenchmarkTelemetry {
    fn offset(&self) -> u128 {
        self.started.elapsed().as_millis()
    }

    fn fail(&self, index: usize) {
        self.events
            .lock()
            .expect("benchmark events mutex poisoned")
            .failed_indices
            .insert(index);
    }

    fn record_sample(
        &self,
        sample: resources::BenchmarkSample,
        reserve: u64,
        baseline_pswpout: Option<u64>,
        baseline_own_vmswap_kib: u64,
    ) -> Result<()> {
        let new_swap_out =
            resources::new_system_swap_out(baseline_pswpout, sample.system_pswpout_pages)?;
        let low_memory = sample.mem_available_mib < reserve;
        let own_swap_growth = sample.own_vmswap_kib > baseline_own_vmswap_kib;
        let system_pressure = new_swap_out
            && sample.psi_memory_full_avg10_percent
                >= resources::MEMORY_FULL_AVG10_THRESHOLD_PERCENT;
        let mut events = self.events.lock().expect("benchmark events mutex poisoned");
        if low_memory {
            events.resource_stop_reason = Some(format!(
                "effective MemAvailable/cgroup headroom {} MiB fell below {reserve} MiB reserve",
                sample.mem_available_mib
            ));
        } else if own_swap_growth {
            events.resource_stop_reason = Some(format!(
                "worker process-tree VmSwap {} KiB exceeded prelaunch baseline {baseline_own_vmswap_kib} KiB",
                sample.own_vmswap_kib
            ));
        } else if system_pressure {
            events.resource_stop_reason = Some(format!(
                "memory PSI full avg10 {}% reached 1.0% with new system-wide pswpout; threshold is not causal attribution to workers",
                sample.psi_memory_full_avg10_percent
            ));
        }
        events.samples.push(sample);
        if system_pressure || own_swap_growth || low_memory {
            self.stop.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    fn record_result(
        &self,
        worker_id: usize,
        index: usize,
        sent: Instant,
        received: Instant,
    ) -> Result<()> {
        ensure!(
            received < self.deadline && !self.stop.load(Ordering::Relaxed),
            "late benchmark result excluded"
        );
        let mut events = self.events.lock().expect("benchmark events mutex poisoned");
        ensure!(
            !events.windows.iter().any(|window| window.index == index),
            "duplicate benchmark result {index}"
        );
        let order = events.windows.len();
        events.windows.push(WindowTiming {
            worker_id,
            index,
            sent_offset_ms: sent.duration_since(self.started).as_millis(),
            received_offset_ms: received.duration_since(self.started).as_millis(),
            latency_ms: received.duration_since(sent).as_millis(),
            completion_order: order,
        });
        if let Some(worker) = events.workers.get_mut(&worker_id) {
            worker
                .first_result_offset_ms
                .get_or_insert(received.duration_since(self.started).as_millis());
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct WaitingResources {
    role: String,
    available_mib: u64,
    settings: crate::config::Resources,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordSpan {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub text: String,
    pub language: Option<String>,
    #[serde(default)]
    pub segments: Vec<WordSpan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub out_of_bounds_segments: Vec<WordSpan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub processing_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diarization {
    pub turns: Vec<Turn>,
    pub exclusive_turns: Vec<Turn>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FinalSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: Option<String>,
    pub text: String,
    pub language: String,
    pub gap: bool,
    pub hypotheses: Vec<(String, Hypothesis)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Transcript {
    pub schema_version: u32,
    pub source_sha256: String,
    pub duration: f64,
    pub quality_status: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub limitations: Vec<String>,
    pub models: Vec<crate::models::ModelSpec>,
    pub segments: Vec<FinalSegment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct JobManifest {
    pub(crate) schema_version: u32,
    pub(crate) input_sha256: String,
    pub(crate) models: Vec<crate::models::ModelSpec>,
    pub(crate) asr: BTreeMap<String, Adapter>,
    pub(crate) asr_script_sha256: BTreeMap<String, String>,
    pub(crate) diarization: Adapter,
    pub(crate) diarization_script_sha256: String,
    pub(crate) llm: crate::config::Llm,
    #[serde(default)]
    pub(crate) review_rules: Option<PathBuf>,
    pub(crate) review_rules_sha256: Option<String>,
    #[serde(
        default = "crate::config::default_asr_roles",
        skip_serializing_if = "crate::config::is_default_asr_roles"
    )]
    pub(crate) asr_roles: Vec<String>,
}

#[cfg(test)]
fn same_job_after_model_move(existing: &JobManifest, current: &JobManifest) -> bool {
    if existing.models.len() != current.models.len() {
        return false;
    }
    let mut normalized = existing.clone();
    for (old, new) in normalized.models.iter_mut().zip(&current.models) {
        if old.role != new.role || old.repo != new.repo || old.revision != new.revision {
            return false;
        }
        old.path = new.path.clone();
    }
    normalized == *current
}

#[derive(Debug, Serialize, Deserialize)]
struct Decision {
    index: usize,
    text: String,
    language: String,
    gap: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct Decisions {
    segments: Vec<Decision>,
}

#[derive(Serialize)]
struct ReviewFlag {
    index: usize,
    flags: Vec<&'static str>,
}

#[derive(Serialize)]
struct BenchmarkReport<'a> {
    role: &'a str,
    engine: &'a str,
    audio_seconds: f64,
    elapsed_seconds: f64,
    real_time_factor: f64,
    window_count: usize,
    worker_count: usize,
    estimated_worker_peak_mib: Option<u64>,
    reserve_mib: u64,
    resource_guard_semantics: &'static str,
    memory_full_avg10_threshold_percent: f64,
    empty_window_count: usize,
    total_characters: usize,
    offload: bool,
    threads: usize,
    max_in_flight: usize,
    adapter_args: &'a [String],
    model_revision: &'a str,
    adapter_sha256: String,
    under_ten_minutes: bool,
    status: &'a str,
    stop_reason: Option<String>,
    completed_indices: Vec<usize>,
    missing_indices: Vec<usize>,
    failed_indices: Vec<usize>,
    preparation_seconds: f64,
    process_cleanup_seconds: f64,
    report_write_offset_ms: u128,
    workers: Vec<WorkerTiming>,
    windows: Vec<WindowTiming>,
    resource_samples: Vec<resources::BenchmarkSample>,
    peak_aggregate_rss_mib: Option<u64>,
    min_mem_available_mib: Option<u64>,
    corpus_manifest: Option<&'a Path>,
    worker_ready_semantics: &'static str,
}

const RESOURCE_GUARD_SEMANTICS: &str = "Stop below 4096 MiB effective MemAvailable/cgroup reserve, on worker process-tree VmSwap growth over prelaunch baseline, or when memory PSI full avg10 reaches 1.0% together with new system pswpout. System pswpout alone is telemetry, not worker attribution. Counter/read errors stop for safety; thresholds do not establish causal attribution.";

#[derive(Serialize)]
struct DiarizationBenchmarkReport<'a> {
    role: &'a str,
    audio_seconds: f64,
    elapsed_seconds: f64,
    real_time_factor: f64,
    turn_count: usize,
    threads: usize,
    adapter_args: &'a [String],
    model_revision: &'a str,
    adapter_sha256: String,
    under_ten_minutes: bool,
}

fn meets_30min_target(elapsed_seconds: f64) -> bool {
    elapsed_seconds < 600.0
}

fn benchmark_status(
    failures: &[String],
    missing: &[usize],
    elapsed: f64,
    resource_stop: bool,
) -> &'static str {
    if resource_stop {
        "resource_stop"
    } else if failures
        .iter()
        .any(|error| error.contains("600-second") || error.contains("late benchmark"))
    {
        "timeout"
    } else if !failures.is_empty() {
        "worker_error"
    } else if !missing.is_empty() || !meets_30min_target(elapsed) {
        "timeout"
    } else {
        "success"
    }
}

pub(crate) fn repeated_fourgram_fraction(text: &str) -> f64 {
    let words: Vec<_> = text.split_whitespace().map(str::to_lowercase).collect();
    let groups = words.windows(4);
    let count = groups.len();
    if count == 0 {
        return 0.0;
    }
    let unique = groups.collect::<BTreeSet<_>>().len();
    (count - unique) as f64 / count as f64
}

fn review_flags(
    chunk: &Interval,
    hypotheses: &[(String, Hypothesis)],
    diarization: &Diarization,
) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if hypotheses
        .iter()
        .any(|(_, hypothesis)| repeated_fourgram_fraction(&hypothesis.text) > 0.5)
    {
        flags.push("repeated_asr_text");
    }
    let speech = diarization
        .turns
        .iter()
        .any(|turn| turn.start < chunk.end && turn.end > chunk.start);
    if !speech
        && hypotheses
            .iter()
            .filter(|(_, hypothesis)| !hypothesis.text.trim().is_empty())
            .count()
            == 1
    {
        flags.push("single_asr_hypothesis_without_diarized_speech");
    }
    flags
}

fn prepare_benchmark_audio(input: &Path, output: &Path) -> Result<(PathBuf, PathBuf)> {
    const AUDIO_SECONDS: f64 = 1800.0;
    ensure!(
        input
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("mp3")),
        "benchmark input must be MP3"
    );
    let input = input
        .canonicalize()
        .context("benchmark MP3 is unavailable")?;
    ensure!(
        audio::duration(&input)? >= AUDIO_SECONDS,
        "benchmark requires at least 30 minutes of audio"
    );
    ensure!(
        !output.exists() || fs::read_dir(output)?.next().is_none(),
        "benchmark output must be a new or empty directory (cached results invalidate timing)"
    );
    fs::create_dir_all(output)?;
    let output = output.canonicalize()?;
    let wav = output.join("audio.wav");
    audio::normalize(&input, &wav)?;
    Ok((output, wav))
}

pub struct BenchmarkOptions<'a> {
    pub offload: bool,
    pub workers: usize,
    pub worker_peak_mib: Option<u64>,
    pub corpus_manifest: Option<&'a Path>,
}

pub fn benchmark_asr(
    config: &Config,
    input: &Path,
    output: &Path,
    role: &str,
    options: BenchmarkOptions<'_>,
) -> Result<()> {
    let BenchmarkOptions {
        offload,
        workers,
        worker_peak_mib,
        corpus_manifest,
    } = options;
    const AUDIO_SECONDS: f64 = 1800.0;
    let engine = config.asr_engine(role)?;
    ensure!(
        !offload
            || config
                .resources
                .offload_roles
                .iter()
                .any(|allowed| allowed == role),
        "disk offload is not enabled for {role} in resources.offload_roles"
    );
    ensure!(
        (1..=4).contains(&workers),
        "benchmark workers must be between 1 and 4"
    );
    ensure!(
        worker_peak_mib.is_none_or(|value| value > 0),
        "--worker-peak-mib must be positive"
    );
    let configured_peak = if offload {
        config.resources.offload_estimated_peak_mib.get(role)
    } else {
        config.resources.estimated_peak_mib.get(role)
    };
    let estimate = worker_peak_mib.or(configured_peak.copied());
    let reserve = config.resources.reserve_mib.max(4096);
    if workers > 1 {
        worker_peak_mib.context(
            "multiple benchmark workers require an explicit --worker-peak-mib measured for this engine",
        )?;
    }
    crate::models::verify_local(config.model(role))?;
    let adapter_sha256 = hash_file(&config.asr(role).script)?;
    let prep_started = Instant::now();
    let corpus_hash = corpus_manifest.map(hash_file).transpose()?;
    let (output, clips, chunks) = if let Some(manifest) = corpus_manifest {
        let (clips, chunks) = verify_corpus(manifest, input, output, true)?;
        fs::create_dir_all(output)?;
        (output.canonicalize()?, clips, chunks)
    } else {
        let (output, wav) = prepare_benchmark_audio(input, output)?;
        let chunks = audio::chunks(AUDIO_SECONDS, 28.0, 0.8)?;
        let clips = output.join("clips");
        fs::create_dir_all(&clips)?;
        for (i, chunk) in chunks.iter().enumerate() {
            audio::extract(&wav, chunk, &clips, i)?;
        }
        (output, clips, chunks)
    };
    let preparation_seconds = prep_started.elapsed().as_secs_f64();
    let required = estimate
        .context("benchmark requires an estimated worker peak")?
        .saturating_mul(workers as u64)
        .saturating_add(reserve);
    let preflight = resources::benchmark_sample(0, &[]);
    let admission = preflight
        .as_ref()
        .ok()
        .map(|sample| sample.mem_available_mib);
    let reason = match &preflight {
        Err(error) => Some(format!(
            "resource sampling failed before worker launch: {error:#}"
        )),
        Ok(sample) if sample.mem_available_mib < required => Some(format!(
            "resource stop before worker launch: required {required} MiB including {reserve} MiB reserve, available {} MiB",
            sample.mem_available_mib
        )),
        Ok(_) => None,
    };
    if let Some(reason) = reason {
        let report = BenchmarkReport {
            role,
            engine,
            audio_seconds: AUDIO_SECONDS,
            elapsed_seconds: 0.0,
            real_time_factor: 0.0,
            window_count: chunks.len(),
            worker_count: workers,
            estimated_worker_peak_mib: estimate,
            reserve_mib: reserve,
            resource_guard_semantics: RESOURCE_GUARD_SEMANTICS,
            memory_full_avg10_threshold_percent: resources::MEMORY_FULL_AVG10_THRESHOLD_PERCENT,
            empty_window_count: 0,
            total_characters: 0,
            offload,
            threads: config.asr(role).threads,
            max_in_flight: config.asr(role).max_in_flight,
            adapter_args: &config.asr(role).args,
            model_revision: &config.model(role).revision,
            adapter_sha256,
            under_ten_minutes: false,
            status: "resource_stop",
            stop_reason: Some(reason.clone()),
            completed_indices: vec![],
            missing_indices: (0..chunks.len()).collect(),
            failed_indices: vec![],
            preparation_seconds,
            process_cleanup_seconds: 0.0,
            report_write_offset_ms: 0,
            workers: vec![],
            windows: vec![],
            resource_samples: preflight.into_iter().collect(),
            peak_aggregate_rss_mib: None,
            min_mem_available_mib: admission,
            corpus_manifest,
            worker_ready_semantics: "unavailable on ASR wire; ready_offset_ms is null",
        };
        write_json(&output.join("benchmark.json"), &report)?;
        bail!(
            "{reason}; benchmark saved at {}",
            output.join("benchmark.json").display()
        );
    }
    let initial_sample = preflight?;
    let baseline_pswpout = initial_sample.system_pswpout_pages;
    let baseline_own_vmswap_kib = initial_sample.own_vmswap_kib;
    let started = Instant::now();
    let deadline = started + Duration::from_secs(600);
    let telemetry = BenchmarkTelemetry {
        started,
        deadline,
        stop: AtomicBool::new(false),
        events: Mutex::new(BenchmarkEvents {
            samples: vec![initial_sample],
            ..BenchmarkEvents::default()
        }),
    };
    let failures = thread::scope(|scope| {
        let telemetry_ref = &telemetry;
        let sampler = scope.spawn(move || {
            while !telemetry_ref.stop.load(Ordering::Relaxed) {
                let pids: Vec<_> = telemetry_ref
                    .events
                    .lock()
                    .expect("benchmark events mutex poisoned")
                    .pids
                    .iter()
                    .copied()
                    .collect();
                match resources::benchmark_sample(telemetry_ref.offset(), &pids) {
                    Ok(sample) => {
                        if let Err(error) = telemetry_ref.record_sample(
                            sample,
                            reserve,
                            baseline_pswpout,
                            baseline_own_vmswap_kib,
                        ) {
                            telemetry_ref
                                .events
                                .lock()
                                .expect("benchmark events mutex poisoned")
                                .monitoring_error = Some(format!("{error:#}"));
                            telemetry_ref.stop.store(true, Ordering::Relaxed);
                        }
                    }
                    Err(error) => {
                        telemetry_ref
                            .events
                            .lock()
                            .expect("benchmark events mutex poisoned")
                            .monitoring_error = Some(format!("{error:#}"));
                        telemetry_ref.stop.store(true, Ordering::Relaxed);
                    }
                }
                thread::sleep(Duration::from_millis(200));
            }
        });
        let clips_ref = clips.as_path();
        let output_ref = output.as_path();
        let chunks_ref = chunks.as_slice();
        let handles: Vec<_> = (0..workers)
            .map(|worker_number| {
                scope.spawn(move || {
                    run_asr(
                        config,
                        role,
                        engine,
                        clips_ref,
                        output_ref,
                        chunks_ref,
                        WorkerMode {
                            concurrent: workers > 1,
                            offload,
                            worker_number,
                            worker_count: workers,
                            deadline: Some(deadline),
                            benchmark: Some(telemetry_ref),
                            execution: None,
                            retry: None,
                            task: None,
                        },
                    )
                })
            })
            .collect();
        let mut failures = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Ok(WorkerOutcome::Complete)) => {}
                Ok(Ok(WorkerOutcome::OutOfMemory)) => failures.push("worker killed by OOM".into()),
                Ok(Ok(WorkerOutcome::Cancelled)) => failures.push("worker cancelled".into()),
                Ok(Err(error)) => failures.push(format!("{error:#}")),
                Err(_) => failures.push("benchmark worker panicked".into()),
            }
        }
        telemetry.stop.store(true, Ordering::Relaxed);
        if sampler.join().is_err() {
            failures.push("resource sampler panicked".into());
        }
        failures
    });
    let total_elapsed = started.elapsed().as_secs_f64();
    let mut failures = failures;
    if let Err(error) = hash_file(&config.asr(role).script).and_then(|hash| {
        ensure!(
            hash == adapter_sha256,
            "ASR adapter script changed during benchmark"
        );
        Ok(())
    }) {
        failures.push(format!("{error:#}"));
    }
    if let Some(manifest) = corpus_manifest {
        match verify_corpus(manifest, input, &output, false).and_then(|_| hash_file(manifest)) {
            Ok(hash) if Some(&hash) == corpus_hash.as_ref() => {}
            Ok(_) => failures.push("corpus manifest changed during benchmark".into()),
            Err(error) => failures.push(format!("corpus changed during benchmark: {error:#}")),
        }
    }
    let mut empty_window_count = 0;
    let mut total_characters = 0;
    let mut events = telemetry
        .events
        .into_inner()
        .expect("benchmark events mutex poisoned");
    let completed: BTreeSet<_> = events.windows.iter().map(|window| window.index).collect();
    for (i, chunk) in chunks
        .iter()
        .enumerate()
        .filter(|(i, _)| completed.contains(i))
    {
        match read_json::<Hypothesis>(&output.join(role).join(format!("{i:05}.json"))).and_then(
            |hypothesis| {
                validate_hypothesis(&hypothesis, chunk.end - chunk.start)?;
                Ok(hypothesis)
            },
        ) {
            Ok(hypothesis) => {
                empty_window_count += usize::from(hypothesis.text.trim().is_empty());
                total_characters += hypothesis.text.chars().count();
            }
            Err(error) => failures.push(format!("window {i}: {error:#}")),
        }
    }
    let missing: Vec<_> = (0..chunks.len())
        .filter(|i| !completed.contains(i))
        .collect();
    let elapsed = if missing.is_empty() {
        events
            .windows
            .iter()
            .map(|window| window.received_offset_ms)
            .max()
            .unwrap_or(0) as f64
            / 1000.0
    } else {
        total_elapsed
    };
    let monitoring_failed = events.monitoring_error.is_some();
    let resource_stop_reason = events.resource_stop_reason.take();
    if let Some(error) = events.monitoring_error.take() {
        failures.push(error);
    }
    if let Some(reason) = &resource_stop_reason {
        failures.push(reason.clone());
    }
    let status = benchmark_status(
        &failures,
        &missing,
        elapsed,
        monitoring_failed
            || resource_stop_reason.is_some()
            || events
                .samples
                .iter()
                .any(|sample| sample.mem_available_mib < reserve),
    );
    let peak_rss = events
        .samples
        .iter()
        .filter(|sample| sample.tracked_process_count > 0)
        .map(|sample| sample.aggregate_rss_mib)
        .max();
    let min_available = events.samples.iter().map(|s| s.mem_available_mib).min();
    let cleanup_seconds = total_elapsed
        - events
            .windows
            .iter()
            .map(|window| window.received_offset_ms)
            .max()
            .unwrap_or(0) as f64
            / 1000.0;
    let report = BenchmarkReport {
        role,
        engine,
        audio_seconds: AUDIO_SECONDS,
        elapsed_seconds: elapsed,
        real_time_factor: elapsed / AUDIO_SECONDS,
        window_count: chunks.len(),
        worker_count: workers,
        estimated_worker_peak_mib: estimate,
        reserve_mib: reserve,
        resource_guard_semantics: RESOURCE_GUARD_SEMANTICS,
        memory_full_avg10_threshold_percent: resources::MEMORY_FULL_AVG10_THRESHOLD_PERCENT,
        empty_window_count,
        total_characters,
        offload,
        threads: config.asr(role).threads,
        max_in_flight: config.asr(role).max_in_flight,
        adapter_args: &config.asr(role).args,
        model_revision: &config.model(role).revision,
        adapter_sha256,
        under_ten_minutes: status == "success",
        status,
        stop_reason: if status == "success" {
            None
        } else if failures.is_empty() {
            Some("incomplete windows or 600-second benchmark limit".into())
        } else {
            Some(failures.join("; "))
        },
        completed_indices: completed.iter().copied().collect(),
        missing_indices: missing,
        failed_indices: events.failed_indices.into_iter().collect(),
        preparation_seconds,
        process_cleanup_seconds: cleanup_seconds,
        report_write_offset_ms: started.elapsed().as_millis(),
        workers: events.workers.into_values().collect(),
        windows: events.windows,
        resource_samples: events.samples,
        peak_aggregate_rss_mib: peak_rss,
        min_mem_available_mib: min_available,
        corpus_manifest,
        worker_ready_semantics: "unavailable on ASR wire; ready_offset_ms is null",
    };
    write_json(&output.join("benchmark.json"), &report)?;
    println!(
        "{role}: 30 minutes audio in {elapsed:.1} s; RTF {:.3}; under 10 min: {}",
        report.real_time_factor, report.under_ten_minutes
    );
    ensure!(
        report.under_ten_minutes,
        "benchmark {status}: {}; saved at {}",
        report
            .stop_reason
            .as_deref()
            .unwrap_or("incomplete or exceeded deadline"),
        output.join("benchmark.json").display()
    );
    Ok(())
}

pub fn benchmark_diarization(config: &Config, input: &Path, output: &Path) -> Result<()> {
    const AUDIO_SECONDS: f64 = 1800.0;
    crate::models::verify_local(config.model("diarization"))?;
    let adapter_sha256 = hash_file(&config.diarization.script)?;
    let (output, wav) = prepare_benchmark_audio(input, output)?;
    let clips = output.join("clips");
    fs::create_dir_all(&clips)?;
    let sample = audio::extract(
        &wav,
        &Interval {
            start: 0.0,
            end: AUDIO_SECONDS,
        },
        &clips,
        0,
    )?;
    let result = output.join("diarization.partial.json");
    let mut command = Command::new(&config.diarization.program);
    command
        .arg(&config.diarization.script)
        .args(&config.diarization.args)
        .arg("--model")
        .arg(&config.model("diarization").path)
        .arg("--input")
        .arg(&sample)
        .arg("--output")
        .arg(&result)
        .arg("--threads")
        .arg(config.diarization.threads.to_string());
    let started = std::time::Instant::now();
    process::run(
        &mut command,
        &output.join("diarization.stdout"),
        &output.join("diarization.stderr"),
        Duration::from_secs(600),
    )?;
    let elapsed = started.elapsed().as_secs_f64();
    ensure!(
        hash_file(&config.diarization.script)? == adapter_sha256,
        "diarization adapter script changed during benchmark; timing provenance is invalid"
    );
    let turns: Diarization = read_json(&result)?;
    validate_diarization(&turns, AUDIO_SECONDS)?;
    fs::rename(result, output.join("diarization.json"))?;
    let report = DiarizationBenchmarkReport {
        role: "diarization",
        audio_seconds: AUDIO_SECONDS,
        elapsed_seconds: elapsed,
        real_time_factor: elapsed / AUDIO_SECONDS,
        turn_count: turns.turns.len(),
        threads: config.diarization.threads,
        adapter_args: &config.diarization.args,
        model_revision: &config.model("diarization").revision,
        adapter_sha256,
        under_ten_minutes: meets_30min_target(elapsed),
    };
    write_json(&output.join("benchmark.json"), &report)?;
    println!(
        "diarization: 30 minutes audio in {elapsed:.1} s; RTF {:.3}; under 10 min: {}",
        report.real_time_factor, report.under_ten_minutes
    );
    ensure!(
        report.under_ten_minutes,
        "diarization exceeded the required 10-minute limit; benchmark saved at {}",
        output.join("benchmark.json").display()
    );
    Ok(())
}

#[cfg(test)]
pub fn transcribe(
    config: &Config,
    input: &Path,
    output: &Path,
    sequential_asr: bool,
) -> Result<()> {
    ensure!(
        transcribe_outcome(config, input, output, sequential_asr)? == Outcome::Complete,
        "project is waiting for LLM review"
    );
    Ok(())
}

pub fn transcribe_outcome(
    config: &Config,
    input: &Path,
    output: &Path,
    sequential_asr: bool,
) -> Result<Outcome> {
    transcribe_step(config, input, output, sequential_asr, None)
}

pub(crate) fn transcribe_step(
    config: &Config,
    input: &Path,
    output: &Path,
    sequential_asr: bool,
    rerun: Option<&str>,
) -> Result<Outcome> {
    transcribe_selected(config, input, output, sequential_asr, rerun, None)
}

pub(crate) fn transcribe_selected(
    config: &Config,
    input: &Path,
    output: &Path,
    sequential_asr: bool,
    rerun: Option<&str>,
    target: Option<workflow::Step>,
) -> Result<Outcome> {
    ensure!(
        input
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("mp3")),
        "input must be MP3"
    );
    let original_input = input.to_path_buf();
    let input = input.canonicalize().context("input MP3 is unavailable")?;
    let hash = hash_file(&input)?;
    fs::create_dir_all(output).with_context(|| format!("cannot create {}", output.display()))?;
    let output = output
        .canonicalize()
        .context("job directory is unavailable")?;
    let meta = output.join("manifest.json");
    let legacy = if meta.exists() {
        Some(read_json::<JobManifest>(&meta)?)
    } else {
        None
    };
    let mut asr_roles = config.select_asr_roles(config.asr_selection.as_deref())?;
    for step in target
        .as_ref()
        .map(|step| step.as_str())
        .into_iter()
        .chain(rerun)
    {
        if let Some(role) = step.strip_prefix("asr:") {
            ensure!(
                asr_roles.iter().any(|selected| selected == role),
                "{step} is outside the project ASR selection; include {role} with --asr"
            );
        }
    }
    let selected = workflow::prerequisite_steps_for_roles(target.as_ref(), &asr_roles);
    let asr_script_sha256 = config
        .asr
        .iter()
        .map(|(role, adapter)| {
            let old = legacy.as_ref();
            if !asr_roles.contains(role) {
                return Ok((
                    role.clone(),
                    old.and_then(|manifest| manifest.asr_script_sha256.get(role))
                        .cloned()
                        .unwrap_or_else(|| "not-selected".into()),
                ));
            }
            Ok((
                role.clone(),
                stage_file_hash(
                    &adapter.script,
                    old.and_then(|m| m.asr.get(role))
                        .map(|a| a.script.as_path()),
                    old.and_then(|m| m.asr_script_sha256.get(role))
                        .map(String::as_str),
                )?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let manifest = JobManifest {
        schema_version: 1,
        input_sha256: hash.clone(),
        models: config.models.clone(),
        asr: config.asr.clone(),
        asr_script_sha256,
        diarization: config.diarization.clone(),
        diarization_script_sha256: stage_file_hash(
            &config.diarization.script,
            legacy.as_ref().map(|m| m.diarization.script.as_path()),
            legacy
                .as_ref()
                .map(|m| m.diarization_script_sha256.as_str()),
        )?,
        llm: config.llm.clone(),
        review_rules: config.review_rules.clone(),
        review_rules_sha256: config
            .review_rules
            .as_deref()
            .map(|path| {
                stage_file_hash(
                    path,
                    legacy.as_ref().and_then(|m| m.review_rules.as_deref()),
                    legacy
                        .as_ref()
                        .and_then(|m| m.review_rules_sha256.as_deref()),
                )
            })
            .transpose()?,
        asr_roles: asr_roles.clone(),
    };
    let mut job =
        workflow::Project::open(&output, &original_input, &manifest, legacy.as_ref(), rerun)?;
    let _interrupt_guard = process::InterruptGuard::install()?;
    let execution = job.execution();
    execution.set_mode_steps(&selected)?;
    execution.set_metadata(config)?;
    let duration = match audio::duration(&input) {
        Ok(duration) => duration,
        Err(error) => {
            job.warning("audio", false, &format!("{error:#}"))?;
            block_required_descendants(&job, "audio", false, "required audio could not be read")?;
            return Err(error);
        }
    };
    execution.set_duration(duration)?;
    let planned_chunks = audio::chunks(duration, 28.0, 0.8)?;
    let audio_total_ms = (duration * 1000.0).ceil() as u64;
    let fragment_ids: Vec<_> = (0..planned_chunks.len()).collect();
    execution.set_units("chunks", &fragment_ids, "fragments")?;
    for role in &asr_roles {
        execution.set_units(&format!("asr:{role}"), &fragment_ids, "fragments")?;
    }
    execution.set_units("review", &fragment_ids, "validated replies")?;
    execution.set_units("review-flags", &fragment_ids, "inspected fragments")?;
    execution.set_units("export", &[0, 1], "verified files")?;
    let terminal = crate::terminal::start(execution.clone())?;
    let outcome = (|| -> Result<Outcome> {
        let _quality_refresh = crate::quality::RefreshOnExit {
            project: &output,
            settings: &config.quality,
        };
        for (role, adapter) in &config.asr {
            if !asr_roles.contains(role) {
                continue;
            }
            let step = format!("asr:{role}");
            if File::open(&adapter.script).is_err() {
                job.cached_warning(
                    &step,
                    "ASR adapter is unavailable; using validated cached output",
                )?;
            }
            if config
                .models
                .iter()
                .find(|m| m.role == *role)
                .is_some_and(|m| !m.path.exists())
            {
                job.cached_warning(
                    &step,
                    "ASR model is unavailable; using validated cached output",
                )?;
            }
        }
        if File::open(&config.diarization.script).is_err() {
            job.cached_warning(
                "diarization",
                "diarization adapter is unavailable; using validated cached output",
            )?;
        }
        if config
            .models
            .iter()
            .find(|m| m.role == "diarization")
            .is_some_and(|m| !m.path.exists())
        {
            job.cached_warning(
                "diarization",
                "diarization model is unavailable; using validated cached output",
            )?;
        }
        if config
            .review_rules
            .as_ref()
            .is_some_and(|path| File::open(path).is_err())
        {
            job.cached_warning(
                "review",
                "review rules are unavailable; using validated cached review",
            )?;
        }
        write_json(&meta, &manifest)?;
        let wav = output.join("audio.wav");
        if execution.cancelled("audio")? || process::interrupted() {
            return required_stage_outcome(
                &job,
                "audio",
                "required audio preparation was cancelled",
            );
        }
        if !job.ready("audio") {
            let result = (|| -> Result<()> {
                job.begin("audio")?;
                let partial = output.join("audio.partial.wav");
                let task = task_context(&execution, "audio", Some(audio_total_ms));
                audio::normalize_observed(&input, &partial, Some(&task))?;
                fs::rename(partial, &wav)?;
                execution.numeric_progress(
                    "audio",
                    audio_total_ms,
                    Some(audio_total_ms),
                    "audio ms",
                )?;
                job.finish("audio")
            })();
            if let Err(error) = result {
                if error.is::<process::TaskCancelled>()
                    || execution.cancelled("audio")?
                    || process::interrupted()
                {
                    return required_stage_outcome(
                        &job,
                        "audio",
                        "required audio preparation was cancelled",
                    );
                }
                job.warning("audio", false, &format!("{error:#}"))?;
                block_required_descendants(
                    &job,
                    "audio",
                    false,
                    "required audio preparation failed",
                )?;
                return Err(error);
            }
        }
        if target == Some(workflow::Step::Audio) {
            return checkpoint_with_checks(
                &mut job,
                config,
                &output,
                "audio",
                Outcome::StageComplete,
            );
        }
        let chunks = if selected.contains("chunks") {
            let result = (|| -> Result<Vec<Interval>> {
                let chunks_path = output.join("chunks.json");
                let chunks: Vec<Interval> = if job.ready("chunks") {
                    read_json(&chunks_path)?
                } else {
                    job.begin("chunks")?;
                    let chunks = audio::chunks(duration, 28.0, 0.8)?;
                    write_json(&chunks_path, &chunks)?;
                    chunks
                };
                let clips = output.join("clips");
                crate::quality::auto_refresh(&output, &config.quality);
                fs::create_dir_all(&clips)?;
                for (i, interval) in chunks.iter().enumerate() {
                    if execution.cancelled("chunks")? || process::interrupted() {
                        return Err(process::TaskCancelled.into());
                    }
                    let clip = clips.join(format!("{i:05}.wav"));
                    if !clip.exists() {
                        let task = task_context(&execution, "chunks", None);
                        audio::extract_observed(&wav, interval, &clips, i, Some(&task))?;
                    }
                    let (_, samples) = pcm_sha256(&clip)?;
                    ensure!(samples > 0, "saved fragment {i} has no PCM samples");
                    execution.unit_done("chunks", i)?;
                }
                if !job.ready("chunks") {
                    job.finish("chunks")?;
                }
                Ok(chunks)
            })();
            match result {
                Ok(chunks) => chunks,
                Err(error) => {
                    if error.is::<process::TaskCancelled>()
                        || execution.cancelled("chunks")?
                        || process::interrupted()
                    {
                        return required_stage_outcome(
                            &job,
                            "chunks",
                            "required audio fragments were cancelled",
                        );
                    }
                    job.warning("chunks", false, &format!("{error:#}"))?;
                    block_required_descendants(
                        &job,
                        "chunks",
                        false,
                        "required audio fragments failed",
                    )?;
                    return Err(error);
                }
            }
        } else {
            Vec::new()
        };
        let clips = output.join("clips");
        if target == Some(workflow::Step::Chunks) {
            return checkpoint_with_checks(
                &mut job,
                config,
                &output,
                "chunks",
                Outcome::StageComplete,
            );
        }
        if let Err(error) =
            run_pending_asr_retries(&mut job, config, &asr_roles, &clips, &output, &chunks)
        {
            if error.is::<process::TaskCancelled>() {
                return cancellation_outcome(&execution);
            }
            return Err(error);
        }
        let asr_run = (|| -> Result<()> {
            if !asr_roles
                .iter()
                .any(|role| selected.contains(format!("asr:{role}").as_str()))
            {
                return Ok(());
            }
            let waiting_path = output.join("waiting-resources.json");
            if waiting_path.exists() {
                let waiting: WaitingResources = read_json(&waiting_path)?;
                crate::config::validate_asr_roles(std::slice::from_ref(&waiting.role))?;
                if selected.contains(format!("asr:{}", waiting.role).as_str()) {
                    let available = resources::memory()?.available_mib;
                    ensure!(
                        (waiting.settings.systemd_memory_limit_mib.is_none()
                            && available > waiting.available_mib.saturating_add(512))
                            || config.resources != waiting.settings,
                        "job is waiting for resources after OOM in {}; available RAM is {} MiB ({} MiB at failure). Free more RAM or adjust [resources] (a fixed systemd memory limit must be changed), then resume",
                        waiting.role,
                        available,
                        waiting.available_mib
                    );
                    fs::remove_file(&waiting_path)?;
                } else {
                    eprintln!(
                        "Ignoring saved resource wait for unselected ASR role {}",
                        waiting.role
                    );
                }
            }
            let mut remaining = Vec::new();
            for role in &asr_roles {
                let step = format!("asr:{role}");
                if selected.contains(step.as_str())
                    && !execution.cancelled(&step)?
                    && (!job.ready(&step)
                        || (0..chunks.len())
                            .any(|i| !output.join(role).join(format!("{i:05}.json")).exists()))
                {
                    remaining.push(role.as_str());
                }
            }
            let mut model_errors = Vec::new();
            remaining.retain(|role| {
                if (0..chunks.len())
                    .all(|i| output.join(role).join(format!("{i:05}.json")).exists())
                {
                    return true;
                }
                match crate::models::verify_local(config.model(role)) {
                    Ok(()) => true,
                    Err(error) => {
                        model_errors.push((*role, format!("{error:#}")));
                        false
                    }
                }
            });
            for (role, error) in &model_errors {
                job.warning(&format!("asr:{role}"), false, error)?;
            }
            let mut oom_attempts = BTreeMap::<&str, usize>::new();
            let mut oom_reasons = BTreeMap::<&str, String>::new();
            let mut force_one_worker = sequential_asr;
            let mut force_offload = BTreeSet::<&str>::new();
            while !remaining.is_empty() {
                let budget = resources::memory()?;
                let batch = resources::choose_batch(
                    &remaining,
                    budget,
                    &config.resources,
                    force_one_worker,
                )?;
                let mut auto_attempts = BTreeMap::new();
                for (role, _) in &batch {
                    let step = format!("asr:{role}");
                    job.begin(&step)?;
                    let snapshot = execution.snapshot()?;
                    let id = snapshot.stages[&step]
                        .active_attempt
                        .as_ref()
                        .context("ASR stage has no active attempt after begin")?;
                    if snapshot.attempts[id].auto_retry {
                        auto_attempts.insert(*role, (id.clone(), Instant::now()));
                    }
                    let operation = oom_reasons
                        .remove(role)
                        .unwrap_or_else(|| "loading backend / waiting for first result".into());
                    execution.operation(&step, &operation)?;
                }
                let concurrent = batch.iter().map(|(_, count)| count).sum::<usize>() > 1;
                let modes: BTreeMap<_, _> = batch
                    .iter()
                    .map(|(role, count)| {
                        let offload = config.resources.offload_dir.is_some()
                            && *count == 1
                            && config
                                .resources
                                .offload_roles
                                .iter()
                                .any(|allowed| allowed == role)
                            && (force_offload.contains(role)
                                || (batch.len() == 1
                                    && resources::single_worker_peak_mib(role, &config.resources)
                                        > budget
                                            .available_mib
                                            .saturating_sub(config.resources.reserve_mib)));
                        (*role, offload)
                    })
                    .collect();
                let offload_roles: Vec<_> = batch
                    .iter()
                    .map(|(role, _)| *role)
                    .filter(|role| modes[role])
                    .collect();
                resources::record_admission(
                    &output,
                    &batch,
                    &offload_roles,
                    budget,
                    &config.resources,
                    force_one_worker,
                )?;
                let outcomes = thread::scope(|scope| -> Result<Vec<(&str, WorkerOutcome)>> {
                    let intervals = &chunks;
                    let clips_ref = clips.as_path();
                    let output_ref = output.as_path();
                    let execution_ref = &execution;
                    let completions: BTreeMap<_, _> = batch
                        .iter()
                        .map(|(role, count)| {
                            (
                                *role,
                                Arc::new((AtomicU64::new(*count as u64), AtomicBool::new(false))),
                            )
                        })
                        .collect();
                    let mut handles = Vec::new();
                    for (role, worker_count) in &batch {
                        let engine = config.asr_engine(role)?;
                        for worker_number in 0..*worker_count {
                            let offload = modes[role];
                            let completion = Arc::clone(&completions[role]);
                            let auto_attempt = auto_attempts.get(role).cloned();
                            handles.push((
                                *role,
                                scope.spawn(move || {
                                    let step = format!("asr:{role}");
                                    let task = if let Some((id, started)) = &auto_attempt {
                                        auto_asr_task_context(execution_ref, &step, id, *started)
                                    } else {
                                        task_context(execution_ref, &step, None)
                                    };
                                    let result = run_asr(
                                        config,
                                        role,
                                        engine,
                                        clips_ref,
                                        output_ref,
                                        intervals,
                                        WorkerMode {
                                            concurrent,
                                            offload,
                                            worker_number,
                                            worker_count: *worker_count,
                                            deadline: None,
                                            benchmark: None,
                                            execution: Some(execution_ref),
                                            retry: None,
                                            task: Some(&task),
                                        },
                                    )
                                    .or_else(|error| cancelled_worker_or_error(&task, error));
                                    if let Some((id, started)) = &auto_attempt {
                                        execution_ref
                                            .record_runtime(id, elapsed_seconds(*started))?;
                                    }
                                    if !matches!(result, Ok(WorkerOutcome::Complete)) {
                                        completion.1.store(true, Ordering::SeqCst);
                                    }
                                    if completion.0.fetch_sub(1, Ordering::SeqCst) == 1
                                        && !completion.1.load(Ordering::SeqCst)
                                    {
                                        if let Err(error) = task.check_cancelled() {
                                            return cancelled_worker_or_error(&task, error);
                                        }
                                        for (index, chunk) in intervals.iter().enumerate() {
                                            let value: Hypothesis = read_json(
                                                &output_ref
                                                    .join(role)
                                                    .join(format!("{index:05}.json")),
                                            )?;
                                            validate_hypothesis(&value, chunk.end - chunk.start)?;
                                        }
                                        if let Err(error) =
                                            execution_ref.finish_stage(&format!("asr:{role}"))
                                        {
                                            return cancelled_worker_or_error(&task, error);
                                        }
                                    }
                                    result
                                }),
                            ));
                        }
                    }
                    let mut errors = Vec::new();
                    let mut outcomes = Vec::new();
                    for (role, handle) in handles {
                        match handle.join() {
                            Ok(Ok(outcome)) => outcomes.push((role, outcome)),
                            Ok(Err(err)) => errors.push(err),
                            Err(_) => errors.push(anyhow::anyhow!("ASR worker panicked")),
                        }
                    }
                    if let Some(index) = errors
                        .iter()
                        .position(|error| error.is::<process::TaskCancelled>())
                    {
                        return Err(errors.swap_remove(index));
                    }
                    ensure!(
                        errors.is_empty(),
                        "ASR failed: {}",
                        errors
                            .iter()
                            .map(|error| format!("{error:#}"))
                            .collect::<Vec<_>>()
                            .join("; ")
                    );
                    Ok(outcomes)
                })?;
                for (role, outcome) in outcomes {
                    let step = format!("asr:{role}");
                    if matches!(
                        outcome,
                        WorkerOutcome::Cancelled | WorkerOutcome::OutOfMemory
                    ) && !process::interrupted()
                    {
                        let snapshot = execution.snapshot()?;
                        if !snapshot.skipped.contains(&step)
                            && snapshot.auto_runtime_seconds
                                >= snapshot.budget.max_auto_runtime_seconds
                            && snapshot.stages[&step]
                                .active_attempt
                                .as_ref()
                                .is_some_and(|id| snapshot.attempts[id].auto_retry)
                        {
                            job.warning(&step, false, "automatic retry runtime budget exhausted")?;
                            write_json(
                                &waiting_path,
                                &WaitingResources {
                                    role: role.to_owned(),
                                    available_mib: resources::memory()?.available_mib,
                                    settings: config.resources.clone(),
                                },
                            )?;
                            bail!(
                                "{role} exceeded the automatic retry runtime budget; job is waiting for resources in {}",
                                waiting_path.display()
                            );
                        }
                    }
                    if outcome == WorkerOutcome::Cancelled {
                        if process::interrupted()
                            || execution.snapshot()?.interrupted
                            || !execution.cancelled(&format!("asr:{role}"))?
                        {
                            return Err(process::TaskCancelled.into());
                        }
                        continue;
                    }
                    if outcome == WorkerOutcome::OutOfMemory {
                        if process::interrupted() || execution.snapshot()?.interrupted {
                            return Err(process::TaskCancelled.into());
                        }
                        if execution.cancelled(&format!("asr:{role}"))? {
                            continue;
                        }
                        eprintln!(
                            "{role}: isolated worker was killed by OOM; previous chunks remain saved"
                        );
                        let attempts = {
                            let count = oom_attempts.entry(role).or_default();
                            *count += 1;
                            *count
                        };
                        let missing: Vec<_> = (0..chunks.len())
                            .filter(|index| {
                                !output.join(role).join(format!("{index:05}.json")).exists()
                            })
                            .collect();
                        if missing.is_empty() {
                            for (index, chunk) in chunks.iter().enumerate() {
                                let hypothesis: Hypothesis =
                                    read_json(&output.join(role).join(format!("{index:05}.json")))?;
                                validate_hypothesis(&hypothesis, chunk.end - chunk.start)?;
                            }
                            execution.finish_stage(&format!("asr:{role}"))?;
                            continue;
                        }
                        let reason = format!(
                            "resource-oom-v1 (initiator: rule): worker OOM; attempt {attempts}; missing fragments {missing:?}"
                        );
                        job.warning(&format!("asr:{role}"), false, &reason)?;
                        oom_reasons.insert(role, reason);
                        force_one_worker = true;
                        if config.resources.offload_dir.is_some()
                            && config
                                .resources
                                .offload_roles
                                .iter()
                                .any(|allowed| allowed == role)
                            && !modes[role]
                        {
                            force_offload.insert(role);
                            eprintln!("{role}: retrying with verified CPU/disk offload");
                        }
                        let snapshot = execution.snapshot()?;
                        let automatic_repeats = snapshot
                            .attempts
                            .values()
                            .filter(|attempt| attempt.auto_retry)
                            .count();
                        let automatic_budget_exhausted = automatic_repeats
                            >= snapshot.budget.max_auto_repeats as usize
                            || snapshot.auto_runtime_seconds
                                >= snapshot.budget.max_auto_runtime_seconds;
                        if attempts > config.resources.oom_retries || automatic_budget_exhausted {
                            write_json(
                                &waiting_path,
                                &WaitingResources {
                                    role: role.to_owned(),
                                    available_mib: resources::memory()?.available_mib,
                                    settings: config.resources.clone(),
                                },
                            )?;
                            bail!(
                                "{role} was killed by OOM after {} attempt(s); job is waiting for resources in {}{}",
                                attempts,
                                waiting_path.display(),
                                if automatic_budget_exhausted {
                                    "; automatic retry budget exhausted"
                                } else {
                                    ""
                                }
                            );
                        }
                        plan_oom_restart(
                            &execution,
                            role,
                            &missing,
                            oom_reasons[role].as_str(),
                            budget.available_mib,
                        )?;
                        let retry_at = Instant::now() + Duration::from_secs(5 * attempts as u64);
                        let task = task_context(&execution, &format!("asr:{role}"), None);
                        while Instant::now() < retry_at {
                            task.check_cancelled()?;
                            thread::sleep(
                                Duration::from_millis(250)
                                    .min(retry_at.saturating_duration_since(Instant::now())),
                            );
                        }
                    }
                }
                let mut next = Vec::new();
                for role in remaining {
                    let step = format!("asr:{role}");
                    if execution.cancelled(&step)? {
                        continue;
                    }
                    let stage_ready = execution
                        .snapshot()?
                        .stages
                        .get(&step)
                        .is_some_and(|stage| stage.status == workflow::State::Ready);
                    if !stage_ready
                        || (0..chunks.len())
                            .any(|i| !output.join(role).join(format!("{i:05}.json")).exists())
                    {
                        next.push(role);
                    }
                }
                remaining = next;
            }
            ensure!(
                model_errors.is_empty(),
                "ASR model integrity failed: {}",
                model_errors
                    .iter()
                    .map(|(role, error)| format!("{role}: {error}"))
                    .collect::<Vec<_>>()
                    .join("; ")
            );
            Ok(())
        })();
        if let Err(error) = &asr_run
            && error.is::<process::TaskCancelled>()
        {
            return cancellation_outcome(&execution);
        }
        if asr_run.is_ok() {
            job.status = workflow::status(&output)?;
        }
        if asr_run.is_ok()
            && let Err(error) =
                run_pending_asr_retries(&mut job, config, &asr_roles, &clips, &output, &chunks)
        {
            if error.is::<process::TaskCancelled>() {
                return cancellation_outcome(&execution);
            }
            return Err(error);
        }
        if asr_run.is_ok()
            && let Err(error) = run_pending_checks(&mut job, config, &output)
        {
            if error.is::<process::TaskCancelled>() {
                return cancellation_outcome(&execution);
            }
            return Err(error);
        }
        if let Some(step) = target
            .as_ref()
            .filter(|step| step.as_str().starts_with("asr:"))
        {
            asr_run?;
            if execution.cancelled(step.as_str())? || process::interrupted() {
                return cancellation_outcome(&execution);
            }
            let role = step
                .as_str()
                .strip_prefix("asr:")
                .context("invalid ASR stage")?;
            for (i, chunk) in chunks.iter().enumerate() {
                validate_hypothesis(
                    &read_json::<Hypothesis>(&output.join(role).join(format!("{i:05}.json")))?,
                    chunk.end - chunk.start,
                )?;
            }
            if !job.ready(step.as_str()) {
                job.finish(step.as_str())?;
            }
            return checkpoint_with_checks(
                &mut job,
                config,
                &output,
                step.as_str(),
                Outcome::StageComplete,
            );
        }
        let diarization_run = (|| -> Result<Diarization> {
            let diarization_path = output.join("diarization.json");
            if process::interrupted() {
                return Err(process::TaskCancelled.into());
            }
            if execution.is_skipped("diarization")? {
                return Ok(Diarization {
                    turns: Vec::new(),
                    exclusive_turns: Vec::new(),
                });
            }
            if execution.cancelled("diarization")? {
                return Err(process::TaskCancelled.into());
            }
            if !job.ready("diarization") {
                job.begin("diarization")?;
                if !diarization_path.exists() {
                    crate::models::verify_local(config.model("diarization"))?;
                    let partial = output.join("diarization.partial.json");
                    let model = config.model("diarization");
                    let mut command = Command::new(&config.diarization.program);
                    command
                        .arg(&config.diarization.script)
                        .args(&config.diarization.args)
                        .arg("--model")
                        .arg(&model.path)
                        .arg("--input")
                        .arg(&wav)
                        .arg("--output")
                        .arg(&partial)
                        .arg("--threads")
                        .arg(config.diarization.threads.to_string());
                    let task = task_context(&execution, "diarization", None);
                    process::run_observed(
                        &mut command,
                        &output.join("diarization.stdout"),
                        &output.join("diarization.stderr"),
                        Duration::from_secs(4 * 3600),
                        Some(&task),
                    )?;
                    let _: Diarization = read_json(&partial)?;
                    fs::rename(partial, &diarization_path)?;
                }
                let dia: Diarization = read_json(&diarization_path)?;
                validate_diarization(&dia, duration)?;
                job.finish("diarization")?;
            }
            let diarization: Diarization = read_json(&diarization_path)?;
            validate_diarization(&diarization, duration)?;
            Ok(diarization)
        })();
        if let Err(error) = &diarization_run
            && error.is::<process::TaskCancelled>()
        {
            return cancellation_outcome(&execution);
        }
        if asr_run.is_err() {
            let snapshot = execution.snapshot()?;
            if asr_roles.iter().all(|role| {
                snapshot
                    .stages
                    .get(&format!("asr:{role}"))
                    .is_none_or(|stage| stage.status != workflow::State::Ready)
            }) {
                block_required_descendants(
                    &job,
                    "review-flags",
                    true,
                    "no ASR hypotheses are available after ASR failure",
                )?;
            }
        }
        let diarization = match (asr_run, diarization_run) {
            (Ok(()), Ok(diarization)) => diarization,
            (Err(asr), Ok(_)) => return Err(asr),
            (Ok(()), Err(dia)) => return Err(dia),
            (Err(asr), Err(dia)) => bail!("ASR failed: {asr:#}; diarization failed: {dia:#}"),
        };
        if target == Some(workflow::Step::Diarization) {
            let outcome = if execution.is_skipped("diarization")? {
                Outcome::Blocked
            } else {
                Outcome::StageComplete
            };
            return checkpoint_with_checks(&mut job, config, &output, "diarization", outcome);
        }
        let mut available_roles = Vec::new();
        for role in &asr_roles {
            if !execution.is_skipped(&format!("asr:{role}"))?
                && !execution.cancelled(&format!("asr:{role}"))?
            {
                available_roles.push(role.clone());
            }
        }
        asr_roles = available_roles;
        if asr_roles.is_empty() {
            let reason = "no selected ASR hypotheses are available for review";
            execution.operation("review", reason)?;
            block_required_descendants(&job, "review-flags", true, reason)?;
            return checkpoint_with_checks(&mut job, config, &output, "review", Outcome::Blocked);
        }
        let mut hypotheses = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let mut group = Vec::new();
            for role in &asr_roles {
                let path = output.join(role).join(format!("{i:05}.json"));
                let hypothesis: Hypothesis = read_json(&path)?;
                validate_hypothesis(&hypothesis, chunk.end - chunk.start)?;
                group.push((role.to_owned(), hypothesis));
            }
            hypotheses.push(group);
        }
        for role in &asr_roles {
            let step = format!("asr:{role}");
            ensure!(
                job.ready(&step),
                "ASR stage {step} is not ready after validating its hypotheses"
            );
        }
        let new_flags = !job.ready("review-flags");
        if new_flags {
            job.begin("review-flags")?;
        }
        let mut flags_by_chunk = Vec::with_capacity(chunks.len());
        for (i, (chunk, group)) in chunks.iter().zip(&hypotheses).enumerate() {
            if execution.cancelled("review-flags")? || process::interrupted() {
                return cancellation_outcome(&execution);
            }
            flags_by_chunk.push(review_flags(chunk, group, &diarization));
            if new_flags {
                execution.unit_done("review-flags", i)?;
            }
        }
        let review_flags = flags_by_chunk;
        let flagged: Vec<_> = review_flags
            .iter()
            .enumerate()
            .filter(|(_, flags)| !flags.is_empty())
            .map(|(index, flags)| ReviewFlag {
                index,
                flags: flags.clone(),
            })
            .collect();
        if new_flags {
            write_json(&output.join("review-flags.json"), &flagged)?;
            job.finish("review-flags")?;
        }
        if target == Some(workflow::Step::ReviewFlags) {
            if execution.cancelled("review-flags")? || process::interrupted() {
                return cancellation_outcome(&execution);
            }
            return checkpoint_with_checks(
                &mut job,
                config,
                &output,
                "review-flags",
                Outcome::StageComplete,
            );
        }
        if let Err(error) = run_pending_checks(&mut job, config, &output) {
            if error.is::<process::TaskCancelled>() {
                return cancellation_outcome(&execution);
            }
            return Err(error);
        }
        crate::quality::auto_refresh(&output, &config.quality);
        if execution.is_skipped("review")? {
            block_required_descendants(
                &job,
                "review",
                false,
                "review was skipped; verified export requires reviewed text",
            )?;
            return checkpoint_with_checks(&mut job, config, &output, "review", Outcome::Blocked);
        }
        if execution.cancelled("review")? || process::interrupted() {
            return required_stage_outcome(&job, "review", "required review was cancelled");
        }
        let mut segments = Vec::new();
        let review_dir = output.join("reviews");
        fs::create_dir_all(&review_dir)?;
        if !job.ready("review") {
            job.begin("review")?;
            if !config.llm.enabled {
                job.warning("review", true, "LLM is not configured; run `project rerun <project> --step review` after configuring it")?;
                block_required_descendants(
                    &job,
                    "review",
                    false,
                    "review is pending configuration",
                )?;
                return checkpoint_with_checks(
                    &mut job,
                    config,
                    &output,
                    "review",
                    Outcome::PendingReview,
                );
            }
        }
        let review_rules = if job.ready("review") {
            None
        } else {
            config
                .review_rules
                .as_ref()
                .map(|path| {
                    fs::read_to_string(path)
                        .with_context(|| format!("cannot read review rules {}", path.display()))
                })
                .transpose()?
        };
        ensure!(
            review_rules.as_ref().is_none_or(|text| text.len() <= 65536),
            "review rules exceed 64 KiB"
        );
        if !job.ready("review")
            && (0..chunks.len())
                .step_by(8)
                .any(|start| !review_dir.join(format!("{start:05}.json")).exists())
        {
            let cached = crate::llm::cached_readiness(&config.llm);
            if cached.is_none()
                && (matches!(
                    config.llm.provider,
                    crate::config::LlmProvider::Codex | crate::config::LlmProvider::Copilot
                ) || config.llm.probe_args.is_some())
            {
                eprintln!("Checking LLM access before review (this may consume provider quota)...");
            }
            let readiness = cached.unwrap_or_else(|| crate::llm::readiness(&config.llm));
            if !matches!(
                readiness,
                crate::llm::Readiness::Ready | crate::llm::Readiness::Unverified
            ) {
                job.warning(
                "review",
                true,
                &format!(
                    "LLM access check: {} Resume when available; no local results were discarded",
                    readiness.help(config.llm.provider)
                ),
            )?;
                block_required_descendants(&job, "review", false, "review access is pending")?;
                return checkpoint_with_checks(
                    &mut job,
                    config,
                    &output,
                    "review",
                    Outcome::PendingReview,
                );
            }
        }
        let review_result = (|| -> Result<()> {
            for start in (0..chunks.len()).step_by(8) {
                let task = task_context(&execution, "review", None);
                task.check_cancelled()?;
                task.report_operation("reviewing transcript")?;
                let end = (start + 8).min(chunks.len());
                let decisions_path = review_dir.join(format!("{start:05}.json"));
                let partial = decisions_path.with_extension("partial.json");
                let new_decisions = !decisions_path.exists();
                let mut normalize_cached = false;
                let decisions: Decisions = if decisions_path.exists() {
                    let response = fs::read_to_string(&decisions_path)?;
                    normalize_cached = response
                        .trim_matches([' ', '\t', '\r', '\n'])
                        .starts_with("```");
                    if normalize_cached {
                        let raw = original_response_file(&decisions_path)?;
                        fs::write(&raw, response.as_bytes())?;
                    }
                    parse_review_response(&response).with_context(|| {
                        format!("invalid saved LLM response {}", decisions_path.display())
                    })?
                } else {
                    let prompt = serde_json::json!({
                        "instruction": "Do not use tools or read files. Review the selected ASR hypotheses for each audio interval; any positive number of hypotheses may be available. With one hypothesis there is no cross-model corroboration. With several hypotheses, agreement does not prove accuracy or model independence. Use only their text and advisory review flags, never invent words. A missing diarization turn does not prove there was no speech: do not discard a unique hypothesis on that basis. For German including Swiss dialect, return standard German without changing meaning. Keep Russian in Russian. If speech cannot be recovered, set gap=true and put [пропуск] in text. Return strictly JSON: {\"segments\":[{\"index\":integer,\"text\":string,\"language\":\"de|ru|mixed|unknown\",\"gap\":boolean}]}. One entry per index. Never claim to have heard the audio.",
                        "review_rules": review_rules,
                        "intervals": (start..end).map(|i| serde_json::json!({"index": i, "start": chunks[i].start, "end": chunks[i].end, "hypotheses": hypotheses[i], "review_flags": review_flags[i]})).collect::<Vec<_>>()
                    });
                    if partial.exists() {
                        fs::remove_file(&partial)?;
                    }
                    llm_observed(
                        config,
                        &prompt.to_string(),
                        &partial,
                        &review_dir.join(format!("{start:05}.stderr")),
                        Some(&task),
                    )?;
                    read_json(&partial)?
                };
                ensure!(
                    decisions.segments.len() == end - start,
                    "LLM returned wrong number of segments in batch {start}"
                );
                for i in start..end {
                    let decision = decisions
                        .segments
                        .iter()
                        .find(|d| d.index == i)
                        .context("LLM omitted segment")?;
                    ensure!(
                        decisions.segments.iter().filter(|d| d.index == i).count() == 1,
                        "LLM duplicated segment {i}"
                    );
                    ensure!(
                        ["de", "ru", "mixed", "unknown"].contains(&decision.language.as_str()),
                        "LLM returned invalid language"
                    );
                    ensure!(
                        !decision.text.trim().is_empty(),
                        "LLM returned empty text for segment {i}"
                    );
                    let missing = hypotheses[i].iter().all(|(_, h)| h.text.trim().is_empty());
                    ensure!(
                        !missing || decision.gap,
                        "all ASR outputs empty for speech interval {i}, but LLM did not mark a gap"
                    );
                    ensure!(
                        !decision.gap || decision.text.contains("[пропуск]"),
                        "gap must be explicit in text"
                    );
                    let midpoint = (chunks[i].start + chunks[i].end) / 2.0;
                    let speaker = diarization
                        .exclusive_turns
                        .iter()
                        .find(|t| t.start <= midpoint && midpoint < t.end)
                        .map(|t| t.speaker.clone());
                    let mut text = decision.text.clone();
                    if let Some(previous) = segments.last() {
                        let previous: &FinalSegment = previous;
                        if previous.end > chunks[i].start && previous.speaker == speaker {
                            text = trim_repeated_prefix(&previous.text, &text);
                        }
                    }
                    if text.is_empty() && !decision.gap {
                        continue;
                    }
                    segments.push(FinalSegment {
                        start: chunks[i].start,
                        end: chunks[i].end,
                        speaker,
                        text,
                        language: decision.language.clone(),
                        gap: decision.gap,
                        hypotheses: hypotheses[i].clone(),
                    });
                }
                if new_decisions {
                    fs::rename(&partial, &decisions_path)?;
                } else if normalize_cached {
                    write_json(&decisions_path, &decisions)?;
                }
                for i in start..end {
                    execution.unit_done("review", i)?;
                }
            }
            Ok(())
        })();
        if let Err(err) = review_result {
            if err.is::<process::TaskCancelled>()
                || execution.cancelled("review")?
                || process::interrupted()
            {
                return required_stage_outcome(&job, "review", "required review was cancelled");
            }
            job.warning(
                "review",
                true,
                &format!("{err:#}; resume to retry review, or rerun --step review"),
            )?;
            block_required_descendants(&job, "review", false, "review validation is pending")?;
            return checkpoint_with_checks(
                &mut job,
                config,
                &output,
                "review",
                Outcome::PendingReview,
            );
        }
        if !job.ready("review") {
            if execution.cancelled("review")? || process::interrupted() {
                return required_stage_outcome(&job, "review", "required review was cancelled");
            }
            if let Err(error) = job.finish("review") {
                if execution.cancelled("review")? || process::interrupted() {
                    return required_stage_outcome(&job, "review", "required review was cancelled");
                }
                job.warning("review", true, &format!("{error:#}"))?;
                block_required_descendants(&job, "review", false, "review could not be finalized")?;
                return Err(error);
            }
        }
        if target == Some(workflow::Step::Review) {
            return checkpoint_with_checks(
                &mut job,
                config,
                &output,
                "review",
                Outcome::StageComplete,
            );
        }
        let status = if segments.iter().any(|s| s.gap) || !flagged.is_empty() {
            "needs_audio_review"
        } else {
            "text_reviewed_audio_not_verified"
        };
        let diarization_skipped = execution.is_skipped("diarization")?;
        let transcript = Transcript {
            schema_version: 1,
            source_sha256: hash,
            duration,
            quality_status: status.to_owned(),
            limitations: if diarization_skipped {
                vec!["Diarization was explicitly skipped; speaker labels are unavailable.".into()]
            } else {
                Vec::new()
            },
            models: config
                .models
                .iter()
                .filter(|model| {
                    (model.role == "diarization" && !diarization_skipped)
                        || asr_roles.contains(&model.role)
                })
                .cloned()
                .collect(),
            segments,
        };
        if job.ready("export") {
            if execution.cancelled("export")? || process::interrupted() {
                return cancellation_outcome(&execution);
            }
            return checkpoint_with_checks(&mut job, config, &output, "export", Outcome::Complete);
        }
        if execution.cancelled("export")? || process::interrupted() {
            return cancellation_outcome(&execution);
        }
        job.begin("export")?;
        let exports = job.status.export_paths(&output)?;
        fs::create_dir_all(exports.json.parent().context("export has no directory")?)?;
        write_json_with_temporary(
            &exports.json,
            &export_partial_path(&exports.json)?,
            &transcript,
        )?;
        let _: Transcript = read_json(&exports.json)?;
        execution.unit_done("export", 0)?;
        let mut text = String::new();
        for s in &transcript.segments {
            text.push_str(&format!(
                "[{:02}:{:02}:{:02}] {}: {}\n",
                (s.start as u64) / 3600,
                ((s.start as u64) / 60) % 60,
                (s.start as u64) % 60,
                s.speaker.as_deref().unwrap_or("неизвестный"),
                s.text
            ));
        }
        let text_partial = export_partial_path(&exports.txt)?;
        fs::write(&text_partial, &text)?;
        fs::rename(text_partial, &exports.txt)?;
        ensure!(
            fs::read_to_string(&exports.txt)? == text,
            "exported transcript text differs from verified content"
        );
        execution.unit_done("export", 1)?;
        if execution.cancelled("export")? || process::interrupted() {
            return cancellation_outcome(&execution);
        }
        job.finish("export")?;
        checkpoint_with_checks(&mut job, config, &output, "export", Outcome::Complete)
    })()?;
    terminal.finish()?;
    Ok(outcome)
}

fn run_asr(
    config: &Config,
    role: &str,
    engine: &str,
    clips: &Path,
    output: &Path,
    chunks: &[Interval],
    mode: WorkerMode,
) -> Result<WorkerOutcome> {
    let model = config.model(role);
    let adapter = config.asr(role);
    let restart = if mode.retry.is_none() {
        mode.execution
            .map(|execution| {
                let snapshot = execution.snapshot()?;
                let step = format!("asr:{role}");
                let view = snapshot.stages.get(&step).context("missing ASR stage")?;
                Ok::<_, anyhow::Error>(
                    view.active_attempt
                        .as_ref()
                        .and_then(|id| snapshot.attempts.get(id))
                        .filter(|attempt| attempt.kind == "restart")
                        .map(|attempt| {
                            (
                                attempt.fragments.iter().copied().collect::<BTreeSet<_>>(),
                                attempt.parameters.max_in_flight,
                            )
                        }),
                )
            })
            .transpose()?
            .flatten()
    } else {
        None
    };
    let max_in_flight = restart
        .as_ref()
        .and_then(|(_, limit)| *limit)
        .unwrap_or(adapter.max_in_flight);
    let folder = mode
        .retry
        .zip(mode.execution)
        .map(|(retry, execution)| execution.retry_output(&retry.attempt_id))
        .unwrap_or_else(|| output.join(role));
    fs::create_dir_all(&folder)?;
    let mut pending = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        if restart
            .as_ref()
            .is_some_and(|(assigned, _)| !assigned.contains(&i))
        {
            if mode.worker_number == 0 {
                let value: Hypothesis = read_json(&folder.join(format!("{i:05}.json")))?;
                validate_hypothesis(&value, chunk.end - chunk.start)?;
            }
            continue;
        }
        if mode
            .retry
            .is_some_and(|retry| !retry.fragments.contains(&i))
        {
            continue;
        }
        if i % mode.worker_count != mode.worker_number {
            continue;
        }
        let path = folder.join(format!("{i:05}.json"));
        if path.exists() {
            let value: Hypothesis = read_json(&path)?;
            validate_hypothesis(&value, chunk.end - chunk.start)?;
            observe_asr_unit(mode, role, i)?;
            continue;
        }
        pending.push((i, chunk));
    }
    if pending.is_empty() {
        return Ok(WorkerOutcome::Complete);
    }
    let offload_dir = if mode.offload {
        let path = config
            .resources
            .offload_dir
            .as_deref()
            .context("offload directory not configured")?;
        resources::prepare_offload_dir(path, config.resources.offload_min_free_mib)?;
        Some(path)
    } else {
        None
    };
    let started = std::time::Instant::now();
    let stderr = folder.join(if mode.worker_count == 1 {
        "worker.stderr".to_owned()
    } else {
        format!("worker-{:02}.stderr", mode.worker_number)
    });
    let unit = config.resources.supervise_asr_with_systemd.then(|| {
        format!(
            "shum-asr-{role}-{}-{}",
            std::process::id(),
            WORKER_ID.fetch_add(1, Ordering::Relaxed)
        )
    });
    let mut command = if let Some(unit) = &unit {
        let mut command = Command::new("systemd-run");
        command.args([
            "--user",
            "--quiet",
            "--pipe",
            "--wait",
            "--unit",
            unit,
            "--property",
            "MemoryAccounting=yes",
        ]);
        if let Some(limit) = config.resources.systemd_memory_limit_mib {
            command.args(["--property", &format!("MemoryMax={limit}M")]);
        }
        if let Some(limit) = config.resources.systemd_swap_limit_mib {
            command.args(["--property", &format!("MemorySwapMax={limit}M")]);
        }
        command.arg(&adapter.program);
        command
    } else {
        Command::new(&adapter.program)
    };
    command
        .arg(&adapter.script)
        .args(&adapter.args)
        .arg("--engine")
        .arg(engine)
        .arg("--model")
        .arg(&model.path)
        .arg("--threads")
        .arg(adapter.threads.to_string())
        .arg("--serve");
    if let Some(path) = offload_dir {
        command.arg("--offload-dir").arg(path);
    }
    if let Some(task) = mode.task {
        task.check_cancelled()?;
    }
    process::configure_owned_command(&mut command);
    let child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(File::create(&stderr)?)
        .spawn()
        .with_context(|| format!("cannot start {role} worker"))?;
    let pid = child.id();
    let mut owned = process::OwnedProcess::new(child, unit.clone());
    if let Some(benchmark) = mode.benchmark {
        benchmark
            .events
            .lock()
            .expect("benchmark events mutex poisoned")
            .workers
            .insert(
                mode.worker_number,
                WorkerTiming {
                    worker_id: mode.worker_number,
                    pid,
                    start_offset_ms: started.duration_since(benchmark.started).as_millis(),
                    ready_offset_ms: None,
                    first_result_offset_ms: None,
                    finish_offset_ms: None,
                },
            );
        benchmark
            .events
            .lock()
            .expect("benchmark events mutex poisoned")
            .pids
            .insert(pid);
    }
    let mut stdin = owned
        .child_mut()
        .stdin
        .take()
        .context("ASR worker stdin missing")?;
    let stdout = owned
        .child_mut()
        .stdout
        .take()
        .context("ASR worker stdout missing")?;
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    let mut observed_peak_mib: Option<u64> = None;
    let run = (|| -> Result<()> {
        let mut pending = pending.into_iter().peekable();
        let mut in_flight = BTreeMap::new();
        let mut pause_for_ram = false;
        while (!pause_for_ram && pending.peek().is_some()) || !in_flight.is_empty() {
            if let Some(task) = mode.task {
                task.check_cancelled()?;
            }
            while !pause_for_ram
                && in_flight.len() < max_in_flight
                && let Some((i, chunk)) = pending.next()
            {
                if let Some(task) = mode.task {
                    task.check_cancelled()?;
                }
                if mode.deadline.is_some_and(|limit| Instant::now() >= limit) {
                    bail!("{role} exceeded the 600-second benchmark limit");
                }
                if mode
                    .benchmark
                    .is_some_and(|benchmark| benchmark.stop.load(Ordering::Relaxed))
                {
                    bail!(
                        "{role} resource stop: MemAvailable fell below benchmark reserve or sampling failed"
                    );
                }
                let clip = clips.join(format!("{i:05}.wav")).canonicalize()?;
                if max_in_flight > 1 || mode.benchmark.is_some() {
                    writeln!(stdin, "{}", serde_json::json!({"index": i, "input": clip}))?;
                } else {
                    writeln!(stdin, "{}", serde_json::json!({"input": clip}))?;
                }
                stdin.flush()?;
                in_flight.insert(i, (chunk, Instant::now()));
            }
            let waiting = *in_flight
                .keys()
                .next()
                .context("no ASR request in flight")?;
            let window_started = in_flight
                .get(&waiting)
                .context("missing in-flight ASR window")?
                .1;
            let timeout = Duration::from_millis(250)
                .min(Duration::from_secs(1800).saturating_sub(window_started.elapsed()))
                .min(mode.deadline.map_or(Duration::MAX, |limit| {
                    limit.saturating_duration_since(Instant::now())
                }));
            let line = match receiver.recv_timeout(timeout) {
                Ok(line) => {
                    line.with_context(|| format!("{role} chunk {waiting}: worker stdout failed"))?
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if let Some(task) = mode.task {
                        task.check_cancelled()?;
                    }
                    if window_started.elapsed() >= Duration::from_secs(1800) {
                        bail!(
                            "{role} chunk {waiting}: worker timed out after 1800 seconds; see {}",
                            stderr.display()
                        );
                    }
                    if mode.deadline.is_some_and(|limit| Instant::now() >= limit) {
                        bail!("{role} exceeded the 600-second benchmark limit on chunk {waiting}");
                    }
                    if mode
                        .benchmark
                        .is_some_and(|benchmark| benchmark.stop.load(Ordering::Relaxed))
                    {
                        bail!("{role} resource stop: MemAvailable below reserve or sampler failed");
                    }
                    continue;
                }
                Err(error) => bail!(
                    "{role} chunk {waiting}: worker timed out or exited ({error}); see {}",
                    stderr.display()
                ),
            };
            let received = Instant::now();
            if let Some(task) = mode.task {
                task.check_cancelled()?;
            }
            if mode.deadline.is_some_and(|limit| received >= limit) {
                bail!("{role} exceeded the 600-second benchmark limit on chunk {waiting}");
            }
            if mode
                .benchmark
                .is_some_and(|benchmark| benchmark.stop.load(Ordering::Relaxed))
            {
                bail!("{role} resource stop before result acceptance");
            }
            let response: serde_json::Value = serde_json::from_str(&line).with_context(|| {
                if let Some(benchmark) = mode.benchmark {
                    benchmark.fail(waiting);
                }
                format!(
                    "{role} chunk {waiting}: invalid worker JSON; see {}",
                    stderr.display()
                )
            })?;
            if max_in_flight > 1
                && response.get("index").is_none()
                && let Some(error) = response.get("error")
            {
                if let Some(benchmark) = mode.benchmark {
                    benchmark.fail(waiting);
                }
                bail!(
                    "{role} worker failed while chunk {waiting} was in flight: {error}; see {}",
                    stderr.display()
                );
            }
            let i = if max_in_flight > 1
                || (mode.benchmark.is_some() && response.get("index").is_some())
            {
                response
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .with_context(|| {
                        if let Some(benchmark) = mode.benchmark {
                            benchmark.fail(waiting);
                        }
                        "ASR response must contain a nonnegative window index"
                    })?
            } else {
                waiting
            };
            let (chunk, window_started) = in_flight.remove(&i).with_context(|| {
                if let Some(benchmark) = mode.benchmark {
                    benchmark.fail(i);
                }
                format!("{role} returned unexpected or duplicate chunk {i}")
            })?;
            if let Some(error) = response.get("error") {
                if let Some(benchmark) = mode.benchmark {
                    benchmark.fail(i);
                }
                bail!("{role} chunk {i}: {error}; see {}", stderr.display());
            }
            let mut value: Hypothesis = serde_json::from_value(response).with_context(|| {
                if let Some(benchmark) = mode.benchmark {
                    benchmark.fail(i);
                }
                format!("{role} chunk {i}: invalid hypothesis")
            })?;
            value.processing_mode = Some(
                if mode.offload {
                    "disk_offload"
                } else {
                    "standard"
                }
                .to_owned(),
            );
            value.model_revision = Some(model.revision.clone());
            if role == "flix" {
                let discarded = clip_whisper_timestamps(&mut value, chunk.end - chunk.start);
                if discarded > 0 {
                    eprintln!(
                        "flix chunk {i}: {discarded} timestamp(s) exceeded the audio; original text and timestamps preserved for review"
                    );
                }
            }
            validate_hypothesis(&value, chunk.end - chunk.start).inspect_err(|_| {
                if let Some(benchmark) = mode.benchmark {
                    benchmark.fail(i);
                }
            })?;
            let result_path = folder.join(format!("{i:05}.json"));
            write_json(&result_path, &value).inspect_err(|_| {
                if let Some(benchmark) = mode.benchmark {
                    benchmark.fail(i);
                }
            })?;
            observe_asr_unit(mode, role, i)?;
            if let Some(benchmark) = mode.benchmark
                && let Err(error) =
                    benchmark.record_result(mode.worker_number, i, window_started, received)
            {
                fs::remove_file(&result_path).context("cannot remove excluded benchmark result")?;
                return Err(error);
            }
            resources::record_window(
                output,
                resources::WindowEvent {
                    role,
                    index: i,
                    audio_seconds: chunk.end - chunk.start,
                    elapsed_ms: window_started.elapsed().as_millis(),
                    mode: if mode.offload {
                        "disk_offload"
                    } else {
                        "standard"
                    },
                    characters: value.text.chars().count(),
                    repeated_fourgram_fraction: repeated_fourgram_fraction(&value.text),
                },
            )?;
            if let Some(unit) = &unit
                && let Some(peak) = resources::systemd_memory_peak(unit)?
            {
                observed_peak_mib = Some(observed_peak_mib.unwrap_or(0).max(peak));
            }
            if mode.benchmark.is_none()
                && mode.concurrent
                && resources::memory()?.available_mib < config.resources.reserve_mib
            {
                eprintln!(
                    "{role}: RAM below reserve after chunk {i}; draining in-flight requests before next batch"
                );
                pause_for_ram = true;
            }
        }
        Ok(())
    })();
    drop(stdin);
    let exit = if run.is_err() {
        owned.stop()?;
        owned.child_mut().wait()?
    } else {
        match process::wait_child_observed(&mut owned, Duration::from_secs(60), mode.task)? {
            Some(exit) => exit,
            None => {
                owned.stop()?;
                bail!(
                    "{role} worker exceeded the 60-second shutdown timeout; see {}",
                    stderr.display()
                );
            }
        }
    };
    if let Some(benchmark) = mode.benchmark {
        let mut events = benchmark
            .events
            .lock()
            .expect("benchmark events mutex poisoned");
        events.pids.remove(&pid);
        if let Some(worker) = events.workers.get_mut(&mode.worker_number) {
            worker.finish_offset_ms = Some(benchmark.offset());
        }
    }
    reader
        .join()
        .map_err(|_| anyhow::anyhow!("ASR reader thread panicked"))?;
    let service_result = unit
        .as_ref()
        .map(|unit| resources::systemd_result(unit))
        .transpose()?;
    if let Some(unit) = &unit
        && let Some(peak) = resources::systemd_memory_peak(unit)?
    {
        observed_peak_mib = Some(observed_peak_mib.unwrap_or(0).max(peak));
    }
    let outcome = if service_result.as_deref() == Some("oom-kill") {
        "oom-kill"
    } else if run
        .as_ref()
        .err()
        .is_some_and(|error| error.is::<process::TaskCancelled>())
    {
        "cancelled"
    } else if run.is_err() || !exit.success() {
        "error"
    } else {
        "success"
    };
    resources::record_worker(
        output,
        role,
        started.elapsed().as_millis(),
        observed_peak_mib,
        if mode.offload {
            "disk_offload"
        } else {
            "standard"
        },
        outcome,
    )?;
    if outcome == "oom-kill" {
        return Ok(WorkerOutcome::OutOfMemory);
    }
    run?;
    ensure!(
        exit.success(),
        "{role} worker failed with {exit}; see {}",
        stderr.display()
    );
    Ok(WorkerOutcome::Complete)
}

fn observe_asr_unit(mode: WorkerMode<'_>, role: &str, index: usize) -> Result<()> {
    if let Some(execution) = mode.execution {
        if let Some(retry) = mode.retry {
            execution.retry_unit_done(&retry.attempt_id, index)?;
        } else {
            execution.unit_done(&format!("asr:{role}"), index)?;
        }
    }
    Ok(())
}

#[cfg(test)]
fn llm(config: &Config, prompt: &str, output: &Path, stderr: &Path) -> Result<()> {
    llm_observed(config, prompt, output, stderr, None)
}

fn llm_observed(
    config: &Config,
    prompt: &str,
    output: &Path,
    stderr: &Path,
    task: Option<&process::TaskContext>,
) -> Result<()> {
    let raw = original_response_file(output)?;
    crate::llm::review_observed(&config.llm, prompt, &raw, stderr, task)?;
    let response = fs::read_to_string(&raw)
        .with_context(|| format!("cannot read original LLM response {}", raw.display()))?;
    let decisions = parse_review_response(&response).with_context(|| {
        format!(
            "invalid LLM response; original preserved at {}",
            raw.display()
        )
    })?;
    write_json(output, &decisions)?;
    Ok(())
}

fn original_response_file(output: &Path) -> Result<PathBuf> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let raw = output.with_extension(format!(
        "response-{stamp}-{}.txt",
        WORKER_ID.fetch_add(1, Ordering::Relaxed)
    ));
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&raw)?;
    Ok(raw)
}

fn parse_review_response(response: &str) -> Result<Decisions> {
    let text = response.trim_matches([' ', '\t', '\r', '\n']);
    let content = if let Some(body) = text
        .strip_prefix("```json\n")
        .or_else(|| text.strip_prefix("```json\r\n"))
        .or_else(|| text.strip_prefix("```\n"))
        .or_else(|| text.strip_prefix("```\r\n"))
    {
        body.strip_suffix("\n```")
            .or_else(|| body.strip_suffix("\r\n```"))
            .context("LLM JSON fence must wrap the entire response and have a closing fence")?
    } else {
        text
    };
    serde_json::from_str(content)
        .context("LLM response must be pure JSON or one fully fenced json block")
}

pub(crate) fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0_u8; 65536];
    loop {
        let len = file.read(&mut buf)?;
        if len == 0 {
            break;
        }
        hash.update(&buf[..len]);
    }
    Ok(hex::encode(hash.finalize()))
}

fn stage_file_hash(
    path: &Path,
    previous_path: Option<&Path>,
    previous_hash: Option<&str>,
) -> Result<String> {
    match hash_file(path) {
        Ok(hash) => Ok(hash),
        Err(error)
            if error.downcast_ref::<std::io::Error>().is_some_and(|io| {
                matches!(
                    io.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
                )
            }) =>
        {
            if previous_path == Some(path)
                && let Some(hash) = previous_hash
            {
                return Ok(hash.to_owned());
            }
            Ok(format!("unavailable:{}", path.display()))
        }
        Err(error) => Err(error).with_context(|| format!("cannot fingerprint {}", path.display())),
    }
}

fn trim_repeated_prefix(previous: &str, current: &str) -> String {
    let left: Vec<_> = previous.split_whitespace().collect();
    let right: Vec<_> = current.split_whitespace().collect();
    for count in (1..=left.len().min(right.len()).min(12)).rev() {
        if left[left.len() - count..]
            .iter()
            .zip(&right[..count])
            .all(|(a, b)| {
                a.trim_matches(|c: char| !c.is_alphanumeric())
                    .eq_ignore_ascii_case(b.trim_matches(|c: char| !c.is_alphanumeric()))
            })
        {
            return right[count..].join(" ");
        }
    }
    current.to_owned()
}

fn clip_whisper_timestamps(value: &mut Hypothesis, duration: f64) -> usize {
    let mut in_bounds = Vec::new();
    for mut segment in std::mem::take(&mut value.segments) {
        if segment.start >= duration {
            value.out_of_bounds_segments.push(segment);
        } else {
            if segment.end > duration {
                // Subtracting absolute clip bounds can lose sub-sample precision.
                if segment.end > duration + 1e-6 {
                    value.out_of_bounds_segments.push(segment.clone());
                }
                segment.end = duration;
            }
            in_bounds.push(segment);
        }
    }
    if !value.out_of_bounds_segments.is_empty() {
        value.raw_text = Some(std::mem::take(&mut value.text));
        value.text = in_bounds.iter().map(|s| s.text.as_str()).collect();
    }
    value.segments = in_bounds;
    value.out_of_bounds_segments.len()
}

pub(crate) fn validate_hypothesis(value: &Hypothesis, duration: f64) -> Result<()> {
    for segment in &value.segments {
        ensure!(
            segment.start.is_finite()
                && segment.end.is_finite()
                && segment.start >= 0.0
                && segment.end >= segment.start
                && segment.end <= duration + 0.5,
            "ASR returned invalid clip-relative timestamp"
        );
    }
    Ok(())
}

pub(crate) fn validate_diarization(value: &Diarization, duration: f64) -> Result<()> {
    for turn in value.turns.iter().chain(&value.exclusive_turns) {
        ensure!(
            turn.start.is_finite()
                && turn.end.is_finite()
                && turn.start >= 0.0
                && turn.end > turn.start
                && turn.end <= duration + 0.5
                && !turn.speaker.trim().is_empty(),
            "diarization returned invalid speaker turn"
        );
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_reader(
        File::open(path).with_context(|| format!("missing {}", path.display()))?,
    )
    .with_context(|| format!("invalid JSON in {}", path.display()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let temp = path.with_extension("partial");
    write_json_with_temporary(path, &temp, value)
}

fn export_partial_path(path: &Path) -> Result<PathBuf> {
    let mut name = path
        .file_name()
        .context("export has no filename")?
        .to_os_string();
    name.push(".partial");
    Ok(path.with_file_name(name))
}

fn write_json_with_temporary<T: Serialize>(path: &Path, temp: &Path, value: &T) -> Result<()> {
    let mut file = File::create(temp)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.sync_all()?;
    fs::rename(temp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::hash_file;
    use super::{
        Decisions, Diarization, Hypothesis, JobManifest, Transcript, Turn, WordSpan, WorkerMode,
        WorkerOutcome, auto_asr_task_context, benchmark_asr, benchmark_diarization,
        cancelled_worker_or_error, clip_whisper_timestamps, elapsed_seconds, llm,
        parse_review_response, plan_oom_restart, read_json, review_flags, run_asr,
        same_job_after_model_move, transcribe, trim_repeated_prefix, validate_hypothesis,
    };
    use crate::audio::Interval;
    use crate::config::{Adapter, Config, Llm};
    use crate::execution::{self, Initiator};
    use crate::models::ModelSpec;
    use crate::process::TaskContext;
    use crate::workflow::{self, Outcome, Stage, State};
    use sha2::Digest;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::Instant;

    #[test]
    fn cancelled_asr_finalization_does_not_mask_unrelated_errors() {
        let cancelled = TaskContext::new(Some(std::sync::Arc::new(|| Ok(true))), None, None);
        assert_eq!(
            cancelled_worker_or_error(&cancelled, anyhow::anyhow!("attempt is not running"))
                .unwrap(),
            WorkerOutcome::Cancelled
        );
        let active = TaskContext::default();
        let error =
            cancelled_worker_or_error(&active, anyhow::anyhow!("disk write failed")).unwrap_err();
        assert_eq!(error.to_string(), "disk write failed");
    }

    #[test]
    fn oom_restart_records_rule_decision_runs_only_missing_units_and_honors_budget() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let status = workflow::ProjectStatus {
            schema_version: 2,
            input: root.join("source.mp3"),
            input_sha256: "input".into(),
            export_stem: Some("source".into()),
            stages: workflow::steps_for_roles(&["primeline".into()])
                .into_iter()
                .map(|step| {
                    (
                        step.clone(),
                        Stage {
                            status: State::NotStarted,
                            attempts: 0,
                            fingerprint: step,
                            output_sha256: None,
                            warning: None,
                        },
                    )
                })
                .collect(),
            asr_roles: vec!["primeline".into()],
        };
        fs::write(
            workflow::state_path(root),
            serde_json::to_vec(&status).unwrap(),
        )
        .unwrap();
        let handle = execution::Handle::open(root, &status).unwrap();
        let chunks = (0..3)
            .map(|i| Interval {
                start: i as f64,
                end: i as f64 + 1.0,
            })
            .collect::<Vec<_>>();
        fs::write(
            root.join("chunks.json"),
            serde_json::to_vec(&chunks).unwrap(),
        )
        .unwrap();
        handle
            .set_units("asr:primeline", &[0, 1, 2], "fragments")
            .unwrap();
        let failed = handle.begin_stage("asr:primeline").unwrap();
        handle
            .fail_stage("asr:primeline", State::Failed, "mock worker OOM")
            .unwrap();
        let folder = root.join("primeline");
        fs::create_dir(&folder).unwrap();
        for index in [0, 2] {
            fs::write(
                folder.join(format!("{index:05}.json")),
                br#"{"text":"cached","language":"de","segments":[]}"#,
            )
            .unwrap();
        }
        let reason = "resource-oom-v1: mock worker OOM";
        let id = plan_oom_restart(&handle, "primeline", &[1], reason, 2048).unwrap();
        let snapshot = handle.snapshot().unwrap();
        let decision = snapshot.decisions.last().unwrap();
        assert!(matches!(
            &decision.initiator,
            Initiator::Rule { rule } if rule == "resource-recovery"
        ));
        assert_eq!(decision.reason, reason);
        assert_eq!(decision.fragments, vec![1]);
        assert!(
            decision
                .evidence
                .iter()
                .any(|item| item["failed_attempt"] == failed)
        );
        assert_eq!(snapshot.attempts[&id].kind, "restart");
        assert!(snapshot.attempts[&id].auto_retry);
        assert_eq!(handle.begin_stage("asr:primeline").unwrap(), id);
        assert_eq!(
            handle.snapshot().unwrap().stages["asr:primeline"].units,
            vec![1]
        );

        let clips = root.join("clips");
        fs::create_dir(&clips).unwrap();
        for index in 0..3 {
            fs::write(clips.join(format!("{index:05}.wav")), b"mock clip").unwrap();
        }
        let script = root.join("asr.py");
        fs::write(
            &script,
            r#"import json,pathlib,sys
for line in sys.stdin:
    clip=pathlib.Path(json.loads(line)["input"])
    with (pathlib.Path(__file__).parent / "requests.log").open("a") as log:
        log.write(clip.stem + "\n")
    print(json.dumps({"text":"fresh","language":"de","segments":[]}),flush=True)
"#,
        )
        .unwrap();
        let model = root.join("model");
        fs::create_dir(&model).unwrap();
        let adapter = Adapter {
            program: "python3".into(),
            script,
            engine: None,
            threads: 1,
            max_in_flight: 2,
            args: Vec::new(),
        };
        let config = Config {
            models: vec![ModelSpec {
                role: "primeline".into(),
                repo: "mock".into(),
                revision: "test".into(),
                path: model,
            }],
            asr: BTreeMap::from([("primeline".into(), adapter.clone())]),
            diarization: adapter,
            llm: Llm::default(),
            review_rules: None,
            resources: Default::default(),
            quality: Default::default(),
            asr_selection: None,
        };
        let started = Instant::now();
        let task = auto_asr_task_context(&handle, "asr:primeline", &id, started);
        assert_eq!(
            run_asr(
                &config,
                "primeline",
                "mock",
                &clips,
                root,
                &chunks,
                WorkerMode {
                    concurrent: false,
                    offload: false,
                    worker_number: 0,
                    worker_count: 1,
                    deadline: None,
                    benchmark: None,
                    execution: Some(&handle),
                    retry: None,
                    task: Some(&task),
                },
            )
            .unwrap(),
            WorkerOutcome::Complete
        );
        handle
            .record_runtime(&id, elapsed_seconds(started))
            .unwrap();
        handle.finish_stage("asr:primeline").unwrap();
        assert_eq!(
            fs::read_to_string(root.join("requests.log")).unwrap(),
            "00001\n"
        );
        assert_eq!(
            handle.snapshot().unwrap().stages["asr:primeline"].done,
            [1].into()
        );
        let budget = handle.snapshot().unwrap().budget.max_auto_runtime_seconds;
        handle.record_runtime(&id, budget).unwrap();
        assert!(
            plan_oom_restart(
                &handle,
                "primeline",
                &[0],
                "resource-oom-v1: second mock worker OOM",
                2048,
            )
            .unwrap_err()
            .to_string()
            .contains("automatic retry runtime budget exhausted")
        );
    }

    #[test]
    fn llm_accepts_only_json_or_one_complete_json_fence() {
        let json = r#"{"segments":[{"index":0,"text":"Guten Tag","language":"de","gap":false}]}"#;
        for response in [
            json.to_owned(),
            format!(" \n{json}\r\n"),
            format!("```json\n{json}\n```"),
            format!("```json\r\n{json}\r\n```\r\n"),
            format!("```\n{json}\n```"),
        ] {
            let parsed = parse_review_response(&response).unwrap();
            assert_eq!(parsed.segments.len(), 1);
            assert_eq!(parsed.segments[0].text, "Guten Tag");
        }
        for response in [
            format!("Here is the result:\n```json\n{json}\n```"),
            format!("```json\n{json}\n```\nDone."),
            format!("```json\n{json}\n```\n```json\n{json}\n```"),
            format!("```json\n{json}"),
            format!("```python\n{json}\n```"),
            format!("{json}\n{json}"),
            "```json\n{\"segments\":null}\n```".into(),
            "```json\n{\"segments\":[{\"index\":\"0\"}]}\n```".into(),
        ] {
            assert!(parse_review_response(&response).is_err(), "{response}");
        }
    }

    #[test]
    fn original_llm_reply_is_preserved_on_valid_and_invalid_wrappers() {
        let temp = tempfile::tempdir().unwrap();
        let script = temp.path().join("reply.py");
        let json = r#"{"segments":[{"index":0,"text":"Guten Tag","language":"de","gap":false}]}"#;
        let adapter = Adapter {
            program: "python3".into(),
            script: "unused.py".into(),
            engine: None,
            threads: 1,
            max_in_flight: 1,
            args: vec![],
        };
        let mut config = Config {
            models: vec![],
            asr: BTreeMap::new(),
            diarization: adapter,
            llm: crate::config::Llm {
                enabled: true,
                provider: crate::config::LlmProvider::Custom,
                program: "python3".into(),
                args: vec![script.to_string_lossy().into_owned(), "{output}".into()],
                ..Default::default()
            },
            review_rules: None,
            resources: Default::default(),
            quality: Default::default(),
            asr_selection: None,
        };
        for (iteration, reply) in [
            format!("```json\n{json}\n```"),
            format!("Prefix\n```json\n{json}\n```"),
        ]
        .iter()
        .enumerate()
        {
            fs::write(
                &script,
                format!(
                    "import pathlib,sys\npathlib.Path(sys.argv[1]).write_text({:?})\n",
                    reply
                ),
            )
            .unwrap();
            config.llm.args[0] = script.to_string_lossy().into_owned();
            let output = temp.path().join(format!("batch{iteration}.json"));
            let result = llm(&config, "{}", &output, &temp.path().join("stderr"));
            assert_eq!(result.is_ok(), iteration == 0);
            let originals: Vec<_> = fs::read_dir(temp.path())
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| {
                    path.file_name()
                        .unwrap()
                        .to_string_lossy()
                        .starts_with(&format!("batch{iteration}.response-"))
                })
                .collect();
            assert_eq!(originals.len(), 1);
            assert_eq!(fs::read_to_string(&originals[0]).unwrap(), *reply);
            if iteration == 0 {
                let parsed: Decisions = read_json(&output).unwrap();
                assert_eq!(parsed.segments[0].text, "Guten Tag");
            } else {
                assert!(!output.exists());
            }
        }
    }

    #[test]
    fn resuming_a_moved_project_only_ignores_model_storage_paths() {
        let adapter = Adapter {
            program: PathBuf::from("/bin/python"),
            script: PathBuf::from("/app/worker.py"),
            engine: None,
            threads: 4,
            max_in_flight: 1,
            args: vec![],
        };
        let previous = JobManifest {
            schema_version: 1,
            input_sha256: "input".into(),
            models: vec![ModelSpec {
                role: "primeline".into(),
                repo: "example/model".into(),
                revision: "revision".into(),
                path: PathBuf::from("/old/models/primeline/revision"),
            }],
            asr: [("primeline".into(), adapter.clone())].into(),
            asr_script_sha256: [("primeline".into(), "script".into())].into(),
            diarization: adapter,
            diarization_script_sha256: "diarization-script".into(),
            llm: Llm {
                program: PathBuf::from("codex"),
                args: vec![],
                ..Llm::default()
            },
            review_rules: None,
            review_rules_sha256: None,
            asr_roles: crate::config::default_asr_roles(),
        };
        let mut relocated = previous.clone();
        relocated.models[0].path = PathBuf::from("/new/models/primeline/revision");
        assert!(same_job_after_model_move(&previous, &relocated));
        relocated.models[0].revision = "different".into();
        assert!(!same_job_after_model_move(&previous, &relocated));
        relocated.models[0].revision = "revision".into();
        relocated.input_sha256 = "different".into();
        assert!(!same_job_after_model_move(&previous, &relocated));
    }

    #[test]
    fn strips_only_shared_boundary() {
        assert_eq!(
            trim_repeated_prefix("Ich gehe heute", "heute nach Hause"),
            "nach Hause"
        );
        assert_eq!(
            trim_repeated_prefix("Ich gehe heute", "morgen nach Hause"),
            "morgen nach Hause"
        );
    }

    #[test]
    fn benchmark_enforces_strict_ten_minutes_and_machine_readable_result() {
        assert!(super::meets_30min_target(599.999));
        assert!(!super::meets_30min_target(600.0));
        let report = super::BenchmarkReport {
            role: "flix",
            engine: "whisper",
            audio_seconds: 1800.0,
            elapsed_seconds: 599.999,
            real_time_factor: 599.999 / 1800.0,
            window_count: 67,
            worker_count: 1,
            estimated_worker_peak_mib: Some(3000),
            reserve_mib: 4096,
            resource_guard_semantics: super::RESOURCE_GUARD_SEMANTICS,
            memory_full_avg10_threshold_percent:
                crate::resources::MEMORY_FULL_AVG10_THRESHOLD_PERCENT,
            empty_window_count: 0,
            total_characters: 1000,
            offload: false,
            threads: 12,
            max_in_flight: 1,
            adapter_args: &[],
            model_revision: "pinned",
            adapter_sha256: "script-hash".to_owned(),
            under_ten_minutes: true,
            status: "success",
            stop_reason: None,
            completed_indices: (0..67).collect(),
            missing_indices: vec![],
            failed_indices: vec![],
            preparation_seconds: 1.0,
            process_cleanup_seconds: 0.0,
            report_write_offset_ms: 599999,
            workers: vec![],
            windows: vec![],
            resource_samples: vec![],
            peak_aggregate_rss_mib: None,
            min_mem_available_mib: None,
            corpus_manifest: None,
            worker_ready_semantics: "unavailable on ASR wire; ready_offset_ms is null",
        };
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["audio_seconds"], 1800.0);
        assert_eq!(value["under_ten_minutes"], true);
        assert_eq!(value["window_count"], 67);
        assert_eq!(value["model_revision"], "pinned");
        assert_eq!(value["threads"], 12);
        assert_eq!(value["adapter_sha256"], "script-hash");
    }

    #[test]
    fn benchmark_rejects_late_and_duplicate_results_without_counting_them() {
        use std::sync::{Mutex, atomic::AtomicBool};
        use std::time::{Duration, Instant};
        let started = Instant::now();
        let telemetry = super::BenchmarkTelemetry {
            started,
            deadline: started + Duration::from_secs(600),
            stop: AtomicBool::new(false),
            events: Mutex::new(super::BenchmarkEvents::default()),
        };
        telemetry.events.lock().unwrap().workers.insert(
            0,
            super::WorkerTiming {
                worker_id: 0,
                pid: 123,
                start_offset_ms: 0,
                ready_offset_ms: None,
                first_result_offset_ms: None,
                finish_offset_ms: None,
            },
        );
        telemetry
            .record_result(0, 3, started, started + Duration::from_millis(10))
            .unwrap();
        assert!(
            telemetry
                .record_result(1, 3, started, started + Duration::from_millis(20))
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        assert!(
            telemetry
                .record_result(0, 4, started, started + Duration::from_secs(600))
                .unwrap_err()
                .to_string()
                .contains("late")
        );
        assert_eq!(telemetry.events.lock().unwrap().windows.len(), 1);
        let events = telemetry.events.lock().unwrap();
        assert_eq!(events.workers[&0].ready_offset_ms, None);
        assert_eq!(events.workers[&0].first_result_offset_ms, Some(10));
        drop(events);
        telemetry.fail(5);
        assert_eq!(
            telemetry
                .events
                .lock()
                .unwrap()
                .failed_indices
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![5]
        );
    }

    #[test]
    fn benchmark_classifies_partial_timeout_worker_error_and_resource_stop() {
        let missing = [3, 5];
        assert_eq!(
            super::benchmark_status(&[], &missing, 599.0, false),
            "timeout"
        );
        assert_eq!(
            super::benchmark_status(&["worker stdout failed".into()], &missing, 30.0, false),
            "worker_error"
        );
        assert_eq!(
            super::benchmark_status(
                &["600-second benchmark limit".into()],
                &missing,
                600.1,
                false
            ),
            "timeout"
        );
        assert_eq!(
            super::benchmark_status(&["memory below reserve".into()], &missing, 40.0, true),
            "resource_stop"
        );
        assert_eq!(super::benchmark_status(&[], &[], 599.999, false), "success");
        assert_eq!(super::benchmark_status(&[], &[], 600.0, false), "timeout");
    }

    #[test]
    fn benchmark_guard_requires_own_swap_or_psi_with_system_swapout() {
        use std::sync::{
            Mutex,
            atomic::{AtomicBool, Ordering},
        };
        use std::time::{Duration, Instant};
        let started = Instant::now();
        let telemetry = super::BenchmarkTelemetry {
            started,
            deadline: started + Duration::from_secs(600),
            stop: AtomicBool::new(false),
            events: Mutex::new(super::BenchmarkEvents::default()),
        };
        let sample =
            |pswpin, pswpout, psi, own_swap, available| crate::resources::BenchmarkSample {
                offset_ms: 10,
                tracked_process_count: 1,
                aggregate_rss_mib: 519,
                own_vmswap_kib: own_swap,
                mem_available_mib: available,
                psi_memory_full_avg10_percent: psi,
                system_pswpin_pages: Some(pswpin),
                system_pswpout_pages: Some(pswpout),
                load_average_1m: None,
                cpu_frequency_mhz: None,
                temperature_millidegrees: None,
                throttled_periods: None,
            };
        telemetry
            .record_sample(sample(200, 183, 0.0, 0, 12_000), 4096, Some(183), 0)
            .unwrap();
        telemetry
            .record_sample(sample(225, 184, 0.99, 0, 12_000), 4096, Some(183), 0)
            .unwrap();
        assert!(!telemetry.stop.load(Ordering::Relaxed));
        assert!(
            telemetry
                .events
                .lock()
                .unwrap()
                .resource_stop_reason
                .is_none()
        );
        telemetry
            .record_sample(sample(225, 184, 1.0, 0, 12_000), 4096, Some(183), 0)
            .unwrap();
        assert!(telemetry.stop.load(Ordering::Relaxed));
        let events = telemetry.events.lock().unwrap();
        let reason = events.resource_stop_reason.as_ref().unwrap();
        assert!(reason.contains("PSI full avg10 1% reached 1.0%"));
        assert!(reason.contains("not causal attribution"));
        assert_eq!(events.samples.len(), 3);
        assert_eq!(
            super::benchmark_status(std::slice::from_ref(reason), &[0], 4.84, true),
            "resource_stop"
        );
        drop(events);
        let fresh = || super::BenchmarkTelemetry {
            started,
            deadline: started + Duration::from_secs(600),
            stop: AtomicBool::new(false),
            events: Mutex::new(super::BenchmarkEvents::default()),
        };
        let own = fresh();
        own.record_sample(sample(225, 184, 0.0, 1, 12_000), 4096, Some(183), 0)
            .unwrap();
        assert!(own.stop.load(Ordering::Relaxed));
        assert!(
            own.events
                .lock()
                .unwrap()
                .resource_stop_reason
                .as_ref()
                .unwrap()
                .contains("VmSwap")
        );
        let reserve = fresh();
        reserve
            .record_sample(sample(225, 183, 0.0, 0, 4095), 4096, Some(183), 0)
            .unwrap();
        assert!(reserve.stop.load(Ordering::Relaxed));
        let swap_alone = fresh();
        swap_alone
            .record_sample(sample(225, 184, 0.0, 0, 12_000), 4096, Some(183), 0)
            .unwrap();
        assert!(!swap_alone.stop.load(Ordering::Relaxed));
        assert!(
            swap_alone
                .record_sample(sample(225, 182, 0.0, 0, 12_000), 4096, Some(183), 0)
                .is_err()
        );
    }

    #[test]
    fn prepared_corpus_checks_hashes_boundaries_and_never_modifies_clips() {
        let root = tempfile::tempdir().unwrap();
        let base = root.path().join("corpus");
        std::fs::create_dir_all(base.join("clips")).unwrap();
        let source = root.path().join("input.mp3");
        std::fs::write(&source, b"fixture source").unwrap();
        let data = vec![0_u8; 28 * 16000 * 2];
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&[1, 0, 1, 0]);
        wav.extend_from_slice(&16000_u32.to_le_bytes());
        wav.extend_from_slice(&32000_u32.to_le_bytes());
        wav.extend_from_slice(&[2, 0, 16, 0]);
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        let source_wav = base.join("audio.wav");
        let full_audio_bytes = 1801_u32 * 16000 * 2;
        let mut full_header = wav[..44].to_vec();
        full_header[4..8].copy_from_slice(&(36 + full_audio_bytes).to_le_bytes());
        full_header[40..44].copy_from_slice(&full_audio_bytes.to_le_bytes());
        std::fs::write(&source_wav, full_header).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&source_wav)
            .unwrap()
            .set_len(44 + u64::from(full_audio_bytes))
            .unwrap();
        let chunks = crate::audio::chunks(1800.0, 28.0, 0.8).unwrap();
        let windows: Vec<_> = chunks
            .iter()
            .enumerate()
            .map(|(index, interval)| {
                let path = base.join("clips").join(format!("{index:05}.wav"));
                let samples = ((interval.end - interval.start) * 16000.0).round() as usize;
                let pcm = vec![0_u8; samples * 2];
                let mut clip = wav[..40].to_vec();
                clip[4..8].copy_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
                clip.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
                clip.extend_from_slice(&pcm);
                std::fs::write(&path, &clip).unwrap();
                serde_json::json!({
                    "index": index, "start": interval.start, "end": interval.end,
                    "samples": samples, "wav_path": format!("corpus/clips/{index:05}.wav"),
                    "wav_sha256": super::hash_file(&path).unwrap(),
                    "pcm_sha256": hex::encode(sha2::Sha256::digest(&pcm))
                })
            })
            .collect();
        let manifest = root.path().join("corpus.json");
        let mut value = serde_json::json!({
            "schema_version": 1, "source_sha256": super::hash_file(&source).unwrap(),
            "audio_seconds": 1800.0,
            "wav": {"path":"corpus/audio.wav", "sha256":super::hash_file(&source_wav).unwrap()},
            "windows": windows
        });
        super::write_json(&manifest, &value).unwrap();
        let output = root.path().join("run");
        assert_eq!(
            super::verify_corpus(&manifest, &source, &output, true)
                .unwrap()
                .1
                .len(),
            67
        );
        assert!(!output.exists());
        let parent_windows: Vec<_> = value["windows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|window| {
                serde_json::json!({
                    "index": window["index"], "start": window["start"], "end": window["end"],
                    "frames": window["samples"], "path": window["wav_path"],
                    "sha256": window["wav_sha256"], "pcm_sha256": window["pcm_sha256"]
                })
            })
            .collect();
        let parent_manifest = root.path().join("study-corpus.json");
        super::write_json(
            &parent_manifest,
            &serde_json::json!({
                "schema": 1, "audio_path": source_wav,
                "audio_sha256": super::hash_file(&source_wav).unwrap(),
                "duration_seconds": 1800.0, "audio_duration_seconds": 1801.0,
                "clips": parent_windows,
                "source_model": {"repo": "fixture", "revision": "fixture"}
            }),
        )
        .unwrap();
        assert!(super::verify_corpus(&parent_manifest, &source_wav, &output, true).is_ok());
        assert!(super::verify_corpus(&parent_manifest, &source, &output, true).is_err());
        value["wav"]["path"] = serde_json::json!(source_wav);
        super::write_json(&manifest, &value).unwrap();
        assert!(super::verify_corpus(&manifest, &source, &output, true).is_ok());
        value["windows"][23]["start"] = serde_json::json!(0.0);
        super::write_json(&manifest, &value).unwrap();
        assert!(super::verify_corpus(&manifest, &source, &output, true).is_err());
        value["windows"][23]["start"] = serde_json::json!(chunks[23].start);
        super::write_json(&manifest, &value).unwrap();
        std::fs::write(base.join("clips/00023.wav"), b"tampered").unwrap();
        assert!(super::verify_corpus(&manifest, &source, &output, true).is_err());
    }

    #[test]
    fn repetition_metric_flags_loops_without_deleting_transcript() {
        assert_eq!(super::repeated_fourgram_fraction("eins zwei drei"), 0.0);
        assert_eq!(
            super::repeated_fourgram_fraction("eins zwei drei vier"),
            0.0
        );
        assert!(
            super::repeated_fourgram_fraction(
                "eins zwei drei vier eins zwei drei vier eins zwei drei vier"
            ) > 0.5
        );
    }

    #[test]
    fn flags_unique_speech_claims_without_diarization_but_preserves_them() {
        let hypothesis = |text: &str| Hypothesis {
            text: text.into(),
            language: None,
            segments: vec![],
            raw_text: None,
            out_of_bounds_segments: vec![],
            processing_mode: None,
            model_revision: None,
        };
        let chunk = Interval {
            start: 0.0,
            end: 28.0,
        };
        let group = vec![
            ("primeline".into(), hypothesis("")),
            ("flix".into(), hypothesis("")),
            ("qwen".into(), hypothesis("unique words")),
        ];
        let no_speech = Diarization {
            turns: vec![],
            exclusive_turns: vec![],
        };
        assert_eq!(
            review_flags(&chunk, &group, &no_speech),
            ["single_asr_hypothesis_without_diarized_speech"]
        );
        assert_eq!(group[2].1.text, "unique words");
        let speech = Diarization {
            turns: vec![Turn {
                start: 1.0,
                end: 2.0,
                speaker: "SPEAKER_00".into(),
            }],
            exclusive_turns: vec![],
        };
        assert!(review_flags(&chunk, &group, &speech).is_empty());
        let repeating = vec![(
            "flix".into(),
            hypothesis("eins zwei drei vier eins zwei drei vier eins zwei drei vier"),
        )];
        assert!(review_flags(&chunk, &repeating, &speech).contains(&"repeated_asr_text"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn failed_disk_write_does_not_publish_a_checkpoint() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("00000.json");
        symlink("/dev/full", dir.path().join("00000.partial")).unwrap();
        let error =
            super::write_json(&final_path, &serde_json::json!({"text": "saved"})).unwrap_err();
        assert!(
            error.to_string().contains("No space left on device"),
            "{error:#}"
        );
        assert!(!final_path.exists());
    }

    #[test]
    fn rejects_out_of_clip_timestamps() {
        let value = Hypothesis {
            text: "test".into(),
            language: None,
            segments: vec![WordSpan {
                start: 0.0,
                end: 3.0,
                text: "test".into(),
            }],
            raw_text: None,
            out_of_bounds_segments: Vec::new(),
            processing_mode: None,
            model_revision: None,
        };
        assert!(validate_hypothesis(&value, 1.0).is_err());
    }

    #[test]
    fn whisper_clip_boundary_roundoff_is_not_a_timing_anomaly() {
        let duration = 84.0_f64 - 81.60000000000001_f64;
        assert!(duration < 2.4);
        let mut value = Hypothesis {
            text: "ending".into(),
            language: Some("de".into()),
            segments: vec![WordSpan {
                start: 0.14,
                end: 2.4,
                text: "ending".into(),
            }],
            raw_text: None,
            out_of_bounds_segments: Vec::new(),
            processing_mode: None,
            model_revision: None,
        };
        assert_eq!(clip_whisper_timestamps(&mut value, duration), 0);
        assert_eq!(value.segments[0].end, duration);
        assert_eq!(value.text, "ending");
        assert!(value.raw_text.is_none());
        assert!(value.out_of_bounds_segments.is_empty());
    }

    #[test]
    fn preserves_whisper_hallucinations_outside_the_audio_for_review() {
        let mut value = Hypothesis {
            text: " real partial invented".into(),
            language: Some("german".into()),
            segments: vec![
                WordSpan {
                    start: 0.0,
                    end: 26.0,
                    text: " real".into(),
                },
                WordSpan {
                    start: 26.0,
                    end: 29.0,
                    text: " partial".into(),
                },
                WordSpan {
                    start: 28.0,
                    end: 30.0,
                    text: " invented".into(),
                },
            ],
            raw_text: None,
            out_of_bounds_segments: Vec::new(),
            processing_mode: None,
            model_revision: None,
        };
        assert_eq!(clip_whisper_timestamps(&mut value, 28.0), 2);
        assert_eq!(value.text, " real partial");
        assert_eq!(value.raw_text.as_deref(), Some(" real partial invented"));
        assert_eq!(value.out_of_bounds_segments.len(), 2);
        assert_eq!(value.segments[1].end, 28.0);
        assert!(validate_hypothesis(&value, 28.0).is_ok());
    }

    #[test]
    fn matches_out_of_order_responses_to_their_own_windows() {
        let temp = tempfile::tempdir().unwrap();
        let script = temp.path().join("worker.py");
        fs::write(
            &script,
            r#"
import json, sys
requests = [json.loads(sys.stdin.readline()) for _ in range(3)]
for request in reversed(requests):
    print(json.dumps({"index": request["index"], "text": str(request["index"]), "language": "de", "segments": []}), flush=True)
"#,
        )
        .unwrap();
        let clips = temp.path().join("clips");
        fs::create_dir(&clips).unwrap();
        for index in 0..3 {
            fs::write(clips.join(format!("{index:05}.wav")), b"sample").unwrap();
        }
        let config = Config {
            models: ["primeline", "flix", "qwen", "diarization"]
                .iter()
                .map(|role| ModelSpec {
                    role: (*role).into(),
                    repo: (*role).into(),
                    revision: "pinned".into(),
                    path: temp.path().join(role),
                })
                .collect(),
            asr: [(
                "qwen".into(),
                Adapter {
                    program: "python3".into(),
                    script,
                    engine: None,
                    threads: 1,
                    max_in_flight: 3,
                    args: vec![],
                },
            )]
            .into(),
            diarization: Adapter {
                program: "python3".into(),
                script: temp.path().join("unused.py"),
                engine: None,
                threads: 1,
                max_in_flight: 1,
                args: vec![],
            },
            llm: Llm {
                program: "unused".into(),
                args: vec![],
                ..Llm::default()
            },
            review_rules: None,
            resources: Default::default(),
            quality: Default::default(),
            asr_selection: None,
        };
        let chunks = vec![
            Interval {
                start: 0.0,
                end: 1.0,
            };
            3
        ];
        run_asr(
            &config,
            "qwen",
            "qwen",
            &clips,
            temp.path(),
            &chunks,
            WorkerMode {
                concurrent: false,
                offload: false,
                worker_number: 0,
                worker_count: 1,
                deadline: None,
                benchmark: None,
                execution: None,
                retry: None,
                task: None,
            },
        )
        .unwrap();
        for index in 0..3 {
            let value: Hypothesis =
                super::read_json(&temp.path().join("qwen").join(format!("{index:05}.json")))
                    .unwrap();
            assert_eq!(value.text, index.to_string());
            assert_eq!(value.model_revision.as_deref(), Some("pinned"));
        }
        fs::remove_dir_all(temp.path().join("qwen")).unwrap();
        fs::write(
            &config.asr("qwen").script,
            r#"
import json, sys
for _ in range(3):
    json.loads(sys.stdin.readline())
print(json.dumps({"index": 99, "text": "wrong window", "segments": []}), flush=True)
"#,
        )
        .unwrap();
        let error = run_asr(
            &config,
            "qwen",
            "qwen",
            &clips,
            temp.path(),
            &chunks,
            WorkerMode {
                concurrent: false,
                offload: false,
                worker_number: 0,
                worker_count: 1,
                deadline: None,
                benchmark: None,
                execution: None,
                retry: None,
                task: None,
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unexpected or duplicate chunk 99")
        );
        assert!(!temp.path().join("qwen/00000.json").exists());
    }

    #[test]
    fn selected_stages_run_only_prerequisites_and_never_descendants() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let input = root.join("input.mp3");
        assert!(
            Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=duration=2",
                    "-y"
                ])
                .arg(&input)
                .status()
                .unwrap()
                .success()
        );
        let asr = root.join("asr.py");
        fs::write(
            &asr,
            r#"
import argparse,json,sys
p=argparse.ArgumentParser()
for key in ("engine","model","threads"):p.add_argument("--"+key)
p.add_argument("--serve",action="store_true")
p.parse_args()
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({"text":"Guten Tag","language":"de","segments":[]}),flush=True)
"#,
        )
        .unwrap();
        let dia = root.join("dia.py");
        fs::write(
            &dia,
            r#"
import argparse,json,pathlib
p=argparse.ArgumentParser()
for key in ("model","input","output","threads"):p.add_argument("--"+key)
a=p.parse_args()
pathlib.Path(a.output).write_text(json.dumps({"turns":[],"exclusive_turns":[]}))
"#,
        )
        .unwrap();
        let models = ["primeline","flix","qwen","diarization"].into_iter().map(|role| {
            let path = root.join(format!("model-{role}"));
            if role == "primeline" || role == "diarization" {
                fs::create_dir(&path).unwrap();
                fs::write(path.join("weights"),b"test").unwrap();
                fs::write(path.join(".shum-manifest.json"),
                    serde_json::to_vec(&serde_json::json!({"repo":role,"revision":"test","files":{"weights":hash_file(&path.join("weights")).unwrap()}})).unwrap()).unwrap();
            }
            ModelSpec {role:role.into(),repo:role.into(),revision:"test".into(),path}
        }).collect();
        let adapter = Adapter {
            program: "python3".into(),
            script: asr,
            engine: None,
            threads: 1,
            max_in_flight: 1,
            args: vec![],
        };
        let config = Config {
            models,
            asr: ["primeline", "flix", "qwen"]
                .into_iter()
                .map(|r| (r.into(), adapter.clone()))
                .collect(),
            diarization: Adapter {
                script: dia,
                ..adapter
            },
            llm: Llm::default(),
            review_rules: None,
            quality: Default::default(),
            asr_selection: None,
            resources: crate::config::Resources {
                reserve_mib: 0,
                supervise_asr_with_systemd: false,
                estimated_peak_mib: [("primeline".into(), 1)].into(),
                ..Default::default()
            },
        };
        let output = root.join("stages");
        for step in [
            workflow::Step::Audio,
            workflow::Step::Chunks,
            workflow::Step::AsrPrimeline,
        ] {
            assert_eq!(
                workflow::run_stage(&config, &output, step, Some(&input), true).unwrap(),
                Outcome::StageComplete
            );
        }
        assert!(output.join("primeline/00000.json").is_file());
        assert!(!output.join("flix").exists());
        assert!(!output.join("qwen").exists());
        assert!(!output.join("diarization.json").exists());
        assert!(!output.join("reviews").exists());
        assert!(!workflow::export_paths(&output).unwrap().txt.exists());
        assert_eq!(
            workflow::status(&output).unwrap().stages["asr:primeline"].status,
            workflow::State::Ready
        );
        let separate = root.join("dia-only");
        assert_eq!(
            workflow::run_stage(
                &config,
                &separate,
                workflow::Step::Diarization,
                Some(&input),
                true
            )
            .unwrap(),
            Outcome::StageComplete
        );
        assert!(separate.join("diarization.json").exists());
        assert!(!separate.join("chunks.json").exists());
        assert!(!separate.join("primeline").exists());
    }

    #[test]
    fn runs_three_asr_processes_in_parallel_and_resumes() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let asr = root.join("asr.py");
        fs::write(
            &asr,
            r#"
import argparse, json, pathlib, sys, time
p = argparse.ArgumentParser()
for name in ("engine", "model", "threads"):
    p.add_argument("--" + name)
p.add_argument("--offload-dir")
p.add_argument("--serve", action="store_true")
a = p.parse_args()
assert a.serve
marker = pathlib.Path(a.model).parent / (a.engine + ".started")
with marker.open("a") as stream:
    stream.write("started\n")
if a.offload_dir:
    assert pathlib.Path(a.offload_dir).is_dir()
    (marker.parent / (a.engine + ".offloaded")).write_text(a.offload_dir)
sequential = marker.parent / "sequential"
running = marker.parent / "running"
if a.engine == "qwen" and (marker.parent / "force-oom").exists():
    block = bytearray(200 * 1024 * 1024)
    block[:] = b"x" * len(block)
    time.sleep(2)
if a.engine == "qwen" and (marker.parent / "force-error").exists():
    print(json.dumps({"error": "decode timeout"}), flush=True)
    sys.exit(1)
if sequential.exists():
    assert not running.exists(), "ASR processes overlapped in sequential mode"
    running.write_text(a.engine)
deadline = time.monotonic() + 5
while not sequential.exists() and len(list(marker.parent.glob("*.started"))) != 3:
    if time.monotonic() > deadline:
        raise RuntimeError("ASR processes were not running in parallel")
    time.sleep(0.02)
for request in sys.stdin:
    clip = pathlib.Path(json.loads(request)["input"])
    assert clip.is_file()
    print(json.dumps({"text":"Guten Tag " + clip.stem, "language":"de", "segments":[]}), flush=True)
if sequential.exists():
    running.unlink()
"#,
        )
        .unwrap();
        let dia = root.join("dia.py");
        fs::write(&dia, r#"
import argparse, json, pathlib
p = argparse.ArgumentParser()
for name in ("model", "input", "output", "threads"):
    p.add_argument("--" + name)
a = p.parse_args()
pathlib.Path(a.output).write_text(json.dumps({"turns":[{"start":0.0,"end":70.0,"speaker":"SPEAKER_00"}],"exclusive_turns":[{"start":0.0,"end":70.0,"speaker":"SPEAKER_00"}]}))
"#).unwrap();
        let llm = root.join("llm.py");
        fs::write(&llm, r#"
import json, pathlib, sys
prompt = json.load(sys.stdin)
assert len(prompt["intervals"][0]["hypotheses"]) == 3
pathlib.Path(sys.argv[1]).write_text(json.dumps({"segments":[{"index":s["index"],"text":"Guten Tag " + str(s["index"]),"language":"de","gap":False} for s in prompt["intervals"]]}))
"#).unwrap();
        let input = root.join("sample.mp3");
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=70",
                "-y",
            ])
            .arg(&input)
            .status()
            .unwrap();
        assert!(status.success());
        for role in ["primeline", "flix", "qwen", "diarization"] {
            let model_dir = root.join(role);
            fs::create_dir_all(&model_dir).unwrap();
            let weights = model_dir.join("weights.bin");
            fs::write(&weights, b"mock model weights").unwrap();
            fs::write(
                model_dir.join(".shum-manifest.json"),
                serde_json::to_vec(&serde_json::json!({
                    "repo": role,
                    "revision": "test-revision",
                    "files": { "weights.bin": super::hash_file(&weights).unwrap() }
                }))
                .unwrap(),
            )
            .unwrap();
        }
        let mut config = Config {
            models: ["primeline", "flix", "qwen", "diarization"]
                .iter()
                .map(|role| ModelSpec {
                    role: (*role).to_owned(),
                    repo: role.to_string(),
                    revision: "test-revision".to_owned(),
                    path: root.join(role),
                })
                .collect(),
            asr: ["primeline", "flix", "qwen"]
                .iter()
                .map(|role| {
                    (
                        (*role).to_owned(),
                        Adapter {
                            program: PathBuf::from("python3"),
                            script: asr.clone(),
                            engine: None,
                            threads: 1,
                            max_in_flight: 1,
                            args: Vec::new(),
                        },
                    )
                })
                .collect(),
            diarization: Adapter {
                program: PathBuf::from("python3"),
                script: dia,
                engine: None,
                threads: 1,
                max_in_flight: 1,
                args: Vec::new(),
            },
            llm: Llm {
                program: PathBuf::from("python3"),
                args: vec![llm.to_string_lossy().into_owned(), "{output}".into()],
                enabled: true,
                ..Llm::default()
            },
            review_rules: None,
            quality: Default::default(),
            asr_selection: None,
            resources: crate::config::Resources {
                reserve_mib: 0,
                max_asr_workers: 3,
                max_parallel_asr_models: None,
                sequential_asr_roles: false,
                estimated_peak_mib: ["primeline", "flix", "qwen"]
                    .into_iter()
                    .map(|role| (role.to_owned(), 1))
                    .collect(),
                asr_worker_counts: Default::default(),
                asr_worker_peak_mib: Default::default(),
                supervise_asr_with_systemd: false,
                systemd_memory_limit_mib: None,
                systemd_swap_limit_mib: None,
                oom_retries: 2,
                offload_dir: None,
                offload_roles: vec!["flix".to_owned(), "qwen".to_owned()],
                offload_min_free_mib: 8192,
                offload_estimated_peak_mib: [("flix".to_owned(), 6500), ("qwen".to_owned(), 7000)]
                    .into(),
            },
        };
        let output = root.join("job");
        transcribe(&config, &input, &output, false).unwrap();
        let unavailable_asr = root.join("asr.unavailable");
        let unavailable_dia = root.join("dia.unavailable");
        fs::rename(&asr, &unavailable_asr).unwrap();
        fs::rename(&config.diarization.script, &unavailable_dia).unwrap();
        for role in ["primeline", "flix", "qwen", "diarization"] {
            fs::rename(root.join(role), root.join(format!("{role}.unavailable"))).unwrap();
        }
        assert_eq!(
            crate::workflow::rerun(&config, &output, crate::workflow::Step::Review, true).unwrap(),
            crate::workflow::Outcome::Complete
        );
        assert_eq!(
            crate::workflow::rerun(&config, &output, crate::workflow::Step::Export, true).unwrap(),
            crate::workflow::Outcome::Complete
        );
        assert!(
            crate::workflow::status(&output).unwrap().stages["asr:flix"]
                .warning
                .is_some()
        );
        for engine in ["nemo", "whisper", "qwen"] {
            assert_eq!(
                fs::read_to_string(root.join(format!("{engine}.started"))).unwrap(),
                "started\n"
            );
        }
        fs::rename(unavailable_asr, &asr).unwrap();
        fs::rename(&unavailable_dia, &config.diarization.script).unwrap();
        for role in ["primeline", "flix", "qwen", "diarization"] {
            fs::rename(root.join(format!("{role}.unavailable")), root.join(role)).unwrap();
        }
        fs::rename(&config.diarization.script, &unavailable_dia).unwrap();
        assert!(
            crate::workflow::rerun(&config, &output, crate::workflow::Step::Diarization, true)
                .unwrap_err()
                .to_string()
                .contains("diarization")
        );
        assert_eq!(
            crate::workflow::status(&output).unwrap().stages["asr:flix"].status,
            crate::workflow::State::Ready
        );
        fs::rename(unavailable_dia, &config.diarization.script).unwrap();
        assert_eq!(
            crate::workflow::resume(&config, &output, true).unwrap(),
            crate::workflow::Outcome::Complete
        );
        config.llm.enabled = false;
        assert_eq!(
            crate::workflow::rerun(&config, &output, crate::workflow::Step::Review, true).unwrap(),
            crate::workflow::Outcome::PendingReview
        );
        assert!(!workflow::export_paths(&output).unwrap().txt.exists());
        assert!(output.join("review-flags.json").exists());
        assert_eq!(
            crate::workflow::status(&output).unwrap().stages["review"].status,
            crate::workflow::State::PendingExternal
        );
        for engine in ["nemo", "whisper", "qwen"] {
            assert_eq!(
                fs::read_to_string(root.join(format!("{engine}.started"))).unwrap(),
                "started\n"
            );
        }
        config.llm.enabled = true;
        config.llm.probe_args = Some(vec![llm.to_string_lossy().into_owned()]);
        assert_eq!(
            crate::workflow::resume(&config, &output, true).unwrap(),
            crate::workflow::Outcome::PendingReview
        );
        assert!(
            crate::workflow::status(&output).unwrap().stages["review"]
                .warning
                .as_ref()
                .unwrap()
                .contains("LLM access check")
        );
        assert!(!workflow::export_paths(&output).unwrap().txt.exists());
        config.llm.probe_args = None;
        assert_eq!(
            crate::workflow::resume(&config, &output, true).unwrap(),
            crate::workflow::Outcome::Complete
        );
        let probe = root.join("probe.py");
        let probe_calls = root.join("probe.calls");
        fs::write(
            &probe,
            format!(
                "import pathlib, sys\nassert sys.stdin.read() == 'Reply with exactly SHUM_READY. Do not call tools or read files.'\nwith pathlib.Path({:?}).open('a') as stream:\n    stream.write('1')\nprint('SHUM_READY')\n",
                probe_calls.to_string_lossy()
            ),
        )
        .unwrap();
        config.llm.probe_args = Some(vec![probe.to_string_lossy().into_owned()]);
        assert_eq!(
            crate::llm::readiness(&config.llm),
            crate::llm::Readiness::Ready
        );
        assert_eq!(
            crate::workflow::rerun(&config, &output, crate::workflow::Step::Review, true).unwrap(),
            crate::workflow::Outcome::Complete
        );
        assert_eq!(fs::read_to_string(&probe_calls).unwrap(), "1");
        config.llm.probe_args = None;
        let healthy_llm = fs::read(&llm).unwrap();
        fs::write(&llm, "print('invalid response')\n").unwrap();
        assert_eq!(
            crate::workflow::rerun(&config, &output, crate::workflow::Step::Review, true).unwrap(),
            crate::workflow::Outcome::PendingReview
        );
        assert!(!workflow::export_paths(&output).unwrap().json.exists());
        assert!(output.join("review-flags.json").exists());
        fs::write(&llm, healthy_llm).unwrap();
        assert_eq!(
            crate::workflow::resume(&config, &output, true).unwrap(),
            crate::workflow::Outcome::Complete
        );
        transcribe(&config, &input, &output, true).unwrap();
        let transcript: Transcript = serde_json::from_slice(
            &fs::read(workflow::export_paths(&output).unwrap().json).unwrap(),
        )
        .unwrap();
        assert!(transcript.segments.len() >= 2);
        assert_eq!(
            transcript.segments[0].speaker.as_deref(),
            Some("SPEAKER_00")
        );
        assert_eq!(transcript.segments[0].hypotheses.len(), 3);
        assert_eq!(transcript.segments[0].text, "Guten Tag 0");
        for engine in ["nemo", "whisper", "qwen"] {
            assert_eq!(
                fs::read_to_string(root.join(format!("{engine}.started"))).unwrap(),
                "started\n"
            );
        }
        for role in ["primeline", "flix", "qwen"] {
            fs::remove_dir_all(output.join(role)).unwrap();
        }
        fs::write(root.join("sequential"), "").unwrap();
        transcribe(&config, &input, &output, true).unwrap();
        for engine in ["nemo", "whisper", "qwen"] {
            assert_eq!(
                fs::read_to_string(root.join(format!("{engine}.started"))).unwrap(),
                "started\nstarted\n"
            );
        }
        for role in ["primeline", "flix", "qwen"] {
            fs::remove_dir_all(output.join(role)).unwrap();
            config
                .resources
                .estimated_peak_mib
                .insert(role.to_owned(), u64::MAX / 4);
        }
        transcribe(&config, &input, &output, false).unwrap();
        for engine in ["nemo", "whisper", "qwen"] {
            assert_eq!(
                fs::read_to_string(root.join(format!("{engine}.started"))).unwrap(),
                "started\nstarted\nstarted\n"
            );
        }
        fs::remove_file(root.join("sequential")).unwrap();
        for role in ["primeline", "flix", "qwen"] {
            fs::remove_dir_all(output.join(role)).unwrap();
        }
        config.resources.asr_worker_counts.insert("flix".into(), 3);
        config
            .resources
            .asr_worker_peak_mib
            .insert("flix".into(), 1);
        transcribe(&config, &input, &output, false).unwrap();
        assert_eq!(
            fs::read_to_string(root.join("whisper.started")).unwrap(),
            "started\n".repeat(6)
        );
        for index in 0..3 {
            assert!(
                output
                    .join("flix")
                    .join(format!("{index:05}.json"))
                    .exists()
            );
            assert!(
                output
                    .join("flix")
                    .join(format!("worker-{index:02}.stderr"))
                    .exists()
            );
        }
        let log = fs::read_to_string(output.join("resource-events.jsonl")).unwrap();
        let batches: Vec<serde_json::Value> = log
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(batches.iter().any(|event| {
            event["type"] == "window"
                && event["role"] == "primeline"
                && event["index"] == 0
                && event["audio_seconds"] == 28.0
                && event["characters"].as_u64().is_some_and(|chars| chars > 0)
                && event["repeated_fourgram_fraction"].as_f64().is_some()
        }));
        assert!(
            batches
                .iter()
                .any(|event| { event["worker_counts"] == serde_json::json!([["flix", 3]]) })
        );
        let last_admission = batches
            .iter()
            .rev()
            .find(|event| event.get("roles").is_some())
            .unwrap();
        assert_eq!(last_admission["roles"], serde_json::json!(["qwen"]));
        assert_eq!(last_admission["using_swap_as_safety_net"], true);
        if std::env::var_os("SHUM_TEST_SYSTEMD").is_some() {
            let started_before: std::collections::BTreeMap<_, _> = ["nemo", "whisper", "qwen"]
                .into_iter()
                .map(|engine| {
                    (
                        engine,
                        fs::read_to_string(root.join(format!("{engine}.started"))).unwrap(),
                    )
                })
                .collect();
            for role in ["primeline", "flix", "qwen"] {
                fs::remove_dir_all(output.join(role)).unwrap();
            }
            config.resources.supervise_asr_with_systemd = true;
            config.resources.systemd_swap_limit_mib = Some(0);
            transcribe(&config, &input, &output, true).unwrap();
            let events = fs::read_to_string(output.join("resource-events.jsonl")).unwrap();
            assert!(events.lines().any(|line| {
                let event: serde_json::Value = serde_json::from_str(line).unwrap();
                event["type"] == "worker"
                    && event["role"] == "primeline"
                    && event["memory_peak_mib"]
                        .as_u64()
                        .is_some_and(|peak| peak > 0)
            }));
            for engine in ["nemo", "whisper", "qwen"] {
                assert_eq!(
                    fs::read_to_string(root.join(format!("{engine}.started"))).unwrap(),
                    format!("{}started\n", started_before[engine])
                );
            }
            fs::remove_dir_all(output.join("qwen")).unwrap();
            fs::write(root.join("force-oom"), "").unwrap();
            config.resources.offload_dir = Some(root.join("private-offload"));
            config.resources.offload_min_free_mib = 1;
            config
                .resources
                .estimated_peak_mib
                .insert("qwen".to_owned(), 1);
            config.resources.systemd_memory_limit_mib = Some(64);
            config.resources.oom_retries = 1;
            assert!(
                transcribe(&config, &input, &output, true)
                    .unwrap_err()
                    .to_string()
                    .contains("waiting for resources")
            );
            assert!(output.join("waiting-resources.json").exists());
            let events = fs::read_to_string(output.join("resource-events.jsonl")).unwrap();
            assert!(events.lines().any(|line| {
                let event: serde_json::Value = serde_json::from_str(line).unwrap();
                event["role"] == "qwen"
                    && event["mode"] == "disk_offload"
                    && event["result"] == "oom-kill"
            }));
            assert!(
                transcribe(&config, &input, &output, true)
                    .unwrap_err()
                    .to_string()
                    .contains("waiting for resources")
            );
            fs::remove_file(root.join("force-oom")).unwrap();
            config.resources.systemd_memory_limit_mib = None;
            transcribe(&config, &input, &output, true).unwrap();
            assert!(!output.join("waiting-resources.json").exists());
            fs::remove_dir_all(output.join("qwen")).unwrap();
            fs::write(root.join("force-error"), "").unwrap();
            assert!(
                transcribe(&config, &input, &output, true)
                    .unwrap_err()
                    .to_string()
                    .contains("decode timeout")
            );
            assert!(!output.join("waiting-resources.json").exists());
            fs::remove_file(root.join("force-error")).unwrap();
            transcribe(&config, &input, &output, true).unwrap();
        }
        fs::remove_dir_all(output.join("flix")).unwrap();
        config.resources.asr_worker_counts.remove("flix");
        config.resources.asr_worker_peak_mib.remove("flix");
        config.resources.offload_dir = Some(root.join("private-offload"));
        config.resources.offload_min_free_mib = 1;
        transcribe(&config, &input, &output, false).unwrap();
        let offloaded: Hypothesis = super::read_json(&output.join("flix/00000.json")).unwrap();
        assert_eq!(offloaded.processing_mode.as_deref(), Some("disk_offload"));
        assert_eq!(offloaded.model_revision.as_deref(), Some("test-revision"));
        assert_eq!(
            fs::read_to_string(root.join("whisper.offloaded")).unwrap(),
            root.join("private-offload").to_string_lossy()
        );
        fs::remove_dir_all(output.join("qwen")).unwrap();
        config
            .resources
            .estimated_peak_mib
            .insert("qwen".to_owned(), u64::MAX / 4);
        config.resources.offload_min_free_mib = u64::MAX;
        assert!(
            transcribe(&config, &input, &output, false)
                .unwrap_err()
                .to_string()
                .contains("offload directory")
        );
        assert!(!output.join("qwen/00000.json").exists());
        config.resources.offload_min_free_mib = 1;
        transcribe(&config, &input, &output, false).unwrap();
        let offloaded: Hypothesis = super::read_json(&output.join("qwen/00000.json")).unwrap();
        assert_eq!(offloaded.processing_mode.as_deref(), Some("disk_offload"));
        assert_eq!(
            fs::read_to_string(root.join("qwen.offloaded")).unwrap(),
            root.join("private-offload").to_string_lossy()
        );
        if std::env::var_os("SHUM_TEST_BENCHMARK").is_some() {
            fs::remove_file(root.join("sequential")).unwrap();
            let long_input = root.join("30-minutes.mp3");
            assert!(
                Command::new("ffmpeg")
                    .args([
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=16000:cl=mono",
                        "-t",
                        "1801",
                        "-y",
                    ])
                    .arg(&long_input)
                    .status()
                    .unwrap()
                    .success()
            );
            let benchmark = root.join("benchmark");
            let blocked = root.join("blocked-benchmark");
            assert!(
                benchmark_asr(
                    &config,
                    &long_input,
                    &blocked,
                    "primeline",
                    super::BenchmarkOptions {
                        offload: false,
                        workers: 2,
                        worker_peak_mib: Some(u64::MAX),
                        corpus_manifest: None,
                    },
                )
                .unwrap_err()
                .to_string()
                .contains("resource stop before worker launch")
            );
            let blocked_report: serde_json::Value =
                super::read_json(&blocked.join("benchmark.json")).unwrap();
            assert_eq!(blocked_report["status"], "resource_stop");
            assert_eq!(blocked_report["completed_indices"], serde_json::json!([]));
            assert_eq!(
                blocked_report["missing_indices"].as_array().unwrap().len(),
                67
            );
            assert_eq!(blocked_report["workers"], serde_json::json!([]));
            benchmark_asr(
                &config,
                &long_input,
                &benchmark,
                "primeline",
                super::BenchmarkOptions {
                    offload: false,
                    workers: 2,
                    worker_peak_mib: Some(1),
                    corpus_manifest: None,
                },
            )
            .unwrap();
            let report: serde_json::Value =
                super::read_json(&benchmark.join("benchmark.json")).unwrap();
            assert_eq!(report["audio_seconds"], 1800.0);
            assert_eq!(report["window_count"], 67);
            assert_eq!(report["worker_count"], 2);
            assert_eq!(report["under_ten_minutes"], true);
            assert_eq!(
                (0..67)
                    .filter(|i| benchmark
                        .join("primeline")
                        .join(format!("{i:05}.json"))
                        .exists())
                    .count(),
                67
            );
            let expired = root.join("expired-benchmark");
            let chunks = crate::audio::chunks(1800.0, 28.0, 0.8).unwrap();
            let error = super::run_asr(
                &config,
                "primeline",
                "test",
                &benchmark.join("clips"),
                &expired,
                &chunks,
                super::WorkerMode {
                    concurrent: false,
                    offload: false,
                    worker_number: 0,
                    worker_count: 1,
                    deadline: Some(std::time::Instant::now()),
                    benchmark: None,
                    execution: None,
                    retry: None,
                    task: None,
                },
            )
            .unwrap_err();
            assert!(error.to_string().contains("600-second benchmark limit"));
            assert!(!expired.join("primeline/00000.json").exists());
            assert!(
                benchmark_asr(
                    &config,
                    &long_input,
                    &benchmark,
                    "primeline",
                    super::BenchmarkOptions {
                        offload: false,
                        workers: 2,
                        worker_peak_mib: Some(1),
                        corpus_manifest: None,
                    },
                )
                .is_err()
            );
            let diarization_benchmark = root.join("diarization-benchmark");
            benchmark_diarization(&config, &long_input, &diarization_benchmark).unwrap();
            let report: serde_json::Value =
                super::read_json(&diarization_benchmark.join("benchmark.json")).unwrap();
            assert_eq!(report["audio_seconds"], 1800.0);
            assert_eq!(report["role"], "diarization");
            assert_eq!(report["under_ten_minutes"], true);
            assert!(diarization_benchmark.join("diarization.json").exists());
        }
        let flix_script = root.join("flix.py");
        config.asr.get_mut("flix").unwrap().script = flix_script.clone();
        fs::remove_file(output.join("diarization.json")).unwrap();
        assert!(
            crate::workflow::rerun(&config, &output, crate::workflow::Step::AsrFlix, true).is_err()
        );
        let stages = crate::workflow::status(&output).unwrap().stages;
        assert_eq!(stages["diarization"].status, crate::workflow::State::Ready);
        assert_eq!(
            stages["asr:primeline"].status,
            crate::workflow::State::Ready
        );
        assert_eq!(stages["asr:flix"].status, crate::workflow::State::Failed);
        assert!(!workflow::export_paths(&output).unwrap().txt.exists());
        fs::copy(&asr, &flix_script).unwrap();
        assert_eq!(
            crate::workflow::resume(&config, &output, true).unwrap(),
            crate::workflow::Outcome::Complete
        );
        let weights = root.join("flix/weights.bin");
        fs::write(&weights, b"corrupted").unwrap();
        assert!(
            crate::workflow::rerun(&config, &output, crate::workflow::Step::AsrFlix, true)
                .unwrap_err()
                .to_string()
                .contains("checksum")
        );
        let stages = crate::workflow::status(&output).unwrap().stages;
        assert_eq!(stages["asr:flix"].status, crate::workflow::State::Failed);
        assert_eq!(stages["diarization"].status, crate::workflow::State::Ready);
        fs::write(weights, b"mock model weights").unwrap();
        assert_eq!(
            crate::workflow::resume(&config, &output, true).unwrap(),
            crate::workflow::Outcome::Complete
        );
    }
}
