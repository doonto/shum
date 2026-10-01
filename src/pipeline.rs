use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use wait_timeout::ChildExt;

use crate::{
    audio::{self, Interval},
    config::{Adapter, Config},
    process, resources,
};

const ROLES: [(&str, &str); 3] = [("primeline", "nemo"), ("flix", "whisper"), ("qwen", "qwen")];
static WORKER_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, PartialEq)]
enum WorkerOutcome {
    Complete,
    OutOfMemory,
}

#[derive(Clone, Copy)]
struct WorkerMode {
    concurrent: bool,
    offload: bool,
    worker_number: usize,
    worker_count: usize,
    deadline: Option<Instant>,
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
    pub models: Vec<crate::models::ModelSpec>,
    pub segments: Vec<FinalSegment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct JobManifest {
    schema_version: u32,
    input_sha256: String,
    models: Vec<crate::models::ModelSpec>,
    asr: BTreeMap<String, Adapter>,
    asr_script_sha256: BTreeMap<String, String>,
    diarization: Adapter,
    diarization_script_sha256: String,
    llm: crate::config::Llm,
    review_rules_sha256: Option<String>,
}

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

#[derive(Debug, Deserialize)]
struct Decision {
    index: usize,
    text: String,
    language: String,
    gap: bool,
}

#[derive(Debug, Deserialize)]
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
    empty_window_count: usize,
    total_characters: usize,
    offload: bool,
    threads: usize,
    max_in_flight: usize,
    adapter_args: &'a [String],
    model_revision: &'a str,
    adapter_sha256: String,
    under_ten_minutes: bool,
}

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

fn repeated_fourgram_fraction(text: &str) -> f64 {
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

pub fn benchmark_asr(
    config: &Config,
    input: &Path,
    output: &Path,
    role: &str,
    offload: bool,
    workers: usize,
    worker_peak_mib: Option<u64>,
) -> Result<()> {
    const AUDIO_SECONDS: f64 = 1800.0;
    let engine = ROLES
        .iter()
        .find(|(name, _)| *name == role)
        .map(|(_, engine)| *engine)
        .context("benchmark role must be primeline, flix, or qwen")?;
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
    if workers > 1 {
        let per_worker = worker_peak_mib.context(
            "multiple benchmark workers require an explicit --worker-peak-mib measured for this engine",
        )?;
        let budget = resources::memory()?.available_mib;
        let required = per_worker
            .saturating_mul(workers as u64)
            .saturating_add(config.resources.reserve_mib);
        ensure!(
            required <= budget,
            "{workers} benchmark workers need an estimated {required} MiB including reserve; only {budget} MiB available"
        );
    }
    let adapter_sha256 = hash_file(&config.asr(role).script)?;
    let (output, wav) = prepare_benchmark_audio(input, output)?;
    let chunks = audio::chunks(AUDIO_SECONDS, 28.0, 0.8)?;
    let clips = output.join("clips");
    fs::create_dir_all(&clips)?;
    for (i, chunk) in chunks.iter().enumerate() {
        audio::extract(&wav, chunk, &clips, i)?;
    }
    let started = Instant::now();
    let deadline = started + Duration::from_secs(600);
    thread::scope(|scope| -> Result<()> {
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
                Ok(Err(error)) => failures.push(format!("{error:#}")),
                Err(_) => failures.push("benchmark worker panicked".into()),
            }
        }
        ensure!(
            failures.is_empty(),
            "benchmark failed: {}",
            failures.join("; ")
        );
        Ok(())
    })?;
    let elapsed = started.elapsed().as_secs_f64();
    ensure!(
        hash_file(&config.asr(role).script)? == adapter_sha256,
        "ASR adapter script changed during benchmark; timing provenance is invalid"
    );
    let mut empty_window_count = 0;
    let mut total_characters = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        let hypothesis: Hypothesis = read_json(&output.join(role).join(format!("{i:05}.json")))?;
        validate_hypothesis(&hypothesis, chunk.end - chunk.start)?;
        empty_window_count += usize::from(hypothesis.text.trim().is_empty());
        total_characters += hypothesis.text.chars().count();
    }
    let report = BenchmarkReport {
        role,
        engine,
        audio_seconds: AUDIO_SECONDS,
        elapsed_seconds: elapsed,
        real_time_factor: elapsed / AUDIO_SECONDS,
        window_count: chunks.len(),
        worker_count: workers,
        estimated_worker_peak_mib: estimate,
        empty_window_count,
        total_characters,
        offload,
        threads: config.asr(role).threads,
        max_in_flight: config.asr(role).max_in_flight,
        adapter_args: &config.asr(role).args,
        model_revision: &config.model(role).revision,
        adapter_sha256,
        under_ten_minutes: meets_30min_target(elapsed),
    };
    write_json(&output.join("benchmark.json"), &report)?;
    println!(
        "{role}: 30 minutes audio in {elapsed:.1} s; RTF {:.3}; under 10 min: {}",
        report.real_time_factor, report.under_ten_minutes
    );
    ensure!(
        report.under_ten_minutes,
        "{role} exceeded the required 10-minute limit; benchmark saved at {}",
        output.join("benchmark.json").display()
    );
    Ok(())
}

pub fn benchmark_diarization(config: &Config, input: &Path, output: &Path) -> Result<()> {
    const AUDIO_SECONDS: f64 = 1800.0;
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

pub fn transcribe(
    config: &Config,
    input: &Path,
    output: &Path,
    sequential_asr: bool,
) -> Result<()> {
    ensure!(
        input
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("mp3")),
        "input must be MP3"
    );
    let input = input.canonicalize().context("input MP3 is unavailable")?;
    let hash = hash_file(&input)?;
    fs::create_dir_all(output).with_context(|| format!("cannot create {}", output.display()))?;
    let output = output
        .canonicalize()
        .context("job directory is unavailable")?;
    let meta = output.join("manifest.json");
    let asr_script_sha256 = config
        .asr
        .iter()
        .map(|(role, adapter)| Ok((role.clone(), hash_file(&adapter.script)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let manifest = JobManifest {
        schema_version: 1,
        input_sha256: hash.clone(),
        models: config.models.clone(),
        asr: config.asr.clone(),
        asr_script_sha256,
        diarization: config.diarization.clone(),
        diarization_script_sha256: hash_file(&config.diarization.script)?,
        llm: config.llm.clone(),
        review_rules_sha256: config.review_rules.as_deref().map(hash_file).transpose()?,
    };
    if meta.exists() {
        let existing: JobManifest = read_json(&meta)?;
        ensure!(
            same_job_after_model_move(&existing, &manifest),
            "job input, adapter, LLM or pinned model revision changed; use another output directory"
        );
    } else {
        write_json(&meta, &manifest)?;
    }
    let duration = audio::duration(&input)?;
    let wav = output.join("audio.wav");
    if !wav.exists() {
        let partial = output.join("audio.partial.wav");
        audio::normalize(&input, &partial)?;
        fs::rename(partial, &wav)?;
    }
    let diarization_path = output.join("diarization.json");
    if !diarization_path.exists() {
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
        process::run(
            &mut command,
            &output.join("diarization.stdout"),
            &output.join("diarization.stderr"),
            Duration::from_secs(4 * 3600),
        )?;
        let _: Diarization = read_json(&partial)?;
        fs::rename(partial, &diarization_path)?;
    }
    let diarization: Diarization = read_json(&diarization_path)?;
    validate_diarization(&diarization, duration)?;
    let chunks_path = output.join("chunks.json");
    let chunks: Vec<Interval> = if chunks_path.exists() {
        read_json(&chunks_path)?
    } else {
        let chunks = audio::chunks(duration, 28.0, 0.8)?;
        write_json(&chunks_path, &chunks)?;
        chunks
    };
    let clips = output.join("clips");
    fs::create_dir_all(&clips)?;
    for (i, interval) in chunks.iter().enumerate() {
        let clip = clips.join(format!("{i:05}.wav"));
        if !clip.exists() {
            audio::extract(&wav, interval, &clips, i)?;
        }
    }
    let waiting_path = output.join("waiting-resources.json");
    if waiting_path.exists() {
        let waiting: WaitingResources = read_json(&waiting_path)?;
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
    }
    let mut remaining: Vec<_> = ROLES
        .iter()
        .filter(|(role, _)| {
            (0..chunks.len()).any(|i| !output.join(role).join(format!("{i:05}.json")).exists())
        })
        .map(|(role, _)| *role)
        .collect();
    let mut oom_attempts = BTreeMap::<&str, usize>::new();
    let mut force_one_worker = sequential_asr;
    let mut force_offload = BTreeSet::<&str>::new();
    while !remaining.is_empty() {
        let budget = resources::memory()?;
        let batch =
            resources::choose_batch(&remaining, budget, &config.resources, force_one_worker)?;
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
            let mut handles = Vec::new();
            for (role, worker_count) in &batch {
                let engine = ROLES
                    .iter()
                    .find(|(name, _)| name == role)
                    .map(|(_, engine)| *engine)
                    .context("unknown ASR role in resource batch")?;
                for worker_number in 0..*worker_count {
                    let offload = modes[role];
                    handles.push((
                        *role,
                        scope.spawn(move || {
                            run_asr(
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
                                },
                            )
                        }),
                    ));
                }
            }
            let mut errors = Vec::new();
            let mut outcomes = Vec::new();
            for (role, handle) in handles {
                match handle.join() {
                    Ok(Ok(outcome)) => outcomes.push((role, outcome)),
                    Ok(Err(err)) => errors.push(format!("{err:#}")),
                    Err(_) => errors.push("ASR worker panicked".to_string()),
                }
            }
            ensure!(errors.is_empty(), "ASR failed: {}", errors.join("; "));
            Ok(outcomes)
        })?;
        for (role, outcome) in outcomes {
            if outcome == WorkerOutcome::OutOfMemory {
                eprintln!(
                    "{role}: isolated worker was killed by OOM; previous chunks remain saved"
                );
                let attempts = oom_attempts.entry(role).or_default();
                *attempts += 1;
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
                if *attempts > config.resources.oom_retries {
                    write_json(
                        &waiting_path,
                        &WaitingResources {
                            role: role.to_owned(),
                            available_mib: resources::memory()?.available_mib,
                            settings: config.resources.clone(),
                        },
                    )?;
                    bail!(
                        "{role} was killed by OOM after {} attempt(s); job is waiting for resources in {}",
                        attempts,
                        waiting_path.display()
                    );
                }
                thread::sleep(Duration::from_secs(5 * *attempts as u64));
            }
        }
        remaining.retain(|role| {
            (0..chunks.len()).any(|i| !output.join(role).join(format!("{i:05}.json")).exists())
        });
    }
    let mut hypotheses = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let mut group = Vec::new();
        for (role, _) in ROLES {
            let path = output.join(role).join(format!("{i:05}.json"));
            let hypothesis: Hypothesis = read_json(&path)?;
            validate_hypothesis(&hypothesis, chunk.end - chunk.start)?;
            group.push((role.to_owned(), hypothesis));
        }
        hypotheses.push(group);
    }
    let review_flags: Vec<_> = chunks
        .iter()
        .zip(&hypotheses)
        .map(|(chunk, group)| review_flags(chunk, group, &diarization))
        .collect();
    let flagged: Vec<_> = review_flags
        .iter()
        .enumerate()
        .filter(|(_, flags)| !flags.is_empty())
        .map(|(index, flags)| ReviewFlag {
            index,
            flags: flags.clone(),
        })
        .collect();
    write_json(&output.join("review-flags.json"), &flagged)?;
    let mut segments = Vec::new();
    let review_rules = config
        .review_rules
        .as_ref()
        .map(|path| {
            fs::read_to_string(path)
                .with_context(|| format!("cannot read review rules {}", path.display()))
        })
        .transpose()?;
    ensure!(
        review_rules.as_ref().is_none_or(|text| text.len() <= 65536),
        "review rules exceed 64 KiB"
    );
    let review_dir = output.join("reviews");
    fs::create_dir_all(&review_dir)?;
    for start in (0..chunks.len()).step_by(8) {
        let end = (start + 8).min(chunks.len());
        let decisions_path = review_dir.join(format!("{start:05}.json"));
        let decisions: Decisions = if decisions_path.exists() {
            read_json(&decisions_path)?
        } else {
            let prompt = serde_json::json!({
                "instruction": "Do not use tools or read files. Compare three ASR hypotheses for each audio interval; use only their text and advisory review flags, never invent words. A missing diarization turn does not prove there was no speech: do not discard a unique hypothesis on that basis. For German including Swiss dialect, return standard German without changing meaning. Keep Russian in Russian. If speech cannot be recovered, set gap=true and put [пропуск] in text. Return strictly JSON: {\"segments\":[{\"index\":integer,\"text\":string,\"language\":\"de|ru|mixed|unknown\",\"gap\":boolean}]}. One entry per index. Never claim to have heard the audio.",
                "review_rules": review_rules,
                "intervals": (start..end).map(|i| serde_json::json!({"index": i, "start": chunks[i].start, "end": chunks[i].end, "hypotheses": hypotheses[i], "review_flags": review_flags[i]})).collect::<Vec<_>>()
            });
            llm(
                config,
                &prompt.to_string(),
                &decisions_path,
                &review_dir.join(format!("{start:05}.stderr")),
            )?;
            read_json(&decisions_path)?
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
    }
    let status = if segments.iter().any(|s| s.gap) || !flagged.is_empty() {
        "needs_audio_review"
    } else {
        "text_reviewed_audio_not_verified"
    };
    let transcript = Transcript {
        schema_version: 1,
        source_sha256: hash,
        duration,
        quality_status: status.to_owned(),
        models: config.models.clone(),
        segments,
    };
    write_json(&output.join("transcript.json"), &transcript)?;
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
    fs::write(output.join("transcript.txt"), text)?;
    Ok(())
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
    let folder = output.join(role);
    fs::create_dir_all(&folder)?;
    let mut pending = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        if i % mode.worker_count != mode.worker_number {
            continue;
        }
        let path = folder.join(format!("{i:05}.json"));
        if path.exists() {
            let value: Hypothesis = read_json(&path)?;
            validate_hypothesis(&value, chunk.end - chunk.start)?;
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
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(File::create(&stderr)?)
        .spawn()
        .with_context(|| format!("cannot start {role} worker"))?;
    let mut stdin = child.stdin.take().context("ASR worker stdin missing")?;
    let stdout = child.stdout.take().context("ASR worker stdout missing")?;
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
            while !pause_for_ram
                && in_flight.len() < adapter.max_in_flight
                && let Some((i, chunk)) = pending.next()
            {
                if mode.deadline.is_some_and(|limit| Instant::now() >= limit) {
                    bail!("{role} exceeded the 600-second benchmark limit");
                }
                let clip = clips.join(format!("{i:05}.wav")).canonicalize()?;
                if adapter.max_in_flight > 1 {
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
            let timeout = mode.deadline.map_or(Duration::from_secs(1800), |limit| {
                Duration::from_secs(1800).min(limit.saturating_duration_since(Instant::now()))
            });
            let line = receiver
                .recv_timeout(timeout)
                .with_context(|| {
                    if mode.deadline.is_some_and(|limit| Instant::now() >= limit) {
                        format!("{role} exceeded the 600-second benchmark limit on chunk {waiting}")
                    } else {
                        format!(
                            "{role} chunk {waiting}: worker timed out or exited; see {}",
                            stderr.display()
                        )
                    }
                })?
                .with_context(|| format!("{role} chunk {waiting}: worker stdout failed"))?;
            let response: serde_json::Value = serde_json::from_str(&line).with_context(|| {
                format!(
                    "{role} chunk {waiting}: invalid worker JSON; see {}",
                    stderr.display()
                )
            })?;
            if adapter.max_in_flight > 1
                && response.get("index").is_none()
                && let Some(error) = response.get("error")
            {
                bail!(
                    "{role} worker failed while chunk {waiting} was in flight: {error}; see {}",
                    stderr.display()
                );
            }
            let i = if adapter.max_in_flight > 1 {
                response
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .context("concurrent ASR response must contain a nonnegative window index")?
            } else {
                waiting
            };
            let (chunk, window_started) = in_flight
                .remove(&i)
                .with_context(|| format!("{role} returned unexpected or duplicate chunk {i}"))?;
            if let Some(error) = response.get("error") {
                bail!("{role} chunk {i}: {error}; see {}", stderr.display());
            }
            let mut value: Hypothesis = serde_json::from_value(response)
                .with_context(|| format!("{role} chunk {i}: invalid hypothesis"))?;
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
            validate_hypothesis(&value, chunk.end - chunk.start)?;
            write_json(&folder.join(format!("{i:05}.json")), &value)?;
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
            if mode.concurrent && resources::memory()?.available_mib < config.resources.reserve_mib
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
    if run.is_err() && child.try_wait()?.is_none() {
        if let Some(unit) = &unit {
            resources::stop_service(unit)?;
        }
        child.kill().context("cannot stop failed ASR worker")?;
    }
    let status = child.wait_timeout(Duration::from_secs(60))?;
    let exit = match status {
        Some(exit) => exit,
        None => {
            if let Some(unit) = &unit {
                resources::stop_service(unit)?;
            }
            child
                .kill()
                .context("cannot stop ASR worker after exit timeout")?;
            child.wait()?
        }
    };
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

fn llm(config: &Config, prompt: &str, output: &Path, stderr: &Path) -> Result<()> {
    let stdout = output.with_extension("stdout");
    let partial = output.with_extension("partial.json");
    if partial.exists() {
        fs::remove_file(&partial)?;
    }
    let mut command = Command::new(&config.llm.program);
    let review_dir = output
        .parent()
        .context("review path has no parent")?
        .canonicalize()?;
    let absolute_output = review_dir.join(
        partial
            .file_name()
            .context("review path has no file name")?,
    );
    command.current_dir(review_dir);
    for arg in &config.llm.args {
        command.arg(arg.replace("{output}", &absolute_output.to_string_lossy()));
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(File::create(&stdout)?)
        .stderr(File::create(stderr)?)
        .spawn()
        .with_context(|| format!("cannot start LLM CLI {:?}", config.llm.program))?;
    child
        .stdin
        .take()
        .context("LLM stdin missing")?
        .write_all(prompt.as_bytes())?;
    match child.wait_timeout(Duration::from_secs(300))? {
        Some(status) if status.success() => {}
        Some(status) => bail!("LLM exited with {status}; see {}", stderr.display()),
        None => {
            child.kill()?;
            child.wait()?;
            bail!("LLM timed out; see {}", stderr.display());
        }
    }
    if !partial.exists() {
        fs::rename(&stdout, &partial).context("LLM did not produce a response")?;
    }
    let _: Decisions = read_json(&partial)?;
    fs::rename(partial, output)?;
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
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
                value.out_of_bounds_segments.push(segment.clone());
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

fn validate_hypothesis(value: &Hypothesis, duration: f64) -> Result<()> {
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

fn validate_diarization(value: &Diarization, duration: f64) -> Result<()> {
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
    serde_json::to_writer_pretty(File::create(&temp)?, value)?;
    fs::rename(temp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Diarization, Hypothesis, JobManifest, Transcript, Turn, WordSpan, WorkerMode,
        benchmark_asr, benchmark_diarization, clip_whisper_timestamps, review_flags, run_asr,
        same_job_after_model_move, transcribe, trim_repeated_prefix, validate_hypothesis,
    };
    use crate::audio::Interval;
    use crate::config::{Adapter, Config, Llm};
    use crate::models::ModelSpec;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    fn resuming_a_moved_project_only_ignores_model_storage_paths() {
        let adapter = Adapter {
            program: PathBuf::from("/bin/python"),
            script: PathBuf::from("/app/worker.py"),
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
            },
            review_rules_sha256: None,
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
            empty_window_count: 0,
            total_characters: 1000,
            offload: false,
            threads: 12,
            max_in_flight: 1,
            adapter_args: &[],
            model_revision: "pinned",
            adapter_sha256: "script-hash".to_owned(),
            under_ten_minutes: true,
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
                    threads: 1,
                    max_in_flight: 3,
                    args: vec![],
                },
            )]
            .into(),
            diarization: Adapter {
                program: "python3".into(),
                script: temp.path().join("unused.py"),
                threads: 1,
                max_in_flight: 1,
                args: vec![],
            },
            llm: Llm {
                program: "unused".into(),
                args: vec![],
            },
            review_rules: None,
            resources: Default::default(),
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
                threads: 1,
                max_in_flight: 1,
                args: Vec::new(),
            },
            llm: Llm {
                program: PathBuf::from("python3"),
                args: vec![llm.to_string_lossy().into_owned(), "{output}".into()],
            },
            review_rules: None,
            resources: crate::config::Resources {
                reserve_mib: 0,
                max_asr_workers: 3,
                estimated_peak_mib: ["primeline", "flix", "qwen"]
                    .into_iter()
                    .map(|role| (role.to_owned(), 1))
                    .collect(),
                asr_worker_counts: Default::default(),
                asr_worker_peak_mib: Default::default(),
                supervise_asr_with_systemd: false,
                systemd_memory_limit_mib: None,
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
        transcribe(&config, &input, &output, true).unwrap();
        let transcript: Transcript =
            serde_json::from_slice(&fs::read(output.join("transcript.json")).unwrap()).unwrap();
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
            for role in ["primeline", "flix", "qwen"] {
                fs::remove_dir_all(output.join(role)).unwrap();
            }
            config.resources.supervise_asr_with_systemd = true;
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
                    "started\n".repeat(if engine == "whisper" { 7 } else { 4 })
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
            assert!(
                benchmark_asr(
                    &config,
                    &long_input,
                    &benchmark,
                    "primeline",
                    false,
                    2,
                    Some(u64::MAX)
                )
                .unwrap_err()
                .to_string()
                .contains("benchmark workers need")
            );
            assert!(!benchmark.exists());
            benchmark_asr(
                &config,
                &long_input,
                &benchmark,
                "primeline",
                false,
                2,
                Some(1),
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
                    false,
                    2,
                    Some(1)
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
    }
}
