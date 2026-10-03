pub mod audio;
pub mod text;

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{audio::Interval, pipeline::Hypothesis, workflow};

const DIRECTORY: &str = ".quality";
const SCHEMA_VERSION: u32 = 1;
const RULES_VERSION: &str = "advisory-v1";
const MAX_JSON_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub enabled: bool,
    pub max_chunks: usize,
    pub inspect_source_channels: bool,
    pub audio: audio::Settings,
    pub text: text::Settings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_chunks: 10_000,
            inspect_source_channels: true,
            audio: audio::Settings::default(),
            text: text::Settings::default(),
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=100_000).contains(&self.max_chunks),
            "quality.max_chunks must be between 1 and 100000"
        );
        self.audio.validate()?;
        self.text.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    pub status: String,
    pub value: Option<Value>,
    pub error: Option<String>,
}

impl Section {
    fn unavailable(reason: &str) -> Self {
        Self {
            status: "unavailable".into(),
            value: None,
            error: Some(reason.into()),
        }
    }

    fn from_result<T: Serialize>(result: Result<T>) -> Self {
        match result.and_then(|value| serde_json::to_value(value).map_err(Into::into)) {
            Ok(value) => {
                let unavailable = value.get("available") == Some(&json!(false));
                let error = if unavailable {
                    Some(
                        value["unavailable_reason"]
                            .as_str()
                            .unwrap_or("inspection unavailable")
                            .to_owned(),
                    )
                } else {
                    None
                };
                Self {
                    status: if unavailable {
                        "unavailable"
                    } else {
                        "available"
                    }
                    .into(),
                    value: Some(value),
                    error,
                }
            }
            Err(error) => Self {
                status: "failed".into(),
                value: None,
                error: Some(format!("{error:#}")),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Window {
    pub index: usize,
    pub start: f64,
    pub end: f64,
    pub available_roles: Vec<String>,
    pub missing_roles: Vec<String>,
    #[serde(default)]
    pub skipped_roles: Vec<String>,
    pub errors: Vec<String>,
    pub analysis: Section,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFinding {
    pub code: String,
    pub severity: String,
    pub window_index: Option<usize>,
    pub evidence: Value,
    pub recommendation: String,
    pub limitation: String,
}

fn audio_findings(section: &Section, settings: &audio::Settings) -> Vec<AudioFinding> {
    let Some(value) = section
        .value
        .as_ref()
        .filter(|_| section.status == "available")
    else {
        return Vec::new();
    };
    let mut findings = Vec::new();
    let mut check = |metrics: &Value, index: Option<usize>| {
        if metrics["zero_signal"] == true {
            findings.push(AudioFinding {
                code: "zero_pcm_signal".into(),
                severity: "warning".into(),
                window_index: index,
                evidence: json!({"sample_count": metrics["sample_count"]}),
                recommendation: "inspect_fragment".into(),
                limitation:
                    "Zero measured PCM is not a semantic assessment of the original recording."
                        .into(),
            });
        } else if metrics["low_rms"] == true {
            findings.push(AudioFinding {
                code: "low_audio_level".into(),
                severity: "advisory".into(),
                window_index: index,
                evidence: json!({"rms_dbfs": metrics["rms_dbfs"], "threshold_dbfs": settings.low_rms_dbfs}),
                recommendation: "inspect_fragment".into(),
                limitation: "Low energy does not establish missing speech or justify gain automatically.".into(),
            });
        }
        if metrics["full_scale_fraction"]
            .as_f64()
            .is_some_and(|fraction| fraction > 0.0)
        {
            findings.push(AudioFinding {
                code: "near_full_scale_samples".into(),
                severity: "advisory".into(),
                window_index: index,
                evidence: json!({"fraction": metrics["full_scale_fraction"], "amplitude_threshold": settings.full_scale_threshold}),
                recommendation: "inspect_fragment".into(),
                limitation: "Samples near full scale indicate possible clipping, not proven waveform distortion.".into(),
            });
        }
    };
    check(&value["metrics"], None);
    if let Some(windows) = value["windows"].as_array() {
        for (index, window) in windows.iter().enumerate() {
            if window["metrics"]["sample_count"]
                .as_u64()
                .is_some_and(|count| count > 0)
            {
                check(&window["metrics"], Some(index));
            }
        }
    }
    findings
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    pub rules_version: String,
    pub policy_sha256: String,
    pub policy: Settings,
    pub sources: BTreeMap<String, String>,
    pub execution_status: String,
    pub accuracy_status: String,
    pub automatic_actions: bool,
    #[serde(default = "crate::config::default_asr_roles")]
    pub asr_roles: Vec<String>,
    pub total_windows: usize,
    pub analyzed_windows: usize,
    pub audio: Section,
    pub source_channels: Section,
    pub diarization: Section,
    pub resources: Section,
    pub windows: Vec<Window>,
    pub audio_findings: Vec<AudioFinding>,
    pub errors: Vec<String>,
    pub elapsed_ms: u128,
}

pub fn report_path(project: &Path) -> PathBuf {
    project.join(DIRECTORY).join("report.json")
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    ensure!(
        !fs::symlink_metadata(path)?.file_type().is_symlink(),
        "quality input must not be a symlink: {}",
        path.display()
    );
    ensure!(
        fs::metadata(path)?.len() <= MAX_JSON_BYTES,
        "quality JSON exceeds 16 MiB: {}",
        path.display()
    );
    serde_json::from_reader(BufReader::new(File::open(path)?))
        .with_context(|| format!("invalid quality input {}", path.display()))
}

fn contained_input(project: &Path, path: &Path) -> Result<()> {
    ensure!(
        path.canonicalize()?.starts_with(project.canonicalize()?),
        "quality artifact leaves the project directory: {}",
        path.display()
    );
    let mut current = path.parent();
    while let Some(parent) = current {
        if parent == project {
            break;
        }
        ensure!(
            !parent.is_symlink(),
            "quality artifact directory must not be symlinked"
        );
        current = parent.parent();
    }
    Ok(())
}

fn hash_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn file_signature(path: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file(),
        "not a regular quality input: {}",
        path.display()
    );
    let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?;
    Ok(format!(
        "metadata:{}:{}",
        metadata.len(),
        modified.as_nanos()
    ))
}

fn fingerprint(path: &Path, sources: &mut BTreeMap<String, String>, name: &str) -> Result<()> {
    if path.exists() {
        ensure!(
            fs::symlink_metadata(path)?.is_file(),
            "quality input is not a regular file: {}",
            path.display()
        );
        ensure!(
            fs::metadata(path)?.len() <= MAX_JSON_BYTES,
            "quality input exceeds 16 MiB: {}",
            path.display()
        );
        sources.insert(name.into(), crate::pipeline::hash_file(path)?);
    }
    Ok(())
}

fn directory(project: &Path) -> Result<PathBuf> {
    let path = project.join(DIRECTORY);
    if path.exists() {
        ensure!(
            fs::symlink_metadata(&path)?.file_type().is_dir(),
            "quality sidecar is not a regular directory"
        );
        ensure!(
            fs::read(path.join(".owner"))? == b"shum-quality-v1\n",
            "quality sidecar belongs to another application"
        );
    } else {
        fs::create_dir(&path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
        }
        fs::write(path.join(".owner"), b"shum-quality-v1\n")?;
    }
    Ok(path)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let partial = path.with_extension("json.partial");
    ensure!(
        !path.is_symlink() && !partial.is_symlink(),
        "quality report paths must not be symlinks"
    );
    let bytes = serde_json::to_vec(value)?;
    ensure!(
        bytes.len() as u64 <= MAX_JSON_BYTES,
        "quality report exceeds 16 MiB"
    );
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&partial)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(partial, path)?;
    Ok(())
}

fn resource_events(project: &Path) -> Result<Vec<Value>> {
    let path = project.join("resource-events.jsonl");
    if !path.exists() {
        return Ok(Vec::new());
    }
    ensure!(
        fs::metadata(&path)?.len() <= MAX_JSON_BYTES && !path.is_symlink(),
        "resource telemetry exceeds budget or is symlinked"
    );
    BufReader::new(File::open(&path)?)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(&line?)
                .with_context(|| format!("invalid resource telemetry line {}", index + 1))
        })
        .collect()
}

pub fn inspect_locked(project: &Path, settings: &Settings, fresh: bool) -> Result<Report> {
    inspect_observed(project, settings, fresh, None, &|_| Ok(()))
}

pub fn inspect_observed(
    project: &Path,
    settings: &Settings,
    fresh: bool,
    scope: Option<&[usize]>,
    completed: &dyn Fn(usize) -> Result<()>,
) -> Result<Report> {
    inspect_inputs(project, settings, fresh, scope, None, completed)
}

pub fn inspect_inputs(
    project: &Path,
    settings: &Settings,
    fresh: bool,
    scope: Option<&[usize]>,
    input_folders: Option<&BTreeMap<String, PathBuf>>,
    completed: &dyn Fn(usize) -> Result<()>,
) -> Result<Report> {
    settings.validate()?;
    let started = Instant::now();
    let mut state = workflow::status(project)?;
    if let Some(inputs) = input_folders {
        ensure!(!inputs.is_empty(), "check has no ASR inputs");
        state.asr_roles = inputs.keys().cloned().collect();
        for folder in inputs.values() {
            contained_input(project, folder)?;
        }
    }
    contained_input(project, &project.join("workflow.json"))?;
    let sidecar = if let Some(inputs) = input_folders {
        let key = hash_bytes(&serde_json::to_vec(&(inputs, scope))?);
        let folder = project.join(".execution").join("quality").join(key);
        fs::create_dir_all(&folder)?;
        contained_input(project, &folder)?;
        folder
    } else {
        directory(project)?
    };
    let report_file = sidecar.join("report.json");
    let policy_sha256 = hash_bytes(&serde_json::to_vec(settings)?);
    let mut sources = BTreeMap::new();
    sources.insert(
        "recorded_original_input_sha256".into(),
        state.input_sha256.clone(),
    );
    let chunks_path = project.join("chunks.json");
    for name in ["chunks.json", "resource-events.jsonl", "diarization.json"] {
        let path = project.join(name);
        if path.exists() {
            contained_input(project, &path)?;
        }
    }
    let chunks: Vec<Interval> = if chunks_path.exists() {
        read_json(&chunks_path)?
    } else {
        Vec::new()
    };
    if let Some(scope) = scope {
        ensure!(
            !scope.is_empty()
                && scope.iter().all(|index| *index < chunks.len())
                && scope
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == scope.len(),
            "quality scope must contain unique existing fragment indices"
        );
        sources.insert(
            "fragment_scope".into(),
            hash_bytes(&serde_json::to_vec(scope)?),
        );
        ensure!(
            scope.len() <= settings.max_chunks,
            "quality scope exceeds max_chunks budget"
        );
    }
    let window_indices = scope.map_or_else(
        || (0..chunks.len().min(settings.max_chunks)).collect::<Vec<_>>(),
        |scope| scope.to_vec(),
    );
    ensure!(
        chunks.iter().all(|chunk| chunk.start.is_finite()
            && chunk.end.is_finite()
            && chunk.start >= 0.0
            && chunk.end > chunk.start),
        "invalid diagnostic chunk bounds"
    );
    fingerprint(&chunks_path, &mut sources, "chunks.json")?;
    fingerprint(
        &project.join("workflow.json"),
        &mut sources,
        "workflow.json",
    )?;
    fingerprint(
        &project.join("resource-events.jsonl"),
        &mut sources,
        "resource-events.jsonl",
    )?;
    fingerprint(
        &project.join("diarization.json"),
        &mut sources,
        "diarization.json",
    )?;
    let wav = project.join("audio.wav");
    if wav.exists() {
        contained_input(project, &wav)?;
        sources.insert("audio.wav_cache_metadata".into(), file_signature(&wav)?);
    }
    if state.input.exists() {
        sources.insert(
            "source_channels_cache_metadata".into(),
            file_signature(&state.input)?,
        );
    }
    for role in &state.asr_roles {
        for index in &window_indices {
            let name = format!("{role}/{index:05}.json");
            let file = input_folders
                .and_then(|inputs| inputs.get(role))
                .map_or_else(
                    || project.join(&name),
                    |folder| folder.join(format!("{index:05}.json")),
                );
            if file.exists() {
                contained_input(project, &file)?;
            }
            fingerprint(&file, &mut sources, &name)?;
        }
    }
    let previous: Option<Report> = if report_file.exists() {
        Some(read_json(&report_file)?)
    } else {
        None
    };
    let original_signature = sources.get("source_channels_cache_metadata");
    if state.input.exists()
        && (fresh
            || previous.as_ref().is_none_or(|previous| {
                previous.sources.get("source_channels_cache_metadata") != original_signature
            }))
    {
        let actual = crate::pipeline::hash_file(&state.input)?;
        ensure!(
            actual == state.input_sha256,
            "original recording content changed; quality measurements would not match project hypotheses"
        );
        sources.insert("verified_original_input_sha256".into(), actual);
    } else if let Some(actual) = previous
        .as_ref()
        .and_then(|previous| previous.sources.get("verified_original_input_sha256"))
    {
        sources.insert("verified_original_input_sha256".into(), actual.clone());
    }
    if !fresh
        && let Some(previous) = &previous
        && previous.schema_version == SCHEMA_VERSION
        && previous.rules_version == RULES_VERSION
        && previous.execution_status != "failed"
        && previous.policy_sha256 == policy_sha256
        && previous.sources == sources
    {
        for window in &previous.windows {
            completed(window.index)?;
        }
        return Ok(previous.clone());
    }
    let audio_cached = !fresh
        && previous.as_ref().is_some_and(|previous| {
            previous.policy_sha256 == policy_sha256
                && previous.sources.get("audio.wav_cache_metadata")
                    == sources.get("audio.wav_cache_metadata")
                && previous.sources.get("chunks.json") == sources.get("chunks.json")
        });
    let audio = if audio_cached {
        previous
            .as_ref()
            .context("missing audio cache")?
            .audio
            .clone()
    } else if wav.exists() {
        Section::from_result(audio::analyze(
            &wav,
            &chunks[..chunks.len().min(settings.max_chunks)],
            &settings.audio,
        ))
    } else {
        Section::unavailable("prepared audio.wav is not available; no decoding was started")
    };
    let source_cached = !fresh
        && previous.as_ref().is_some_and(|previous| {
            previous.policy_sha256 == policy_sha256
                && previous.sources.get("source_channels_cache_metadata")
                    == sources.get("source_channels_cache_metadata")
        });
    let source_channels = if !settings.inspect_source_channels {
        Section::unavailable("original-channel inspection disabled by policy")
    } else if source_cached {
        previous
            .as_ref()
            .context("missing source-channel cache")?
            .source_channels
            .clone()
    } else if state.input.exists() {
        Section::from_result(audio::analyze_channels(&state.input, &settings.audio))
    } else {
        Section::unavailable(
            "original recording unavailable; prepared audio is analyzed separately",
        )
    };
    let resources = Section::from_result(
        resource_events(project).and_then(|events| text::analyze_resources(&events)),
    );
    let diarization = if project.join("diarization.json").exists()
        && state
            .stages
            .get("diarization")
            .is_some_and(|stage| stage.status == workflow::State::Ready)
    {
        Section::from_result(read_json::<crate::pipeline::Diarization>(
            &project.join("diarization.json"),
        ))
    } else {
        Section::unavailable("current diarization is not available; absence does not prove silence")
    };
    let mut windows = Vec::new();
    let mut report_bytes =
        serde_json::to_vec(&(&audio, &source_channels, &resources, &diarization))?.len();
    ensure!(
        report_bytes < MAX_JSON_BYTES as usize / 2,
        "audio/resource diagnostics exceed report budget"
    );
    for index in window_indices {
        let chunk = &chunks[index];
        let mut hypotheses = Vec::new();
        let mut missing_roles = Vec::new();
        let mut errors = Vec::new();
        for role in &state.asr_roles {
            let path = input_folders
                .and_then(|inputs| inputs.get(role))
                .map_or_else(|| project.join(role), PathBuf::clone)
                .join(format!("{index:05}.json"));
            let stage_current = input_folders.is_some()
                || state
                    .stages
                    .get(&format!("asr:{role}"))
                    .is_some_and(|stage| {
                        matches!(
                            stage.status,
                            workflow::State::Ready | workflow::State::Running
                        )
                    });
            if !path.exists() || !stage_current {
                missing_roles.push(role.into());
            } else {
                match read_json::<Hypothesis>(&path) {
                    Ok(hypothesis) => hypotheses.push((role.to_owned(), hypothesis)),
                    Err(error) => errors.push(format!("{role}: {error:#}")),
                }
            }
        }
        let window = Window {
            index,
            start: chunk.start,
            end: chunk.end,
            available_roles: hypotheses.iter().map(|(role, _)| role.clone()).collect(),
            missing_roles,
            skipped_roles: state
                .stages
                .keys()
                .filter_map(|step| step.strip_prefix("asr:"))
                .filter(|role| !state.asr_roles.iter().any(|active| active == role))
                .map(str::to_owned)
                .collect(),
            errors,
            analysis: Section::from_result(text::analyze(
                &hypotheses,
                chunk.end - chunk.start,
                &settings.text,
            )),
        };
        report_bytes = report_bytes.saturating_add(serde_json::to_vec(&window)?.len());
        if report_bytes > MAX_JSON_BYTES as usize / 2 {
            break;
        }
        windows.push(window);
        completed(index)?;
    }
    let mut errors = [
        ("audio", &audio),
        ("source_channels", &source_channels),
        ("resources", &resources),
        ("diarization", &diarization),
    ]
    .into_iter()
    .filter(|(_, section)| section.status == "failed")
    .map(|(name, section)| {
        format!(
            "{name}: {}",
            section
                .error
                .as_deref()
                .unwrap_or("unknown diagnostic failure")
        )
    })
    .collect::<Vec<_>>();
    for window in &windows {
        errors.extend(
            window
                .errors
                .iter()
                .map(|error| format!("window {}: {error}", window.index)),
        );
        if window.analysis.status == "failed" {
            errors.push(format!(
                "window {}: {}",
                window.index,
                window
                    .analysis
                    .error
                    .as_deref()
                    .unwrap_or("diagnostics failed")
            ));
        }
    }
    let failed = !errors.is_empty()
        || windows
            .iter()
            .any(|window| !window.errors.is_empty() || window.analysis.status == "failed");
    let partial = audio.status == "unavailable"
        || chunks.is_empty()
        || chunks.len() > settings.max_chunks
        || windows.len() < chunks.len()
        || audio
            .value
            .as_ref()
            .is_some_and(|value| value["partial_coverage"] == true)
        || source_channels
            .value
            .as_ref()
            .is_some_and(|value| value["partial_coverage"] == true)
        || (settings.inspect_source_channels && source_channels.status == "unavailable")
        || windows
            .iter()
            .any(|window| !window.missing_roles.is_empty());
    let mut audio_findings = audio_findings(&audio, &settings.audio);
    if source_channels
        .value
        .as_ref()
        .is_some_and(|value| value["possible_phase_cancellation"] == true)
    {
        audio_findings.push(AudioFinding {
            code: "possible_mono_phase_cancellation".into(),
            severity: "advisory".into(),
            window_index: None,
            evidence: json!({"measurement_source": "decoded_original_channels",
                "stereo_correlation": source_channels.value.as_ref().map(|value| &value["stereo_correlation"])}),
            recommendation: "consider_channel_review".into(),
            limitation: "Correlation is not speech detection and does not select a better channel automatically.".into(),
        });
    }
    let report = Report {
        schema_version: SCHEMA_VERSION,
        rules_version: RULES_VERSION.into(),
        policy_sha256,
        policy: settings.clone(),
        sources,
        execution_status: if failed {
            "failed"
        } else if partial {
            "partial"
        } else {
            "complete"
        }
        .into(),
        accuracy_status: "unverified".into(),
        automatic_actions: false,
        asr_roles: state.asr_roles,
        total_windows: chunks.len(),
        analyzed_windows: windows.len(),
        audio_findings,
        audio,
        source_channels,
        diarization,
        resources,
        windows,
        errors,
        elapsed_ms: started.elapsed().as_millis(),
    };
    write_json(&sidecar.join("report.json"), &report)?;
    let failure = sidecar.join("failure.json");
    if failure.exists() {
        ensure!(
            !failure.is_symlink(),
            "quality failure record must not be symlinked"
        );
        fs::remove_file(failure)?;
    }
    Ok(report)
}

pub fn inspect(project: &Path, settings: &Settings) -> Result<Report> {
    ensure!(project.is_dir(), "project is not a directory");
    ensure!(
        !project.join(".workflow.lock").is_symlink(),
        "project lock must not be symlinked"
    );
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(project.join(".workflow.lock"))?;
    lock.try_lock_exclusive()
        .context("project is already in use")?;
    inspect_locked(project, settings, true).inspect_err(|error| {
        if let Err(write_error) = persist_failure(project, error) {
            eprintln!("Cannot persist quality diagnostic failure: {write_error:#}");
        }
    })
}

fn persist_failure(project: &Path, error: &anyhow::Error) -> Result<()> {
    let directory = directory(project)?;
    write_json(
        &directory.join("failure.json"),
        &json!({
            "schema_version": SCHEMA_VERSION,
            "execution_status": "failed",
            "accuracy_status": "unverified",
            "error": format!("{error:#}")
        }),
    )
}

pub fn summary(report: &Report, project: &Path) -> Value {
    let finding_count = report
        .windows
        .iter()
        .map(|window| {
            window
                .analysis
                .value
                .as_ref()
                .and_then(|value| value["findings"].as_array())
                .map_or(0, Vec::len)
        })
        .sum::<usize>();
    json!({
        "schema_version": report.schema_version,
        "rules_version": report.rules_version,
        "execution_status": report.execution_status,
        "accuracy_status": report.accuracy_status,
        "automatic_actions": false,
        "asr_roles": report.asr_roles,
        "total_windows": report.total_windows,
        "analyzed_windows": report.analyzed_windows,
        "asr_finding_count": finding_count,
        "audio_finding_count": report.audio_findings.len(),
        "resource_finding_count": report.resources.value.as_ref().and_then(|value| value["findings"].as_array()).map_or(0, Vec::len),
        "report": report_path(project),
        "errors": report.errors,
        "elapsed_ms": report.elapsed_ms,
        "cache_validation": "ASR/policy content hashes; prepared audio metadata fast cache; explicit check-quality rescans"
    })
}

pub fn saved_summary(project: &Path) -> Result<Option<Value>> {
    let failure = project.join(DIRECTORY).join("failure.json");
    if failure.exists() {
        contained_input(project, &failure)?;
        return Ok(Some(read_json::<Value>(&failure)?));
    }
    let path = report_path(project);
    if !path.exists() {
        return Ok(None);
    }
    contained_input(project, &path)?;
    let report: Report = read_json(&path)?;
    let mut value = summary(&report, project);
    value["freshness"] = json!("not_revalidated");
    Ok(Some(value))
}

pub fn read_page(project: &Path, offset: usize, limit: usize) -> Result<Value> {
    ensure!(
        (1..=50).contains(&limit),
        "quality page limit must be between 1 and 50"
    );
    ensure!(
        !project.join(DIRECTORY).join("failure.json").exists(),
        "last quality inspection failed; run project check-quality before reading an older report"
    );
    contained_input(project, &report_path(project))?;
    let report: Report = read_json(&report_path(project))?;
    ensure!(
        offset <= report.windows.len(),
        "quality page offset exceeds available windows"
    );
    let mut audio = report.audio.clone();
    if let Some(value) = audio.value.as_mut().and_then(Value::as_object_mut) {
        value.remove("windows");
        value.remove("chunks");
    }
    let mut diarization = report.diarization.clone();
    if diarization.value.is_some() {
        diarization.value = Some(
            json!({"available": true, "detail": "stored in project diarization.json; not repeated in paginated response"}),
        );
    }
    Ok(json!({
        "summary": summary(&report, project),
        "audio": audio,
        "diarization": diarization,
        "resources": report.resources,
        "source_channels": report.source_channels,
        "audio_findings": report.audio_findings.iter().filter(|finding| finding.window_index.is_none_or(|index| index >= offset && index < offset.saturating_add(limit))).collect::<Vec<_>>(),
        "offset": offset,
        "windows": report.windows.iter().skip(offset).take(limit).collect::<Vec<_>>(),
        "next_offset": (offset.saturating_add(limit) < report.windows.len()).then_some(offset + limit),
        "freshness": "not_revalidated"
    }))
}

pub fn auto_refresh(project: &Path, settings: &Settings) {
    if !settings.enabled {
        return;
    }
    match inspect_locked(project, settings, false) {
        Ok(report) if report.execution_status == "failed" => {
            eprintln!(
                "Quality diagnostics failed; transcription outputs unchanged: {}",
                report_path(project).display()
            );
            for error in &report.errors {
                eprintln!("Quality diagnostics: {error}");
            }
        }
        Ok(_) => {}
        Err(error) => {
            eprintln!("Quality diagnostics failed; transcription outputs unchanged: {error:#}");
            if let Err(write_error) = persist_failure(project, &error) {
                eprintln!("Cannot persist quality diagnostic failure: {write_error:#}");
            }
        }
    }
}

pub struct RefreshOnExit<'a> {
    pub project: &'a Path,
    pub settings: &'a Settings,
}

impl Drop for RefreshOnExit<'_> {
    fn drop(&mut self) {
        auto_refresh(self.project, self.settings);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_checks_report_only_assigned_fragments_and_cached_units() {
        let project = project();
        let chunks = vec![
            crate::audio::Interval {
                start: 0.0,
                end: 0.0000625
            };
            5
        ];
        fs::write(
            project.path().join("chunks.json"),
            serde_json::to_vec(&chunks).unwrap(),
        )
        .unwrap();
        let settings = Settings::default();
        let seen = std::sync::Mutex::new(Vec::new());
        let observe = |index| {
            seen.lock().unwrap().push(index);
            Ok(())
        };
        let report =
            inspect_observed(project.path(), &settings, false, Some(&[1, 4]), &observe).unwrap();
        assert_eq!(*seen.lock().unwrap(), [1, 4]);
        assert_eq!(report.analyzed_windows, 2);
        assert_eq!(
            report
                .windows
                .iter()
                .map(|window| window.index)
                .collect::<Vec<_>>(),
            [1, 4]
        );
        seen.lock().unwrap().clear();
        inspect_observed(project.path(), &settings, false, Some(&[1, 4]), &observe).unwrap();
        assert_eq!(*seen.lock().unwrap(), [1, 4]);
        assert!(inspect_observed(project.path(), &settings, false, Some(&[5]), &observe).is_err());
        assert!(
            inspect_observed(project.path(), &settings, false, Some(&[1, 1]), &observe).is_err()
        );
        let narrow = Settings {
            max_chunks: 1,
            ..settings
        };
        let late = inspect_observed(project.path(), &narrow, false, Some(&[4]), &observe).unwrap();
        assert_eq!(late.windows[0].index, 4);
        assert!(inspect_observed(project.path(), &narrow, false, Some(&[1, 4]), &observe).is_err());
    }

    fn project() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&38_u32.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&16000_u32.to_le_bytes());
        wav.extend_from_slice(&32000_u32.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&2_u32.to_le_bytes());
        wav.extend_from_slice(&0_i16.to_le_bytes());
        fs::write(root.path().join("audio.wav"), &wav).unwrap();
        fs::write(
            root.path().join("chunks.json"),
            r#"[{"start":0.0,"end":0.0000625}]"#,
        )
        .unwrap();
        fs::write(
            root.path().join("workflow.json"),
            serde_json::to_vec(&json!({
                "schema_version": 2, "input": root.path().join("missing.mp3"),
                "input_sha256": "recorded-input-hash", "stages": {}
            }))
            .unwrap(),
        )
        .unwrap();
        root
    }

    #[test]
    fn partial_checks_are_cached_and_leave_checkpoints_unchanged() {
        let root = project();
        let original = fs::read(root.path().join("workflow.json")).unwrap();
        let settings = Settings {
            inspect_source_channels: false,
            ..Settings::default()
        };
        let report = inspect(root.path(), &settings).unwrap();
        assert_eq!(report.execution_status, "partial");
        assert_eq!(report.accuracy_status, "unverified");
        assert!(!report.automatic_actions);
        assert_eq!(report.windows[0].missing_roles.len(), 3);
        let cached = inspect_locked(root.path(), &settings, false).unwrap();
        assert_eq!(
            serde_json::to_value(report).unwrap(),
            serde_json::to_value(cached).unwrap()
        );
        assert_eq!(
            fs::read(root.path().join("workflow.json")).unwrap(),
            original
        );
        assert!(read_page(root.path(), 0, 1).is_ok());
        assert!(read_page(root.path(), 2, 1).is_err());
        assert!(read_page(root.path(), 0, 51).is_err());
    }

    #[test]
    fn diagnostics_respect_project_lock_and_foreign_sidecars() {
        let root = project();
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.path().join(".workflow.lock"))
            .unwrap();
        lock.try_lock_exclusive().unwrap();
        assert!(
            inspect(root.path(), &Settings::default())
                .unwrap_err()
                .to_string()
                .contains("already in use")
        );
        drop(lock);
        fs::create_dir(root.path().join(DIRECTORY)).unwrap();
        assert!(inspect(root.path(), &Settings::default()).is_err());
    }

    #[test]
    fn disabled_auto_checks_and_invalid_settings_are_explicit() {
        let root = project();
        auto_refresh(
            root.path(),
            &Settings {
                enabled: false,
                ..Settings::default()
            },
        );
        assert!(!root.path().join(DIRECTORY).exists());
        let settings = Settings {
            max_chunks: 0,
            ..Settings::default()
        };
        assert!(settings.validate().is_err());
        assert!(inspect(root.path(), &settings).is_err());
    }

    #[test]
    fn audio_findings_are_advisory_and_do_not_recommend_gain_from_energy_alone() {
        let section = Section {
            status: "available".into(),
            value: Some(json!({"metrics": {
                "sample_count": 16, "zero_signal": false, "low_rms": true,
                "rms_dbfs": -60.0, "full_scale_fraction": 0.01
            }, "windows": []})),
            error: None,
        };
        let findings = audio_findings(&section, &audio::Settings::default());
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].code, "low_audio_level");
        assert_eq!(findings[0].recommendation, "inspect_fragment");
        assert_eq!(findings[1].code, "near_full_scale_samples");
    }

    #[cfg(unix)]
    #[test]
    fn diagnostic_artifacts_cannot_escape_the_project() {
        let root = project();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("chunks.json"), b"[]").unwrap();
        fs::remove_file(root.path().join("chunks.json")).unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("chunks.json"),
            root.path().join("chunks.json"),
        )
        .unwrap();
        assert!(inspect(root.path(), &Settings::default()).is_err());
        assert_eq!(
            saved_summary(root.path()).unwrap().unwrap()["execution_status"],
            "failed"
        );
    }

    #[test]
    fn failure_record_overrides_old_success_and_can_be_repaired() {
        let root = project();
        let settings = Settings {
            inspect_source_channels: false,
            ..Settings::default()
        };
        inspect(root.path(), &settings).unwrap();
        fs::write(
            root.path().join(DIRECTORY).join("failure.json"),
            r#"{"execution_status":"failed","error":"previous scan failed"}"#,
        )
        .unwrap();
        assert_eq!(
            saved_summary(root.path()).unwrap().unwrap()["execution_status"],
            "failed"
        );
        assert!(read_page(root.path(), 0, 1).is_err());
        inspect(root.path(), &settings).unwrap();
        assert!(!root.path().join(DIRECTORY).join("failure.json").exists());
        assert_eq!(
            saved_summary(root.path()).unwrap().unwrap()["execution_status"],
            "partial"
        );
    }
}
