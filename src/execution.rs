use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;

use crate::workflow::{self, ProjectStatus, State};

const VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Initiator {
    User,
    Rule {
        rule: String,
    },
    Model {
        model: String,
        policy_version: String,
        /// An optional probability in [0, 1], not an authorization threshold.
        confidence: Option<f64>,
    },
}
impl Handle {
    pub fn plan_restart(
        &self,
        step: &str,
        fragments: &[usize],
        parameters: RetryParameters,
        reason: &str,
        evidence: Vec<Value>,
    ) -> Result<String> {
        self.transact(|s| {
            ensure!(
                step.starts_with("asr:") && s.snapshot.stages.contains_key(step),
                "unknown ASR step"
            );
            ensure!(
                !s.snapshot.skipped.contains(step),
                "step is skipped; restore it explicitly"
            );
            ensure!(
                !reason.trim().is_empty() && !fragments.is_empty(),
                "restart requires reason and missing fragments"
            );
            ensure!(
                fragments.iter().collect::<BTreeSet<_>>().len() == fragments.len(),
                "duplicate restart fragments"
            );
            let chunks: Vec<crate::audio::Interval> =
                serde_json::from_slice(&fs::read(self.root.join("chunks.json"))?)?;
            ensure!(
                fragments.iter().all(|index| *index < chunks.len()),
                "restart fragment out of bounds"
            );
            ensure!(
                s.snapshot
                    .attempts
                    .values()
                    .filter(|attempt| attempt.auto_retry)
                    .count()
                    < s.snapshot.budget.max_auto_repeats as usize,
                "automatic retry budget exhausted"
            );
            ensure!(
                s.snapshot.auto_runtime_seconds < s.snapshot.budget.max_auto_runtime_seconds,
                "automatic retry runtime budget exhausted"
            );
            ensure!(
                !s.snapshot
                    .attempts
                    .values()
                    .any(|attempt| attempt.step == step
                        && attempt.kind == "restart"
                        && matches!(attempt.status, State::Running | State::NotStarted)),
                "restart already scheduled"
            );
            let failed = s
                .snapshot
                .attempts
                .values()
                .filter(|attempt| {
                    attempt.step == step
                        && matches!(attempt.status, State::Failed | State::Cancelled)
                })
                .max_by_key(|attempt| {
                    attempt
                        .id
                        .strip_prefix('a')
                        .and_then(|id| id.parse::<u64>().ok())
                })
                .map(|attempt| attempt.id.clone())
                .context("restart requires a stopped prior attempt")?;
            let mut deps = dependencies(s, step)?;
            if s.snapshot.attempts[&failed].artifact.is_some() {
                deps.push(failed.clone());
            }
            let id = new_attempt(
                s,
                step,
                deps,
                fragments.to_vec(),
                parameters.clone(),
                State::NotStarted,
                true,
            );
            s.snapshot.attempts.get_mut(&id).unwrap().kind = "restart".into();
            let mut evidence = evidence;
            evidence.push(serde_json::json!({"failed_attempt":failed}));
            s.snapshot.decisions.push(Decision {
                request_id: format!("resource-{id}"),
                initiator: Initiator::Rule {
                    rule: "resource-recovery".into(),
                },
                reason: reason.into(),
                evidence,
                action: ControlAction::Retry {
                    step: step.into(),
                    fragments: fragments.to_vec(),
                    parameters: parameters.clone(),
                },
                affected_steps: affected(s, step)?,
                fragments: fragments.to_vec(),
                parameters: Some(parameters),
            });
            emit(s, "retry_planned", Some(step), Some(&id), Some(reason));
            s.events.last_mut().unwrap().payload["decision"] =
                serde_json::to_value(s.snapshot.decisions.last())?;
            Ok(id)
        })
    }
    pub fn record_runtime(&self, id: &str, seconds: u64) -> Result<()> {
        self.transact(|s| {
            let previous = s.snapshot.auto_runtime_seconds;
            let previous_runtime = s
                .snapshot
                .attempts
                .get(id)
                .context("unknown attempt")?
                .runtime_seconds;
            charge_runtime(&mut s.snapshot, id, Some(seconds))?;
            if previous != s.snapshot.auto_runtime_seconds
                || previous_runtime != s.snapshot.attempts[id].runtime_seconds
            {
                emit(s, "runtime_observed", None, Some(id), None);
            }
            Ok(())
        })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RetryParameters {
    pub threads: Option<usize>,
    pub max_in_flight: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ControlAction {
    Retry {
        step: String,
        fragments: Vec<usize>,
        parameters: RetryParameters,
    },
    Skip {
        step: String,
    },
    CancelAttempt {
        attempt_id: String,
    },
    Restore {
        step: String,
    },
    AddCheck {
        step: String,
        dependencies: Vec<String>,
        fragments: Vec<usize>,
    },
    Select {
        attempt_id: String,
    },
    Invalidate {
        step: String,
    },
    Dependencies {
        attempt_id: String,
        dependencies: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlRequest {
    pub version: u32,
    pub request_id: String,
    pub expected_revision: Option<u64>,
    pub initiator: Initiator,
    pub reason: String,
    #[serde(default)]
    pub evidence: Vec<Value>,
    #[serde(default)]
    pub confirmed: bool,
    pub action: ControlAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    #[serde(default = "event_version")]
    pub version: u32,
    pub sequence: u64,
    pub kind: String,
    pub step: Option<String>,
    pub attempt_id: Option<String>,
    pub message: Option<String>,
    #[serde(default)]
    pub payload: Value,
}

fn event_version() -> u32 {
    VERSION
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
    pub fingerprint: String,
    pub base_attempt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attempt {
    pub id: String,
    pub step: String,
    #[serde(default = "stage_kind")]
    pub kind: String,
    pub dependencies: Vec<String>,
    pub fragments: Vec<usize>,
    pub parameters: RetryParameters,
    pub status: State,
    pub artifact: Option<Artifact>,
    pub done: BTreeSet<usize>,
    pub auto_retry: bool,
    pub started_at_seconds: Option<u64>,
    #[serde(default)]
    pub runtime_seconds: Option<u64>,
}

fn stage_kind() -> String {
    "stage".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageView {
    pub status: State,
    pub selected_attempt: Option<String>,
    pub active_attempt: Option<String>,
    pub units: Vec<usize>,
    pub done: BTreeSet<usize>,
    pub unit: String,
    pub numeric_done: u64,
    pub numeric_total: Option<u64>,
    pub operation: Option<String>,
}

impl StageView {
    fn new(status: State) -> Self {
        Self {
            status,
            selected_attempt: None,
            active_attempt: None,
            units: Vec::new(),
            done: BTreeSet::new(),
            unit: String::new(),
            numeric_done: 0,
            numeric_total: None,
            operation: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub max_auto_repeats: u32,
    pub max_auto_runtime_seconds: u64,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            max_auto_repeats: 3,
            max_auto_runtime_seconds: 600,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub request_id: String,
    pub initiator: Initiator,
    pub reason: String,
    pub evidence: Vec<Value>,
    pub action: ControlAction,
    pub affected_steps: Vec<String>,
    pub fragments: Vec<usize>,
    pub parameters: Option<RetryParameters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    pub request_id: String,
    pub revision: u64,
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preview {
    pub revision: u64,
    pub affected_steps: Vec<String>,
    pub fragments: Vec<usize>,
    pub parameters: Option<RetryParameters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub schema_version: u32,
    pub revision: u64,
    pub duration: Option<f64>,
    pub stages: BTreeMap<String, StageView>,
    pub attempts: BTreeMap<String, Attempt>,
    pub decisions: Vec<Decision>,
    pub skipped: BTreeSet<String>,
    pub mode_steps: BTreeSet<String>,
    pub budget: Budget,
    #[serde(default)]
    pub auto_runtime_seconds: u64,
    pub queued: Vec<Receipt>,
    #[serde(default)]
    pub interrupted: bool,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub project_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Store {
    snapshot: Snapshot,
    events: Vec<Event>,
    requests: BTreeMap<String, ControlRequest>,
    queue: Vec<String>,
    next_attempt: u64,
    #[serde(skip)]
    dirty: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryWork {
    pub attempt_id: String,
    pub step: String,
    pub fragments: Vec<usize>,
    pub parameters: RetryParameters,
}

#[derive(Debug, Clone)]
pub struct CheckWork {
    pub attempt_id: String,
    pub step: String,
    pub fragments: Vec<usize>,
    pub input_folders: BTreeMap<String, PathBuf>,
}

#[derive(Clone)]
pub struct Handle {
    root: PathBuf,
    guard: Arc<Mutex<()>>,
}

fn directory(root: &Path) -> PathBuf {
    root.join(".execution")
}
fn path(root: &Path) -> PathBuf {
    directory(root).join("state.json")
}

fn private_directory(root: &Path) -> Result<PathBuf> {
    let dir = directory(root);
    if !dir.exists() {
        fs::DirBuilder::new().mode(0o700).create(&dir)?;
    }
    ensure!(
        !fs::symlink_metadata(&dir)?.file_type().is_symlink(),
        "execution directory must not be a symlink"
    );
    ensure!(dir.is_dir(), "execution state path must be a directory");
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
    Ok(dir)
}

fn load(root: &Path) -> Result<Store> {
    ensure!(
        !fs::symlink_metadata(directory(root))?
            .file_type()
            .is_symlink(),
        "execution directory must not be a symlink"
    );
    ensure!(
        !fs::symlink_metadata(path(root))?.file_type().is_symlink(),
        "execution state must not be a symlink"
    );
    let store: Store =
        serde_json::from_slice(&fs::read(path(root))?).context("invalid execution state")?;
    ensure!(
        store.snapshot.schema_version == VERSION,
        "unsupported execution schema"
    );
    ensure!(
        store.snapshot.revision == store.events.last().map_or(0, |event| event.sequence),
        "execution journal sequence mismatch"
    );
    ensure!(
        store
            .events
            .iter()
            .enumerate()
            .all(|(index, event)| event.version == VERSION && event.sequence == index as u64 + 1),
        "execution journal is not contiguous or has an unsupported version"
    );
    for (id, attempt) in &store.snapshot.attempts {
        let number = id
            .strip_prefix('a')
            .and_then(|digits| digits.parse::<u64>().ok())
            .context("invalid attempt ID in execution history")?;
        ensure!(
            number <= store.next_attempt && attempt.id == *id,
            "attempt history ID mismatch"
        );
        for dependency in &attempt.dependencies {
            let parent = dependency
                .strip_prefix('a')
                .and_then(|digits| digits.parse::<u64>().ok())
                .context("invalid dependency attempt ID")?;
            ensure!(
                parent < number && store.snapshot.attempts.contains_key(dependency),
                "execution attempt graph is cyclic or missing a dependency"
            );
        }
    }
    Ok(store)
}

fn write(root: &Path, store: &Store) -> Result<()> {
    let dir = private_directory(root)?;
    let partial = dir.join(format!(
        "state.{}.{}.partial",
        std::process::id(),
        now_nanos()
    ));
    let result = (|| {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&partial)?;
        serde_json::to_writer(&file, store)?;
        file.sync_all()?;
        fs::rename(&partial, path(root))?;
        File::open(dir)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

fn emit(
    store: &mut Store,
    kind: &str,
    step: Option<&str>,
    attempt: Option<&str>,
    message: Option<&str>,
) {
    store.snapshot.revision += 1;
    let attempt_view = attempt.and_then(|id| store.snapshot.attempts.get(id));
    let stage_name = step.or_else(|| attempt_view.map(|attempt| attempt.step.as_str()));
    let payload = serde_json::json!({
        "stage": stage_name.and_then(|name| store.snapshot.stages.get(name)),
        "attempt": attempt_view,
        "duration": store.snapshot.duration,
        "mode_steps": store.snapshot.mode_steps,
        "skipped": store.snapshot.skipped,
        "budget": store.snapshot.budget,
        "auto_runtime_seconds": store.snapshot.auto_runtime_seconds,
        "interrupted": store.snapshot.interrupted,
    });
    store.events.push(Event {
        version: VERSION,
        sequence: store.snapshot.revision,
        kind: kind.into(),
        step: step.map(str::to_owned),
        attempt_id: attempt.map(str::to_owned),
        message: message.map(str::to_owned),
        payload,
    });
}

impl Handle {
    pub fn open(root: &Path, status: &ProjectStatus) -> Result<Self> {
        let handle = Self {
            root: root.to_path_buf(),
            guard: Arc::new(Mutex::new(())),
        };
        handle.transact_optional(|existing| {
            let mut store = match existing {
                Some(store) => store,
                None => {
                    let mut snapshot = Snapshot {
                        schema_version: VERSION,
                        revision: 0,
                        duration: None,
                        stages: BTreeMap::new(),
                        attempts: BTreeMap::new(),
                        decisions: Vec::new(),
                        skipped: BTreeSet::new(),
                        mode_steps: status.stages.keys().cloned().collect(),
                        budget: Budget::default(),
                        auto_runtime_seconds: 0,
                        queued: Vec::new(),
                        interrupted: false,
                        metadata: Value::Null,
                        project_root: Some(root.into()),
                    };
                    let mut next_attempt = 0;
                    for step in workflow::steps_for_roles(&status.asr_roles)
                        .into_iter()
                        .chain(
                            status
                                .stages
                                .keys()
                                .filter(|step| {
                                    !workflow::steps_for_roles(&status.asr_roles).contains(step)
                                })
                                .cloned(),
                        )
                    {
                        let stage = status
                            .stages
                            .get(&step)
                            .cloned()
                            .unwrap_or(workflow::Stage {
                                status: State::NotStarted,
                                attempts: 0,
                                fingerprint: String::new(),
                                output_sha256: None,
                                warning: None,
                            });
                        let mut view = StageView::new(stage.status);
                        let archive_root = root.join("attempts").join(step.replace(':', "-"));
                        if archive_root.is_dir() {
                            let mut archived = fs::read_dir(&archive_root)?
                                .map(|entry| entry.map(|entry| entry.path()))
                                .collect::<std::io::Result<Vec<_>>>()?;
                            archived.sort_by_key(|path| {
                                path.file_name()
                                    .and_then(|name| name.to_str())
                                    .and_then(|name| name.parse::<u64>().ok())
                            });
                            for archive in archived {
                                ensure!(
                                    archive.is_dir()
                                        && archive
                                            .file_name()
                                            .and_then(|name| name.to_str())
                                            .is_some_and(|name| name.parse::<u64>().is_ok()),
                                    "invalid legacy attempt archive"
                                );
                                if let Some(hash) = hash_directory(&archive)? {
                                    next_attempt += 1;
                                    let id = format!("a{next_attempt}");
                                    let historical_fingerprint =
                                        if archive.join(".stage.json").exists() {
                                            let stage: workflow::Stage = serde_json::from_slice(
                                                &fs::read(archive.join(".stage.json"))?,
                                            )?;
                                            stage.fingerprint
                                        } else {
                                            String::new()
                                        };
                                    snapshot.attempts.insert(
                                        id.clone(),
                                        Attempt {
                                            id: id.clone(),
                                            step: step.clone(),
                                            kind: stage_kind(),
                                            dependencies: Vec::new(),
                                            fragments: Vec::new(),
                                            parameters: RetryParameters::default(),
                                            status: State::Ready,
                                            artifact: Some(Artifact {
                                                path: archive,
                                                sha256: hash,
                                                fingerprint: historical_fingerprint,
                                                base_attempt: None,
                                            }),
                                            done: BTreeSet::new(),
                                            auto_retry: false,
                                            started_at_seconds: None,
                                            runtime_seconds: None,
                                        },
                                    );
                                }
                            }
                        }
                        if stage.status == State::Ready {
                            next_attempt += 1;
                            let id = format!("a{next_attempt}");
                            let dependencies = workflow::parents(&step, &status.asr_roles)
                                .iter()
                                .filter_map(|parent| {
                                    snapshot
                                        .stages
                                        .get(parent)
                                        .and_then(|view| view.selected_attempt.clone())
                                })
                                .collect();
                            view.selected_attempt = Some(id.clone());
                            let artifact_path = freeze_asr(root, &step, &id)?;
                            snapshot.attempts.insert(
                                id.clone(),
                                Attempt {
                                    id: id.clone(),
                                    step: step.clone(),
                                    kind: stage_kind(),
                                    dependencies: Vec::new(),
                                    fragments: Vec::new(),
                                    parameters: RetryParameters::default(),
                                    status: State::Ready,
                                    artifact: stage.output_sha256.as_ref().map(|hash| Artifact {
                                        path: artifact_path,
                                        sha256: hash.clone(),
                                        fingerprint: stage.fingerprint.clone(),
                                        base_attempt: None,
                                    }),
                                    done: BTreeSet::new(),
                                    auto_retry: false,
                                    started_at_seconds: None,
                                    runtime_seconds: None,
                                },
                            );
                            snapshot
                                .attempts
                                .get_mut(view.selected_attempt.as_ref().unwrap())
                                .unwrap()
                                .dependencies = dependencies;
                        }
                        snapshot.stages.insert(step.clone(), view);
                    }
                    let mut store = Store {
                        snapshot,
                        events: Vec::new(),
                        requests: BTreeMap::new(),
                        queue: Vec::new(),
                        next_attempt,
                        dirty: false,
                    };
                    emit(&mut store, "plan_created", None, None, None);
                    store
                }
            };
            if let Some(previous) = &store.snapshot.project_root
                && previous != root
            {
                for attempt in store.snapshot.attempts.values_mut() {
                    if let Some(artifact) = &mut attempt.artifact {
                        let relative = artifact
                            .path
                            .strip_prefix(previous)
                            .context("artifact is outside recorded project root")?;
                        artifact.path = root.join(relative);
                    }
                }
                store.snapshot.project_root = Some(root.into());
                emit(&mut store, "project_relocated", None, None, None);
            } else if store.snapshot.project_root.is_none() {
                store.snapshot.project_root = Some(root.into());
                emit(&mut store, "project_root_recorded", None, None, None);
            }
            let mut changed = false;
            for (step, legacy) in &status.stages {
                if !store.snapshot.stages.contains_key(step) {
                    store
                        .snapshot
                        .stages
                        .insert(step.clone(), StageView::new(legacy.status));
                    changed = true;
                }
            }
            if changed {
                emit(&mut store, "plan_updated", None, None, None);
            }
            let chunks_path = root.join("chunks.json");
            if status
                .stages
                .get("chunks")
                .is_some_and(|stage| stage.status == State::Ready)
                && chunks_path.exists()
            {
                let chunks: Vec<crate::audio::Interval> =
                    serde_json::from_slice(&fs::read(chunks_path)?)?;
                let units = (0..chunks.len()).collect::<Vec<_>>();
                let mut initialized = false;
                for (step, view) in &mut store.snapshot.stages {
                    if step.starts_with("asr:") && view.units.is_empty() {
                        view.units = units.clone();
                        view.unit = "windows".into();
                        if view.status == State::Ready {
                            view.done = units.iter().copied().collect();
                        }
                        initialized = true;
                    }
                }
                if initialized {
                    emit(&mut store, "units_initialized", None, None, None);
                }
            }
            let interrupted = store
                .snapshot
                .stages
                .iter()
                .filter_map(|(step, view)| {
                    view.active_attempt
                        .as_ref()
                        .filter(|id| {
                            store
                                .snapshot
                                .attempts
                                .get(*id)
                                .is_some_and(|attempt| attempt.status == State::Running)
                        })
                        .map(|id| (step.clone(), id.clone()))
                })
                .collect::<Vec<_>>();
            for (step, id) in interrupted {
                let legacy = status.stages.get(&step);
                let attempt = store.snapshot.attempts.get_mut(&id).unwrap();
                let view = store.snapshot.stages.get_mut(&step).unwrap();
                view.active_attempt = None;
                if let Some(legacy) = legacy
                    && legacy.status == State::Ready
                    && let Some(hash) = &legacy.output_sha256
                {
                    attempt.status = State::Ready;
                    attempt.artifact = Some(Artifact {
                        path: freeze_asr(root, &step, &id)?,
                        sha256: hash.clone(),
                        fingerprint: legacy.fingerprint.clone(),
                        base_attempt: None,
                    });
                    view.selected_attempt = Some(id.clone());
                    view.status = State::Ready;
                    emit(
                        &mut store,
                        "attempt_recovered",
                        Some(&step),
                        Some(&id),
                        None,
                    );
                } else {
                    attempt.status = State::Failed;
                    view.status = State::Failed;
                    emit(
                        &mut store,
                        "attempt_interrupted",
                        Some(&step),
                        Some(&id),
                        None,
                    );
                }
            }
            if store.snapshot.interrupted {
                store.snapshot.interrupted = false;
                emit(&mut store, "execution_resumed", None, None, None);
            }
            Ok((store, ()))
        })?;
        Ok(handle)
    }

    fn transact_optional<T>(
        &self,
        f: impl FnOnce(Option<Store>) -> Result<(Store, T)>,
    ) -> Result<T> {
        let _guard = self.guard.lock().expect("execution mutex poisoned");
        private_directory(&self.root)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(directory(&self.root).join("lock"))?;
        lock.lock_exclusive()?;
        let old = if path(&self.root).exists() {
            Some(load(&self.root)?)
        } else {
            None
        };
        let old_revision = old.as_ref().map(|s| s.snapshot.revision);
        let (store, value) = f(old)?;
        if old_revision != Some(store.snapshot.revision) || store.dirty {
            write(&self.root, &store)?;
        }
        Ok(value)
    }

    fn transact<T>(&self, f: impl FnOnce(&mut Store) -> Result<T>) -> Result<T> {
        self.transact_optional(|old| {
            let mut store = old.context("project has no execution state")?;
            let value = f(&mut store)?;
            Ok((store, value))
        })
    }

    pub fn snapshot(&self) -> Result<Snapshot> {
        Ok(load(&self.root)?.snapshot)
    }
    pub fn project(&self) -> &Path {
        &self.root
    }
    pub fn interrupt(&self) -> Result<()> {
        self.transact(|s| {
            if !s.snapshot.interrupted {
                s.snapshot.interrupted = true;
                emit(s, "execution_interrupted", None, None, None);
            }
            Ok(())
        })
    }
    pub fn set_duration(&self, duration: f64) -> Result<()> {
        ensure!(duration.is_finite() && duration >= 0.0, "invalid duration");
        self.transact(|s| {
            if s.snapshot.duration != Some(duration) {
                s.snapshot.duration = Some(duration);
                emit(s, "duration", None, None, None);
            }
            Ok(())
        })
    }
    pub fn set_metadata(&self, config: &crate::config::Config) -> Result<()> {
        let state = workflow::status(&self.root)?;
        let mut models = BTreeMap::new();
        for role in &state.asr_roles {
            let adapter = config
                .asr
                .get(role)
                .context("missing configured ASR adapter")?;
            models.insert(
                role.clone(),
                serde_json::json!({
                    "engine": config.asr_engine(role)?,
                    "threads": adapter.threads,
                    "max_in_flight": adapter.max_in_flight,
                }),
            );
        }
        let metadata = serde_json::json!({
            "models": models,
            "max_parallel_asr_models": config.resources.max_parallel_asr_models,
            "max_asr_workers": config.resources.max_asr_workers,
            "external_review": if config.llm.enabled {
                Some(&config.llm.program)
            } else { None },
        });
        self.transact(|s| {
            if s.snapshot.metadata != metadata {
                s.snapshot.metadata = metadata.clone();
                emit(s, "plan_metadata", None, None, None);
                s.events.last_mut().unwrap().payload["metadata"] = metadata;
            }
            Ok(())
        })
    }
    pub fn set_mode_steps(&self, steps: &BTreeSet<String>) -> Result<()> {
        self.transact(|s| {
            for step in steps {
                ensure!(s.snapshot.stages.contains_key(step), "unknown step {step}");
            }
            let mut steps = steps.clone();
            steps.extend(
                s.snapshot
                    .stages
                    .keys()
                    .filter(|step| step.starts_with("check:"))
                    .cloned(),
            );
            if s.snapshot.mode_steps != steps {
                s.snapshot.mode_steps = steps;
                emit(s, "mode_steps", None, None, None);
            }
            Ok(())
        })
    }
    pub fn set_units(&self, step: &str, units: &[usize], unit: &str) -> Result<()> {
        self.transact(|s| {
            let view = s.snapshot.stages.get_mut(step).context("unknown step")?;
            let unique = units.iter().copied().collect::<BTreeSet<_>>();
            ensure!(unique.len() == units.len(), "duplicate fragment IDs");
            if view.units != units || view.unit != unit {
                view.units = units.to_vec();
                view.unit = unit.to_owned();
                view.done.retain(|id| unique.contains(id));
                view.numeric_total = Some(units.len() as u64);
                view.numeric_done = view.done.len() as u64;
                emit(s, "units", Some(step), None, None);
            }
            Ok(())
        })
    }
    pub fn unit_done(&self, step: &str, index: usize) -> Result<()> {
        self.transact(|s| {
            let view = s.snapshot.stages.get_mut(step).context("unknown step")?;
            ensure!(
                view.units.contains(&index),
                "fragment {index} is not assigned to {step}"
            );
            if view.done.insert(index) {
                view.numeric_done = view.done.len() as u64;
                if let Some(id) = &view.active_attempt {
                    s.snapshot
                        .attempts
                        .get_mut(id)
                        .context("missing active attempt")?
                        .done
                        .insert(index);
                }
                emit(s, "unit_done", Some(step), None, Some(&index.to_string()));
            }
            Ok(())
        })
    }
    pub fn numeric_progress(
        &self,
        step: &str,
        done: u64,
        total: Option<u64>,
        unit: &str,
    ) -> Result<()> {
        ensure!(
            total.is_none_or(|total| done <= total),
            "progress exceeds total"
        );
        self.transact(|s| {
            let view = s.snapshot.stages.get_mut(step).context("unknown step")?;
            if (view.numeric_done, view.numeric_total, view.unit.as_str()) != (done, total, unit) {
                let old_done = view.numeric_done;
                let old_total = view.numeric_total;
                let old_unit = view.unit.clone();
                view.numeric_done = done;
                view.numeric_total = total;
                view.unit = unit.to_owned();
                let significant = old_total != total
                    || old_unit != unit
                    || match total {
                        Some(total) if total > 0 => {
                            done == total
                                || done.saturating_mul(20) / total
                                    != old_done.saturating_mul(20) / total
                        }
                        _ => done == 1 || done.is_power_of_two(),
                    };
                if significant {
                    emit(s, "progress", Some(step), None, None);
                } else {
                    s.dirty = true;
                }
            }
            Ok(())
        })
    }
    pub fn operation(&self, step: &str, message: &str) -> Result<()> {
        self.transact(|s| {
            let view = s.snapshot.stages.get_mut(step).context("unknown step")?;
            if view.operation.as_deref() != Some(message) {
                view.operation = Some(message.to_owned());
                emit(s, "operation", Some(step), None, Some(message));
            }
            Ok(())
        })
    }
    pub fn cancelled(&self, step: &str) -> Result<bool> {
        self.apply_controls()?;
        let snapshot = self.snapshot()?;
        let view = snapshot.stages.get(step).context("unknown step")?;
        if snapshot.interrupted {
            return Ok(true);
        }
        if let Some(attempt) = view
            .active_attempt
            .as_ref()
            .and_then(|id| snapshot.attempts.get(id))
            && attempt.auto_retry
            && attempt.started_at_seconds.is_some_and(|start| {
                snapshot
                    .auto_runtime_seconds
                    .saturating_add(if attempt.runtime_seconds.is_some() {
                        0
                    } else {
                        now_seconds().saturating_sub(start)
                    })
                    >= snapshot.budget.max_auto_runtime_seconds
            })
        {
            self.transact(|s| {
                let id = attempt.id.clone();
                let retry = s
                    .snapshot
                    .attempts
                    .get_mut(&id)
                    .context("unknown attempt")?;
                if retry.status == State::Running {
                    retry.status = State::Cancelled;
                    charge_runtime(&mut s.snapshot, &id, None)?;
                    emit(s, "retry_budget_exhausted", Some(step), Some(&id), None);
                }
                Ok(())
            })?;
            return Ok(true);
        }
        Ok(snapshot.skipped.contains(step)
            || view
                .active_attempt
                .as_ref()
                .and_then(|id| snapshot.attempts.get(id))
                .is_some_and(|attempt| attempt.status == State::Cancelled))
    }
    pub fn attempt_cancelled(&self, id: &str) -> Result<bool> {
        self.apply_controls()?;
        let snapshot = self.snapshot()?;
        let attempt = snapshot.attempts.get(id).context("unknown attempt")?;
        if snapshot.interrupted
            || snapshot.skipped.contains(&attempt.step)
            || attempt.status == State::Cancelled
        {
            return Ok(true);
        }
        self.cancelled(&attempt.step)
    }
    pub fn begin_stage(&self, step: &str) -> Result<String> {
        self.transact(|s| {
            ensure!(!s.snapshot.skipped.contains(step), "step is skipped");
            let view = s.snapshot.stages.get(step).context("unknown step")?;
            if let Some(id) = &view.active_attempt
                && s.snapshot.attempts[id].status == State::Running
            {
                return Ok(id.clone());
            }
            let pending_restart = s
                .snapshot
                .attempts
                .values()
                .find(|attempt| {
                    attempt.step == step
                        && attempt.kind == "restart"
                        && attempt.status == State::NotStarted
                })
                .map(|attempt| attempt.id.clone());
            let id = if let Some(id) = pending_restart {
                ensure!(
                    s.snapshot.auto_runtime_seconds < s.snapshot.budget.max_auto_runtime_seconds,
                    "automatic retry runtime budget exhausted"
                );
                let attempt = s.snapshot.attempts.get_mut(&id).unwrap();
                attempt.status = State::Running;
                attempt.started_at_seconds = Some(now_seconds());
                id
            } else {
                let dependencies = dependencies(s, step)?;
                let fragments = view.units.clone();
                new_attempt(
                    s,
                    step,
                    dependencies,
                    fragments,
                    RetryParameters::default(),
                    State::Running,
                    false,
                )
            };
            let view = s.snapshot.stages.get_mut(step).unwrap();
            view.units = s.snapshot.attempts[&id].fragments.clone();
            view.numeric_total = (!view.units.is_empty()).then_some(view.units.len() as u64);
            view.active_attempt = Some(id.clone());
            view.status = State::Running;
            view.done.clear();
            view.numeric_done = 0;
            workflow::project_started_stage(&self.root, step)?;
            emit(s, "attempt_started", Some(step), Some(&id), None);
            Ok(id)
        })
    }
    pub fn finish_stage(&self, step: &str) -> Result<()> {
        let (hash, fingerprint) = workflow::completed_stage_artifact(&self.root, step)?;
        self.transact(|s| {
            let id = s
                .snapshot
                .stages
                .get(step)
                .context("unknown step")?
                .active_attempt
                .clone()
                .context("no active attempt")?;
            let attempt = s.snapshot.attempts.get_mut(&id).unwrap();
            ensure!(attempt.status == State::Running, "attempt is not running");
            let artifact_path = freeze_asr(&self.root, step, &id)?;
            attempt.status = State::Ready;
            attempt.artifact = Some(Artifact {
                path: artifact_path,
                sha256: hash.clone(),
                fingerprint,
                base_attempt: None,
            });
            let view = s.snapshot.stages.get_mut(step).unwrap();
            view.active_attempt = None;
            view.selected_attempt = Some(id.clone());
            view.status = State::Ready;
            charge_runtime(&mut s.snapshot, &id, None)?;
            emit(s, "attempt_completed", Some(step), Some(&id), None);
            workflow::project_completed_stage(&self.root, step, hash.clone())?;
            Ok(())
        })
    }
    pub fn fail_stage(&self, step: &str, status: State, message: &str) -> Result<()> {
        self.transact(|s| {
            let view = s.snapshot.stages.get_mut(step).context("unknown step")?;
            let status = if s.snapshot.skipped.contains(step) {
                State::Skipped
            } else if view
                .active_attempt
                .as_ref()
                .and_then(|id| s.snapshot.attempts.get(id))
                .is_some_and(|attempt| attempt.status == State::Cancelled)
            {
                State::Cancelled
            } else {
                status
            };
            view.status = if status == State::Cancelled && view.selected_attempt.is_some() {
                State::Ready
            } else {
                status
            };
            let projected_status = view.status;
            if let Some(id) = view.active_attempt.take() {
                s.snapshot.attempts.get_mut(&id).unwrap().status = status;
                capture_partial_asr(&self.root, &mut s.snapshot, &id)?;
                charge_runtime(&mut s.snapshot, &id, None)?;
                emit(s, "attempt_stopped", Some(step), Some(&id), Some(message));
            } else {
                emit(s, "stage_stopped", Some(step), None, Some(message));
            }
            workflow::project_stage_warning(&self.root, step, projected_status, message)
        })
    }
    pub fn cached_warning(&self, step: &str, message: &str) -> Result<()> {
        self.transact(|s| {
            let stage = s.snapshot.stages.get(step).context("unknown step")?;
            if stage.status == State::Ready {
                workflow::project_stage_warning(&self.root, step, State::Ready, message)?;
                emit(s, "stage_warning", Some(step), None, Some(message));
            }
            Ok(())
        })
    }
    pub fn project_legacy(&self) -> Result<()> {
        self.transact(|s| {
            let previous = s.snapshot.clone();
            workflow::project_control_state(&self.root, &previous, &mut s.snapshot)
        })
    }
    pub fn is_skipped(&self, step: &str) -> Result<bool> {
        let snapshot = self.snapshot()?;
        ensure!(snapshot.stages.contains_key(step), "unknown step {step}");
        Ok(snapshot.skipped.contains(step))
    }
    pub fn pending_retries(&self) -> Result<Vec<RetryWork>> {
        let s = self.snapshot()?;
        Ok(s.attempts
            .values()
            .filter(|attempt| {
                attempt.kind == "retry"
                    && attempt.status == State::NotStarted
                    && !s.skipped.contains(&attempt.step)
                    && s.stages[&attempt.step].active_attempt.is_none()
                    && s.stages[&attempt.step].active_attempt.is_none()
                    && attempt
                        .dependencies
                        .iter()
                        .all(|id| s.attempts[id].status == State::Ready)
            })
            .map(|attempt| RetryWork {
                attempt_id: attempt.id.clone(),
                step: attempt.step.clone(),
                fragments: attempt.fragments.clone(),
                parameters: attempt.parameters.clone(),
            })
            .collect())
    }
    pub fn begin_retry(&self, id: &str) -> Result<()> {
        self.transact(|s| {
            let attempt = s.snapshot.attempts.get(id).context("unknown attempt")?;
            ensure!(attempt.kind == "retry", "attempt is not a retry");
            ensure!(attempt.status == State::NotStarted, "retry is not pending");
            ensure!(
                !s.snapshot.skipped.contains(&attempt.step),
                "step is skipped"
            );
            ensure!(
                attempt
                    .dependencies
                    .iter()
                    .all(|parent| s.snapshot.attempts[parent].status == State::Ready),
                "retry dependencies are not ready"
            );
            let step = attempt.step.clone();
            let view = s.snapshot.stages.get_mut(&step).unwrap();
            ensure!(view.active_attempt.is_none(), "another attempt is running");
            view.active_attempt = Some(id.into());
            let attempt = s.snapshot.attempts.get_mut(id).unwrap();
            attempt.status = State::Running;
            attempt.started_at_seconds = Some(now_seconds());
            emit(s, "retry_started", Some(&step), Some(id), None);
            Ok(())
        })
    }
    pub fn retry_unit_done(&self, id: &str, index: usize) -> Result<()> {
        self.transact(|s| {
            let attempt = s.snapshot.attempts.get_mut(id).context("unknown attempt")?;
            ensure!(attempt.kind == "retry", "attempt is not a retry");
            ensure!(
                attempt.status == State::Running && attempt.fragments.contains(&index),
                "fragment is not assigned to running retry"
            );
            if attempt.done.insert(index) {
                emit(
                    s,
                    "retry_unit_done",
                    None,
                    Some(id),
                    Some(&index.to_string()),
                );
            }
            Ok(())
        })
    }
    pub fn finish_retry(&self, id: &str) -> Result<()> {
        self.transact(|s| {
            let attempt = s.snapshot.attempts.get_mut(id).context("unknown attempt")?;
            ensure!(attempt.kind == "retry", "attempt is not a retry");
            ensure!(attempt.status == State::Running, "retry is not running");
            ensure!(
                attempt
                    .fragments
                    .iter()
                    .all(|index| attempt.done.contains(index)),
                "retry has unfinished fragments"
            );
            let step = attempt.step.clone();
            let folder = retry_output_path(&self.root, id)?;
            ensure!(folder.is_dir(), "retry output directory is missing");
            for index in &attempt.fragments {
                ensure!(
                    folder.join(format!("{index:05}.json")).is_file(),
                    "retry output for fragment {index} is missing"
                );
            }
            let mut hasher = sha2::Sha256::new();
            for index in &attempt.fragments {
                let file = folder.join(format!("{index:05}.json"));
                sha2::Digest::update(&mut hasher, index.to_le_bytes());
                sha2::Digest::update(&mut hasher, crate::pipeline::hash_file(&file)?.as_bytes());
            }
            let base_attempt = s.snapshot.stages[&step].selected_attempt.clone();
            ensure!(
                base_attempt
                    .as_ref()
                    .is_none_or(|base| attempt.dependencies.contains(base)),
                "selected base changed since retry was planned"
            );
            attempt.artifact = Some(Artifact {
                path: folder,
                sha256: hex::encode(sha2::Digest::finalize(hasher)),
                fingerprint: workflow::status(&self.root)?.stages[&step]
                    .fingerprint
                    .clone(),
                base_attempt,
            });
            attempt.status = State::Ready;
            charge_runtime(&mut s.snapshot, id, None)?;
            s.snapshot.stages.get_mut(&step).unwrap().active_attempt = None;
            emit(s, "retry_completed", Some(&step), Some(id), None);
            Ok(())
        })
    }
    pub fn select_retry(&self, id: &str) -> Result<()> {
        let snapshot = self.snapshot()?;
        let step = snapshot
            .attempts
            .get(id)
            .context("unknown attempt")?
            .step
            .clone();
        if snapshot.stages[&step].selected_attempt.as_deref() == Some(id) {
            return Ok(());
        }
        self.transact(|s| {
            let previous = s.snapshot.clone();
            select(s, id, &self.root)?;
            workflow::project_selected_stage(&self.root, &step)?;
            workflow::project_control_state(&self.root, &previous, &mut s.snapshot)
        })
    }
    pub fn retry_output(&self, id: &str) -> PathBuf {
        retry_output_path(&self.root, id).expect("attempt IDs must be generated by execution")
    }
    pub fn pending_checks(&self) -> Result<Vec<CheckWork>> {
        let s = self.snapshot()?;
        let mut work = Vec::new();
        for attempt in s.attempts.values().filter(|attempt| {
            attempt.kind == "check"
                && attempt.status == State::NotStarted
                && !s.skipped.contains(&attempt.step)
                && attempt
                    .dependencies
                    .iter()
                    .all(|parent| s.attempts[parent].status == State::Ready)
        }) {
            let input_folders = check_inputs(&self.root, &s, attempt)?;
            work.push(CheckWork {
                attempt_id: attempt.id.clone(),
                step: attempt.step.clone(),
                fragments: attempt.fragments.clone(),
                input_folders,
            });
        }
        Ok(work)
    }
    pub fn begin_check(&self, id: &str) -> Result<()> {
        self.transact(|s| {
            let attempt = s.snapshot.attempts.get(id).context("unknown check")?;
            ensure!(
                attempt.kind == "check" && attempt.status == State::NotStarted,
                "check is not pending"
            );
            ensure!(
                !s.snapshot.skipped.contains(&attempt.step),
                "check is skipped"
            );
            ensure!(
                attempt
                    .dependencies
                    .iter()
                    .all(|parent| s.snapshot.attempts[parent].status == State::Ready),
                "check dependencies are not ready"
            );
            let step = attempt.step.clone();
            let attempt = s.snapshot.attempts.get_mut(id).unwrap();
            attempt.status = State::Running;
            attempt.started_at_seconds = Some(now_seconds());
            let stage = s.snapshot.stages.get_mut(&step).unwrap();
            stage.active_attempt = Some(id.into());
            stage.status = State::Running;
            stage.units = s.snapshot.attempts[id].fragments.clone();
            stage.done.clear();
            stage.numeric_done = 0;
            stage.numeric_total = Some(stage.units.len() as u64);
            emit(s, "check_started", Some(&step), Some(id), None);
            Ok(())
        })
    }
    pub fn check_unit_done(&self, id: &str, index: usize) -> Result<()> {
        let snapshot = self.snapshot()?;
        let attempt = snapshot.attempts.get(id).context("unknown check")?;
        ensure!(
            attempt.kind == "check" && attempt.status == State::Running,
            "check is not running"
        );
        self.unit_done(&attempt.step, index)
    }
    pub fn finish_check(&self, id: &str, report: &Path) -> Result<()> {
        let report = report.canonicalize()?;
        ensure!(
            report.starts_with(self.root.canonicalize()?.join(".execution")),
            "check report must be owned by execution"
        );
        let hash = crate::pipeline::hash_file(&report)?;
        self.transact(|s| {
            let attempt = s.snapshot.attempts.get_mut(id).context("unknown check")?;
            ensure!(
                attempt.kind == "check" && attempt.status == State::Running,
                "check is not running"
            );
            ensure!(
                attempt
                    .fragments
                    .iter()
                    .all(|index| attempt.done.contains(index)),
                "check has unfinished fragments"
            );
            attempt.status = State::Ready;
            attempt.artifact = Some(Artifact {
                path: report,
                sha256: hash,
                fingerprint: String::new(),
                base_attempt: None,
            });
            let step = attempt.step.clone();
            charge_runtime(&mut s.snapshot, id, None)?;
            let stage = s.snapshot.stages.get_mut(&step).unwrap();
            stage.active_attempt = None;
            stage.selected_attempt = Some(id.into());
            stage.status = State::Ready;
            emit(s, "check_completed", Some(&step), Some(id), None);
            Ok(())
        })
    }
    pub fn apply_controls(&self) -> Result<()> {
        self.transact(|s| {
            let queued = std::mem::take(&mut s.queue);
            for id in queued {
                let request = s
                    .requests
                    .get(&id)
                    .context("missing queued request")?
                    .clone();
                let mut candidate = s.clone();
                let outcome = apply(&mut candidate, &request, &self.root);
                match outcome {
                    Ok(()) => {
                        let projected =
                            if let ControlAction::Select { attempt_id } = &request.action {
                                let step = &candidate.snapshot.attempts[attempt_id].step;
                                if s.snapshot.stages[step].selected_attempt.as_deref()
                                    == Some(attempt_id)
                                {
                                    Ok(())
                                } else {
                                    workflow::project_selected_stage(&self.root, step)
                                }
                            } else {
                                Ok(())
                            };
                        if let Err(err) = projected.and_then(|()| {
                            workflow::project_control_state(
                                &self.root,
                                &s.snapshot,
                                &mut candidate.snapshot,
                            )
                        }) {
                            let message = format!("{err:#}");
                            if let Some(receipt) =
                                s.snapshot.queued.iter_mut().find(|r| r.request_id == id)
                            {
                                receipt.state = "rejected".into();
                                receipt.error = Some(message.clone());
                            }
                            emit(s, "control_rejected", None, None, Some(&message));
                            continue;
                        }
                        *s = candidate;
                    }
                    Err(err) => {
                        let message = format!("{err:#}");
                        if let Some(receipt) =
                            s.snapshot.queued.iter_mut().find(|r| r.request_id == id)
                        {
                            receipt.state = "rejected".into();
                            receipt.error = Some(message.clone());
                        }
                        emit(s, "control_rejected", None, None, Some(&message));
                        continue;
                    }
                }
                if let Some(receipt) = s.snapshot.queued.iter_mut().find(|r| r.request_id == id) {
                    receipt.state = "applied".into();
                }
                emit(s, "control_applied", None, None, Some(&id));
            }
            Ok(())
        })
    }
    pub fn reconcile(&self, legacy: &ProjectStatus) -> Result<()> {
        self.transact(|s| {
            for (step, stage) in &legacy.stages {
                if matches!(stage.status, State::Stale | State::NotStarted)
                    && let Some(id) = s
                        .snapshot
                        .stages
                        .get(step)
                        .and_then(|view| view.selected_attempt.clone())
                    && let Some(artifact) = s
                        .snapshot
                        .attempts
                        .get_mut(&id)
                        .and_then(|attempt| attempt.artifact.as_mut())
                    && artifact.path == self.root
                {
                    let archived = self
                        .root
                        .join("attempts")
                        .join(step.replace(':', "-"))
                        .join(stage.attempts.to_string());
                    if archived.is_dir() {
                        artifact.path = archived;
                    }
                }
                if stage.status == State::Ready
                    && stage.output_sha256.is_some()
                    && let Some(id) = s
                        .snapshot
                        .stages
                        .get(step)
                        .and_then(|view| view.selected_attempt.clone())
                    && let Some(attempt) = s.snapshot.attempts.get_mut(&id)
                    && attempt.artifact.is_none()
                    && let Some(hash) = &stage.output_sha256
                {
                    attempt.artifact = Some(Artifact {
                        path: freeze_asr(&self.root, step, &id)?,
                        sha256: hash.clone(),
                        fingerprint: stage.fingerprint.clone(),
                        base_attempt: None,
                    });
                    emit(s, "artifact_recovered", Some(step), Some(&id), None);
                }
                if stage.status == State::Ready
                    && stage.output_sha256.is_some()
                    && s.snapshot.stages.get(step).is_some_and(|view| {
                        view.selected_attempt.is_none() && view.active_attempt.is_none()
                    })
                {
                    let deps = dependencies(s, step)?;
                    let id = new_attempt(
                        s,
                        step,
                        deps,
                        Vec::new(),
                        RetryParameters::default(),
                        State::Ready,
                        false,
                    );
                    s.snapshot.attempts.get_mut(&id).unwrap().artifact = Some(Artifact {
                        path: freeze_asr(&self.root, step, &id)?,
                        sha256: stage.output_sha256.clone().unwrap(),
                        fingerprint: stage.fingerprint.clone(),
                        base_attempt: None,
                    });
                    s.snapshot.stages.get_mut(step).unwrap().selected_attempt = Some(id.clone());
                    emit(s, "cached_attempt_imported", Some(step), Some(&id), None);
                }
                let Some(view) = s.snapshot.stages.get_mut(step) else {
                    continue;
                };
                if s.snapshot.skipped.contains(step) {
                    continue;
                }
                if view.status != stage.status && view.active_attempt.is_none() {
                    view.status = stage.status;
                    if matches!(stage.status, State::Stale | State::NotStarted) {
                        view.selected_attempt = None;
                    }
                    emit(s, "stage_reconciled", Some(step), None, None);
                }
            }
            Ok(())
        })
    }
}

fn dependencies(s: &Store, step: &str) -> Result<Vec<String>> {
    let roles = s
        .snapshot
        .stages
        .keys()
        .filter(|name| s.snapshot.mode_steps.contains(*name) && !s.snapshot.skipped.contains(*name))
        .filter_map(|name| name.strip_prefix("asr:").map(str::to_owned))
        .collect::<Vec<_>>();
    let mut selected = Vec::new();
    for parent in workflow::parents(step, &roles) {
        if let Some(id) = &s
            .snapshot
            .stages
            .get(&parent)
            .context("missing dependency step")?
            .selected_attempt
        {
            selected.push(id.clone());
        }
    }
    Ok(selected)
}

fn new_attempt(
    s: &mut Store,
    step: &str,
    dependencies: Vec<String>,
    fragments: Vec<usize>,
    parameters: RetryParameters,
    status: State,
    auto_retry: bool,
) -> String {
    s.next_attempt += 1;
    let id = format!("a{}", s.next_attempt);
    s.snapshot.attempts.insert(
        id.clone(),
        Attempt {
            id: id.clone(),
            step: step.into(),
            kind: stage_kind(),
            dependencies,
            fragments,
            parameters,
            status,
            artifact: None,
            done: BTreeSet::new(),
            auto_retry,
            started_at_seconds: (status == State::Running).then(now_seconds),
            runtime_seconds: None,
        },
    );
    id
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_secs()
}

fn charge_runtime(snapshot: &mut Snapshot, id: &str, measured: Option<u64>) -> Result<()> {
    let attempt = snapshot.attempts.get_mut(id).context("unknown attempt")?;
    let seconds = measured.or(attempt.runtime_seconds).unwrap_or_else(|| {
        attempt
            .started_at_seconds
            .map_or(0, |started| now_seconds().saturating_sub(started))
    });
    let prior = attempt.runtime_seconds.unwrap_or(0);
    if attempt.auto_retry {
        snapshot.auto_runtime_seconds = snapshot
            .auto_runtime_seconds
            .saturating_add(seconds.saturating_sub(prior));
    }
    attempt.runtime_seconds = Some(seconds.max(prior));
    Ok(())
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos()
}

fn hash_directory(directory: &Path) -> Result<Option<String>> {
    let mut files = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in fs::read_dir(path)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    if files.is_empty() {
        return Ok(None);
    }
    files.sort();
    let mut hash = sha2::Sha256::new();
    for file in files {
        hash.update(file.strip_prefix(directory)?.to_string_lossy().as_bytes());
        hash.update(crate::pipeline::hash_file(&file)?.as_bytes());
    }
    Ok(Some(hex::encode(hash.finalize())))
}

fn copy_artifact(source: &Path, destination: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source)?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "artifact must not contain symlinks"
    );
    if metadata.is_dir() {
        fs::create_dir_all(destination)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            copy_artifact(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else {
        ensure!(metadata.is_file(), "artifact must be a regular file");
        if let Ok(metadata) = fs::symlink_metadata(destination) {
            ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "unsafe artifact destination"
            );
        }
        fs::copy(source, destination)?;
        fs::set_permissions(destination, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn freeze_asr(root: &Path, step: &str, id: &str) -> Result<PathBuf> {
    let Some(role) = step.strip_prefix("asr:") else {
        return Ok(root.into());
    };
    crate::config::validate_asr_roles(&[role.into()])?;
    let source = root.join(role);
    if !source.exists() {
        // Legacy metadata without files remains readable; it cannot supply a checked input.
        return Ok(root.into());
    }
    let destination = directory(root).join("artifacts").join(id);
    if !destination.exists() {
        copy_artifact(&source, &destination.join(role))?;
    }
    Ok(destination)
}

fn capture_partial_asr(root: &Path, snapshot: &mut Snapshot, id: &str) -> Result<()> {
    let attempt = snapshot.attempts.get_mut(id).context("unknown attempt")?;
    let Some(role) = attempt.step.strip_prefix("asr:") else {
        return Ok(());
    };
    if !matches!(attempt.kind.as_str(), "stage" | "restart")
        || attempt.done.is_empty()
        || attempt.artifact.is_some()
    {
        return Ok(());
    }
    let destination = freeze_asr(root, &attempt.step, id)?;
    ensure!(
        destination != root,
        "saved ASR units have no artifact directory"
    );
    let sha256 = hash_directory(&destination.join(role))?.context("empty partial ASR artifact")?;
    let legacy = workflow::status(root)?;
    attempt.artifact = Some(Artifact {
        path: destination,
        sha256,
        fingerprint: legacy
            .stages
            .get(&attempt.step)
            .context("missing ASR stage")?
            .fingerprint
            .clone(),
        base_attempt: None,
    });
    Ok(())
}

fn materialize_asr(root: &Path, snapshot: &Snapshot, id: &str, target: &Path) -> Result<()> {
    let attempt = snapshot.attempts.get(id).context("missing input attempt")?;
    let role = attempt
        .step
        .strip_prefix("asr:")
        .context("input is not ASR")?;
    let artifact = attempt
        .artifact
        .as_ref()
        .context("input attempt has no artifact")?;
    if attempt.kind == "retry" {
        let base = artifact
            .base_attempt
            .as_ref()
            .context("retry input has no full base artifact")?;
        materialize_asr(root, snapshot, base, target)?;
        for index in &attempt.fragments {
            copy_artifact(
                &artifact.path.join(format!("{index:05}.json")),
                &target.join(format!("{index:05}.json")),
            )?;
        }
    } else {
        ensure!(
            artifact.path.starts_with(root),
            "input artifact is outside project"
        );
        copy_artifact(&artifact.path.join(role), target)?;
    }
    Ok(())
}

fn check_inputs(
    root: &Path,
    snapshot: &Snapshot,
    attempt: &Attempt,
) -> Result<BTreeMap<String, PathBuf>> {
    let mut inputs = BTreeMap::new();
    let mut pending = attempt.dependencies.clone();
    let mut seen = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let input = snapshot
            .attempts
            .get(&id)
            .context("missing check dependency")?;
        if let Some(role) = input.step.strip_prefix("asr:") {
            ensure!(
                !inputs.contains_key(role),
                "ambiguous ASR versions in check dependencies"
            );
            let destination = directory(root)
                .join("check-inputs")
                .join(&attempt.id)
                .join(role);
            materialize_asr(root, snapshot, &id, &destination)?;
            inputs.insert(role.into(), destination);
        } else {
            pending.extend(input.dependencies.iter().cloned());
        }
    }
    ensure!(
        !inputs.is_empty(),
        "check requires at least one concrete ASR artifact"
    );
    Ok(inputs)
}

fn retry_output_path(root: &Path, id: &str) -> Result<PathBuf> {
    ensure!(
        !id.is_empty()
            && id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "invalid attempt ID"
    );
    let base = directory(root).join("retries");
    match fs::symlink_metadata(&base) {
        Ok(metadata) => ensure!(
            !metadata.file_type().is_symlink(),
            "retry directory must not be a symlink"
        ),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }
    let output = base.join(id);
    match fs::symlink_metadata(&output) {
        Ok(metadata) => ensure!(
            !metadata.file_type().is_symlink(),
            "retry output must not be a symlink"
        ),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }
    Ok(output)
}

fn affected(s: &Store, step: &str) -> Result<Vec<String>> {
    let roles = s
        .snapshot
        .stages
        .keys()
        .filter_map(|name| name.strip_prefix("asr:").map(str::to_owned))
        .collect::<Vec<_>>();
    ensure!(s.snapshot.stages.contains_key(step), "unknown step {step}");
    if step.starts_with("check:") {
        return Ok(vec![step.into()]);
    }
    Ok(workflow::descendants_for_roles(step, &roles)?
        .into_iter()
        .collect())
}

fn validate(s: &Store, request: &ControlRequest) -> Result<Preview> {
    ensure!(request.version == VERSION, "unsupported control version");
    ensure!(
        serde_json::to_vec(request)?.len() <= 65_536,
        "control request is too large"
    );
    ensure!(
        !request.request_id.is_empty()
            && request.request_id.len() <= 128
            && request
                .request_id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
        "invalid request ID"
    );
    ensure!(
        !request.reason.trim().is_empty(),
        "control reason is required"
    );
    ensure!(
        request
            .expected_revision
            .is_none_or(|rev| rev == s.snapshot.revision),
        "stale execution revision"
    );
    match &request.initiator {
        Initiator::Rule { rule } => ensure!(!rule.trim().is_empty(), "rule identity is required"),
        Initiator::Model {
            model,
            policy_version,
            confidence,
        } => {
            ensure!(
                !model.trim().is_empty() && !policy_version.trim().is_empty(),
                "model identity and policy version are required"
            );
            ensure!(
                confidence.is_none_or(|c| c.is_finite() && (0.0..=1.0).contains(&c)),
                "confidence is a probability from 0 to 1"
            );
        }
        Initiator::User => {}
    }
    if matches!(
        request.action,
        ControlAction::Skip { .. } | ControlAction::CancelAttempt { .. }
    ) {
        ensure!(
            request.confirmed && matches!(request.initiator, Initiator::User),
            "skip and cancellation require user confirmation"
        );
    }
    if matches!(request.action, ControlAction::Restore { .. }) {
        ensure!(
            matches!(request.initiator, Initiator::User),
            "only a user may restore a skipped step"
        );
    }
    let (steps, fragments, parameters) = match &request.action {
        ControlAction::Retry {
            step,
            fragments,
            parameters,
        } => {
            ensure!(
                step.starts_with("asr:"),
                "fragment retries are supported for ASR steps"
            );
            ensure!(
                !s.snapshot.skipped.contains(step),
                "step is skipped; restore it explicitly"
            );
            ensure!(
                parameters.threads.is_none_or(|n| n > 0)
                    && parameters
                        .max_in_flight
                        .is_none_or(|n| (1..=4).contains(&n)),
                "invalid retry parameters"
            );
            ensure!(!fragments.is_empty(), "retry requires assigned fragments");
            let unique = fragments.iter().copied().collect::<BTreeSet<_>>();
            ensure!(unique.len() == fragments.len(), "duplicate fragment IDs");
            let units = &s.snapshot.stages.get(step).context("unknown step")?.units;
            ensure!(
                fragments.iter().all(|index| units.contains(index)),
                "retry fragment outside assigned units"
            );
            let repeats = s
                .snapshot
                .attempts
                .values()
                .filter(|attempt| attempt.auto_retry)
                .count();
            if !matches!(request.initiator, Initiator::User) {
                ensure!(
                    repeats < s.snapshot.budget.max_auto_repeats as usize,
                    "automatic retry budget exhausted"
                );
                ensure!(
                    s.snapshot.auto_runtime_seconds < s.snapshot.budget.max_auto_runtime_seconds,
                    "automatic retry runtime budget exhausted"
                );
            }
            (
                affected(s, step)?,
                fragments.clone(),
                Some(parameters.clone()),
            )
        }
        ControlAction::Skip { step }
        | ControlAction::Restore { step }
        | ControlAction::Invalidate { step } => (affected(s, step)?, Vec::new(), None),
        ControlAction::CancelAttempt { attempt_id } | ControlAction::Select { attempt_id } => {
            let attempt = s
                .snapshot
                .attempts
                .get(attempt_id)
                .context("unknown attempt")?;
            if matches!(request.action, ControlAction::Select { .. }) {
                ensure!(
                    !s.snapshot.skipped.contains(&attempt.step),
                    "step is skipped; restore it explicitly"
                );
                ensure!(
                    attempt.status == State::Ready && attempt.artifact.is_some(),
                    "only a completed attempt with artifact can be selected"
                );
            } else {
                ensure!(
                    attempt.status == State::Running || attempt.status == State::NotStarted,
                    "attempt cannot be cancelled"
                );
            }
            (
                affected(s, &attempt.step)?,
                attempt.fragments.clone(),
                Some(attempt.parameters.clone()),
            )
        }
        ControlAction::AddCheck {
            step,
            dependencies,
            fragments,
        } => {
            ensure!(s.snapshot.stages.contains_key(step), "unknown step");
            ensure!(
                step.starts_with("asr:") || step == "review-flags",
                "checks must target ASR or review-flags"
            );
            ensure!(
                !s.snapshot
                    .skipped
                    .contains(&format!("check:{}", step.replace(':', "-"))),
                "quality check stage is skipped; restore it explicitly"
            );
            ensure!(
                !s.snapshot.skipped.contains(step),
                "step is skipped; restore it explicitly"
            );
            ensure!(
                dependencies
                    .iter()
                    .all(|id| s.snapshot.attempts.contains_key(id)),
                "unknown dependency attempt"
            );
            ensure!(
                dependencies.iter().collect::<BTreeSet<_>>().len() == dependencies.len(),
                "duplicate dependency attempt"
            );
            let units = &s.snapshot.stages[step].units;
            ensure!(
                fragments.iter().collect::<BTreeSet<_>>().len() == fragments.len(),
                "duplicate check fragment"
            );
            ensure!(
                fragments.iter().all(|index| units.contains(index)),
                "unknown check fragment"
            );
            ensure!(
                !dependencies.is_empty(),
                "check requires concrete input attempts"
            );
            if !matches!(request.initiator, Initiator::User) {
                ensure!(
                    s.snapshot
                        .attempts
                        .values()
                        .filter(|attempt| attempt.auto_retry)
                        .count()
                        < s.snapshot.budget.max_auto_repeats as usize,
                    "automatic retry budget exhausted"
                );
                ensure!(
                    s.snapshot.auto_runtime_seconds < s.snapshot.budget.max_auto_runtime_seconds,
                    "automatic retry runtime budget exhausted"
                );
            }
            (vec![step.clone()], fragments.clone(), None)
        }
        ControlAction::Dependencies {
            attempt_id,
            dependencies,
        } => {
            let attempt = s
                .snapshot
                .attempts
                .get(attempt_id)
                .context("unknown attempt")?;
            ensure!(
                attempt.status == State::NotStarted,
                "only a pending attempt's dependencies may change"
            );
            let number = attempt_id
                .strip_prefix('a')
                .context("invalid attempt ID")?
                .parse::<u64>()?;
            ensure!(
                dependencies.iter().collect::<BTreeSet<_>>().len() == dependencies.len(),
                "duplicate dependency attempt"
            );
            for parent in dependencies {
                ensure!(
                    s.snapshot.attempts.contains_key(parent),
                    "unknown dependency attempt"
                );
                let parent_number = parent
                    .strip_prefix('a')
                    .context("invalid dependency ID")?
                    .parse::<u64>()?;
                ensure!(
                    parent_number < number,
                    "dependency change would create a cycle"
                );
            }
            (vec![attempt.step.clone()], attempt.fragments.clone(), None)
        }
    };
    Ok(Preview {
        revision: s.snapshot.revision,
        affected_steps: steps,
        fragments,
        parameters,
    })
}

fn invalidate(s: &mut Store, step: &str, reason: &str, include_self: bool) -> Result<()> {
    for name in affected(s, step)? {
        if !include_self && name == step {
            continue;
        }
        let view = s.snapshot.stages.get_mut(&name).unwrap();
        if view.status == State::Ready || view.status == State::Running {
            view.status = State::Stale;
            view.selected_attempt = None;
            if let Some(id) = &view.active_attempt {
                s.snapshot.attempts.get_mut(id).unwrap().status = State::Cancelled;
            }
            emit(s, "stage_invalidated", Some(&name), None, Some(reason));
        }
    }
    Ok(())
}

fn select(s: &mut Store, id: &str, root: &Path) -> Result<()> {
    let attempt = s.snapshot.attempts.get(id).context("unknown attempt")?;
    ensure!(
        attempt.status == State::Ready && attempt.artifact.is_some(),
        "attempt has no completed artifact"
    );
    ensure!(
        s.snapshot.stages[&attempt.step].active_attempt.is_none(),
        "cannot select while stage is running"
    );
    if s.snapshot.stages[&attempt.step].selected_attempt.as_deref() == Some(id) {
        return Ok(());
    }
    if attempt.step.starts_with("asr:") {
        let state: ProjectStatus = serde_json::from_slice(&fs::read(workflow::state_path(root))?)?;
        ensure!(
            attempt.artifact.as_ref().unwrap().fingerprint
                == state.stages[&attempt.step].fingerprint,
            "attempt belongs to different model weights or parameters"
        );
        publish_asr(root, &s.snapshot, id)?;
    }
    if attempt.kind == "retry" {
        let role = attempt
            .step
            .strip_prefix("asr:")
            .context("retry is not an ASR step")?;
        let output = retry_output_path(root, id)?;
        for index in &attempt.fragments {
            let name = format!("{index:05}.json");
            let delta = output.join(&name);
            let selected = root.join(role).join(name);
            ensure!(
                selected.is_file()
                    && crate::pipeline::hash_file(&selected)?
                        == crate::pipeline::hash_file(&delta)?,
                "retry fragment {index} has not been published"
            );
        }
    } else if let Some(artifact) = &attempt.artifact
        && artifact.path != root
    {
        ensure!(
            artifact.path.starts_with(root.join("attempts"))
                || artifact.path.starts_with(directory(root).join("artifacts")),
            "historical artifact path is outside the project archive"
        );
        let archive_base = if artifact.path.starts_with(root.join("attempts")) {
            root.join("attempts")
        } else {
            directory(root).join("artifacts")
        };
        ensure!(
            !fs::symlink_metadata(archive_base)?.file_type().is_symlink(),
            "historical archive must not be a symlink"
        );
        ensure!(
            !fs::symlink_metadata(&artifact.path)?
                .file_type()
                .is_symlink(),
            "historical artifact must not be a symlink"
        );
        let mut files = vec![artifact.path.clone()];
        let mut checked = false;
        while let Some(path) = files.pop() {
            for entry in fs::read_dir(path)? {
                let path = entry?.path();
                if path.file_name().is_some_and(|name| name == ".stage.json") {
                    continue;
                }
                ensure!(
                    !fs::symlink_metadata(&path)?.file_type().is_symlink(),
                    "historical artifact must not contain symlinks"
                );
                if path.is_dir() {
                    files.push(path);
                } else {
                    let selected = root.join(path.strip_prefix(&artifact.path)?);
                    ensure!(
                        selected.is_file()
                            && crate::pipeline::hash_file(&selected)?
                                == crate::pipeline::hash_file(&path)?,
                        "historical attempt must be republished before selection"
                    );
                    checked = true;
                }
            }
        }
        ensure!(checked, "historical artifact has no files");
    }
    let step = attempt.step.clone();
    let view = s.snapshot.stages.get_mut(&step).unwrap();
    ensure!(
        view.active_attempt.is_none(),
        "cannot select while stage is running"
    );
    if view.selected_attempt.as_deref() == Some(id) {
        return Ok(());
    }
    view.selected_attempt = Some(id.into());
    view.status = State::Ready;
    emit(s, "attempt_selected", Some(&step), Some(id), None);
    invalidate(s, &step, "selected artifact changed", false)
}

fn publish_asr(root: &Path, snapshot: &Snapshot, id: &str) -> Result<()> {
    let attempt = &snapshot.attempts[id];
    let role = attempt.step.strip_prefix("asr:").context("not ASR")?;
    let artifact = attempt.artifact.as_ref().context("missing artifact")?;
    ensure!(
        artifact.path.starts_with(root),
        "artifact is outside project"
    );
    if attempt.kind == "retry" {
        let mut hash = sha2::Sha256::new();
        for index in &attempt.fragments {
            hash.update(index.to_le_bytes());
            hash.update(
                crate::pipeline::hash_file(&artifact.path.join(format!("{index:05}.json")))?
                    .as_bytes(),
            );
        }
        ensure!(
            hex::encode(hash.finalize()) == artifact.sha256,
            "retry artifact content changed"
        );
    } else if artifact.path != root {
        ensure!(
            hash_directory(&artifact.path)?.as_deref() == Some(artifact.sha256.as_str()),
            "ASR artifact content changed"
        );
    }
    let candidate = directory(root)
        .join("selected")
        .join(format!("{id}-{}", now_nanos()))
        .join(role);
    materialize_asr(root, snapshot, id, &candidate)?;
    let chunks: Vec<crate::audio::Interval> =
        serde_json::from_slice(&fs::read(root.join("chunks.json"))?)?;
    for (index, chunk) in chunks.iter().enumerate() {
        let hypothesis: crate::pipeline::Hypothesis =
            serde_json::from_slice(&fs::read(candidate.join(format!("{index:05}.json")))?)?;
        crate::pipeline::validate_hypothesis(&hypothesis, chunk.end - chunk.start)?;
    }
    let selected = root.join(role);
    let displaced = directory(root)
        .join("displaced")
        .join(format!("{id}-{}", now_nanos()))
        .join(role);
    let had_previous = selected.exists();
    if had_previous {
        fs::create_dir_all(displaced.parent().context("missing displaced parent")?)?;
        fs::rename(&selected, &displaced)?;
    }
    if let Err(error) = fs::rename(&candidate, &selected) {
        if had_previous {
            fs::rename(&displaced, &selected)
                .context("could not restore previous ASR publication")?;
        }
        return Err(error).context("could not publish selected ASR artifact");
    }
    Ok(())
}

fn apply(s: &mut Store, request: &ControlRequest, root: &Path) -> Result<()> {
    let mut latest = request.clone();
    latest.expected_revision = None;
    let preview = validate(s, &latest)?;
    let mut params = preview.parameters.clone();
    match &request.action {
        ControlAction::Retry {
            step,
            fragments,
            parameters,
        } => {
            let mut deps = dependencies(s, step)?;
            if let Some(base) = s.snapshot.stages[step].selected_attempt.as_ref().or(s
                .snapshot
                .stages[step]
                .active_attempt
                .as_ref())
            {
                deps.push(base.clone());
            }
            let id = new_attempt(
                s,
                step,
                deps,
                fragments.clone(),
                parameters.clone(),
                State::NotStarted,
                !matches!(request.initiator, Initiator::User),
            );
            s.snapshot.attempts.get_mut(&id).unwrap().kind = "retry".into();
            emit(
                s,
                "retry_planned",
                Some(step),
                Some(&id),
                Some(&request.reason),
            );
        }
        ControlAction::Skip { step } => {
            s.snapshot.skipped.insert(step.clone());
            let view = s.snapshot.stages.get_mut(step).unwrap();
            view.status = State::Skipped;
            if let Some(id) = &view.active_attempt {
                s.snapshot.attempts.get_mut(id).unwrap().status = State::Cancelled;
            }
            for attempt in s
                .snapshot
                .attempts
                .values_mut()
                .filter(|attempt| attempt.step == *step)
            {
                if attempt.status == State::NotStarted {
                    attempt.status = State::Skipped;
                } else if attempt.status == State::Running {
                    attempt.status = State::Cancelled;
                }
            }
            emit(s, "step_skipped", Some(step), None, Some(&request.reason));
            invalidate(s, step, &request.reason, false)?;
        }
        ControlAction::Restore { step } => {
            ensure!(s.snapshot.skipped.remove(step), "step was not skipped");
            let legacy: ProjectStatus =
                serde_json::from_slice(&fs::read(workflow::state_path(root))?)?;
            let has_cached = legacy
                .stages
                .get(step)
                .is_some_and(|stage| stage.output_sha256.is_some());
            let view = s.snapshot.stages.get_mut(step).unwrap();
            view.status =
                if view.selected_attempt.is_some() && (has_cached || step.starts_with("check:")) {
                    State::Ready
                } else {
                    State::NotStarted
                };
            if step.starts_with("check:") && view.selected_attempt.is_none() {
                let previous = s
                    .snapshot
                    .attempts
                    .values()
                    .filter(|attempt| attempt.step == *step)
                    .max_by_key(|attempt| {
                        attempt
                            .id
                            .strip_prefix('a')
                            .and_then(|id| id.parse::<u64>().ok())
                    })
                    .cloned()
                    .context("check has no previous attempt")?;
                let id = new_attempt(
                    s,
                    step,
                    previous.dependencies,
                    previous.fragments,
                    previous.parameters,
                    State::NotStarted,
                    false,
                );
                s.snapshot.attempts.get_mut(&id).unwrap().kind = "check".into();
                emit(
                    s,
                    "check_planned",
                    Some(step),
                    Some(&id),
                    Some("user restored quality check"),
                );
            }
            emit(s, "step_restored", Some(step), None, Some(&request.reason));
            invalidate(s, step, &request.reason, false)?;
        }
        ControlAction::CancelAttempt { attempt_id } => {
            let attempt = s.snapshot.attempts.get_mut(attempt_id).unwrap();
            attempt.status = State::Cancelled;
            let step = attempt.step.clone();
            let view = s.snapshot.stages.get_mut(&step).unwrap();
            if view.active_attempt.as_deref() == Some(attempt_id) {
                view.status = if view.selected_attempt.is_some() {
                    State::Ready
                } else {
                    State::Cancelled
                };
            }
            emit(
                s,
                "attempt_cancelled",
                Some(&step),
                Some(attempt_id),
                Some(&request.reason),
            );
        }
        ControlAction::AddCheck {
            step,
            dependencies,
            fragments,
        } => {
            let fragments = if fragments.is_empty() {
                s.snapshot.stages[step].units.clone()
            } else {
                fragments.clone()
            };
            ensure!(!fragments.is_empty(), "check requires known fragment units");
            let id = new_attempt(
                s,
                step,
                dependencies.clone(),
                fragments.clone(),
                RetryParameters::default(),
                State::NotStarted,
                !matches!(request.initiator, Initiator::User),
            );
            let check_step = format!("check:{}", step.replace(':', "-"));
            let attempt = s.snapshot.attempts.get_mut(&id).unwrap();
            attempt.kind = "check".into();
            attempt.step = check_step.clone();
            let mut stage = StageView::new(State::NotStarted);
            stage.units = fragments;
            stage.unit = "windows".into();
            s.snapshot.stages.entry(check_step.clone()).or_insert(stage);
            s.snapshot.mode_steps.insert(check_step.clone());
            emit(
                s,
                "check_planned",
                Some(&check_step),
                Some(&id),
                Some(&request.reason),
            );
        }
        ControlAction::Select { attempt_id } => select(s, attempt_id, root)?,
        ControlAction::Invalidate { step } => invalidate(s, step, &request.reason, true)?,
        ControlAction::Dependencies {
            attempt_id,
            dependencies,
        } => {
            s.snapshot
                .attempts
                .get_mut(attempt_id)
                .unwrap()
                .dependencies = dependencies.clone();
            emit(
                s,
                "dependencies_changed",
                None,
                Some(attempt_id),
                Some(&request.reason),
            );
        }
    }
    s.snapshot.decisions.push(Decision {
        request_id: request.request_id.clone(),
        initiator: request.initiator.clone(),
        reason: request.reason.clone(),
        evidence: request.evidence.clone(),
        action: request.action.clone(),
        affected_steps: preview.affected_steps,
        fragments: preview.fragments,
        parameters: params.take(),
    });
    emit(s, "decision_recorded", None, None, Some(&request.reason));
    s.events.last_mut().unwrap().payload["decision"] =
        serde_json::to_value(s.snapshot.decisions.last())?;
    Ok(())
}

pub fn snapshot(project: &Path) -> Result<Snapshot> {
    Ok(load(project)?.snapshot)
}
pub fn events(project: &Path, after: u64, limit: usize) -> Result<Vec<Event>> {
    ensure!(
        (1..=1000).contains(&limit),
        "event page limit must be 1..=1000"
    );
    Ok(load(project)?
        .events
        .into_iter()
        .filter(|event| event.sequence > after)
        .take(limit)
        .collect())
}
pub fn preview(project: &Path, request: &ControlRequest) -> Result<Preview> {
    validate(&load(project)?, request)
}
pub fn enqueue(project: &Path, request: &ControlRequest) -> Result<Receipt> {
    let handle = Handle {
        root: project.to_path_buf(),
        guard: Arc::new(Mutex::new(())),
    };
    handle.transact(|s| {
        if let Some(existing) = s.requests.get(&request.request_id) {
            ensure!(
                existing == request,
                "duplicate request ID with different control"
            );
            return Ok(s
                .snapshot
                .queued
                .iter()
                .find(|r| r.request_id == request.request_id)
                .context("missing receipt")?
                .clone());
        }
        validate(s, request)?;
        s.requests
            .insert(request.request_id.clone(), request.clone());
        s.queue.push(request.request_id.clone());
        emit(s, "control_queued", None, None, Some(&request.request_id));
        let receipt = Receipt {
            request_id: request.request_id.clone(),
            revision: s.snapshot.revision,
            state: "queued".into(),
            error: None,
        };
        s.snapshot.queued.push(receipt.clone());
        Ok(receipt)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::Stage;

    fn setup() -> (tempfile::TempDir, Handle) {
        let root = tempfile::tempdir().unwrap();
        let stages = [
            "audio",
            "chunks",
            "asr:primeline",
            "review-flags",
            "review",
            "export",
        ]
        .into_iter()
        .map(|step| {
            (
                step.into(),
                Stage {
                    status: State::Ready,
                    attempts: 1,
                    fingerprint: step.into(),
                    output_sha256: Some(step.into()),
                    warning: None,
                },
            )
        })
        .collect();
        let mut status = ProjectStatus {
            schema_version: 2,
            input: root.path().join("source.mp3"),
            input_sha256: "input".into(),
            export_stem: Some("source".into()),
            stages,
            asr_roles: vec!["primeline".into()],
        };
        fs::create_dir(root.path().join("primeline")).unwrap();
        for index in 0..5 {
            fs::write(
                root.path()
                    .join("primeline")
                    .join(format!("{index:05}.json")),
                br#"{"text":"old","language":"de","segments":[]}"#,
            )
            .unwrap();
        }
        fs::write(
            workflow::state_path(root.path()),
            serde_json::to_vec(&status).unwrap(),
        )
        .unwrap();
        let (hash, _) = workflow::completed_stage_artifact(root.path(), "asr:primeline").unwrap();
        status
            .stages
            .get_mut("asr:primeline")
            .unwrap()
            .output_sha256 = Some(hash);
        fs::write(
            workflow::state_path(root.path()),
            serde_json::to_vec(&status).unwrap(),
        )
        .unwrap();
        let handle = Handle::open(root.path(), &status).unwrap();
        let chunks = (0..5)
            .map(|index| crate::audio::Interval {
                start: index as f64,
                end: index as f64 + 1.0,
            })
            .collect::<Vec<_>>();
        fs::write(
            root.path().join("chunks.json"),
            serde_json::to_vec(&chunks).unwrap(),
        )
        .unwrap();
        (root, handle)
    }

    fn control(id: &str, action: ControlAction) -> ControlRequest {
        ControlRequest {
            version: 1,
            request_id: id.into(),
            expected_revision: None,
            initiator: Initiator::User,
            reason: "reviewed".into(),
            evidence: vec![],
            confirmed: true,
            action,
        }
    }

    #[test]
    fn resource_restarts_preserve_scope_history_and_runtime_budget() {
        let (root, handle) = setup();
        let step = "asr:primeline";
        handle
            .set_units(step, &[0, 1, 2, 3, 4], "fragments")
            .unwrap();
        let failed = handle.begin_stage(step).unwrap();
        handle.unit_done(step, 0).unwrap();
        handle
            .fail_stage(step, State::Failed, "out of memory")
            .unwrap();
        let restart = handle
            .plan_restart(
                step,
                &[2, 4],
                RetryParameters {
                    threads: Some(1),
                    max_in_flight: Some(1),
                },
                "reduce resource usage",
                vec![serde_json::json!({"oom":true})],
            )
            .unwrap();
        assert_eq!(handle.begin_stage(step).unwrap(), restart);
        assert!(handle.unit_done(step, 0).is_err());
        let snapshot = handle.snapshot().unwrap();
        assert_eq!(snapshot.attempts[&failed].status, State::Failed);
        assert_eq!(snapshot.attempts[&failed].done, BTreeSet::from([0]));
        assert!(
            snapshot.attempts[&failed]
                .artifact
                .as_ref()
                .unwrap()
                .path
                .join("primeline/00000.json")
                .is_file()
        );
        assert!(snapshot.attempts[&restart].dependencies.contains(&failed));
        assert_eq!(snapshot.attempts[&restart].fragments, [2, 4]);
        assert_eq!(
            snapshot
                .attempts
                .values()
                .filter(|attempt| attempt.auto_retry)
                .count(),
            1
        );
        let decision = snapshot.decisions.last().unwrap();
        assert!(
            matches!(&decision.initiator, Initiator::Rule { rule } if rule == "resource-recovery")
        );
        assert_eq!(decision.evidence.last().unwrap()["failed_attempt"], failed);
        handle.record_runtime(&restart, 10).unwrap();
        let revision = handle.snapshot().unwrap().revision;
        handle.record_runtime(&restart, 10).unwrap();
        assert_eq!(handle.snapshot().unwrap().revision, revision);
        handle.record_runtime(&restart, 20).unwrap();
        handle.record_runtime(&restart, 5).unwrap();
        handle.unit_done(step, 2).unwrap();
        handle.unit_done(step, 4).unwrap();
        handle.finish_stage(step).unwrap();
        assert_eq!(handle.snapshot().unwrap().auto_runtime_seconds, 20);
        let reopened = Handle::open(root.path(), &workflow::status(root.path()).unwrap()).unwrap();
        assert_eq!(reopened.snapshot().unwrap().auto_runtime_seconds, 20);
        reopened.begin_stage(step).unwrap();
        reopened
            .fail_stage(step, State::Failed, "out of memory")
            .unwrap();
        let next = reopened
            .plan_restart(
                step,
                &[4],
                RetryParameters::default(),
                "recover remaining fragment",
                vec![],
            )
            .unwrap();
        reopened.begin_stage(step).unwrap();
        reopened.record_runtime(&next, 580).unwrap();
        assert!(reopened.cancelled(step).unwrap());
        let snapshot = reopened.snapshot().unwrap();
        assert_eq!(snapshot.auto_runtime_seconds, 600);
        assert_eq!(snapshot.attempts[&next].status, State::Cancelled);
        assert!(
            reopened
                .plan_restart(
                    step,
                    &[4],
                    RetryParameters::default(),
                    "budget cannot be overridden",
                    vec![]
                )
                .is_err()
        );
    }

    #[test]
    fn imported_graph_replay_and_persistent_skip() {
        let (root, handle) = setup();
        let initial = handle.snapshot().unwrap();
        let asr = initial.stages["asr:primeline"]
            .selected_attempt
            .as_ref()
            .unwrap();
        assert_eq!(
            initial.attempts[asr].dependencies,
            vec![initial.stages["chunks"].selected_attempt.clone().unwrap()]
        );
        let request = control(
            "skip-1",
            ControlAction::Skip {
                step: "asr:primeline".into(),
            },
        );
        let receipt = enqueue(root.path(), &request).unwrap();
        assert_eq!(receipt.state, "queued");
        assert_eq!(
            enqueue(root.path(), &request).unwrap().revision,
            receipt.revision
        );
        assert!(!handle.is_skipped("asr:primeline").unwrap());
        handle.apply_controls().unwrap();
        assert!(handle.is_skipped("asr:primeline").unwrap());
        assert_eq!(
            workflow::status(root.path()).unwrap().stages["asr:primeline"].status,
            State::Skipped
        );
        assert_eq!(
            handle.snapshot().unwrap().stages["review"].status,
            State::Stale
        );
        let reopened = Handle::open(root.path(), &workflow::status(root.path()).unwrap()).unwrap();
        assert!(reopened.is_skipped("asr:primeline").unwrap());
        let restore = control(
            "restore-1",
            ControlAction::Restore {
                step: "asr:primeline".into(),
            },
        );
        enqueue(root.path(), &restore).unwrap();
        reopened.apply_controls().unwrap();
        assert!(!reopened.is_skipped("asr:primeline").unwrap());
        let snapshot = reopened.snapshot().unwrap();
        let page = events(root.path(), 0, 2).unwrap();
        assert_eq!(page.len(), 2);
        let continuation = events(root.path(), page[1].sequence, 100).unwrap();
        assert_eq!(continuation.last().unwrap().sequence, snapshot.revision);
        assert!(
            page.into_iter()
                .chain(continuation)
                .map(|e| e.sequence)
                .eq(1..=snapshot.revision)
        );
    }

    #[test]
    fn scoped_retries_lineage_budget_and_selection() {
        let (root, handle) = setup();
        handle
            .set_units("asr:primeline", &[0, 2, 4], "windows")
            .unwrap();
        assert!(handle.unit_done("asr:primeline", 7).is_err());
        handle.unit_done("asr:primeline", 2).unwrap();
        let one_done = handle.snapshot().unwrap().revision;
        handle.unit_done("asr:primeline", 2).unwrap();
        assert_eq!(handle.snapshot().unwrap().revision, one_done);
        assert_eq!(
            handle.snapshot().unwrap().stages["asr:primeline"]
                .done
                .len(),
            1
        );
        assert!(
            handle
                .numeric_progress("chunks", 101, Some(100), "clips")
                .is_err()
        );
        handle
            .numeric_progress("chunks", 1, Some(100), "clips")
            .unwrap();
        assert_ne!(
            handle.snapshot().unwrap().stages["chunks"].numeric_done,
            100
        );
        let old = handle.snapshot().unwrap().stages["asr:primeline"]
            .selected_attempt
            .clone()
            .unwrap();
        let make_retry = |id: &str| {
            control(
                id,
                ControlAction::Retry {
                    step: "asr:primeline".into(),
                    fragments: vec![2],
                    parameters: RetryParameters {
                        threads: Some(2),
                        max_in_flight: Some(1),
                    },
                },
            )
        };
        assert!(
            enqueue(
                root.path(),
                &control(
                    "bad",
                    ControlAction::Retry {
                        step: "asr:primeline".into(),
                        fragments: vec![3],
                        parameters: RetryParameters::default()
                    }
                )
            )
            .is_err()
        );
        enqueue(root.path(), &make_retry("retry-1")).unwrap();
        handle.apply_controls().unwrap();
        let work = handle.pending_retries().unwrap().pop().unwrap();
        assert_eq!(work.fragments, [2]);
        assert!(
            handle.snapshot().unwrap().attempts[&work.attempt_id]
                .dependencies
                .contains(&old)
        );
        handle.begin_retry(&work.attempt_id).unwrap();
        assert!(handle.retry_unit_done(&work.attempt_id, 3).is_err());
        let folder = handle.retry_output(&work.attempt_id);
        fs::create_dir_all(&folder).unwrap();
        fs::write(
            folder.join("00002.json"),
            br#"{"text":"new","language":"de","segments":[]}"#,
        )
        .unwrap();
        handle.retry_unit_done(&work.attempt_id, 2).unwrap();
        handle.retry_unit_done(&work.attempt_id, 2).unwrap();
        handle.finish_retry(&work.attempt_id).unwrap();
        assert_eq!(
            handle.snapshot().unwrap().stages["asr:primeline"]
                .selected_attempt
                .as_deref(),
            Some(old.as_str())
        );
        fs::write(root.path().join("review-flags.json"), b"[]").unwrap();
        fs::create_dir_all(root.path().join("reviews")).unwrap();
        fs::write(root.path().join("reviews/00000.json"), b"{}").unwrap();
        fs::write(root.path().join("source.json"), b"{}").unwrap();
        fs::write(root.path().join("source.txt"), b"old transcript").unwrap();
        let old_review = handle.snapshot().unwrap().stages["review"]
            .selected_attempt
            .clone()
            .unwrap();
        handle.select_retry(&work.attempt_id).unwrap();
        assert!(
            fs::read_to_string(root.path().join("primeline/00002.json"))
                .unwrap()
                .contains("new")
        );
        assert!(
            fs::read_to_string(root.path().join("primeline/00000.json"))
                .unwrap()
                .contains("old")
        );
        assert!(
            root.path()
                .join("attempts/review/1/reviews/00000.json")
                .exists()
        );
        assert!(root.path().join("attempts/export/1/source.txt").exists());
        assert_eq!(
            handle.snapshot().unwrap().attempts[&old_review]
                .artifact
                .as_ref()
                .unwrap()
                .path,
            root.path().join("attempts/review/1")
        );
        enqueue(
            root.path(),
            &control(
                "published-select",
                ControlAction::Select {
                    attempt_id: work.attempt_id.clone(),
                },
            ),
        )
        .unwrap();
        handle.apply_controls().unwrap();
        let projected = workflow::status(root.path()).unwrap();
        assert_eq!(projected.stages["asr:primeline"].attempts, 2);
        assert_eq!(projected.stages["review"].status, State::Stale);
        let snapshot = handle.snapshot().unwrap();
        assert_eq!(
            snapshot.attempts[&work.attempt_id]
                .artifact
                .as_ref()
                .unwrap()
                .base_attempt
                .as_deref(),
            Some(old.as_str())
        );
        assert_eq!(snapshot.stages["review"].status, State::Stale);

        for index in 0..3 {
            let mut automatic = make_retry(&format!("auto-{index}"));
            automatic.initiator = Initiator::Rule {
                rule: "quality-v1".into(),
            };
            enqueue(root.path(), &automatic).unwrap();
            handle.apply_controls().unwrap();
        }
        let mut over_budget = make_retry("auto-3");
        over_budget.initiator = Initiator::Rule {
            rule: "quality-v1".into(),
        };
        assert!(enqueue(root.path(), &over_budget).is_err());
        handle
            .transact(|s| {
                s.snapshot.auto_runtime_seconds = 600;
                emit(s, "budget_updated", None, None, None);
                Ok(())
            })
            .unwrap();
        let mut automatic = make_retry("auto-runtime");
        automatic.initiator = Initiator::Rule {
            rule: "quality-v1".into(),
        };
        assert!(enqueue(root.path(), &automatic).is_err());
        automatic.initiator = Initiator::Model {
            model: "m".into(),
            policy_version: "v1".into(),
            confidence: Some(f64::NAN),
        };
        assert!(enqueue(root.path(), &automatic).is_err());
    }

    #[test]
    fn stale_revision_and_invalid_request_do_not_enter_journal() {
        let (root, handle) = setup();
        let before = handle.snapshot().unwrap().revision;
        let mut request = control(
            "bad-id",
            ControlAction::Skip {
                step: "chunks".into(),
            },
        );
        request.expected_revision = Some(before + 1);
        assert!(enqueue(root.path(), &request).is_err());
        request.expected_revision = Some(before);
        request.confirmed = false;
        assert!(enqueue(root.path(), &request).is_err());
        assert_eq!(handle.snapshot().unwrap().revision, before);
    }

    #[test]
    fn archived_legacy_attempts_remain_addressable() {
        let root = tempfile::tempdir().unwrap();
        let archive = root.path().join("attempts").join("asr-primeline").join("1");
        fs::create_dir_all(&archive).unwrap();
        fs::write(archive.join("00000.json"), b"{\"text\":\"old\"}").unwrap();
        let stages = BTreeMap::from([(
            "asr:primeline".into(),
            Stage {
                status: State::Ready,
                attempts: 2,
                fingerprint: "v1".into(),
                output_sha256: Some("current".into()),
                warning: None,
            },
        )]);
        let status = ProjectStatus {
            schema_version: 2,
            input: root.path().join("source.mp3"),
            input_sha256: "input".into(),
            export_stem: Some("source".into()),
            stages,
            asr_roles: vec!["primeline".into()],
        };
        let handle = Handle::open(root.path(), &status).unwrap();
        let snapshot = handle.snapshot().unwrap();
        let attempts = snapshot
            .attempts
            .values()
            .filter(|a| a.step == "asr:primeline")
            .collect::<Vec<_>>();
        assert_eq!(attempts.len(), 2);
        assert!(attempts.iter().any(|attempt| {
            attempt
                .artifact
                .as_ref()
                .is_some_and(|artifact| artifact.path == archive)
        }));
        assert!(snapshot.stages["asr:primeline"].selected_attempt.is_some());
    }

    #[test]
    fn finishing_a_handle_projects_the_legacy_stage_immediately() {
        let (root, handle) = setup();
        fs::write(root.path().join("audio.wav"), b"new normalized audio").unwrap();
        let previous = handle.snapshot().unwrap().stages["audio"]
            .selected_attempt
            .clone();
        handle.begin_stage("audio").unwrap();
        handle.finish_stage("audio").unwrap();
        let stage = &workflow::status(root.path()).unwrap().stages["audio"];
        assert_eq!(stage.status, State::Ready);
        assert_eq!(stage.attempts, 2);
        let selected = handle.snapshot().unwrap().stages["audio"]
            .selected_attempt
            .clone();
        assert_ne!(selected, previous);
        assert_eq!(
            handle.snapshot().unwrap().attempts[&selected.unwrap()]
                .artifact
                .as_ref()
                .unwrap()
                .sha256,
            stage.output_sha256.clone().unwrap()
        );
    }

    #[test]
    fn checks_link_only_existing_attempts_and_cannot_override_user_skip() {
        let (root, handle) = setup();
        handle.set_units("asr:primeline", &[1], "windows").unwrap();
        let parent = handle.snapshot().unwrap().stages["asr:primeline"]
            .selected_attempt
            .clone()
            .unwrap();
        let check = control(
            "check-1",
            ControlAction::AddCheck {
                step: "asr:primeline".into(),
                dependencies: vec![parent.clone()],
                fragments: vec![1],
            },
        );
        enqueue(root.path(), &check).unwrap();
        handle.apply_controls().unwrap();
        let snapshot = handle.snapshot().unwrap();
        let child = snapshot
            .attempts
            .values()
            .find(|a| a.kind == "check")
            .unwrap();
        assert_eq!(child.dependencies, [parent]);
        assert!(handle.pending_retries().unwrap().is_empty());
        let invalid = control(
            "check-cycle",
            ControlAction::AddCheck {
                step: "asr:primeline".into(),
                dependencies: vec!["a999".into()],
                fragments: vec![1],
            },
        );
        assert!(enqueue(root.path(), &invalid).is_err());
        enqueue(
            root.path(),
            &control(
                "skip-check",
                ControlAction::Skip {
                    step: "asr:primeline".into(),
                },
            ),
        )
        .unwrap();
        handle.apply_controls().unwrap();
        let mut automatic = control(
            "restore-rule",
            ControlAction::Restore {
                step: "asr:primeline".into(),
            },
        );
        automatic.initiator = Initiator::Rule {
            rule: "quality-v1".into(),
        };
        assert!(enqueue(root.path(), &automatic).is_err());
        assert!(handle.is_skipped("asr:primeline").unwrap());
    }

    #[test]
    fn quality_inputs_and_selection_preserve_exact_artifact_versions() {
        let (root, handle) = setup();
        handle
            .set_units("asr:primeline", &[0, 1, 2, 3, 4], "windows")
            .unwrap();
        let original = handle.snapshot().unwrap().stages["asr:primeline"]
            .selected_attempt
            .clone()
            .unwrap();
        enqueue(
            root.path(),
            &control(
                "retry-version",
                ControlAction::Retry {
                    step: "asr:primeline".into(),
                    fragments: vec![2],
                    parameters: RetryParameters::default(),
                },
            ),
        )
        .unwrap();
        handle.apply_controls().unwrap();
        let retry = handle.pending_retries().unwrap().pop().unwrap();
        enqueue(
            root.path(),
            &control(
                "check-old-version",
                ControlAction::AddCheck {
                    step: "asr:primeline".into(),
                    dependencies: vec![original.clone()],
                    fragments: vec![2],
                },
            ),
        )
        .unwrap();
        handle.apply_controls().unwrap();
        handle.begin_retry(&retry.attempt_id).unwrap();
        let folder = handle.retry_output(&retry.attempt_id);
        fs::create_dir_all(&folder).unwrap();
        fs::write(
            folder.join("00002.json"),
            br#"{"text":"new","language":"de","segments":[]}"#,
        )
        .unwrap();
        handle.retry_unit_done(&retry.attempt_id, 2).unwrap();
        handle.finish_retry(&retry.attempt_id).unwrap();
        handle.select_retry(&retry.attempt_id).unwrap();
        let check = handle.pending_checks().unwrap().pop().unwrap();
        assert!(check.step.starts_with("check:"));
        assert!(
            fs::read_to_string(check.input_folders["primeline"].join("00002.json"))
                .unwrap()
                .contains("old")
        );
        handle.begin_check(&check.attempt_id).unwrap();
        let report = crate::quality::inspect_inputs(
            root.path(),
            &crate::quality::Settings::default(),
            false,
            Some(&check.fragments),
            Some(&check.input_folders),
            &|index| handle.check_unit_done(&check.attempt_id, index),
        )
        .unwrap();
        assert_eq!(report.windows.len(), 1);
        assert_eq!(report.windows[0].index, 2);
        let report_file = folder.join("quality.json");
        fs::write(&report_file, serde_json::to_vec(&report).unwrap()).unwrap();
        handle
            .finish_check(&check.attempt_id, &report_file)
            .unwrap();
        assert_eq!(
            handle.snapshot().unwrap().stages[&check.step].status,
            State::Ready
        );
        handle.select_retry(&original).unwrap();
        assert!(
            fs::read_to_string(root.path().join("primeline/00002.json"))
                .unwrap()
                .contains("old")
        );
        assert!(
            fs::read_to_string(folder.join("00002.json"))
                .unwrap()
                .contains("new")
        );
        assert_eq!(
            handle.snapshot().unwrap().stages["asr:primeline"]
                .selected_attempt
                .as_deref(),
            Some(original.as_str())
        );
    }

    #[test]
    fn execution_state_cannot_escape_through_directory_symlinks() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join(".execution")).unwrap();
        let status = ProjectStatus {
            schema_version: 2,
            input: root.path().join("source.mp3"),
            input_sha256: "input".into(),
            export_stem: Some("source".into()),
            stages: BTreeMap::new(),
            asr_roles: vec!["primeline".into()],
        };
        assert!(Handle::open(root.path(), &status).is_err());
        assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
    }
}
