use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    config::Config,
    execution,
    pipeline::{self, JobManifest},
};

#[cfg(test)]
const STEPS: [&str; 9] = [
    "audio",
    "chunks",
    "asr:primeline",
    "asr:flix",
    "asr:qwen",
    "diarization",
    "review-flags",
    "review",
    "export",
];

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Step {
    Audio,
    Chunks,
    AsrPrimeline,
    AsrFlix,
    AsrQwen,
    Asr(String),
    Diarization,
    ReviewFlags,
    Review,
    Export,
}

impl Step {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Audio => "audio",
            Self::Chunks => "chunks",
            Self::AsrPrimeline => "asr:primeline",
            Self::AsrFlix => "asr:flix",
            Self::AsrQwen => "asr:qwen",
            Self::Asr(role) => role,
            Self::Diarization => "diarization",
            Self::ReviewFlags => "review-flags",
            Self::Review => "review",
            Self::Export => "export",
        }
    }
}

impl AsRef<str> for Step {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::str::FromStr for Step {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        Ok(match value {
            "audio" => Self::Audio,
            "chunks" => Self::Chunks,
            "asr:primeline" => Self::AsrPrimeline,
            "asr:flix" => Self::AsrFlix,
            "asr:qwen" => Self::AsrQwen,
            "diarization" => Self::Diarization,
            "review-flags" => Self::ReviewFlags,
            "review" => Self::Review,
            "export" => Self::Export,
            _ if value.starts_with("asr:") => {
                let role = value.strip_prefix("asr:").unwrap();
                crate::config::validate_asr_roles(&[role.to_owned()])?;
                Self::Asr(value.to_owned())
            }
            _ => bail!(
                "unknown project step {value:?}; expected audio, chunks, asr:<role>, diarization, review-flags, review or export"
            ),
        })
    }
}

pub fn steps_for_roles(roles: &[String]) -> Vec<String> {
    let mut steps = vec!["audio".into(), "chunks".into()];
    steps.extend(roles.iter().map(|role| format!("asr:{role}")));
    steps.extend(["diarization", "review-flags", "review", "export"].map(str::to_owned));
    steps
}

pub(crate) fn parents(step: &str, roles: &[String]) -> Vec<String> {
    match step {
        "audio" => vec![],
        "chunks" | "diarization" => vec!["audio".into()],
        s if s.starts_with("asr:") => vec!["chunks".into()],
        "review-flags" => roles
            .iter()
            .map(|r| format!("asr:{r}"))
            .chain(std::iter::once("diarization".into()))
            .collect(),
        "review" => vec!["review-flags".into()],
        "export" => vec!["review".into()],
        _ => vec![],
    }
}

pub fn descendants_for_roles(step: &str, roles: &[String]) -> Result<BTreeSet<String>> {
    let steps = steps_for_roles(roles);
    ensure!(
        steps.iter().any(|s| s == step),
        "unknown or unselected project step {step:?}"
    );
    let mut selected = BTreeSet::from([step.to_owned()]);
    loop {
        let previous = selected.len();
        for next in &steps {
            if parents(next, roles)
                .iter()
                .any(|parent| selected.contains(parent))
            {
                selected.insert(next.clone());
            }
        }
        if selected.len() == previous {
            break;
        }
    }
    Ok(selected)
}

#[cfg(test)]
fn descendants(step: &str) -> Result<BTreeSet<String>> {
    descendants_for_roles(step, &crate::config::default_asr_roles())
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    NotStarted,
    Running,
    Ready,
    Failed,
    PendingExternal,
    Stale,
    Skipped,
    Blocked,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    pub status: State,
    pub attempts: u64,
    pub fingerprint: String,
    pub output_sha256: Option<String>,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStatus {
    pub schema_version: u32,
    pub input: PathBuf,
    pub input_sha256: String,
    #[serde(default)]
    pub export_stem: Option<String>,
    pub stages: BTreeMap<String, Stage>,
    #[serde(default = "crate::config::default_asr_roles")]
    pub asr_roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExportPaths {
    pub txt: PathBuf,
    pub json: PathBuf,
}

impl ExportPaths {
    fn legacy(project: &Path) -> Self {
        Self {
            txt: project.join("transcript.txt"),
            json: project.join("transcript.json"),
        }
    }
}

fn source_stem(input: &Path) -> Result<String> {
    Ok(input
        .file_stem()
        .and_then(|s| s.to_str())
        .context("original audio filename must have a UTF-8 stem")?
        .to_owned())
}

impl ProjectStatus {
    pub fn export_paths(&self, project: &Path) -> Result<ExportPaths> {
        let Some(stem) = &self.export_stem else {
            return Ok(ExportPaths::legacy(project));
        };
        let json = format!("{stem}.json");
        ensure!(
            !stem.is_empty()
                && !stem.contains('\0')
                && matches!(
                    Path::new(&json).components().collect::<Vec<_>>().as_slice(),
                    [std::path::Component::Normal(_)]
                ),
            "invalid original audio filename in workflow state"
        );
        let directory = if [
            "workflow.json",
            "manifest.json",
            "chunks.json",
            "diarization.json",
            "review-flags.json",
        ]
        .iter()
        .any(|name| name.eq_ignore_ascii_case(&json))
        {
            project.join("exports")
        } else {
            project.to_path_buf()
        };
        Ok(ExportPaths {
            txt: directory.join(format!("{stem}.txt")),
            json: directory.join(json),
        })
    }
}

pub fn export_paths(project: &Path) -> Result<ExportPaths> {
    status(project)?.export_paths(project)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Outcome {
    Complete,
    PendingReview,
    StageComplete,
    Blocked,
    Interrupted,
}

pub fn prerequisite_steps_for_roles(target: Option<&Step>, roles: &[String]) -> BTreeSet<String> {
    let mut selected = BTreeSet::from([target.unwrap_or(&Step::Export).as_str().to_owned()]);
    loop {
        let previous = selected.len();
        for stage in selected.clone() {
            selected.extend(parents(&stage, roles));
        }
        if selected.len() == previous {
            return selected;
        }
    }
}

pub(crate) fn state_path(project: &Path) -> PathBuf {
    project.join("workflow.json")
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let partial = path.with_extension("partial");
    let result = (|| {
        let mut file = File::create(&partial)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.sync_all()?;
        fs::rename(&partial, path)?;
        File::open(path.parent().context("state has no parent")?)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

fn digest(value: &impl Serialize) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}

pub(crate) fn output_paths(project: &Path, step: &str, exports: &ExportPaths) -> Vec<PathBuf> {
    match step {
        "audio" => vec![project.join("audio.wav")],
        "chunks" => vec![project.join("chunks.json"), project.join("clips")],
        "diarization" => vec![project.join("diarization.json")],
        s if s.starts_with("asr:") => vec![project.join(s.strip_prefix("asr:").unwrap())],
        "review-flags" => vec![project.join("review-flags.json")],
        "review" => vec![project.join("reviews")],
        "export" => vec![exports.json.clone(), exports.txt.clone()],
        _ => vec![],
    }
}

fn tree_hash(root: &Path, paths: &[PathBuf]) -> Result<Option<String>> {
    let mut hash = Sha256::new();
    let mut count = 0usize;
    let mut stack = paths.to_vec();
    while let Some(path) = stack.pop() {
        if !path.exists() {
            return Ok(None);
        }
        if path.is_dir() {
            let mut entries = fs::read_dir(&path)?
                .map(|entry| entry.map(|e| e.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            entries.sort();
            stack.extend(entries.into_iter().rev());
        } else {
            hash.update(path.strip_prefix(root)?.to_string_lossy().as_bytes());
            hash.update(pipeline::hash_file(&path)?.as_bytes());
            count += 1;
        }
    }
    if count == 0 {
        return Ok(None);
    }
    Ok(Some(hex::encode(hash.finalize())))
}

fn model<'a>(manifest: &'a JobManifest, role: &str) -> Result<(&'a str, &'a str)> {
    let model = manifest
        .models
        .iter()
        .find(|m| m.role == role)
        .with_context(|| format!("missing model {role}"))?;
    Ok((&model.repo, &model.revision))
}

fn stage_fingerprint(manifest: &JobManifest, step: &str, deps: &[String]) -> Result<String> {
    let params = match step {
        "audio" => serde_json::json!(["audio-v1", manifest.input_sha256]),
        "chunks" => serde_json::json!(["chunks-v1", 28.0, 0.8]),
        "diarization" => serde_json::json!([
            manifest.diarization.program,
            manifest.diarization.args,
            manifest.diarization.threads,
            manifest.diarization.max_in_flight,
            manifest.diarization_script_sha256,
            model(manifest, "diarization")?
        ]),
        s if s.starts_with("asr:") => {
            let role = step.strip_prefix("asr:").unwrap();
            let adapter = manifest
                .asr
                .get(role)
                .with_context(|| format!("missing adapter {role}"))?;
            let mut params = serde_json::json!([
                adapter.program,
                adapter.args,
                adapter.threads,
                adapter.max_in_flight,
                manifest.asr_script_sha256.get(role),
                model(manifest, role)?
            ]);
            if let Some(engine) = &adapter.engine
                && Some(engine.as_str()) != crate::config::legacy_asr_engine(role)
            {
                params
                    .as_array_mut()
                    .expect("fingerprint parameters are an array")
                    .push(serde_json::json!(engine));
            }
            params
        }
        "review-flags" => serde_json::json!("review-flags-v1"),
        "review" => serde_json::json!([manifest.llm, manifest.review_rules_sha256, "review-v1"]),
        "export" => serde_json::json!("export-source-name-v2"),
        _ => bail!("unknown project step {step}"),
    };
    digest(&(step, params, deps))
}

fn fingerprints(manifest: &JobManifest) -> Result<BTreeMap<String, String>> {
    let mut result: BTreeMap<String, String> = BTreeMap::new();
    for step in steps_for_roles(&manifest.asr_roles) {
        let deps = parents(&step, &manifest.asr_roles)
            .iter()
            .map(|parent| result[parent].clone())
            .collect::<Vec<_>>();
        result.insert(step.clone(), stage_fingerprint(manifest, &step, &deps)?);
    }
    Ok(result)
}

fn archive(project: &Path, step: &str, attempt: u64, exports: &ExportPaths) -> Result<()> {
    let existing = output_paths(project, step, exports)
        .into_iter()
        .filter(|path| path.exists())
        .collect::<Vec<_>>();
    if existing.is_empty() {
        return Ok(());
    }
    let destination = project
        .join("attempts")
        .join(step.replace(':', "-"))
        .join(attempt.to_string());
    fs::create_dir_all(&destination)?;
    for path in existing {
        let target = destination.join(path.file_name().context("output has no name")?);
        ensure!(
            !target.exists(),
            "attempt archive already exists: {}",
            target.display()
        );
        fs::rename(path, target)?;
    }
    if state_path(project).exists() {
        let state: ProjectStatus = serde_json::from_slice(&fs::read(state_path(project))?)?;
        if let Some(stage) = state.stages.get(step) {
            atomic_json(&destination.join(".stage.json"), stage)?;
        }
    }
    Ok(())
}

fn validate_legacy(project: &Path, input: &Path, step: &str) -> Result<bool> {
    let chunks = || -> Result<Vec<crate::audio::Interval>> {
        Ok(serde_json::from_slice(&fs::read(
            project.join("chunks.json"),
        )?)?)
    };
    let valid = match step {
        "audio" => File::open(project.join("audio.wav")).is_ok_and(|mut file| {
            let mut header = [0u8; 12];
            file.metadata().is_ok_and(|meta| meta.len() > 44)
                && file.read_exact(&mut header).is_ok()
                && header.starts_with(b"RIFF")
                && &header[8..12] == b"WAVE"
        }),
        "chunks" => chunks().is_ok_and(|items| {
            let expected = crate::audio::duration(input)
                .and_then(|duration| crate::audio::chunks(duration, 28.0, 0.8));
            expected.is_ok_and(|expected| {
                items.len() == expected.len()
                    && items.iter().enumerate().all(|(i, c)| {
                        (c.start - expected[i].start).abs() < 0.001
                            && (c.end - expected[i].end).abs() < 0.001
                            && project.join("clips").join(format!("{i:05}.wav")).exists()
                    })
            })
        }),
        "diarization" => serde_json::from_slice::<pipeline::Diarization>(
            &fs::read(project.join("diarization.json")).unwrap_or_default(),
        )
        .is_ok_and(|dia| {
            crate::audio::duration(input)
                .is_ok_and(|duration| pipeline::validate_diarization(&dia, duration).is_ok())
        }),
        s if s.starts_with("asr:") => chunks().is_ok_and(|items| {
            items.iter().enumerate().all(|(i, c)| {
                let path = project
                    .join(step.strip_prefix("asr:").unwrap())
                    .join(format!("{i:05}.json"));
                serde_json::from_slice::<pipeline::Hypothesis>(&fs::read(path).unwrap_or_default())
                    .is_ok_and(|hyp| pipeline::validate_hypothesis(&hyp, c.end - c.start).is_ok())
            })
        }),
        "review-flags" => serde_json::from_slice::<Vec<serde_json::Value>>(
            &fs::read(project.join("review-flags.json")).unwrap_or_default(),
        )
        .is_ok(),
        "review" => chunks().is_ok_and(|items| {
            (0..items.len()).step_by(8).all(|start| {
                let path = project.join("reviews").join(format!("{start:05}.json"));
                serde_json::from_slice::<serde_json::Value>(&fs::read(path).unwrap_or_default())
                    .ok()
                    .and_then(|data| data["segments"].as_array().cloned())
                    .is_some_and(|segments| {
                        segments.len() == (items.len() - start).min(8)
                            && segments.iter().enumerate().all(|(j, entry)| {
                                entry["index"].as_u64() == Some((start + j) as u64)
                                    && entry["text"]
                                        .as_str()
                                        .is_some_and(|text| !text.trim().is_empty())
                                    && entry["language"].as_str().is_some_and(|lang| {
                                        ["de", "ru", "mixed", "unknown"].contains(&lang)
                                    })
                                    && entry["gap"].is_boolean()
                            })
                    })
            })
        }),
        "export" => {
            serde_json::from_slice::<pipeline::Transcript>(
                &fs::read(project.join("transcript.json")).unwrap_or_default(),
            )
            .is_ok()
                && fs::metadata(project.join("transcript.txt")).is_ok()
        }
        _ => false,
    };
    Ok(valid
        && tree_hash(
            project,
            &output_paths(project, step, &ExportPaths::legacy(project)),
        )?
        .is_some())
}

struct ProjectLock(File);

impl Drop for ProjectLock {
    fn drop(&mut self) {
        // Closing alone leaves flock held while a forked child still has the descriptor.
        if let Err(error) = fs2::FileExt::unlock(&self.0) {
            eprintln!("could not release project lock: {error:#}");
        }
    }
}

pub struct Project {
    root: PathBuf,
    _lock: ProjectLock,
    pub status: ProjectStatus,
    execution: execution::Handle,
}

impl Project {
    pub(crate) fn open(
        root: &Path,
        input: &Path,
        manifest: &JobManifest,
        legacy: Option<&JobManifest>,
        rerun: Option<&str>,
    ) -> Result<Self> {
        fs::create_dir_all(root)?;
        let root = root.canonicalize()?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join(".workflow.lock"))?;
        lock.try_lock_exclusive()
            .context("project is already in use")?;
        let lock = ProjectLock(lock);
        let hash = &manifest.input_sha256;
        let initial_stem = source_stem(input)?;
        let input = input
            .canonicalize()
            .context("project input is unavailable")?;
        ensure!(
            pipeline::hash_file(&input)? == *hash,
            "project input content changed"
        );
        let path = state_path(&root);
        let new_state = !path.exists();
        let mut status: ProjectStatus = if !new_state {
            serde_json::from_slice(&fs::read(&path)?).context("invalid workflow state")?
        } else {
            ProjectStatus {
                schema_version: 2,
                input: input.clone(),
                input_sha256: hash.clone(),
                export_stem: None,
                stages: BTreeMap::new(),
                asr_roles: manifest.asr_roles.clone(),
            }
        };
        ensure!(status.schema_version == 2, "unsupported workflow schema");
        for stage in status.stages.values_mut() {
            if stage.status == State::Running {
                stage.status = State::Failed;
                stage.warning = Some("stage failed or was interrupted; resume to retry".into());
            }
        }
        crate::config::validate_asr_roles(&manifest.asr_roles)?;
        status.asr_roles = manifest.asr_roles.clone();
        ensure!(
            status.input_sha256 == *hash,
            "project input content changed"
        );
        let previous_exports = status.export_paths(&root)?;
        if status.export_stem.is_none() {
            status.export_stem = Some(if new_state {
                initial_stem
            } else {
                source_stem(&status.input)?
            });
        }
        let exports = status.export_paths(&root)?;
        status.input = input.clone();
        let expected = fingerprints(manifest)?;
        if new_state && let Some(old) = legacy {
            ensure!(
                old.schema_version == 1,
                "unsupported legacy manifest schema"
            );
            ensure!(
                old.input_sha256 == *hash,
                "legacy project input content changed"
            );
            let previous = fingerprints(old)?;
            for step in steps_for_roles(&manifest.asr_roles) {
                if step != "export"
                    && previous
                        .get(&step)
                        .is_some_and(|fingerprint| fingerprint == &expected[&step])
                    && parents(&step, &manifest.asr_roles).iter().all(|p| {
                        status
                            .stages
                            .get(p)
                            .is_some_and(|s| s.status == State::Ready)
                    })
                    && validate_legacy(&root, &input, &step)?
                {
                    status.stages.insert(
                        step.clone(),
                        Stage {
                            status: State::Ready,
                            attempts: 1,
                            fingerprint: expected[&step].clone(),
                            output_sha256: tree_hash(&root, &output_paths(&root, &step, &exports))?,
                            warning: None,
                        },
                    );
                }
            }
        }
        let forced = rerun
            .map(|step| descendants_for_roles(step, &manifest.asr_roles))
            .transpose()?
            .unwrap_or_default();
        let mut invalid = BTreeSet::new();
        let inactive = status
            .stages
            .keys()
            .filter_map(|step| step.strip_prefix("asr:"))
            .chain(manifest.asr.keys().map(String::as_str))
            .filter(|role| !manifest.asr_roles.iter().any(|active| active == role))
            .map(|role| format!("asr:{role}"))
            .collect::<BTreeSet<_>>();
        for step in inactive {
            let stage = status.stages.entry(step).or_insert(Stage {
                status: State::Skipped,
                attempts: 0,
                fingerprint: String::new(),
                output_sha256: None,
                warning: None,
            });
            stage.status = State::Skipped;
            stage.warning = Some("ASR role is not selected for this project run".into());
        }
        for step in steps_for_roles(&manifest.asr_roles) {
            let previous = status.stages.get(&step).cloned();
            let stale = forced.contains(&step)
                || parents(&step, &manifest.asr_roles)
                    .iter()
                    .any(|p| invalid.contains(p));
            let fingerprint = &expected[&step];
            let changed = previous
                .as_ref()
                .is_some_and(|s| s.fingerprint != *fingerprint);
            let damaged = if let Some(s) = &previous {
                s.status == State::Ready
                    && tree_hash(&root, &output_paths(&root, &step, &previous_exports))?.as_ref()
                        != s.output_sha256.as_ref()
            } else {
                false
            };
            if stale || changed || damaged {
                invalid.insert(step.clone());
                if let Some(old) = &previous {
                    archive(&root, &step, old.attempts, &previous_exports)?;
                    if step == "export" && exports != previous_exports {
                        archive(&root, &step, old.attempts, &exports)?;
                    }
                }
            }
            let stage = status.stages.entry(step.clone()).or_insert(Stage {
                status: State::NotStarted,
                attempts: 0,
                fingerprint: fingerprint.clone(),
                output_sha256: None,
                warning: None,
            });
            if stale || changed || damaged {
                stage.status = State::Stale;
                stage.output_sha256 = None;
                stage.warning = if damaged {
                    Some("stage output changed or is missing".into())
                } else {
                    None
                };
            } else if stage.status == State::Ready {
                stage.warning = None;
            } else if stage.status == State::Skipped {
                stage.status = if stage.output_sha256.is_some()
                    && tree_hash(&root, &output_paths(&root, &step, &exports))?.as_ref()
                        == stage.output_sha256.as_ref()
                {
                    State::Ready
                } else {
                    stage.output_sha256 = None;
                    State::NotStarted
                };
                stage.warning = None;
            }
            stage.fingerprint = fingerprint.clone();
        }
        if new_state {
            for step in steps_for_roles(&manifest.asr_roles) {
                if status.stages[&step].status != State::Ready {
                    archive(&root, &step, 0, &previous_exports)?;
                    if step == "export" && exports != previous_exports {
                        archive(&root, &step, 0, &exports)?;
                    }
                }
            }
        }
        atomic_json(&path, &status)?;
        let execution = execution::Handle::open(&root, &status)?;
        execution.reconcile(&status)?;
        let mut project = Self {
            root,
            _lock: lock,
            status,
            execution,
        };
        project.apply_controls()?;
        Ok(project)
    }

    pub(crate) fn ready(&self, step: &str) -> bool {
        if self.is_skipped(step) || self.status.stages[step].status != State::Ready {
            return false;
        }
        match self.execution.snapshot() {
            Ok(snapshot) => snapshot
                .stages
                .get(step)
                .is_some_and(|view| view.status == State::Ready),
            Err(err) => {
                eprintln!("cannot read execution state for {step}: {err:#}");
                false
            }
        }
    }

    pub(crate) fn begin(&mut self, step: &str) -> Result<()> {
        ensure!(
            !self.is_skipped(step),
            "step {step} was skipped; restore it before running"
        );
        self.execution.begin_stage(step)?;
        self.status = status(&self.root)?;
        Ok(())
    }

    pub(crate) fn finish(&mut self, step: &str) -> Result<()> {
        self.execution.finish_stage(step)?;
        self.status = status(&self.root)?;
        Ok(())
    }

    pub(crate) fn warning(&mut self, step: &str, pending: bool, message: &str) -> Result<()> {
        let state = if pending {
            State::PendingExternal
        } else {
            State::Failed
        };
        self.execution.fail_stage(step, state, message)?;
        self.status = status(&self.root)?;
        Ok(())
    }

    pub(crate) fn cached_warning(&mut self, step: &str, message: &str) -> Result<()> {
        self.execution.cached_warning(step, message)?;
        self.status = status(&self.root)?;
        Ok(())
    }

    fn save(&self) -> Result<()> {
        atomic_json(&state_path(&self.root), &self.status)
    }

    pub fn execution(&self) -> execution::Handle {
        self.execution.clone()
    }

    pub fn is_skipped(&self, step: &str) -> bool {
        match self.execution.is_skipped(step) {
            Ok(skipped) => skipped,
            Err(err) => {
                eprintln!("cannot read execution skip state: {err:#}");
                true
            }
        }
    }

    pub fn select_retry(&mut self, attempt_id: &str) -> Result<()> {
        let snapshot = self.execution.snapshot()?;
        let attempt = snapshot
            .attempts
            .get(attempt_id)
            .context("unknown retry attempt")?;
        ensure!(attempt.status == State::Ready, "retry has not completed");
        self.execution.select_retry(attempt_id)?;
        self.status = status(&self.root)?;
        Ok(())
    }

    pub fn apply_controls(&mut self) -> Result<()> {
        self.execution.apply_controls()?;
        self.project_execution()
    }

    fn project_execution(&mut self) -> Result<()> {
        self.execution.project_legacy()?;
        self.status = status(&self.root)?;
        Ok(())
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        match self.execution.snapshot() {
            Ok(snapshot) => {
                for attempt in snapshot
                    .attempts
                    .values()
                    .filter(|attempt| attempt.status == State::Running)
                {
                    if let Err(error) = self.execution.fail_stage(
                        &attempt.step,
                        State::Failed,
                        "stage failed or was interrupted; resume to retry",
                    ) {
                        eprintln!("could not persist interrupted attempt: {error:#}");
                    }
                }
            }
            Err(error) => eprintln!("could not read execution state on shutdown: {error:#}"),
        }
        match status(&self.root) {
            Ok(status) => self.status = status,
            Err(error) => eprintln!("could not refresh completed stages on shutdown: {error:#}"),
        }
        let mut changed = false;
        for stage in self.status.stages.values_mut() {
            if stage.status == State::Running {
                stage.status = State::Failed;
                stage.warning = Some("stage failed or was interrupted; resume to retry".into());
                changed = true;
            }
        }
        if changed && let Err(err) = self.save() {
            eprintln!("could not persist failed project stages: {err:#}");
        }
        if changed {
            for (step, stage) in &self.status.stages {
                if stage.status == State::Failed
                    && stage.warning.as_deref()
                        == Some("stage failed or was interrupted; resume to retry")
                    && let Err(err) = self.execution.fail_stage(
                        step,
                        State::Failed,
                        "stage failed or was interrupted; resume to retry",
                    )
                {
                    eprintln!("could not persist interrupted execution: {err:#}");
                }
            }
        }
    }
}

pub fn status(project: &Path) -> Result<ProjectStatus> {
    let mut state: ProjectStatus = serde_json::from_slice(
        &fs::read(state_path(project)).context("project has no workflow state")?,
    )?;
    ensure!(state.schema_version == 2, "unsupported workflow schema");
    crate::config::validate_asr_roles(&state.asr_roles)?;
    state.export_paths(project)?;
    if project.join(".execution").join("state.json").exists() {
        let execution = execution::snapshot(project)?;
        for (step, view) in &execution.stages {
            if let Some(stage) = state.stages.get_mut(step) {
                stage.status = view.status;
                if execution.skipped.contains(step) {
                    stage.warning = Some("step was skipped by user".into());
                }
            }
        }
    }
    Ok(state)
}

pub(crate) fn completed_stage_artifact(project: &Path, step: &str) -> Result<(String, String)> {
    let state: ProjectStatus = serde_json::from_slice(&fs::read(state_path(project))?)?;
    ensure!(state.schema_version == 2, "unsupported workflow schema");
    let fingerprint = state
        .stages
        .get(step)
        .context("unknown step")?
        .fingerprint
        .clone();
    let exports = state.export_paths(project)?;
    let hash = tree_hash(project, &output_paths(project, step, &exports))?
        .with_context(|| format!("missing outputs for {step}"))?;
    Ok((hash, fingerprint))
}

pub(crate) fn project_completed_stage(project: &Path, step: &str, hash: String) -> Result<()> {
    let mut state: ProjectStatus = serde_json::from_slice(&fs::read(state_path(project))?)?;
    ensure!(state.schema_version == 2, "unsupported workflow schema");
    let stage = state.stages.get_mut(step).context("unknown step")?;
    if stage.status != State::Running {
        stage.attempts += 1;
    }
    stage.status = State::Ready;
    stage.output_sha256 = Some(hash);
    stage.warning = None;
    atomic_json(&state_path(project), &state)
}

pub(crate) fn project_started_stage(project: &Path, step: &str) -> Result<()> {
    let mut state: ProjectStatus = serde_json::from_slice(&fs::read(state_path(project))?)?;
    let stage = state.stages.get_mut(step).context("unknown step")?;
    if stage.status != State::Running {
        stage.attempts += 1;
    }
    stage.status = State::Running;
    stage.warning = None;
    atomic_json(&state_path(project), &state)
}

pub(crate) fn project_stage_warning(
    project: &Path,
    step: &str,
    status: State,
    message: &str,
) -> Result<()> {
    let mut state: ProjectStatus = serde_json::from_slice(&fs::read(state_path(project))?)?;
    if step.starts_with("check:") {
        return Ok(());
    }
    let stage = state.stages.get_mut(step).context("unknown step")?;
    stage.status = status;
    stage.warning = Some(message.into());
    atomic_json(&state_path(project), &state)
}

pub(crate) fn project_selected_stage(project: &Path, step: &str) -> Result<()> {
    let mut state: ProjectStatus = serde_json::from_slice(&fs::read(state_path(project))?)?;
    ensure!(state.schema_version == 2, "unsupported workflow schema");
    let exports = state.export_paths(project)?;
    let hash = tree_hash(project, &output_paths(project, step, &exports))?
        .with_context(|| format!("selected outputs for {step} are missing"))?;
    let stage = state.stages.get_mut(step).context("unknown step")?;
    stage.status = State::Ready;
    stage.attempts += 1;
    stage.output_sha256 = Some(hash);
    stage.warning = None;
    atomic_json(&state_path(project), &state)
}

pub(crate) fn project_control_state(
    project: &Path,
    previous: &execution::Snapshot,
    execution: &mut execution::Snapshot,
) -> Result<()> {
    let mut state: ProjectStatus = serde_json::from_slice(&fs::read(state_path(project))?)?;
    ensure!(state.schema_version == 2, "unsupported workflow schema");
    let exports = state.export_paths(project)?;
    let mut changed = false;
    for (step, view) in &execution.stages {
        let Some(stage) = state.stages.get_mut(step) else {
            continue;
        };
        let invalidated = matches!(view.status, State::Stale | State::NotStarted)
            && matches!(stage.status, State::Ready | State::Skipped);
        let retain_partial_asr = step.starts_with("asr:")
            && stage.status == State::Skipped
            && stage.output_sha256.is_none()
            && view.status == State::NotStarted;
        if invalidated && !retain_partial_asr {
            archive(project, step, stage.attempts, &exports)?;
        }
        if let Some(id) = previous
            .stages
            .get(step)
            .and_then(|view| view.selected_attempt.as_ref())
            && execution.stages[step].selected_attempt.as_deref() != Some(id)
            && (invalidated || stage.status == State::Stale)
            && let Some(artifact) = execution
                .attempts
                .get_mut(id)
                .and_then(|attempt| attempt.artifact.as_mut())
        {
            let archived = project
                .join("attempts")
                .join(step.replace(':', "-"))
                .join(stage.attempts.to_string());
            if artifact.path == project && archived.exists() {
                artifact.path = archived;
            }
        }
        if stage.status == view.status {
            continue;
        }
        stage.status = view.status;
        if matches!(view.status, State::Stale | State::NotStarted) {
            stage.output_sha256 = None;
        }
        if view.status == State::Skipped {
            stage.warning = Some("step was skipped by user".into());
        } else if stage.warning.as_deref() == Some("step was skipped by user") {
            stage.warning = None;
        }
        changed = true;
    }
    if changed {
        atomic_json(&state_path(project), &state)?;
    }
    Ok(())
}

pub fn resume(config: &Config, project: &Path, sequential_asr: bool) -> Result<Outcome> {
    let state = status(project)?;
    let input = state
        .input
        .canonicalize()
        .context("project input is unavailable")?;
    ensure!(
        pipeline::hash_file(&input)? == state.input_sha256,
        "project input changed; restore original content before resuming"
    );
    pipeline::transcribe_outcome(config, &input, project, sequential_asr)
}

pub fn resume_with_input(
    config: &Config,
    project: &Path,
    new_input: &Path,
    sequential_asr: bool,
) -> Result<Outcome> {
    let state = status(project)?;
    let input = new_input
        .canonicalize()
        .context("replacement input is unavailable")?;
    ensure!(
        pipeline::hash_file(&input)? == state.input_sha256,
        "replacement input differs from the project's original audio"
    );
    pipeline::transcribe_outcome(config, &input, project, sequential_asr)
}

pub fn rerun(
    config: &Config,
    project: &Path,
    step: impl AsRef<str>,
    sequential_asr: bool,
) -> Result<Outcome> {
    let step: Step = step.as_ref().parse()?;
    if let Some(role) = step.as_str().strip_prefix("asr:") {
        ensure!(
            config.asr.contains_key(role),
            "ASR role {role:?} is not configured"
        );
    }
    let state = status(project)?;
    let input = state
        .input
        .canonicalize()
        .context("project input is unavailable")?;
    ensure!(
        pipeline::hash_file(&input)? == state.input_sha256,
        "project input changed; restore original content before rerunning"
    );
    pipeline::transcribe_step(config, &input, project, sequential_asr, Some(step.as_str()))
}

pub fn run_stage(
    config: &Config,
    project: &Path,
    step: Step,
    input: Option<&Path>,
    sequential_asr: bool,
) -> Result<Outcome> {
    if let Some(role) = step.as_str().strip_prefix("asr:") {
        ensure!(
            config.asr.contains_key(role),
            "ASR role {role:?} is not configured"
        );
    }
    let input = if state_path(project).exists() {
        let state = status(project)?;
        let path = input
            .unwrap_or(&state.input)
            .canonicalize()
            .context("project input is unavailable; supply --input with identical content")?;
        ensure!(
            pipeline::hash_file(&path)? == state.input_sha256,
            "replacement input differs from the project's original audio"
        );
        path
    } else {
        input.context("new project requires --input")?.to_path_buf()
    };
    pipeline::transcribe_selected(config, &input, project, sequential_asr, None, Some(step))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{Adapter, Llm},
        models::ModelSpec,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn parses_public_step_names() {
        for name in STEPS {
            assert_eq!(name.parse::<Step>().unwrap().as_str(), name);
        }
        assert_eq!(
            "asr:custom_5".parse::<Step>().unwrap().as_str(),
            "asr:custom_5"
        );
        assert!("asr:../unsafe".parse::<Step>().is_err());
    }

    #[test]
    fn inactive_partial_asr_is_retained_and_can_be_reenabled() {
        let (path, input, mut manifest) = fixture();
        let mut project = Project::open(&path, &input, &manifest, None, None).unwrap();
        project.begin("asr:qwen").unwrap();
        fs::create_dir(path.join("qwen")).unwrap();
        fs::write(path.join("qwen/partial.json"), b"partial work").unwrap();
        drop(project);
        manifest.asr_roles = vec!["primeline".into(), "flix".into()];
        let project = Project::open(&path, &input, &manifest, None, None).unwrap();
        assert_eq!(project.status.stages["asr:qwen"].status, State::Skipped);
        assert_eq!(project.status.stages["asr:qwen"].attempts, 1);
        assert_eq!(
            fs::read(path.join("qwen/partial.json")).unwrap(),
            b"partial work"
        );
        drop(project);
        manifest.asr_roles = crate::config::default_asr_roles();
        let project = Project::open(&path, &input, &manifest, None, None).unwrap();
        assert_eq!(project.status.stages["asr:qwen"].status, State::NotStarted);
        assert!(project.status.stages["asr:qwen"].warning.is_none());
        assert!(path.join("qwen/partial.json").exists());
        drop(project);
        fs::remove_dir_all(path).unwrap();
    }

    fn fixture() -> (PathBuf, PathBuf, JobManifest) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "target/workflow-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        let path = path.canonicalize().unwrap();
        let input = path.join("input.mp3");
        fs::write(&input, b"original source").unwrap();
        let adapter = Adapter {
            program: "python3".into(),
            script: "adapter.py".into(),
            engine: None,
            args: vec![],
            threads: 1,
            max_in_flight: 1,
        };
        let models = ["primeline", "flix", "qwen", "diarization"]
            .iter()
            .map(|role| ModelSpec {
                role: (*role).into(),
                repo: format!("repo/{role}"),
                revision: "rev1".into(),
                path: format!("/models/{role}").into(),
            })
            .collect();
        let manifest = JobManifest {
            schema_version: 1,
            input_sha256: pipeline::hash_file(&input).unwrap(),
            models,
            asr: ["primeline", "flix", "qwen"]
                .iter()
                .map(|role| ((*role).into(), adapter.clone()))
                .collect(),
            asr_script_sha256: ["primeline", "flix", "qwen"]
                .iter()
                .map(|role| ((*role).into(), "script".into()))
                .collect(),
            diarization: adapter,
            diarization_script_sha256: "script".into(),
            llm: Llm::default(),
            review_rules: None,
            review_rules_sha256: None,
            asr_roles: crate::config::default_asr_roles(),
        };
        (path, input, manifest)
    }

    fn finished(path: &Path, input: &Path, manifest: &JobManifest) {
        let mut project = Project::open(path, input, manifest, None, None).unwrap();
        let exports = project.status.export_paths(path).unwrap();
        for step in steps_for_roles(&manifest.asr_roles) {
            for file in output_paths(path, &step, &exports) {
                if file.extension().is_none() {
                    fs::create_dir_all(&file).unwrap();
                    fs::write(file.join("00000.json"), "payload").unwrap();
                } else if file.file_name().is_some_and(|name| name == "chunks.json") {
                    fs::write(file, r#"[{"start":0.0,"end":1.0}]"#).unwrap();
                } else {
                    fs::write(file, "payload").unwrap();
                }
            }
            project.begin(&step).unwrap();
            project.finish(&step).unwrap();
        }
    }

    #[test]
    fn adding_and_removing_custom_roles_preserves_ready_checkpoints() {
        let (path, input, mut manifest) = fixture();
        let adapter = manifest.asr["flix"].clone();
        manifest.asr.clear();
        manifest.asr.insert("nova".into(), adapter.clone());
        manifest.asr_script_sha256.clear();
        manifest
            .asr_script_sha256
            .insert("nova".into(), "script".into());
        manifest.models.retain(|model| model.role == "diarization");
        manifest.models.push(ModelSpec {
            role: "nova".into(),
            repo: "repo/nova".into(),
            revision: "rev1".into(),
            path: "/models/nova".into(),
        });
        manifest.asr_roles = vec!["nova".into()];
        finished(&path, &input, &manifest);

        let mut expanded = manifest.clone();
        for role in ["a", "b", "c", "d"] {
            expanded.asr.insert(role.into(), adapter.clone());
            expanded
                .asr_script_sha256
                .insert(role.into(), "script".into());
            expanded.models.push(ModelSpec {
                role: role.into(),
                repo: format!("repo/{role}"),
                revision: "rev1".into(),
                path: format!("/models/{role}").into(),
            });
            expanded.asr_roles.push(role.into());
        }
        let project = Project::open(&path, &input, &expanded, None, None).unwrap();
        assert_eq!(project.status.stages["asr:nova"].status, State::Ready);
        for role in ["a", "b", "c", "d"] {
            assert_eq!(
                project.status.stages[&format!("asr:{role}")].status,
                State::NotStarted
            );
        }
        assert_eq!(project.status.stages["review-flags"].status, State::Stale);
        assert_eq!(project.status.stages["export"].status, State::Stale);
        drop(project);

        let project = Project::open(&path, &input, &manifest, None, None).unwrap();
        assert_eq!(project.status.stages["asr:nova"].status, State::Ready);
        assert_eq!(project.status.stages["asr:a"].status, State::Skipped);
        drop(project);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn explicit_legacy_engines_preserve_checkpoint_fingerprints() {
        let (path, _, manifest) = fixture();
        let previous = fingerprints(&manifest).unwrap();
        let mut explicit = manifest.clone();
        for (role, adapter) in &mut explicit.asr {
            adapter.engine = crate::config::legacy_asr_engine(role).map(str::to_owned);
        }
        assert_eq!(fingerprints(&explicit).unwrap(), previous);
        explicit.asr.get_mut("flix").unwrap().engine = Some("qwen".into());
        let changed = fingerprints(&explicit).unwrap();
        for step in [
            "audio",
            "chunks",
            "asr:primeline",
            "asr:qwen",
            "diarization",
        ] {
            assert_eq!(changed[step], previous[step]);
        }
        for step in ["asr:flix", "review-flags", "review", "export"] {
            assert_ne!(changed[step], previous[step]);
        }
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn dropping_project_releases_lock_even_with_an_inherited_descriptor() {
        let (path, input, manifest) = fixture();
        let project = Project::open(&path, &input, &manifest, None, None).unwrap();
        let inherited = project._lock.0.try_clone().unwrap();
        drop(project);
        let reopened = Project::open(&path, &input, &manifest, None, None).unwrap();
        drop(reopened);
        drop(inherited);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn source_named_exports_survive_rebinding_and_reject_traversal() {
        let (path, input, manifest) = fixture();
        let original = path.join("recording.part.MP3");
        fs::rename(input, &original).unwrap();
        let project = Project::open(&path, &original, &manifest, None, None).unwrap();
        assert_eq!(
            project.status.export_stem.as_deref(),
            Some("recording.part")
        );
        assert_eq!(
            project.status.export_paths(&path).unwrap().txt,
            path.join("recording.part.txt")
        );
        drop(project);
        let moved = path.join("another-name.mp3");
        fs::rename(original, &moved).unwrap();
        let project = Project::open(&path, &moved, &manifest, None, None).unwrap();
        assert_eq!(
            project.status.export_stem.as_deref(),
            Some("recording.part")
        );
        assert_eq!(
            project.status.export_paths(&path).unwrap().json,
            path.join("recording.part.json")
        );
        let mut state = project.status.clone();
        drop(project);
        state.export_stem = Some("../escape".into());
        atomic_json(&state_path(&path), &state).unwrap();
        assert!(status(&path).is_err());
        assert!(Project::open(&path, &moved, &manifest, None, None).is_err());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn rerun_invalidates_only_transitive_dependents_and_preserves_attempts() {
        for step in STEPS {
            let (path, input, manifest) = fixture();
            finished(&path, &input, &manifest);
            let project = Project::open(&path, &input, &manifest, None, Some(step)).unwrap();
            let affected = descendants(step).unwrap();
            for name in STEPS {
                let state = &project.status.stages[name];
                assert_eq!(
                    state.status == State::Stale,
                    affected.contains(name),
                    "{step} -> {name}"
                );
                assert_eq!(state.attempts, 1);
                assert_eq!(
                    output_paths(&path, name, &project.status.export_paths(&path).unwrap())[0]
                        .exists(),
                    !affected.contains(name)
                );
            }
            assert!(
                path.join("attempts")
                    .join(step.replace(':', "-"))
                    .join("1")
                    .exists()
            );
            drop(project);
            fs::remove_dir_all(path).unwrap();
        }
    }

    #[test]
    fn provider_or_one_asr_change_only_invalidates_its_descendants() {
        let (path, input, manifest) = fixture();
        finished(&path, &input, &manifest);
        let mut changed = manifest.clone();
        changed.llm.model = Some("new-model".into());
        let project = Project::open(&path, &input, &changed, None, None).unwrap();
        assert_eq!(project.status.stages["review"].status, State::Stale);
        assert_eq!(project.status.stages["export"].status, State::Stale);
        assert_eq!(project.status.stages["asr:flix"].status, State::Ready);
        drop(project);
        fs::remove_dir_all(path).unwrap();

        let (path, input, manifest) = fixture();
        finished(&path, &input, &manifest);
        let mut changed = manifest.clone();
        changed
            .asr_script_sha256
            .insert("flix".into(), "updated".into());
        let project = Project::open(&path, &input, &changed, None, None).unwrap();
        for name in [
            "audio",
            "chunks",
            "diarization",
            "asr:primeline",
            "asr:qwen",
        ] {
            assert_eq!(project.status.stages[name].status, State::Ready);
        }
        for name in ["asr:flix", "review-flags", "review", "export"] {
            assert_eq!(project.status.stages[name].status, State::Stale);
        }
        drop(project);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn project_lock_and_source_hash_are_enforced() {
        let (path, input, manifest) = fixture();
        let project = Project::open(&path, &input, &manifest, None, None).unwrap();
        assert!(Project::open(&path, &input, &manifest, None, None).is_err());
        drop(project);
        fs::write(&input, "changed").unwrap();
        let changed = manifest.clone();
        assert!(Project::open(&path, &input, &changed, None, None).is_err());
        fs::write(&input, "original source").unwrap();
        assert!(Project::open(&path, &input, &changed, None, None).is_ok());
        let mut wrong = changed;
        wrong.input_sha256 = "different".into();
        assert!(Project::open(&path, &input, &wrong, None, None).is_err());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn unknown_legacy_files_never_become_ready_by_presence() {
        let (path, input, manifest) = fixture();
        fs::write(path.join("audio.wav"), b"RIFFlegacy").unwrap();
        let project = Project::open(&path, &input, &manifest, None, None).unwrap();
        assert_eq!(project.status.stages["audio"].status, State::NotStarted);
        assert!(!path.join("audio.wav").exists());
        assert!(path.join("attempts/audio/0/audio.wav").exists());
        drop(project);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn legacy_migration_verifies_artifacts_and_adapter_content_identity() {
        let (path, input, manifest) = fixture();
        let mut wav = b"RIFF".to_vec();
        wav.extend([0u8; 64]);
        wav[8..12].copy_from_slice(b"WAVE");
        fs::write(path.join("audio.wav"), wav).unwrap();
        fs::write(path.join("chunks.json"), b"not valid JSON").unwrap();
        let project = Project::open(&path, &input, &manifest, Some(&manifest), None).unwrap();
        assert_eq!(project.status.stages["audio"].status, State::Ready);
        assert_eq!(project.status.stages["chunks"].status, State::NotStarted);
        assert!(path.join("attempts/chunks/0/chunks.json").exists());
        drop(project);
        fs::remove_dir_all(path).unwrap();

        let (path, input, manifest) = fixture();
        finished(&path, &input, &manifest);
        let mut relocated = manifest.clone();
        relocated.asr.get_mut("flix").unwrap().script = "other/location/adapter.py".into();
        relocated
            .models
            .iter_mut()
            .find(|m| m.role == "flix")
            .unwrap()
            .path = "/elsewhere".into();
        let project = Project::open(&path, &input, &relocated, None, None).unwrap();
        assert!(
            project
                .status
                .stages
                .values()
                .all(|s| s.status == State::Ready)
        );
        drop(project);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn moved_source_with_identical_hash_remains_same_project() {
        let (path, input, manifest) = fixture();
        finished(&path, &input, &manifest);
        let moved = path.join("moved.mp3");
        fs::rename(&input, &moved).unwrap();
        let project = Project::open(&path, &moved, &manifest, None, None).unwrap();
        assert_eq!(project.status.input, moved);
        assert!(
            project
                .status
                .stages
                .values()
                .all(|s| s.status == State::Ready)
        );
        drop(project);
        fs::remove_dir_all(path).unwrap();
    }
}
