use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::models::ModelSpec;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adapter {
    pub program: PathBuf,
    pub script: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default = "default_in_flight")]
    pub max_in_flight: usize,
    #[serde(default)]
    pub args: Vec<String>,
}

fn default_threads() -> usize {
    4
}

fn default_in_flight() -> usize {
    1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Llm {
    #[serde(default)]
    pub program: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_llm_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub provider: LlmProvider,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub probe_args: Option<Vec<String>>,
}

fn default_llm_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LlmProvider {
    #[default]
    Auto,
    Custom,
    Codex,
    Copilot,
}

impl Default for Llm {
    fn default() -> Self {
        Self {
            program: PathBuf::new(),
            args: Vec::new(),
            enabled: false,
            provider: LlmProvider::Auto,
            model: None,
            probe_args: None,
        }
    }
}

impl Llm {
    pub fn preset(provider: LlmProvider) -> Self {
        let program = match provider {
            LlmProvider::Codex => "codex",
            LlmProvider::Copilot => "copilot",
            LlmProvider::Auto | LlmProvider::Custom => "",
        };
        Self {
            program: program.into(),
            enabled: true,
            provider,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub models: Vec<ModelSpec>,
    pub asr: BTreeMap<String, Adapter>,
    pub diarization: Adapter,
    #[serde(default)]
    pub llm: Llm,
    #[serde(default)]
    pub review_rules: Option<PathBuf>,
    #[serde(default)]
    pub resources: Resources,
    #[serde(default)]
    pub quality: crate::quality::Settings,
    #[serde(skip)]
    pub asr_selection: Option<Vec<String>>,
}

pub(crate) fn default_asr_roles() -> Vec<String> {
    ["primeline", "flix", "qwen"].map(str::to_owned).to_vec()
}

pub(crate) fn is_default_asr_roles(roles: &[String]) -> bool {
    roles == default_asr_roles()
}

pub(crate) fn legacy_asr_engine(role: &str) -> Option<&'static str> {
    match role {
        "primeline" => Some("nemo"),
        "flix" => Some("whisper"),
        "qwen" => Some("qwen"),
        _ => None,
    }
}

pub(crate) fn validate_asr_roles(roles: &[String]) -> Result<Vec<String>> {
    ensure!(
        !roles.is_empty(),
        "ASR selection must contain at least one role"
    );
    for (index, role) in roles.iter().enumerate() {
        validate_role_id(role)?;
        ensure!(!roles[..index].contains(role), "duplicate ASR role: {role}");
    }
    let mut ordered = roles.to_vec();
    ordered.sort_by(|left, right| {
        let priority = |role: &str| match role {
            "primeline" => 0,
            "flix" => 1,
            "qwen" => 2,
            _ => 3,
        };
        priority(left).cmp(&priority(right)).then(left.cmp(right))
    });
    Ok(ordered)
}

fn validate_role_id(role: &str) -> Result<()> {
    ensure!(
        !role.is_empty()
            && role.len() <= 64
            && role.as_bytes()[0].is_ascii_alphanumeric()
            && role
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
            && ![
                "diarization",
                "attempts",
                "audio",
                "chunks",
                "clips",
                "reviews",
                "review",
                "exports",
                "export",
                "runtime",
                "models",
                "config",
                "quality",
                "workflow",
                "transcript",
                "input",
                "status",
                "metadata",
                "logs",
                "tmp",
                "temp",
                "bin",
                "projects",
                "asr",
                "manifest",
                "benchmark",
                "resource-events",
                "review-flags",
            ]
            .contains(&role.to_ascii_lowercase().as_str()),
        "invalid or reserved ASR role ID: {role}"
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Resources {
    pub reserve_mib: u64,
    pub max_asr_workers: usize,
    pub max_parallel_asr_models: Option<usize>,
    pub sequential_asr_roles: bool,
    pub estimated_peak_mib: BTreeMap<String, u64>,
    pub asr_worker_counts: BTreeMap<String, usize>,
    pub asr_worker_peak_mib: BTreeMap<String, u64>,
    pub supervise_asr_with_systemd: bool,
    pub systemd_memory_limit_mib: Option<u64>,
    pub systemd_swap_limit_mib: Option<u64>,
    pub oom_retries: usize,
    pub offload_dir: Option<PathBuf>,
    pub offload_roles: Vec<String>,
    pub offload_min_free_mib: u64,
    pub offload_estimated_peak_mib: BTreeMap<String, u64>,
}

impl Default for Resources {
    fn default() -> Self {
        Self {
            reserve_mib: 4096,
            max_asr_workers: 3,
            max_parallel_asr_models: None,
            sequential_asr_roles: false,
            estimated_peak_mib: [
                ("primeline".to_owned(), 6000),
                ("flix".to_owned(), 10500),
                ("qwen".to_owned(), 13500),
            ]
            .into(),
            asr_worker_counts: BTreeMap::new(),
            asr_worker_peak_mib: BTreeMap::new(),
            supervise_asr_with_systemd: false,
            systemd_memory_limit_mib: None,
            systemd_swap_limit_mib: None,
            oom_retries: 2,
            offload_dir: None,
            offload_roles: vec!["flix".to_owned(), "qwen".to_owned()],
            offload_min_free_mib: 8192,
            offload_estimated_peak_mib: [("flix".to_owned(), 6500), ("qwen".to_owned(), 7000)]
                .into(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read config {}", path.display()))?;
        let mut config: Self =
            toml::from_str(&text).with_context(|| format!("invalid config {}", path.display()))?;
        config.quality.validate()?;
        let root = path
            .canonicalize()
            .with_context(|| format!("config path {} is unavailable", path.display()))?
            .parent()
            .context("config has no parent directory")?
            .to_path_buf();
        for model in &mut config.models {
            model.path = resolve(&root, &model.path);
        }
        ensure!(
            !config.asr.is_empty(),
            "config must contain at least one ASR adapter"
        );
        for role in config.asr.keys() {
            validate_role_id(role)?;
            config.asr_engine(role)?;
            ensure!(
                config
                    .resources
                    .estimated_peak_mib
                    .get(role)
                    .is_some_and(|v| *v > 0),
                "resources.estimated_peak_mib must contain a positive value for {role}"
            );
        }
        ensure!(
            config.resources.max_asr_workers > 0,
            "resources.max_asr_workers must be positive"
        );
        ensure!(
            config
                .resources
                .max_parallel_asr_models
                .is_none_or(|limit| limit > 0),
            "resources.max_parallel_asr_models must be positive"
        );
        for (role, &count) in &config.resources.asr_worker_counts {
            if config.asr.contains_key(role) {
                ensure!(
                    (1..=config.resources.max_asr_workers).contains(&count),
                    "resources.asr_worker_counts.{role} must be between 1 and max_asr_workers"
                );
                if count > 1 {
                    ensure!(
                        config
                            .resources
                            .asr_worker_peak_mib
                            .get(role)
                            .is_some_and(|peak| *peak > 0),
                        "resources.asr_worker_peak_mib.{role} must have a measured positive estimate for multiple workers"
                    );
                }
            }
        }
        for (role, &peak) in &config.resources.asr_worker_peak_mib {
            if config.asr.contains_key(role) {
                ensure!(
                    peak > 0,
                    "resources.asr_worker_peak_mib.{role} must be positive"
                );
            }
        }
        ensure!(
            config.resources.oom_retries <= 5,
            "resources.oom_retries must be between 0 and 5"
        );
        ensure!(
            config
                .resources
                .systemd_memory_limit_mib
                .is_none_or(|value| value >= 64 && config.resources.supervise_asr_with_systemd),
            "resources.systemd_memory_limit_mib requires supervise_asr_with_systemd and at least 64 MiB"
        );
        ensure!(
            config.resources.systemd_swap_limit_mib.is_none()
                || config.resources.supervise_asr_with_systemd,
            "resources.systemd_swap_limit_mib requires supervise_asr_with_systemd"
        );
        if let Some(path) = &mut config.resources.offload_dir {
            *path = resolve(&root, path);
            ensure!(
                config.resources.offload_min_free_mib > 0,
                "resources.offload_min_free_mib must be positive"
            );
        }
        for (role, &peak) in &config.resources.offload_estimated_peak_mib {
            if config.asr.contains_key(role) {
                ensure!(
                    peak > 0,
                    "resources.offload_estimated_peak_mib.{role} must be positive"
                );
            }
        }
        for role in &config.resources.offload_roles {
            if let Some(adapter) = config.asr.get(role) {
                ensure!(
                    config
                        .resources
                        .offload_roles
                        .iter()
                        .filter(|candidate| *candidate == role)
                        .count()
                        == 1,
                    "duplicate resources.offload_roles entry: {role}"
                );
                let engine = config.asr_engine(role)?;
                ensure!(
                    matches!(engine, "whisper" | "qwen")
                        && (matches!(role.as_str(), "flix" | "qwen")
                            || adapter.script.ends_with("adapters/asr/main.py")),
                    "resources.offload_roles.{role} requires a whisper/qwen main.py adapter"
                );
                ensure!(
                    config
                        .resources
                        .offload_estimated_peak_mib
                        .get(role)
                        .is_some_and(|peak| *peak > 0),
                    "resources.offload_estimated_peak_mib must contain a positive value for {role}"
                );
            }
        }
        for adapter in config
            .asr
            .values_mut()
            .chain(std::iter::once(&mut config.diarization))
        {
            adapter.program = resolve_command(&root, &adapter.program);
            adapter.script = resolve(&root, &adapter.script);
            ensure!(adapter.threads > 0, "adapter threads must be positive");
            ensure!(
                (1..=4).contains(&adapter.max_in_flight),
                "adapter max_in_flight must be between 1 and 4"
            );
        }
        if config.llm.enabled {
            ensure!(
                !config.llm.program.as_os_str().is_empty(),
                "llm.program must be set when LLM is enabled"
            );
            ensure!(
                config
                    .llm
                    .model
                    .as_ref()
                    .is_none_or(|model| !model.trim().is_empty() && !model.starts_with('-')),
                "llm.model must be a nonempty model name"
            );
            ensure!(
                config.llm.model.is_some()
                    || !config
                        .llm
                        .args
                        .iter()
                        .chain(config.llm.probe_args.iter().flatten())
                        .any(|arg| arg.contains("{model}")),
                "llm.model is required when arguments use {{model}}"
            );
            config.llm.program = resolve_command(&root, &config.llm.program);
        }
        config.review_rules = config.review_rules.map(|path| resolve(&root, &path));
        for role in config
            .asr
            .keys()
            .map(String::as_str)
            .chain(std::iter::once("diarization"))
        {
            ensure!(
                config
                    .models
                    .iter()
                    .filter(|model| model.role == role)
                    .count()
                    == 1,
                "config must contain exactly one model with role {role}"
            );
        }
        for model in &config.models {
            ensure!(
                model.role == "diarization" || config.asr.contains_key(&model.role),
                "model role {} has no configured ASR adapter",
                model.role
            );
        }
        Ok(config)
    }

    pub fn asr_roles(&self) -> Vec<String> {
        let roles = self.asr.keys().cloned().collect::<Vec<_>>();
        validate_asr_roles(&roles).expect("ASR role IDs validated on load")
    }

    pub fn select_asr_roles(&self, selection: Option<&[String]>) -> Result<Vec<String>> {
        let Some(selection) = selection else {
            return Ok(self.asr_roles());
        };
        let roles = validate_asr_roles(selection)?;
        for role in &roles {
            ensure!(
                self.asr.contains_key(role),
                "ASR role {role:?} is not configured"
            );
        }
        Ok(roles)
    }

    pub fn asr_engine(&self, role: &str) -> Result<&str> {
        let adapter = self
            .asr
            .get(role)
            .with_context(|| format!("unknown ASR role: {role}"))?;
        let engine = match adapter.engine.as_deref() {
            Some(engine) => engine,
            None => legacy_asr_engine(role)
                .with_context(|| format!("asr.{role}.engine is required for custom ASR roles"))?,
        };
        ensure!(
            matches!(engine, "nemo" | "whisper" | "qwen"),
            "asr.{role}.engine must be nemo, whisper or qwen"
        );
        Ok(engine)
    }

    pub fn model(&self, role: &str) -> &ModelSpec {
        self.models
            .iter()
            .find(|model| model.role == role)
            .expect("config roles validated on load")
    }

    pub fn asr(&self, role: &str) -> &Adapter {
        self.asr.get(role).expect("ASR roles validated on load")
    }
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn resolve_command(root: &Path, path: &Path) -> PathBuf {
    if path.components().count() == 1 && !path.is_absolute() {
        path.to_path_buf()
    } else {
        resolve(root, path)
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, Llm, LlmProvider, legacy_asr_engine, validate_asr_roles};

    fn load_value(value: &toml::Value) -> anyhow::Result<Config> {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("config.toml");
        std::fs::write(&path, toml::to_string(value).unwrap()).unwrap();
        Config::load(&path)
    }

    fn example_value() -> toml::Value {
        toml::from_str(include_str!("../shum.example.toml")).unwrap()
    }

    #[test]
    fn legacy_engines_resolve_the_same_when_implicit_or_explicit() {
        let implicit = example_value();
        let loaded = load_value(&implicit).unwrap();
        for (role, engine) in [("primeline", "nemo"), ("flix", "whisper"), ("qwen", "qwen")] {
            assert_eq!(legacy_asr_engine(role), Some(engine));
            assert_eq!(loaded.asr_engine(role).unwrap(), engine);
        }
        assert_eq!(legacy_asr_engine("custom"), None);
        let mut explicit = implicit;
        for (role, engine) in [("primeline", "nemo"), ("flix", "whisper"), ("qwen", "qwen")] {
            explicit["asr"][role]
                .as_table_mut()
                .unwrap()
                .insert("engine".into(), engine.into());
        }
        let loaded = load_value(&explicit).unwrap();
        for (role, engine) in [("primeline", "nemo"), ("flix", "whisper"), ("qwen", "qwen")] {
            assert_eq!(loaded.asr_engine(role).unwrap(), engine);
        }
    }

    fn set_resource_entry(value: &mut toml::Value, section: &str, role: &str, entry: toml::Value) {
        value["resources"]
            .as_table_mut()
            .unwrap()
            .entry(section)
            .or_insert_with(|| toml::Value::Table(Default::default()))
            .as_table_mut()
            .unwrap()
            .insert(role.to_owned(), entry);
    }

    #[test]
    fn one_and_five_roles_and_legacy_engine_serialization() {
        let mut one = example_value();
        one["asr"]
            .as_table_mut()
            .unwrap()
            .retain(|role, _| role == "flix");
        one["models"]
            .as_array_mut()
            .unwrap()
            .retain(|model| matches!(model["role"].as_str(), Some("flix" | "diarization")));
        let loaded = load_value(&one).unwrap();
        assert_eq!(loaded.asr_roles(), vec!["flix"]);
        assert_eq!(loaded.select_asr_roles(None).unwrap(), vec!["flix"]);
        assert_eq!(loaded.asr_engine("flix").unwrap(), "whisper");
        assert!(!toml::to_string(&loaded).unwrap().contains("engine ="));

        let flix = one["asr"]["flix"].clone();
        one["asr"]
            .as_table_mut()
            .unwrap()
            .insert("custom_whisper".into(), flix.clone());
        one["asr"]["custom_whisper"]
            .as_table_mut()
            .unwrap()
            .insert("engine".into(), "whisper".into());
        one["asr"]
            .as_table_mut()
            .unwrap()
            .insert("z_nemo".into(), flix);
        one["asr"]["z_nemo"]
            .as_table_mut()
            .unwrap()
            .insert("engine".into(), "nemo".into());
        one["models"].as_array_mut().unwrap().extend([
            toml::Value::try_from(super::ModelSpec {
                role: "custom_whisper".into(),
                repo: "example/whisper".into(),
                revision: "".into(),
                path: "models/custom_whisper".into(),
            })
            .unwrap(),
            toml::Value::try_from(super::ModelSpec {
                role: "z_nemo".into(),
                repo: "example/nemo".into(),
                revision: "".into(),
                path: "models/z_nemo".into(),
            })
            .unwrap(),
        ]);
        let mut five = example_value();
        for role in ["custom_whisper", "z_nemo"] {
            five["asr"]
                .as_table_mut()
                .unwrap()
                .insert(role.into(), one["asr"][role].clone());
            five["models"].as_array_mut().unwrap().push(
                one["models"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|model| model["role"].as_str() == Some(role))
                    .unwrap()
                    .clone(),
            );
            set_resource_entry(&mut five, "estimated_peak_mib", role, 6000.into());
        }
        five["resources"]
            .as_table_mut()
            .unwrap()
            .insert("max_asr_workers".into(), 5.into());
        set_resource_entry(&mut five, "asr_worker_counts", "custom_whisper", 5.into());
        set_resource_entry(
            &mut five,
            "asr_worker_peak_mib",
            "custom_whisper",
            2800.into(),
        );
        five["resources"]["offload_roles"] =
            toml::Value::Array(["custom_whisper".into()].into_iter().collect());
        set_resource_entry(
            &mut five,
            "offload_estimated_peak_mib",
            "custom_whisper",
            3000.into(),
        );
        let loaded = load_value(&five).unwrap();
        assert_eq!(
            loaded.asr_roles(),
            ["primeline", "flix", "qwen", "custom_whisper", "z_nemo"]
        );
        assert_eq!(loaded.asr_engine("z_nemo").unwrap(), "nemo");
        assert_eq!(loaded.asr_engine("custom_whisper").unwrap(), "whisper");
        assert_eq!(
            loaded
                .select_asr_roles(Some(&["z_nemo".into(), "flix".into()]))
                .unwrap(),
            ["flix", "z_nemo"]
        );
        assert!(loaded.select_asr_roles(Some(&[])).is_err());
        assert!(loaded.select_asr_roles(Some(&["absent".into()])).is_err());
        assert!(
            loaded
                .select_asr_roles(Some(&["flix".into(), "flix".into()]))
                .is_err()
        );
        assert!(loaded.asr_engine("unknown").is_err());
        assert_eq!(loaded.resources.asr_worker_counts["custom_whisper"], 5);
        five["asr"]["custom_whisper"]["script"] = "adapters/asr/custom_worker.py".into();
        assert!(
            load_value(&five)
                .unwrap_err()
                .to_string()
                .contains("main.py adapter")
        );
    }

    #[test]
    fn rejects_missing_engine_unmatched_models_and_unsafe_ids() {
        let original = example_value();
        let mut custom = original.clone();
        let flix = custom["asr"]["flix"].clone();
        custom["asr"]
            .as_table_mut()
            .unwrap()
            .insert("more".into(), flix);
        assert!(
            load_value(&custom)
                .unwrap_err()
                .to_string()
                .contains("engine is required")
        );
        custom["asr"]["more"]
            .as_table_mut()
            .unwrap()
            .insert("engine".into(), "other".into());
        assert!(
            load_value(&custom)
                .unwrap_err()
                .to_string()
                .contains("must be nemo")
        );
        let mut extra = original.clone();
        let duplicate = extra["models"][0].clone();
        extra["models"].as_array_mut().unwrap().push(duplicate);
        assert!(
            load_value(&extra)
                .unwrap_err()
                .to_string()
                .contains("exactly one")
        );
        extra["models"].as_array_mut().unwrap().last_mut().unwrap()["role"] = "orphan".into();
        assert!(
            load_value(&extra)
                .unwrap_err()
                .to_string()
                .contains("no configured ASR adapter")
        );
        let mut empty = original.clone();
        empty["asr"].as_table_mut().unwrap().clear();
        assert!(
            load_value(&empty)
                .unwrap_err()
                .to_string()
                .contains("at least one")
        );
        let mut missing = original.clone();
        missing["models"]
            .as_array_mut()
            .unwrap()
            .retain(|model| model["role"].as_str() != Some("flix"));
        assert!(
            load_value(&missing)
                .unwrap_err()
                .to_string()
                .contains("exactly one model with role flix")
        );
        let mut no_models = original.clone();
        no_models["models"].as_array_mut().unwrap().clear();
        assert!(
            load_value(&no_models)
                .unwrap_err()
                .to_string()
                .contains("exactly one model")
        );
        let mut unsafe_id = original.clone();
        let flix = unsafe_id["asr"]["flix"].clone();
        unsafe_id["asr"]
            .as_table_mut()
            .unwrap()
            .insert("clips".into(), flix);
        assert!(
            load_value(&unsafe_id)
                .unwrap_err()
                .to_string()
                .contains("reserved ASR role ID")
        );
        for role in [
            "clips",
            "attempts",
            "manifest",
            "benchmark",
            "resource-events",
            "review-flags",
            "diarization",
            "../bad",
            ".hidden",
            "a/b",
            "-flag",
            "_hidden",
            "é",
            "",
        ] {
            assert!(validate_asr_roles(&[role.to_owned()]).is_err(), "{role}");
        }
        assert!(validate_asr_roles(&["a".repeat(65)]).is_err());
        assert!(validate_asr_roles(&["a".repeat(64)]).is_ok());
        assert!(validate_asr_roles(&["more".into(), "more".into()]).is_err());
        assert_eq!(
            validate_asr_roles(&["z".into(), "flix".into(), "a".into()]).unwrap(),
            ["flix", "a", "z"]
        );
    }

    #[test]
    fn inactive_resource_history_is_retained_but_active_entries_are_validated() {
        let mut value = example_value();
        value["asr"]
            .as_table_mut()
            .unwrap()
            .retain(|role, _| role == "flix");
        value["models"]
            .as_array_mut()
            .unwrap()
            .retain(|model| matches!(model["role"].as_str(), Some("flix" | "diarization")));
        set_resource_entry(&mut value, "estimated_peak_mib", "retired", 0.into());
        set_resource_entry(&mut value, "asr_worker_counts", "retired", 999.into());
        set_resource_entry(&mut value, "asr_worker_peak_mib", "retired", 0.into());
        set_resource_entry(
            &mut value,
            "offload_estimated_peak_mib",
            "retired",
            0.into(),
        );
        value["resources"]["offload_roles"] =
            toml::Value::Array(["retired".into(), "flix".into()].into_iter().collect());
        let loaded = load_value(&value).unwrap();
        assert_eq!(loaded.resources.asr_worker_counts["retired"], 999);
        assert!(loaded.resources.offload_roles.contains(&"retired".into()));
        value["resources"]["estimated_peak_mib"]["flix"] = 0.into();
        assert!(load_value(&value).unwrap_err().to_string().contains("flix"));
        value["resources"]["estimated_peak_mib"]["flix"] = 10500.into();
        set_resource_entry(&mut value, "asr_worker_counts", "flix", 2.into());
        assert!(
            load_value(&value)
                .unwrap_err()
                .to_string()
                .contains("asr_worker_peak_mib.flix")
        );
        set_resource_entry(&mut value, "asr_worker_peak_mib", "flix", 2600.into());
        value["resources"]["offload_estimated_peak_mib"]["flix"] = 0.into();
        assert!(
            load_value(&value)
                .unwrap_err()
                .to_string()
                .contains("offload_estimated_peak_mib.flix")
        );
    }

    #[test]
    fn optional_model_limit_must_be_positive() {
        let mut value = example_value();
        assert_eq!(
            load_value(&value)
                .unwrap()
                .resources
                .max_parallel_asr_models,
            None
        );
        value["resources"]
            .as_table_mut()
            .unwrap()
            .insert("max_parallel_asr_models".into(), 5.into());
        assert_eq!(
            load_value(&value)
                .unwrap()
                .resources
                .max_parallel_asr_models,
            Some(5)
        );
        value["resources"]["max_parallel_asr_models"] = 0.into();
        assert!(
            load_value(&value)
                .unwrap_err()
                .to_string()
                .contains("max_parallel_asr_models must be positive")
        );
    }

    #[test]
    fn old_llm_configs_remain_enabled_and_absent_llm_is_disabled() {
        let old: Llm = toml::from_str("program = 'legacy-cli'\nargs = ['{output}']").unwrap();
        assert!(old.enabled);
        assert_eq!(old.provider, LlmProvider::Auto);
        assert_eq!(old.model, None);
        assert_eq!(old.probe_args, None);
        assert!(!Llm::default().enabled);
        let preset = Llm::preset(LlmProvider::Copilot);
        assert_eq!(preset.program, std::path::PathBuf::from("copilot"));
        assert!(preset.enabled);
    }

    #[test]
    fn disabled_llm_requires_no_cli() {
        let root = tempfile::Builder::new()
            .prefix("shum-config-test-")
            .tempdir_in(".")
            .unwrap();
        let config = root.path().join("shum.toml");
        let example = include_str!("../shum.example.toml");
        let without_llm = example
            .lines()
            .take_while(|line| *line != "[llm]")
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&config, without_llm).unwrap();
        let loaded = Config::load(&config).unwrap();
        assert!(!loaded.llm.enabled);
        assert!(loaded.llm.program.as_os_str().is_empty());
        std::fs::write(
            &config,
            example.replace(
                "program = \"codex\"\nargs = [\"exec\", \"--sandbox\", \"read-only\", \"--skip-git-repo-check\", \"--ephemeral\", \"--output-last-message\", \"{output}\", \"-\"]",
                "enabled = false",
            ),
        )
        .unwrap();
        let loaded = Config::load(&config).unwrap();
        assert!(!loaded.llm.enabled);
    }

    #[test]
    fn parallel_roles_require_measured_per_worker_memory() {
        let root = tempfile::tempdir().unwrap();
        let config = root.path().join("shum.toml");
        let example = include_str!("../shum.example.toml");
        std::fs::write(&config, example).unwrap();
        assert!(Config::load(&config).is_ok());
        std::fs::write(
            &config,
            example.replace(
                "[resources.offload_estimated_peak_mib]",
                "[resources.asr_worker_counts]\nflix = 3\n\n[resources.offload_estimated_peak_mib]",
            ),
        )
        .unwrap();
        let error = Config::load(&config).unwrap_err();
        assert!(error.to_string().contains("asr_worker_peak_mib.flix"));
        std::fs::write(
            &config,
            example.replace(
                "[resources.offload_estimated_peak_mib]",
                "[resources.asr_worker_counts]\nflix = 3\n\n[resources.asr_worker_peak_mib]\nflix = 2600\n\n[resources.offload_estimated_peak_mib]",
            ),
        )
        .unwrap();
        assert_eq!(
            Config::load(&config).unwrap().resources.asr_worker_counts["flix"],
            3
        );
    }

    #[test]
    fn zero_swap_cap_requires_supervision_and_old_configs_keep_defaults() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("shum.toml");
        let example = include_str!("../shum.example.toml");
        let old = example.replace("sequential_asr_roles = false\n", "");
        std::fs::write(&path, &old).unwrap();
        let loaded = Config::load(&path).unwrap();
        assert!(!loaded.resources.sequential_asr_roles);
        assert_eq!(loaded.resources.systemd_swap_limit_mib, None);
        let capped = old.replace(
            "supervise_asr_with_systemd = false",
            "supervise_asr_with_systemd = false\nsystemd_swap_limit_mib = 0",
        );
        std::fs::write(&path, &capped).unwrap();
        assert!(
            Config::load(&path)
                .unwrap_err()
                .to_string()
                .contains("systemd_swap_limit_mib requires")
        );
        std::fs::write(
            &path,
            capped.replace(
                "supervise_asr_with_systemd = false",
                "supervise_asr_with_systemd = true",
            ),
        )
        .unwrap();
        assert_eq!(
            Config::load(&path)
                .unwrap()
                .resources
                .systemd_swap_limit_mib,
            Some(0)
        );
    }
}
