use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::models::ModelSpec;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adapter {
    pub program: PathBuf,
    pub script: PathBuf,
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
    pub program: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub models: Vec<ModelSpec>,
    pub asr: BTreeMap<String, Adapter>,
    pub diarization: Adapter,
    pub llm: Llm,
    #[serde(default)]
    pub review_rules: Option<PathBuf>,
    #[serde(default)]
    pub resources: Resources,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Resources {
    pub reserve_mib: u64,
    pub max_asr_workers: usize,
    pub estimated_peak_mib: BTreeMap<String, u64>,
    pub asr_worker_counts: BTreeMap<String, usize>,
    pub asr_worker_peak_mib: BTreeMap<String, u64>,
    pub supervise_asr_with_systemd: bool,
    pub systemd_memory_limit_mib: Option<u64>,
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
        let root = path
            .canonicalize()
            .with_context(|| format!("config path {} is unavailable", path.display()))?
            .parent()
            .context("config has no parent directory")?
            .to_path_buf();
        for model in &mut config.models {
            model.path = resolve(&root, &model.path);
        }
        for role in ["primeline", "flix", "qwen"] {
            ensure!(
                config.asr.contains_key(role),
                "config has no ASR adapter for role {role}"
            );
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
            (1..=3).contains(&config.resources.max_asr_workers),
            "resources.max_asr_workers must be between 1 and 3"
        );
        for (role, &count) in &config.resources.asr_worker_counts {
            ensure!(
                ["primeline", "flix", "qwen"].contains(&role.as_str()),
                "unknown ASR role in resources.asr_worker_counts: {role}"
            );
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
        for (role, &peak) in &config.resources.asr_worker_peak_mib {
            ensure!(
                ["primeline", "flix", "qwen"].contains(&role.as_str()) && peak > 0,
                "resources.asr_worker_peak_mib contains an unknown role or zero estimate: {role}"
            );
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
        if let Some(path) = &mut config.resources.offload_dir {
            *path = resolve(&root, path);
            ensure!(
                config.resources.offload_min_free_mib > 0,
                "resources.offload_min_free_mib must be positive"
            );
        }
        for role in &config.resources.offload_roles {
            ensure!(
                ["flix", "qwen"].contains(&role.as_str()),
                "resources.offload_roles can only include flix and qwen"
            );
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
            ensure!(
                config
                    .resources
                    .offload_estimated_peak_mib
                    .get(role)
                    .is_some_and(|peak| *peak > 0),
                "resources.offload_estimated_peak_mib must contain a positive value for {role}"
            );
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
        config.llm.program = resolve_command(&root, &config.llm.program);
        config.review_rules = config.review_rules.map(|path| resolve(&root, &path));
        for role in ["primeline", "flix", "qwen", "diarization"] {
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
        Ok(config)
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
    use super::Config;

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
}
