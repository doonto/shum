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
}

fn default_threads() -> usize {
    4
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
        }
        for adapter in config
            .asr
            .values_mut()
            .chain(std::iter::once(&mut config.diarization))
        {
            adapter.program = resolve_command(&root, &adapter.program);
            adapter.script = resolve(&root, &adapter.script);
            ensure!(adapter.threads > 0, "adapter threads must be positive");
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
