use std::env;
use std::fs::{self, DirBuilder, OpenOptions};
use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::models::{self, ModelSpec};

#[path = "installation.rs"]
mod installation;

const SETTINGS_VERSION: u32 = 2;
const METADATA: &str = "shum.installation.toml";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub version: u32,
    pub storage_dir: PathBuf,
    #[serde(default)]
    pub installation_dir: Option<PathBuf>,
    #[serde(default)]
    pub bin_dir: Option<PathBuf>,
    #[serde(default)]
    pub config_dir: Option<PathBuf>,
    #[serde(default)]
    pub projects_dir: Option<PathBuf>,
    #[serde(default)]
    pub models_dir: Option<PathBuf>,
}

impl Settings {
    pub fn projects_dir(&self) -> PathBuf {
        self.projects_dir
            .clone()
            .unwrap_or_else(|| self.storage_dir.join("projects"))
    }

    pub fn models_dir(&self) -> PathBuf {
        self.models_dir
            .clone()
            .unwrap_or_else(|| self.storage_dir.join("models"))
    }

    pub fn config_dir(&self) -> Result<PathBuf> {
        self.config_dir.clone().map(Ok).unwrap_or_else(config_dir)
    }

    pub fn runtime_dir(&self) -> Result<PathBuf> {
        installation::runtime_dir(
            self.installation_dir
                .as_deref()
                .unwrap_or(&self.storage_dir),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Locator {
    version: u32,
    settings: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Locations {
    pub installation: PathBuf,
    pub bin: PathBuf,
    pub config: PathBuf,
    pub projects: PathBuf,
    pub models: PathBuf,
}

#[derive(Debug, Default, Clone)]
pub struct OnboardOptions {
    pub install_dir: Option<PathBuf>,
    pub bin_dir: Option<PathBuf>,
    pub settings_dir: Option<PathBuf>,
    pub projects_dir: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
    pub installation_locator: Option<PathBuf>,
    pub command_dir: Option<PathBuf>,
    pub setup_shell: bool,
}

impl Locations {
    pub fn under(root: &Path) -> Self {
        Self {
            installation: root.to_path_buf(),
            bin: root.join("bin"),
            config: root.join("config"),
            projects: root.join("projects"),
            models: root.join("models"),
        }
    }

    fn normalize(&self) -> Result<Self> {
        Ok(Self {
            installation: canonical_target(&self.installation)?,
            bin: canonical_target(&self.bin)?,
            config: canonical_target(&self.config)?,
            projects: canonical_target(&self.projects)?,
            models: canonical_target(&self.models)?,
        })
    }

    fn validate(&self) -> Result<()> {
        let dirs = [&self.bin, &self.config, &self.projects, &self.models];
        for (i, left) in dirs.iter().enumerate() {
            ensure!(
                *left != &self.installation,
                "select a separate directory for each installation component"
            );
            for right in dirs.iter().skip(i + 1) {
                ensure!(
                    !left.starts_with(right) && !right.starts_with(left),
                    "installation directories overlap: {} and {}",
                    left.display(),
                    right.display()
                );
            }
        }
        for dir in [
            &self.installation,
            &self.bin,
            &self.config,
            &self.projects,
            &self.models,
        ] {
            ensure_not_source_tree(dir)?;
            ensure!(
                !dir.is_file(),
                "{} is a file, not a directory",
                dir.display()
            );
        }
        ensure!(
            !self.installation.starts_with(&self.projects)
                && !self.installation.starts_with(&self.models)
                && !self.installation.starts_with(&self.config)
                && !self.installation.starts_with(&self.bin),
            "the installation root cannot be inside a component directory"
        );
        let runtime = self.installation.join("runtime");
        for dir in dirs {
            ensure!(
                !dir.starts_with(&runtime) && !runtime.starts_with(dir),
                "component directory {} overlaps managed runtime",
                dir.display()
            );
        }
        Ok(())
    }
}

pub(crate) fn ensure_not_source_tree(path: &Path) -> Result<()> {
    for parent in path.ancestors() {
        ensure!(
            !(parent.join("Cargo.toml").is_file()
                && parent.join("shum.example.toml").is_file()
                && parent.join("src/main.rs").is_file()),
            "{} is inside the Shum source checkout; choose a separate installation directory",
            path.display()
        );
    }
    Ok(())
}

pub fn config_dir() -> Result<PathBuf> {
    if let Some(xdg) = env::var_os("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        let path = PathBuf::from(xdg);
        ensure!(
            path.is_absolute(),
            "XDG_CONFIG_HOME must be an absolute path"
        );
        return Ok(path.join("shum"));
    }
    let home = env::var_os("HOME").context("HOME is unset; cannot locate Shum settings")?;
    Ok(PathBuf::from(home).join(".config/shum"))
}

pub fn settings_path() -> Result<PathBuf> {
    if let Some(explicit) = env::var_os("SHUM_SETTINGS") {
        let path = PathBuf::from(explicit);
        ensure!(path.is_absolute(), "SHUM_SETTINGS must be absolute");
        return Ok(path);
    }
    if let Some(exe_path) = adjacent_metadata_path()? {
        return Ok(exe_path);
    }
    let xdg = config_dir()?;
    let locator = xdg.join("installation.toml");
    if locator.exists() {
        let locator: Locator = toml::from_str(&fs::read_to_string(&locator)?)
            .with_context(|| format!("invalid locator {}", locator.display()))?;
        ensure!(
            locator.version == 1 && locator.settings.is_absolute(),
            "invalid Shum locator"
        );
        return Ok(locator.settings);
    }
    Ok(xdg.join("settings.toml"))
}

fn adjacent_metadata_path() -> Result<Option<PathBuf>> {
    let exe = env::current_exe()?;
    let path = exe
        .parent()
        .context("executable has no parent")?
        .join(METADATA);
    if !path.is_file() {
        return Ok(None);
    }
    let locator: Locator = toml::from_str(&fs::read_to_string(&path)?)
        .with_context(|| format!("invalid installation metadata {}", path.display()))?;
    ensure!(
        locator.version == 1 && locator.settings.is_absolute(),
        "invalid installation metadata"
    );
    Ok(Some(locator.settings))
}

pub fn config_path_for(settings: &Settings) -> Result<PathBuf> {
    Ok(settings.config_dir()?.join("config.toml"))
}

pub fn load() -> Result<Settings> {
    read_settings(&settings_path()?)
}

pub fn load_from(locator: Option<&Path>, explicit_config: Option<&Path>) -> Result<Settings> {
    if locator.is_none() && explicit_config.is_none() {
        return load();
    }
    read_settings(&selected_settings_path(locator, explicit_config)?)
}

fn selected_settings_path(
    locator: Option<&Path>,
    explicit_config: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(locator) = locator {
        return settings_path_from(locator);
    }
    if let Some(config) = explicit_config {
        let parent = config.parent().context("config path has no parent")?;
        let nearby = parent.join("settings.toml");
        if nearby.is_file() {
            return Ok(nearby);
        }
    }
    settings_path()
}

pub fn installed_binary_from(
    locator: Option<&Path>,
    explicit_config: Option<&Path>,
) -> Result<PathBuf> {
    let settings = read_settings_for_setup(&selected_settings_path(locator, explicit_config)?)?;
    Ok(settings
        .bin_dir
        .unwrap_or_else(|| settings.storage_dir.join("bin"))
        .join("shum"))
}

pub fn settings_path_from(locator: &Path) -> Result<PathBuf> {
    let path = if locator.is_dir() {
        [
            locator.join("config/settings.toml"),
            locator.join("settings.toml"),
            locator.join("bin").join(METADATA),
        ]
        .into_iter()
        .find(|path| path.is_file())
        .with_context(|| format!("no Shum installation metadata in {}", locator.display()))?
    } else {
        locator.to_path_buf()
    };
    let text = fs::read_to_string(&path)
        .with_context(|| format!("cannot read installation locator {}", path.display()))?;
    if let Ok(pointer) = toml::from_str::<Locator>(&text) {
        ensure!(
            pointer.version == 1 && pointer.settings.is_absolute(),
            "invalid installation locator"
        );
        return Ok(pointer.settings);
    }
    Ok(path)
}

fn read_settings(path: &Path) -> Result<Settings> {
    let settings = read_settings_for_setup(path)?;
    for dir in [
        settings.storage_dir.clone(),
        settings.projects_dir(),
        settings.models_dir(),
        settings.config_dir()?,
    ] {
        ensure!(
            dir.is_dir(),
            "directory {} is missing; run `shum --onboarding`",
            dir.display()
        );
    }
    if settings.version == SETTINGS_VERSION {
        ensure!(
            installation::runtime_is_complete(&settings.runtime_dir()?)?,
            "runtime resources are missing or damaged; run `shum --onboarding` to repair the installation"
        );
    }
    Ok(settings)
}

fn read_settings_for_setup(path: &Path) -> Result<Settings> {
    let text = fs::read_to_string(path).with_context(|| {
        format!(
            "Shum is not configured at {}; run `shum --onboarding`",
            path.display()
        )
    })?;
    let settings: Settings =
        toml::from_str(&text).with_context(|| format!("invalid settings {}", path.display()))?;
    ensure!(
        settings.version == 1 || settings.version == SETTINGS_VERSION,
        "unsupported Shum settings version {}",
        settings.version
    );
    for dir in [
        settings.storage_dir.clone(),
        settings.projects_dir(),
        settings.models_dir(),
        settings.config_dir()?,
    ] {
        ensure!(
            dir.is_absolute(),
            "directory {} must be absolute",
            dir.display()
        );
    }
    Ok(settings)
}

fn expand_home(path: &Path) -> Result<PathBuf> {
    if path == Path::new("~") || path.starts_with("~/") {
        let home = env::var_os("HOME").context("HOME is unset; cannot expand ~")?;
        return Ok(PathBuf::from(home).join(path.strip_prefix("~")?));
    }
    Ok(path.to_path_buf())
}

pub(crate) fn canonical_target(path: &Path) -> Result<PathBuf> {
    let path = expand_home(path)?;
    let path = if path.is_absolute() {
        path
    } else {
        env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for part in path.components() {
        match part {
            Component::RootDir | Component::Prefix(_) | Component::Normal(_) => {
                normalized.push(part.as_os_str())
            }
            Component::CurDir => {}
            Component::ParentDir => {
                ensure!(normalized.pop(), "path escapes filesystem root");
            }
        }
    }
    let mut current = normalized.as_path();
    let mut missing = Vec::new();
    while !current.exists() {
        missing.push(
            current
                .file_name()
                .context("path has no existing ancestor")?
                .to_os_string(),
        );
        current = current.parent().context("path has no parent")?;
    }
    let mut resolved = current.canonicalize()?;
    for part in missing.into_iter().rev() {
        resolved.push(part);
    }
    Ok(resolved)
}

fn private_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        DirBuilder::new().recursive(true).mode(0o700).create(path)?;
    }
    #[cfg(not(unix))]
    fs::create_dir_all(path)?;
    Ok(())
}

fn check_writable(path: &Path) -> Result<()> {
    let mut ancestor = path;
    while !ancestor.exists() {
        ancestor = ancestor
            .parent()
            .context("directory has no existing ancestor")?;
    }
    ensure!(
        ancestor.is_dir(),
        "{} is not a directory",
        ancestor.display()
    );
    let probe = ancestor.join(format!(".shum-write-check-{}", std::process::id()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .with_context(|| format!("directory {} is not writable", ancestor.display()))?;
    drop(file);
    fs::remove_file(probe)?;
    Ok(())
}

pub(crate) fn atomic_private_write(path: &Path, text: &str) -> Result<()> {
    private_dir(path.parent().context("path has no parent")?)?;
    let temporary = path.with_extension(format!("pending-{}", std::process::id()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result.with_context(|| format!("cannot save {}", path.display()))
}

pub(crate) fn backup_file(path: &Path) -> Result<PathBuf> {
    let mut from = fs::File::open(path)?;
    let mut backup = None;
    let mut to = None;
    for number in 0..1000 {
        let candidate = path.with_extension(format!(
            "{}.bak-{}{}",
            path.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("file"),
            std::process::id(),
            if number == 0 {
                String::new()
            } else {
                format!("-{number}")
            }
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&candidate) {
            Ok(file) => {
                backup = Some(candidate);
                to = Some(file);
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error).context(format!("cannot back up {}", path.display())),
        }
    }
    let backup = backup.context("too many existing config backups; remove old backups manually")?;
    let mut to = to.context("backup file not created")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        to.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    let result = (|| -> Result<()> {
        io::copy(&mut from, &mut to)?;
        to.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&backup);
    }
    result?;
    Ok(backup)
}

pub fn save_config(path: &Path, config: &Config) -> Result<()> {
    atomic_private_write(path, &toml::to_string_pretty(config)?)
}

fn rebase(path: &Path, old: &Path, new: &Path) -> PathBuf {
    path.strip_prefix(old)
        .map_or_else(|_| path.to_path_buf(), |suffix| new.join(suffix))
}

fn managed_python(program: &Path) -> bool {
    let program = program.to_string_lossy();
    program.ends_with("/bin/python")
        && (program.contains("/.venv") || program.contains("/runtime/environments/"))
}

fn bundled_script<'a>(
    script: &Path,
    directory: &str,
    names: &'a [&str],
    roots: &[&Path],
    template_dir: Option<&Path>,
) -> Result<Option<&'a str>> {
    let Some(name) = names
        .iter()
        .copied()
        .find(|name| script.ends_with(Path::new(directory).join(name)))
    else {
        return Ok(None);
    };
    let known_path = !script.is_absolute()
        || template_dir.is_some_and(|dir| script == dir.join(directory).join(name))
        || roots
            .iter()
            .any(|root| script.starts_with(root.join("runtime")));
    if !known_path {
        return Ok(None);
    }
    if script.is_file() {
        let contents = fs::read(script)
            .with_context(|| format!("cannot inspect adapter {}", script.display()))?;
        let Some(bundled_contents) = installation::bundled_file(&format!("{directory}/{name}"))
        else {
            return Ok(None);
        };
        if contents != bundled_contents.as_bytes() {
            return Ok(None);
        }
    }
    Ok(Some(name))
}

fn migrate_data(source: &Path, destination: &Path) -> Result<()> {
    if source == destination || !source.exists() {
        return Ok(());
    }
    ensure!(
        !destination.exists(),
        "cannot copy {}: {} already exists; no files were overwritten",
        source.display(),
        destination.display()
    );
    ensure!(
        !destination.starts_with(source) && !source.starts_with(destination),
        "data directories overlap"
    );
    private_dir(destination.parent().context("destination has no parent")?)?;
    let required = directory_bytes(source)?;
    let available = fs2::available_space(destination.parent().unwrap())?;
    ensure!(
        available >= required,
        "not enough free space to copy {} to {} ({} needed, {} available)",
        source.display(),
        destination.display(),
        required,
        available
    );
    let pending = destination.with_extension(format!("pending-{}", std::process::id()));
    ensure!(
        !pending.exists(),
        "staging directory {} already exists",
        pending.display()
    );
    let result = (|| -> Result<()> {
        copy_verified(source, &pending)?;
        fs::rename(&pending, destination)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&pending);
    }
    result
}

fn directory_bytes(dir: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let size = if kind.is_dir() {
            directory_bytes(&entry.path())?
        } else if kind.is_file() {
            entry.metadata()?.len()
        } else {
            anyhow::bail!(
                "unsupported symlink or special file in {}",
                entry.path().display()
            );
        };
        total = total.checked_add(size).context("directory size overflow")?;
    }
    Ok(total)
}

fn preflight_copy(source: &Path, destination: &Path) -> Result<u64> {
    if canonical_target(source)? == canonical_target(destination)? || !source.exists() {
        return Ok(0);
    }
    ensure!(
        !destination.exists(),
        "cannot copy {}: {} already exists",
        source.display(),
        destination.display()
    );
    let bytes = directory_bytes(source)?;
    let mut parent = destination.parent().context("destination has no parent")?;
    while !parent.exists() {
        parent = parent
            .parent()
            .context("destination has no existing ancestor")?;
    }
    ensure!(
        fs2::available_space(parent)? >= bytes,
        "not enough free space for {}",
        destination.display()
    );
    Ok(bytes)
}

fn copy_verified(source: &Path, destination: &Path) -> Result<()> {
    ensure!(source.is_dir(), "{} is not a directory", source.display());
    private_dir(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        let target = destination.join(name);
        let kind = entry.file_type()?;
        if kind.is_dir() {
            copy_verified(&entry.path(), &target)?;
        } else if kind.is_file() {
            fs::copy(entry.path(), &target)?;
            let hash = |path: &Path| -> Result<Vec<u8>> {
                let mut digest = Sha256::new();
                let mut file = fs::File::open(path)?;
                let mut buffer = [0u8; 65536];
                loop {
                    let count = file.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    digest.update(&buffer[..count]);
                }
                Ok(digest.finalize().to_vec())
            };
            ensure!(
                hash(&entry.path())? == hash(&target)?,
                "copy verification failed: {}",
                entry.path().display()
            );
        } else {
            anyhow::bail!(
                "unsupported symlink or special file in {} (copy manually)",
                entry.path().display()
            );
        }
    }
    Ok(())
}

fn prompt_path(label: &str, fallback: &Path) -> Result<PathBuf> {
    print!("{label} [{}]: ", fallback.display());
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().lock().read_line(&mut answer)?;
    Ok(if answer.trim().is_empty() {
        fallback.to_path_buf()
    } else {
        PathBuf::from(answer.trim())
    })
}

enum ModelChoice {
    Skip,
    Latest,
    Pinned(String),
}

fn model_choice(answer: &str, current_revision: &str) -> Result<ModelChoice> {
    let answer = answer.trim();
    if answer.is_empty() || answer.eq_ignore_ascii_case("skip") {
        return Ok(ModelChoice::Skip);
    }
    if answer.eq_ignore_ascii_case("latest") {
        return Ok(ModelChoice::Latest);
    }
    let revision = if answer.eq_ignore_ascii_case("pinned") {
        current_revision
    } else {
        answer
            .strip_prefix("pin ")
            .context("choose skip, latest, pinned, or pin REVISION")?
            .trim()
    };
    ensure!(
        !revision.is_empty()
            && !revision.starts_with('.')
            && !revision.contains("..")
            && revision
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.'),
        "use a pinned revision without path separators (e.g. a commit hash)"
    );
    Ok(ModelChoice::Pinned(revision.to_owned()))
}

pub fn configure_models_at(config: &mut Config, path: &Path) -> Result<()> {
    configure_models_with_save(config, Some(path))
}

fn production_model(config: &Config, spec: &ModelSpec) -> Option<&'static str> {
    if spec.role == "diarization" {
        return config
            .diarization
            .script
            .ends_with("adapters/diarization/main.py")
            .then_some("pyannote Community-1");
    }
    let adapter = config.asr.get(&spec.role)?;
    if !adapter.script.ends_with("adapters/asr/main.py") {
        return None;
    }
    match config.asr_engine(&spec.role).ok()? {
        "nemo" => Some("NeMo"),
        "whisper" => Some("Whisper/Transformers"),
        "qwen" => Some("Qwen3-ASR"),
        _ => None,
    }
}

fn configure_models_from_disk(settings: &Settings) -> Result<()> {
    if !io::stdin().is_terminal() {
        return Ok(());
    }
    let path = config_path_for(settings)?;
    let mut config = Config::load(&path)?;
    configure_models_with_save(&mut config, Some(&path))
}

fn configure_models_with_save(config: &mut Config, save_path: Option<&Path>) -> Result<()> {
    if !io::stdin().is_terminal() {
        return Ok(());
    }
    println!("Model setup: each download requires an explicit choice; Enter skips.");
    for index in 0..config.models.len() {
        let spec = &config.models[index];
        let Some(engine) = production_model(config, spec) else {
            println!(
                "{}: custom/experimental repository or adapter kept unchanged; use a separately tested adapter to install models manually.",
                spec.role
            );
            continue;
        };
        println!(
            "{} ({}): {} (revision {}, {})",
            spec.role,
            engine,
            spec.repo,
            if spec.revision.is_empty() {
                "unconfigured"
            } else {
                &spec.revision
            },
            if spec.path.is_dir() {
                "directory present; integrity not verified"
            } else {
                "not installed"
            },
        );
        print!("  [Enter] keep/skip, latest, pinned, or pin REVISION: ");
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().lock().read_line(&mut answer)?;
        let mut selected = spec.clone();
        let candidate: ModelSpec = match model_choice(&answer, &selected.revision)? {
            ModelChoice::Skip => continue,
            ModelChoice::Latest => {
                println!("Downloading the latest revision of {}...", selected.repo);
                models::update(&selected)?
            }
            ModelChoice::Pinned(revision) => {
                selected.revision = revision;
                println!(
                    "Downloading pinned revision {} of {}...",
                    selected.revision, selected.repo
                );
                models::install_pinned(&selected)?
            }
        };
        config.models[index] = candidate;
        if let Some(path) = save_path {
            save_config(path, config)?;
        }
    }
    Ok(())
}

pub fn onboard_with(options: &OnboardOptions) -> Result<Settings> {
    let isolated_layout = options.installation_locator.is_none()
        && env::var_os("SHUM_SETTINGS").is_none()
        && options.install_dir.is_some()
        && options.bin_dir.is_some()
        && options.settings_dir.is_some()
        && options.projects_dir.is_some()
        && options.models_dir.is_some();
    let path = if isolated_layout {
        canonical_target(options.settings_dir.as_ref().unwrap())?.join("settings.toml")
    } else if let Some(locator) = &options.installation_locator {
        settings_path_from(locator)?
    } else {
        settings_path().or_else(|error| {
            options
                .install_dir
                .as_ref()
                .map(|root| root.join("config/settings.toml"))
                .context(format!("cannot locate Shum settings: {error:#}"))
        })?
    };
    if isolated_layout && let Ok(metadata) = fs::symlink_metadata(&path) {
        ensure!(
            metadata.file_type().is_file(),
            "settings {} is not a regular file; refusing to overwrite it",
            path.display()
        );
    }
    let previous = if !path.is_file() {
        None
    } else {
        match read_settings_for_setup(&path) {
            Ok(settings) => Some(settings),
            Err(error) => {
                ensure!(
                    io::stdin().is_terminal(),
                    "invalid settings {}: {error:#}; interactive repair is required",
                    path.display()
                );
                eprintln!("Invalid settings at {}: {error:#}", path.display());
                print!("Back up these settings and repair the installation? [y/N]: ");
                io::stdout().flush()?;
                let mut answer = String::new();
                io::stdin().lock().read_line(&mut answer)?;
                ensure!(
                    answer.trim().eq_ignore_ascii_case("y"),
                    "settings were not changed"
                );
                let raw = fs::read_to_string(&path)?;
                let salvage = toml::from_str::<Settings>(&raw).ok();
                let backup = backup_file(&path)?;
                eprintln!("Settings backup: {}", backup.display());
                salvage
            }
        }
    };
    let inferred_root = if previous.is_none() && path.is_file() {
        path.parent()
            .filter(|parent| parent.file_name().is_some_and(|name| name == "config"))
            .and_then(Path::parent)
            .map(Path::to_path_buf)
    } else {
        None
    };
    let fallback = options
        .install_dir
        .clone()
        .or_else(|| previous.as_ref().and_then(|s| s.installation_dir.clone()))
        .or(inferred_root)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("HOME").unwrap_or_else(|| ".".into())).join("shum")
        });
    ensure!(
        options.install_dir.is_some()
            || options.installation_locator.is_some()
            || io::stdin().is_terminal(),
        "onboarding requires a terminal; use `shum --onboarding --storage-dir PATH`"
    );
    let root = if options.install_dir.is_some() || !io::stdin().is_terminal() {
        fallback
    } else if previous.is_none() && !path.exists() {
        let current = env::current_dir()?;
        print!(
            "Install Shum in the current directory ({})? [y/N]: ",
            current.display()
        );
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().lock().read_line(&mut answer)?;
        if answer.trim().eq_ignore_ascii_case("y") {
            if let Err(error) = ensure_not_source_tree(&current) {
                eprintln!("{error:#}");
                prompt_path("Installation directory", &fallback)?
            } else {
                current
            }
        } else {
            prompt_path("Installation directory", &fallback)?
        }
    } else {
        prompt_path("Installation directory", &fallback)?
    };
    let mut locations = Locations::under(&root);
    if path.is_file()
        && previous.is_none()
        && let Some(parent) = path.parent()
    {
        locations.config = parent.to_path_buf();
    }
    if let Some(old) = &previous {
        locations.bin = old.bin_dir.clone().unwrap_or(locations.bin);
        locations.config = old.config_dir.clone().unwrap_or(locations.config);
        locations.projects = old.projects_dir();
        locations.models = old.models_dir();
    }
    if let Some(path) = &options.bin_dir {
        locations.bin = path.clone();
    } else if io::stdin().is_terminal() {
        locations.bin = prompt_path("Binary directory", &locations.bin)?;
    }
    if let Some(path) = &options.settings_dir {
        locations.config = path.clone();
    } else if io::stdin().is_terminal() {
        locations.config = prompt_path("Settings directory", &locations.config)?;
    }
    if let Some(path) = &options.projects_dir {
        locations.projects = path.clone();
    } else if io::stdin().is_terminal() {
        locations.projects = prompt_path("Projects directory", &locations.projects)?;
    }
    if let Some(path) = &options.models_dir {
        locations.models = path.clone();
    } else if io::stdin().is_terminal() {
        locations.models = prompt_path("Models directory", &locations.models)?;
    }
    if isolated_layout {
        let normalized = locations.normalize()?;
        if let Some(old) = &previous {
            let old_bin = old
                .bin_dir
                .clone()
                .unwrap_or_else(|| old.storage_dir.join("bin"));
            let old_config = old.config_dir()?;
            let old_projects = old.projects_dir();
            let old_models = old.models_dir();
            for (actual, expected) in [
                (
                    old.installation_dir.as_deref().unwrap_or(&old.storage_dir),
                    &normalized.installation,
                ),
                (old_bin.as_path(), &normalized.bin),
                (old_config.as_path(), &normalized.config),
                (old_projects.as_path(), &normalized.projects),
                (old_models.as_path(), &normalized.models),
            ] {
                ensure!(
                    canonical_target(actual)?.as_path() == expected.as_path(),
                    "existing settings {} describe a different layout; choose matching paths or repair interactively",
                    path.display()
                );
            }
        } else {
            let existing_config = normalized.config.join("config.toml");
            ensure!(
                fs::symlink_metadata(&existing_config)
                    .is_err_and(|error| error.kind() == io::ErrorKind::NotFound),
                "config {} already exists without matching settings; refusing to overwrite it",
                existing_config.display()
            );
        }
    }
    let settings = if isolated_layout {
        onboard_with_locations_in(&locations, previous.as_ref(), &locations.config)?
    } else {
        onboard_with_locations(&locations, previous.as_ref())?
    };
    println!(
        "Shum: {}\nProjects: {}\nModels: {}\nSettings: {}",
        settings.bin_dir.as_ref().unwrap().join("shum").display(),
        settings.projects_dir().display(),
        settings.models_dir().display(),
        settings.config_dir()?.join("config.toml").display()
    );
    configure_command(&settings, options)?;
    if io::stdin().is_terminal() {
        print!("Create Python environments with uv now? (large downloads) [y/N]: ");
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().lock().read_line(&mut answer)?;
        if answer.trim().eq_ignore_ascii_case("y") {
            let config = Config::load(&config_path_for(&settings)?)?;
            installation::prepare_environments(
                &settings.storage_dir,
                &settings.runtime_dir()?,
                &config,
            )?;
        }

        configure_models_from_disk(&settings)?;
    }
    Ok(settings)
}

fn configure_command(settings: &Settings, options: &OnboardOptions) -> Result<()> {
    let binary = settings
        .bin_dir
        .as_ref()
        .context("binary directory is unset")?
        .join("shum");
    if let Some(directory) = &options.command_dir {
        let registration = crate::command::setup(&binary, Some(directory), options.setup_shell)?;
        return crate::command::print_registration(&registration);
    }
    let path = env::var_os("PATH").unwrap_or_default();
    if crate::command::available(&binary, &path) {
        println!("Terminal command `shum` is available on PATH.");
        return Ok(());
    }
    if !io::stdin().is_terminal() {
        eprintln!(
            "Terminal command registration was skipped. Run {} setup-command, or pass --command-dir PATH to onboarding.",
            binary.display()
        );
        return Ok(());
    }
    let directory = crate::command::default_directory(&binary, &path)?;
    print!(
        "Register the `shum` terminal command in {}? [Y/n]: ",
        directory.display()
    );
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().lock().read_line(&mut answer)?;
    if answer.trim().eq_ignore_ascii_case("n") {
        println!(
            "Command registration skipped; use {} explicitly.",
            binary.display()
        );
        return Ok(());
    }
    let registration = crate::command::setup(&binary, Some(&directory), false)?;
    if !registration.available {
        print!("Configure Bash/Zsh PATH for new terminals (backing up existing profiles)? [Y/n]: ");
        io::stdout().flush()?;
        answer.clear();
        io::stdin().lock().read_line(&mut answer)?;
        if !answer.trim().eq_ignore_ascii_case("n") {
            let registration = crate::command::setup(&binary, Some(&directory), true)?;
            return crate::command::print_registration(&registration);
        }
    }
    crate::command::print_registration(&registration)
}

pub fn onboard_with_locations(chosen: &Locations, previous: Option<&Settings>) -> Result<Settings> {
    onboard_with_locations_in(chosen, previous, &config_dir()?)
}

fn config_for_onboarding(source: Option<&Path>, reset_invalid: bool) -> Result<Config> {
    if let Some(path) = source {
        match Config::load(path) {
            Ok(config) => return Ok(config),
            Err(error) => {
                ensure!(
                    reset_invalid,
                    "invalid config {}: {error:#}; existing config was not overwritten",
                    path.display()
                );
                let backup = backup_file(path)?;
                eprintln!("Invalid config backed up at {}", backup.display());
            }
        }
    }
    let mut config: Config =
        toml::from_str(installation::TEMPLATE).context("invalid bundled configuration template")?;
    config.llm.enabled = false;
    Ok(config)
}

fn fresh_config_for(settings: &Settings) -> Result<Config> {
    let mut config = config_for_onboarding(None, false)?;
    for model in &mut config.models {
        model.path = settings.models_dir().join(&model.role);
    }
    let runtime = settings.runtime_dir()?;
    let environments = settings.storage_dir.join("runtime/environments");
    let engines = config
        .asr_roles()
        .into_iter()
        .map(|role| Ok((role.clone(), config.asr_engine(&role)?.to_owned())))
        .collect::<Result<Vec<_>>>()?;
    for (role, adapter) in &mut config.asr {
        adapter.script = runtime.join("adapters/asr/main.py");
        let engine = engines
            .iter()
            .find(|(id, _)| id == role)
            .unwrap()
            .1
            .as_str();
        adapter.program = environments
            .join(installation::asr_environment_name(
                role,
                &adapter.script,
                engine,
            ))
            .join("bin/python");
    }
    config.diarization.script = runtime.join("adapters/diarization/main.py");
    config.diarization.program = environments.join("diarization/bin/python");
    Ok(config)
}

#[derive(Deserialize)]
struct PinIdentity {
    repo: String,
    revision: String,
}

fn local_pin_candidates(settings: &Settings, spec: &ModelSpec) -> Result<Vec<ModelSpec>> {
    let role_dir = settings.models_dir().join(&spec.role);
    if !role_dir.exists() {
        return Ok(Vec::new());
    }
    ensure!(
        role_dir.is_dir() && !fs::symlink_metadata(&role_dir)?.file_type().is_symlink(),
        "model directory {} is not a real directory",
        role_dir.display()
    );
    let mut dirs = vec![role_dir.clone()];
    for entry in fs::read_dir(&role_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            dirs.push(entry.path());
        }
        ensure!(
            dirs.len() <= 128,
            "too many model revisions under {}",
            role_dir.display()
        );
    }
    let mut found = Vec::new();
    for dir in dirs {
        let manifest = dir.join(".shum-manifest.json");
        if !manifest.is_file() || fs::symlink_metadata(&manifest)?.file_type().is_symlink() {
            continue;
        }
        if fs::metadata(&manifest)?.len() > 16 * 1024 * 1024 {
            eprintln!(
                "Model manifest {} exceeds the size limit; not adopted",
                manifest.display()
            );
            continue;
        }
        let identity = match serde_json::from_slice::<PinIdentity>(&fs::read(&manifest)?) {
            Ok(identity) => identity,
            Err(error) => {
                eprintln!(
                    "Invalid model manifest {}: {error}; not adopted",
                    manifest.display()
                );
                continue;
            }
        };
        if identity.repo != spec.repo
            || !matches!(
                model_choice(&format!("pin {}", identity.revision), ""),
                Ok(ModelChoice::Pinned(_))
            )
            || (dir != role_dir
                && dir.file_name().and_then(|name| name.to_str())
                    != Some(identity.revision.as_str()))
        {
            continue;
        }
        found.push(ModelSpec {
            repo: identity.repo,
            revision: identity.revision,
            path: dir,
            role: spec.role.clone(),
        });
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(found)
}

fn adopt_pinned_models(settings: &Settings, config: &mut Config) -> Result<()> {
    for spec in &mut config.models {
        let candidates = local_pin_candidates(settings, spec)?;
        if candidates.is_empty() {
            continue;
        }
        println!("Existing {} model revisions found:", spec.role);
        for (index, candidate) in candidates.iter().enumerate() {
            println!(
                "  {}: {} ({})",
                index + 1,
                candidate.revision,
                candidate.path.display()
            );
        }
        print!("Choose a revision to verify and restore, or Enter to leave this model unpinned: ");
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().lock().read_line(&mut answer)?;
        if answer.trim().is_empty() {
            continue;
        }
        let index: usize = answer
            .trim()
            .parse()
            .context("enter a listed model number")?;
        let candidate = candidates
            .get(index.saturating_sub(1))
            .filter(|_| index != 0)
            .context("unknown model revision number")?;
        models::verify_local(candidate).with_context(|| {
            format!(
                "local checksum verification failed for {}; no model revision was restored",
                candidate.path.display()
            )
        })?;
        *spec = candidate.clone();
    }
    Ok(())
}

pub fn repair_config(settings: &Settings, path: &Path) -> Result<Config> {
    if path.is_file()
        && let Ok(config) = Config::load(path)
    {
        return Ok(config);
    }
    ensure!(
        io::stdin().is_terminal(),
        "config {} is missing or invalid; interactive repair is required",
        path.display()
    );
    eprintln!("Config {} is missing or invalid.", path.display());
    print!(
        "Back up the existing file (if present) and reset settings? Models and projects remain untouched. [y/N]: "
    );
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().lock().read_line(&mut answer)?;
    ensure!(
        answer.trim().eq_ignore_ascii_case("y"),
        "config was not changed"
    );
    if path.exists() {
        let backup = backup_file(path)?;
        eprintln!("Config backup: {}", backup.display());
    }
    let mut config = fresh_config_for(settings)?;
    adopt_pinned_models(settings, &mut config)?;
    save_config(path, &config)?;
    Config::load(path)
}

fn onboard_with_locations_in(
    chosen: &Locations,
    previous: Option<&Settings>,
    legacy_dir: &Path,
) -> Result<Settings> {
    let selected = chosen.normalize()?;
    selected.validate()?;
    for dir in [
        &selected.installation,
        &selected.bin,
        &selected.config,
        &selected.projects,
        &selected.models,
        legacy_dir,
    ] {
        check_writable(dir)?;
    }
    let source_config = previous
        .map(|s| s.config_dir().map(|d| d.join("config.toml")))
        .transpose()?
        .filter(|p| p.is_file())
        .or_else(|| {
            selected
                .config
                .join("config.toml")
                .is_file()
                .then(|| selected.config.join("config.toml"))
        })
        .or_else(|| {
            legacy_dir
                .join("config.toml")
                .is_file()
                .then(|| legacy_dir.join("config.toml"))
        });
    let mut reset_invalid = false;
    if let Some(path) = &source_config
        && let Err(error) = Config::load(path)
    {
        ensure!(
            io::stdin().is_terminal(),
            "invalid config {}: {error:#}; interactive repair is required",
            path.display()
        );
        eprintln!("Invalid config at {}: {error:#}", path.display());
        print!(
            "Back up this config and reset its settings to defaults? Models and projects remain untouched. [y/N]: "
        );
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().lock().read_line(&mut answer)?;
        ensure!(
            answer.trim().eq_ignore_ascii_case("y"),
            "config was not changed"
        );
        reset_invalid = true;
    }
    let mut config = config_for_onboarding(source_config.as_deref(), reset_invalid)?;
    let old_projects = previous.map(Settings::projects_dir);
    let old_models = previous.map(Settings::models_dir);
    if let (Some(old), Some(new)) = (old_projects.as_ref(), old_models.as_ref()) {
        if canonical_target(old)? != selected.projects && old.exists() {
            ensure!(
                !selected.projects.exists(),
                "cannot copy projects: {} already exists",
                selected.projects.display()
            );
        }
        if canonical_target(new)? != selected.models && new.exists() {
            ensure!(
                !selected.models.exists(),
                "cannot copy models: {} already exists",
                selected.models.display()
            );
        }
    }
    let config_path = selected.config.join("config.toml");
    ensure!(
        source_config
            .as_ref()
            .is_some_and(|p| canonical_target(p).ok().as_ref() == Some(&config_path))
            || !config_path.exists(),
        "config {} already exists; refusing to overwrite it",
        config_path.display()
    );
    let exe = env::current_exe()?;
    let target_exe = selected.bin.join("shum");
    let existing_binary = fs::symlink_metadata(&target_exe).ok();
    if let Some(metadata) = &existing_binary {
        ensure!(
            metadata.file_type().is_file(),
            "binary {} is not a regular file; refusing to overwrite it",
            target_exe.display()
        );
    }
    let existing_metadata = selected.bin.join(METADATA);
    let old_settings_path = previous
        .map(|old| old.config_dir().map(|dir| dir.join("settings.toml")))
        .transpose()?;
    let expected_settings_path = selected.config.join("settings.toml");
    let owned_binary = if existing_binary.is_some() && existing_metadata.is_file() {
        let metadata: Locator = toml::from_str(&fs::read_to_string(&existing_metadata)?)
            .with_context(|| {
                format!(
                    "invalid installation metadata {}",
                    existing_metadata.display()
                )
            })?;
        metadata.version == 1
            && metadata.settings.is_absolute()
            && (canonical_target(&metadata.settings)? == expected_settings_path
                || old_settings_path.as_ref().is_some_and(|old| {
                    canonical_target(old).ok() == canonical_target(&metadata.settings).ok()
                }))
    } else {
        false
    };
    ensure!(
        existing_binary.is_none()
            || canonical_target(&exe)? == canonical_target(&target_exe)?
            || owned_binary,
        "binary {} belongs to another installation or lacks matching Shum metadata; refusing to overwrite it",
        target_exe.display()
    );

    let runtime = installation::runtime_dir(&selected.installation)?;
    let repair_runtime = runtime.exists() && !installation::runtime_is_complete(&runtime)?;
    if repair_runtime {
        ensure!(
            io::stdin().is_terminal(),
            "runtime {} is damaged; interactive repair is required",
            runtime.display()
        );
        print!("Preserve damaged runtime resources and restore the bundled version? [y/N]: ");
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        ensure!(
            answer.trim().eq_ignore_ascii_case("y"),
            "runtime resources were not changed"
        );
    }
    for model in &mut config.models {
        let source_default = source_config
            .as_ref()
            .and_then(|path| path.parent())
            .map(|dir| dir.join("models").join(&model.role));
        model.path = if reset_invalid
            || source_config.is_none()
            || source_default.as_ref() == Some(&model.path)
        {
            selected.models.join(&model.role)
        } else if let Some(old) = &old_models {
            rebase(&model.path, old, &selected.models)
        } else {
            model.path.clone()
        };
    }
    if let (Some(old), Some(path)) = (&old_projects, &mut config.resources.offload_dir) {
        *path = rebase(path, old, &selected.projects);
    }
    if let (Some(old), Some(path)) = (&old_projects, &mut config.review_rules) {
        *path = rebase(path, old, &selected.projects);
    }
    let envs = selected.installation.join("runtime/environments");
    let roots: Vec<&Path> = previous
        .map(|old| {
            vec![
                selected.installation.as_path(),
                old.installation_dir.as_deref().unwrap_or(&old.storage_dir),
            ]
        })
        .unwrap_or_else(|| vec![selected.installation.as_path()]);
    let engines = config
        .asr_roles()
        .into_iter()
        .map(|role| Ok((role.clone(), config.asr_engine(&role)?.to_owned())))
        .collect::<Result<Vec<_>>>()?;
    for (role, adapter) in &mut config.asr {
        let engine = engines
            .iter()
            .find(|(id, _)| id == role)
            .unwrap()
            .1
            .as_str();
        let bundled = bundled_script(
            &adapter.script,
            "adapters/asr",
            &[
                "main.py",
                "flix_fast_worker.py",
                "qwen_gguf_worker.py",
                "primeline_onnx_worker.py",
                "turbo_worker.py",
                "turbo_ct2_worker.py",
                "turbo_onnx_worker.py",
                "qwen_native_worker.py",
            ],
            &roots,
            source_config.as_deref().and_then(Path::parent),
        )?;
        if let Some(name) = bundled {
            let profile = installation::asr_environment_name(role, &adapter.script, engine);
            let old_program = adapter.program.clone();
            adapter.script = runtime.join("adapters/asr").join(name);
            let legacy_program = source_config
                .as_ref()
                .and_then(|path| path.parent())
                .map(|parent| parent.join(format!("adapters/asr/.venv-{profile}/bin/python")));
            let legacy_engine_program = source_config
                .as_ref()
                .and_then(|path| path.parent())
                .map(|parent| parent.join(format!("adapters/asr/.venv-{engine}/bin/python")));
            let managed = source_config.is_none() && managed_python(&old_program)
                || roots.iter().any(|root| {
                    old_program
                        == root
                            .join("runtime/environments")
                            .join(profile)
                            .join("bin/python")
                })
                || legacy_program.as_ref() == Some(&old_program)
                || (name == "main.py" && legacy_engine_program.as_ref() == Some(&old_program));
            if managed {
                adapter.program = envs
                    .join(installation::asr_environment_name(
                        role,
                        &adapter.script,
                        engine,
                    ))
                    .join("bin/python");
            }
        } else if repair_runtime && adapter.script.starts_with(&runtime) && adapter.script.is_file()
        {
            adapter.script = installation::repair_backup_path(&runtime)
                .join(adapter.script.strip_prefix(&runtime)?);
        }
    }
    let diarization_script = bundled_script(
        &config.diarization.script,
        "adapters/diarization",
        &["main.py", "onnx_worker.py"],
        &roots,
        source_config.as_deref().and_then(Path::parent),
    )?;
    if let Some(name) = diarization_script {
        let profile = installation::environment_name("diarization", &config.diarization.script);
        let old_program = config.diarization.program.clone();
        config.diarization.script = runtime.join("adapters/diarization").join(name);
        let legacy_program = source_config
            .as_ref()
            .and_then(|path| path.parent())
            .map(|parent| parent.join("adapters/diarization/.venv/bin/python"));
        if source_config.is_none() && managed_python(&old_program)
            || roots.iter().any(|root| {
                old_program
                    == root
                        .join("runtime/environments")
                        .join(profile)
                        .join("bin/python")
            })
            || legacy_program.as_ref() == Some(&old_program)
        {
            config.diarization.program = envs
                .join(installation::environment_name(
                    "diarization",
                    &config.diarization.script,
                ))
                .join("bin/python");
        }
    } else if repair_runtime
        && config.diarization.script.starts_with(&runtime)
        && config.diarization.script.is_file()
    {
        config.diarization.script = installation::repair_backup_path(&runtime)
            .join(config.diarization.script.strip_prefix(&runtime)?);
    }
    validate_config_location(&config_path, &config)?;
    let settings = Settings {
        version: SETTINGS_VERSION,
        storage_dir: selected.installation.clone(),
        installation_dir: Some(selected.installation.clone()),
        bin_dir: Some(selected.bin.clone()),
        config_dir: Some(selected.config.clone()),
        projects_dir: Some(selected.projects.clone()),
        models_dir: Some(selected.models.clone()),
    };
    if let (Some(projects), Some(models)) = (old_projects.as_ref(), old_models.as_ref()) {
        let project_bytes = preflight_copy(projects, &selected.projects)?;
        let model_bytes = preflight_copy(models, &selected.models)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let existing_parent = |path: &Path| -> Result<PathBuf> {
                let mut parent = path.parent().context("directory has no parent")?;
                while !parent.exists() {
                    parent = parent
                        .parent()
                        .context("directory has no existing ancestor")?;
                }
                Ok(parent.to_path_buf())
            };
            let project_parent = existing_parent(&selected.projects)?;
            let model_parent = existing_parent(&selected.models)?;
            if fs::metadata(&project_parent)?.dev() == fs::metadata(&model_parent)?.dev() {
                ensure!(
                    fs2::available_space(&project_parent)?
                        >= project_bytes
                            .checked_add(model_bytes)
                            .context("copy size overflow")?,
                    "not enough free space for model and project migration"
                );
            }
        }
    }
    if let Some(old) = old_projects.as_ref()
        && canonical_target(old)? != selected.projects
    {
        migrate_data(old, &selected.projects)?;
    }
    if let Some(old) = old_models.as_ref()
        && canonical_target(old)? != selected.models
    {
        migrate_data(old, &selected.models)?;
    }
    for dir in [
        &selected.installation,
        &selected.bin,
        &selected.config,
        &selected.projects,
        &selected.models,
    ] {
        private_dir(dir)?;
    }
    if repair_runtime {
        installation::repair_runtime(&selected.installation, true)?;
    } else {
        installation::publish_runtime(&selected.installation)?;
    }
    if canonical_target(&exe)? != canonical_target(&target_exe)? {
        let staged = target_exe.with_extension(format!("pending-{}", std::process::id()));
        let result = (|| -> Result<()> {
            fs::copy(&exe, &staged)?;
            fs::set_permissions(&staged, fs::metadata(&exe)?.permissions())?;
            fs::rename(&staged, &target_exe)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(staged);
        }
        result?;
    }
    if config_path.exists() && !reset_invalid {
        backup_file(&config_path)?;
    }
    save_config(&config_path, &config)?;
    atomic_private_write(
        &selected.config.join("settings.toml"),
        &toml::to_string_pretty(&settings)?,
    )?;
    let locator = Locator {
        version: 1,
        settings: selected.config.join("settings.toml"),
    };
    let metadata = toml::to_string(&locator)?;
    atomic_private_write(&selected.bin.join(METADATA), &metadata)?;
    atomic_private_write(&legacy_dir.join("installation.toml"), &metadata)?;
    Ok(settings)
}

pub fn output_path(settings: &Settings, supplied: Option<&Path>, name: &str) -> Result<PathBuf> {
    let root = settings.projects_dir().canonicalize().with_context(|| {
        format!(
            "projects directory {} is missing",
            settings.projects_dir().display()
        )
    })?;
    let path = match supplied {
        Some(path) if path.is_absolute() => canonical_target(path)?,
        Some(path) => canonical_target(&root.join(path))?,
        None => canonical_target(&root.join(name))?,
    };
    ensure!(
        path.starts_with(&root) && path != root,
        "output {} must be inside the selected projects directory {}",
        path.display(),
        root.display()
    );
    Ok(path)
}

pub fn validate_config_location(path: &Path, config: &Config) -> Result<()> {
    ensure_not_source_tree(&canonical_target(path)?)?;
    ensure!(
        path.is_absolute() || canonical_target(path)?.is_absolute(),
        "config path must be absolute"
    );
    for model in &config.models {
        ensure_not_source_tree(&canonical_target(&model.path)?)?;
        ensure!(
            model.path.is_absolute(),
            "model path must be absolute: {}",
            model.path.display()
        );
    }
    for path in [
        config.resources.offload_dir.as_deref(),
        config.review_rules.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        ensure_not_source_tree(&canonical_target(path)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project_temp() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(".shum-test-")
            .tempdir()
            .unwrap()
    }

    fn settings(root: &Path) -> Settings {
        Settings {
            version: 2,
            storage_dir: root.to_path_buf(),
            installation_dir: None,
            bin_dir: None,
            config_dir: None,
            projects_dir: None,
            models_dir: None,
        }
    }

    #[test]
    fn installation_rejects_source_checkout_without_a_compiled_path() {
        let source = project_temp();
        fs::create_dir(source.path().join("src")).unwrap();
        fs::write(
            source.path().join("Cargo.toml"),
            "[package]\nname = \"shum\"",
        )
        .unwrap();
        fs::write(source.path().join("shum.example.toml"), "").unwrap();
        fs::write(source.path().join("src/main.rs"), "").unwrap();
        let locations = Locations::under(&source.path().join("local"));
        assert!(locations.validate().is_err());
        let outside = project_temp();
        assert!(Locations::under(outside.path()).validate().is_ok());
    }

    #[test]
    fn adopting_fast_flix_does_not_select_the_transformers_environment() {
        let temp = project_temp();
        let locations = Locations::under(&temp.path().join("installation"));
        fs::create_dir_all(&locations.config).unwrap();
        let mut config: Config = toml::from_str(installation::TEMPLATE).unwrap();
        config.asr.get_mut("flix").unwrap().script = "adapters/asr/flix_fast_worker.py".into();
        config.asr.get_mut("flix").unwrap().program =
            "adapters/asr/.venv-flix-fast/bin/python".into();
        save_config(&locations.config.join("config.toml"), &config).unwrap();
        let settings =
            onboard_with_locations_in(&locations, None, &temp.path().join("legacy")).unwrap();
        let installed = Config::load(&config_path_for(&settings).unwrap()).unwrap();
        assert_eq!(
            installed.asr("flix").program,
            locations
                .installation
                .join("runtime/environments/flix-fast/bin/python")
        );
        assert_eq!(
            installed.asr("flix").script,
            settings
                .runtime_dir()
                .unwrap()
                .join("adapters/asr/flix_fast_worker.py")
        );
    }

    #[test]
    fn custom_main_model_can_be_provisioned_but_custom_script_is_skipped() {
        let mut config: Config = toml::from_str(installation::TEMPLATE).unwrap();
        let mut adapter = config.asr("flix").clone();
        adapter.engine = Some("whisper".into());
        config.asr.insert("meeting".into(), adapter);
        let mut model = config.model("flix").clone();
        model.role = "meeting".into();
        config.models.push(model.clone());
        assert_eq!(
            production_model(&config, &model),
            Some("Whisper/Transformers")
        );
        config.asr.get_mut("meeting").unwrap().script = "external/worker.py".into();
        assert_eq!(production_model(&config, &model), None);
    }

    #[test]
    fn shared_engine_custom_roles_use_distinct_models_and_one_environment() {
        let mut config: Config = toml::from_str(installation::TEMPLATE).unwrap();
        for role in ["interview", "meeting"] {
            let mut adapter = config.asr("flix").clone();
            adapter.engine = Some("whisper".into());
            config.asr.insert(role.into(), adapter);
            let mut model = config.model("flix").clone();
            model.role = role.into();
            model.repo = format!("example/{role}");
            config.models.push(model);
            assert_eq!(
                production_model(&config, config.model(role)),
                Some("Whisper/Transformers")
            );
            assert_eq!(
                installation::asr_environment_name(
                    role,
                    Path::new("adapters/asr/main.py"),
                    config.asr_engine(role).unwrap()
                ),
                "whisper"
            );
        }
        assert_ne!(config.model("interview").repo, config.model("meeting").repo);
    }

    #[test]
    fn onboarding_migrates_two_custom_roles_sharing_an_engine() {
        let temp = project_temp();
        let locations = Locations::under(&temp.path().join("installation"));
        fs::create_dir_all(&locations.config).unwrap();
        let mut config: Config = toml::from_str(installation::TEMPLATE).unwrap();
        for role in ["interview", "meeting"] {
            let mut adapter = config.asr("flix").clone();
            adapter.engine = Some("whisper".into());
            adapter.program = "adapters/asr/.venv-whisper/bin/python".into();
            config.asr.insert(role.into(), adapter);
            let mut model = config.model("flix").clone();
            model.role = role.into();
            model.repo = format!("example/{role}");
            config.models.push(model);
            config
                .resources
                .estimated_peak_mib
                .insert(role.into(), 10500);
        }
        save_config(&locations.config.join("config.toml"), &config).unwrap();
        let settings =
            onboard_with_locations_in(&locations, None, &temp.path().join("legacy")).unwrap();
        let installed = Config::load(&config_path_for(&settings).unwrap()).unwrap();
        for role in ["interview", "meeting"] {
            assert_eq!(
                installed.asr(role).program,
                locations
                    .installation
                    .join("runtime/environments/whisper/bin/python")
            );
            assert_eq!(
                installed.asr(role).script,
                settings.runtime_dir().unwrap().join("adapters/asr/main.py")
            );
            assert_eq!(installed.model(role).repo, format!("example/{role}"));
        }
    }

    #[test]
    fn onboarding_preserves_custom_content_inside_old_runtime_and_adapter_settings() {
        let temp = project_temp();
        let locations = Locations::under(&temp.path().join("installation"));
        fs::create_dir_all(&locations.config).unwrap();
        let old_script = locations
            .installation
            .join("runtime/old/adapters/asr/turbo_worker.py");
        fs::create_dir_all(old_script.parent().unwrap()).unwrap();
        fs::write(&old_script, "# locally customized worker\n").unwrap();
        let mut config: Config = toml::from_str(installation::TEMPLATE).unwrap();
        let adapter = config.asr.get_mut("flix").unwrap();
        adapter.script = old_script.clone();
        adapter.program = locations
            .installation
            .join("runtime/environments/turbo/bin/python");
        adapter.engine = Some("whisper".into());
        adapter.threads = 7;
        adapter.max_in_flight = 2;
        adapter.args = vec!["--custom".into()];
        let expected = adapter.clone();
        save_config(&locations.config.join("config.toml"), &config).unwrap();
        let settings =
            onboard_with_locations_in(&locations, None, &temp.path().join("legacy")).unwrap();
        let installed = Config::load(&config_path_for(&settings).unwrap()).unwrap();
        assert_eq!(installed.asr("flix"), &expected);
        assert_eq!(
            fs::read_to_string(old_script).unwrap(),
            "# locally customized worker\n"
        );
    }

    #[test]
    fn adopting_speed_workers_preserves_profiles_and_upgrades_bundled_scripts() {
        let temp = project_temp();
        let locations = Locations::under(&temp.path().join("installation"));
        fs::create_dir_all(&locations.config).unwrap();
        let mut config: Config = toml::from_str(installation::TEMPLATE).unwrap();
        for (role, script, environment, flag) in [
            (
                "primeline",
                "primeline_onnx_worker.py",
                "primeline-onnx",
                "--onnx-model",
            ),
            ("flix", "turbo_worker.py", "turbo", "--ct2-model"),
        ] {
            let adapter = config.asr.get_mut(role).unwrap();
            adapter.script = locations
                .installation
                .join("runtime/old/adapters/asr")
                .join(script);
            adapter.program = locations
                .installation
                .join("runtime/environments")
                .join(environment)
                .join("bin/python");
            adapter.args = vec![
                flag.into(),
                temp.path().join(role).to_string_lossy().into_owned(),
            ];
        }
        save_config(&locations.config.join("config.toml"), &config).unwrap();
        let settings =
            onboard_with_locations_in(&locations, None, &temp.path().join("legacy")).unwrap();
        let installed = Config::load(&config_path_for(&settings).unwrap()).unwrap();
        for (role, script, environment) in [
            ("primeline", "primeline_onnx_worker.py", "primeline-onnx"),
            ("flix", "turbo_worker.py", "turbo"),
        ] {
            assert_eq!(
                installed.asr(role).script,
                settings
                    .runtime_dir()
                    .unwrap()
                    .join("adapters/asr")
                    .join(script)
            );
            assert_eq!(
                installed.asr(role).program,
                locations
                    .installation
                    .join("runtime/environments")
                    .join(environment)
                    .join("bin/python")
            );
            assert_eq!(installed.asr(role).args, config.asr(role).args);
        }
    }

    #[test]
    fn output_paths_stay_under_projects_even_with_symlinks() {
        let temp = project_temp();
        let root = temp.path().canonicalize().unwrap();
        fs::create_dir(root.join("projects")).unwrap();
        let settings = settings(&root);
        assert_eq!(
            output_path(&settings, None, "recording").unwrap(),
            root.join("projects/recording")
        );
        assert!(output_path(&settings, Some(Path::new("../escape")), "").is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(temp.path(), root.join("projects/link")).unwrap();
            assert!(output_path(&settings, Some(Path::new("link/escape")), "").is_err());
        }
    }

    #[test]
    fn copies_data_without_removing_source_and_detects_collisions() {
        let temp = project_temp();
        let old = temp.path().join("old");
        let new = temp.path().join("new");
        fs::create_dir_all(old.join("run")).unwrap();
        fs::write(old.join("run/result.json"), "private").unwrap();
        migrate_data(&old, &new).unwrap();
        assert_eq!(
            fs::read_to_string(new.join("run/result.json")).unwrap(),
            "private"
        );
        assert_eq!(
            fs::read_to_string(old.join("run/result.json")).unwrap(),
            "private"
        );
        assert!(migrate_data(&old, &new).is_err());
    }

    #[test]
    fn location_preflight_detects_symlink_overlaps() {
        let temp = project_temp();
        let mut paths = Locations::under(&temp.path().join("install"));
        fs::create_dir_all(temp.path().join("existing")).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(temp.path().join("existing"), temp.path().join("alias"))
                .unwrap();
            paths.projects = temp.path().join("alias");
            paths.models = temp.path().join("existing/models");
            assert!(paths.normalize().unwrap().validate().is_err());
        }
    }

    #[test]
    fn bundled_runtime_has_no_source_checkout_dependency() {
        let temp = project_temp();
        let runtime = installation::publish_runtime(temp.path()).unwrap();
        assert!(runtime.join("adapters/asr/main.py").is_file());
        assert!(runtime.join("adapters/diarization/main.py").is_file());
        let config: Config = toml::from_str(installation::TEMPLATE).unwrap();
        assert_eq!(config.models.len(), 4);
    }

    #[test]
    fn fresh_install_selects_independent_locations_and_preserves_data() {
        let temp = project_temp();
        let root = temp.path().join("installation");
        let mut paths = Locations::under(&root);
        paths.bin = temp.path().join("programs");
        paths.config = temp.path().join("settings");
        paths.projects = temp.path().join("projects");
        paths.models = temp.path().join("weights");
        fs::create_dir_all(paths.projects.join("existing")).unwrap();
        fs::write(paths.projects.join("existing/transcript.txt"), "kept").unwrap();
        let xdg = temp.path().join("xdg");
        let installed = onboard_with_locations_in(&paths, None, &xdg).unwrap();
        assert_eq!(
            fs::read_to_string(installed.projects_dir().join("existing/transcript.txt")).unwrap(),
            "kept"
        );
        assert_eq!(installed.models_dir(), paths.models);
        assert!(paths.bin.join("shum").is_file());
        assert!(paths.bin.join(METADATA).is_file());
        assert!(xdg.join("installation.toml").is_file());
        let saved = Config::load(&paths.config.join("config.toml")).unwrap();
        assert!(saved.asr("flix").script.is_file());
        assert_eq!(
            saved.model("primeline").path,
            paths.models.join("primeline")
        );
        assert!(!root.join("runtime/environments").exists());
    }

    #[test]
    fn reinstallation_preserves_model_revision_and_copies_data_without_deleting_source() {
        let temp = project_temp();
        let root = temp.path().join("install");
        let mut paths = Locations::under(&root);
        paths.config = temp.path().join("settings");
        let xdg = temp.path().join("xdg");
        let previous = onboard_with_locations_in(&paths, None, &xdg).unwrap();
        fs::create_dir_all(previous.projects_dir().join("take")).unwrap();
        fs::write(previous.projects_dir().join("take/result"), b"original").unwrap();
        let old_models = previous.models_dir().join("primeline");
        fs::create_dir_all(&old_models).unwrap();
        fs::write(old_models.join("model.bin"), b"weights").unwrap();
        let config_path = previous.config_dir().unwrap().join("config.toml");
        let mut config = Config::load(&config_path).unwrap();
        config
            .models
            .iter_mut()
            .find(|m| m.role == "primeline")
            .unwrap()
            .revision = "pinned".into();
        save_config(&config_path, &config).unwrap();
        paths.projects = temp.path().join("new-projects");
        paths.models = temp.path().join("new-models");
        let updated = onboard_with_locations_in(&paths, Some(&previous), &xdg).unwrap();
        assert!(previous.projects_dir().join("take/result").is_file());
        assert!(updated.projects_dir().join("take/result").is_file());
        assert_eq!(
            fs::read(updated.models_dir().join("primeline/model.bin")).unwrap(),
            b"weights"
        );
        let config = Config::load(&config_path).unwrap();
        assert_eq!(config.model("primeline").revision, "pinned");
        assert_eq!(
            config.model("primeline").path,
            updated.models_dir().join("primeline")
        );
    }

    #[test]
    fn repeated_reinstallation_keeps_each_config_backup() {
        let temp = project_temp();
        let paths = Locations::under(&temp.path().join("installation"));
        let xdg = temp.path().join("xdg");
        let installed = onboard_with_locations_in(&paths, None, &xdg).unwrap();
        let config_path = paths.config.join("config.toml");
        let first = fs::read(&config_path).unwrap();
        onboard_with_locations_in(&paths, Some(&installed), &xdg).unwrap();
        onboard_with_locations_in(&paths, Some(&installed), &xdg).unwrap();
        let backups = fs::read_dir(&paths.config)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("config.toml.bak-")
            })
            .collect::<Vec<_>>();
        assert_eq!(backups.len(), 2);
        assert!(
            backups
                .iter()
                .all(|backup| fs::read(backup).unwrap() == first)
        );
    }

    #[test]
    fn binary_metadata_from_another_installation_does_not_authorize_overwrite() {
        let temp = project_temp();
        let paths = Locations::under(&temp.path().join("installation"));
        fs::create_dir_all(&paths.bin).unwrap();
        fs::write(paths.bin.join("shum"), b"unrelated binary").unwrap();
        fs::write(
            paths.bin.join(METADATA),
            toml::to_string(&Locator {
                version: 1,
                settings: temp.path().join("other/settings.toml"),
            })
            .unwrap(),
        )
        .unwrap();
        let error = onboard_with_locations_in(&paths, None, &temp.path().join("xdg"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("belongs to another installation"), "{error}");
        assert_eq!(
            fs::read(paths.bin.join("shum")).unwrap(),
            b"unrelated binary"
        );
        #[cfg(unix)]
        {
            let protected = temp.path().join("protected");
            fs::write(&protected, b"protected").unwrap();
            fs::remove_file(paths.bin.join("shum")).unwrap();
            std::os::unix::fs::symlink(&protected, paths.bin.join("shum")).unwrap();
            assert!(onboard_with_locations_in(&paths, None, &temp.path().join("xdg")).is_err());
            assert_eq!(fs::read(&protected).unwrap(), b"protected");
        }
    }

    #[test]
    fn config_parent_and_locator_find_installation_without_xdg() {
        let temp = project_temp();
        let root = temp.path().join("install");
        let config_dir = root.join("config");
        fs::create_dir_all(config_dir.clone()).unwrap();
        fs::create_dir_all(root.join("projects")).unwrap();
        fs::create_dir_all(root.join("models")).unwrap();
        installation::publish_runtime(&root).unwrap();
        let mut installed = settings(&root);
        installed.config_dir = Some(config_dir.clone());
        fs::write(
            config_dir.join("settings.toml"),
            toml::to_string(&installed).unwrap(),
        )
        .unwrap();
        let pointer = temp.path().join("installation.toml");
        fs::write(
            &pointer,
            toml::to_string(&Locator {
                version: 1,
                settings: config_dir.join("settings.toml"),
            })
            .unwrap(),
        )
        .unwrap();
        let from_config = load_from(None, Some(&config_dir.join("config.toml"))).unwrap();
        let from_pointer = load_from(Some(&pointer), None).unwrap();
        assert_eq!(from_config.storage_dir, root);
        assert_eq!(from_pointer.storage_dir, root);
        assert_eq!(
            config_path_for(&from_pointer).unwrap(),
            config_dir.join("config.toml")
        );
    }

    #[test]
    fn model_downloads_require_an_explicit_choice() {
        assert!(matches!(
            model_choice("", "pinned").unwrap(),
            ModelChoice::Skip
        ));
        assert!(matches!(
            model_choice("skip", "").unwrap(),
            ModelChoice::Skip
        ));
        assert!(matches!(
            model_choice("latest", "").unwrap(),
            ModelChoice::Latest
        ));
        assert!(
            matches!(model_choice("pinned", "abc123").unwrap(), ModelChoice::Pinned(r) if r == "abc123")
        );
        assert!(
            matches!(model_choice("pin deadbeef", "").unwrap(), ModelChoice::Pinned(r) if r == "deadbeef")
        );
        assert!(model_choice("pin ../unsafe", "").is_err());
        assert!(model_choice("pinned", "").is_err());
    }

    #[test]
    fn corrupt_config_is_never_silently_replaced() {
        let temp = project_temp();
        let path = temp.path().join("config.toml");
        fs::write(&path, "invalid = [\n").unwrap();
        assert!(config_for_onboarding(Some(&path), false).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "invalid = [\n");
        let replacement = config_for_onboarding(Some(&path), true).unwrap();
        assert!(!replacement.llm.enabled);
        assert_eq!(fs::read_to_string(&path).unwrap(), "invalid = [\n");
        assert!(
            path.with_extension(format!("toml.bak-{}", std::process::id()))
                .is_file()
        );
    }

    #[test]
    fn missing_runtime_is_reported_for_installed_settings() {
        let temp = project_temp();
        for name in ["config", "projects", "models"] {
            fs::create_dir_all(temp.path().join(name)).unwrap();
        }
        let installed = Settings {
            config_dir: Some(temp.path().join("config")),
            ..settings(temp.path())
        };
        let path = temp.path().join("config/settings.toml");
        fs::write(&path, toml::to_string(&installed).unwrap()).unwrap();
        assert!(
            read_settings(&path)
                .unwrap_err()
                .to_string()
                .contains("runtime resources")
        );
        installation::publish_runtime(temp.path()).unwrap();
        assert!(read_settings(&path).is_ok());
    }

    #[test]
    fn model_wizard_accepts_custom_repositories_for_supported_engines() {
        let mut config: Config = toml::from_str(installation::TEMPLATE).unwrap();
        let qwen = config
            .models
            .iter()
            .find(|model| model.role == "qwen")
            .unwrap()
            .clone();
        assert_eq!(production_model(&config, &qwen), Some("Qwen3-ASR"));
        let mut incompatible = qwen.clone();
        incompatible.repo = "Qwen/Qwen3-ASR-1.7B".into();
        assert_eq!(production_model(&config, &incompatible), Some("Qwen3-ASR"));
        config.asr.get_mut("qwen").unwrap().script = "adapters/asr/qwen_gguf_worker.py".into();
        assert!(production_model(&config, &qwen).is_none());
    }

    #[test]
    fn repair_template_uses_installed_runtime_and_disabled_llm() {
        let temp = project_temp();
        let installed = Settings {
            installation_dir: Some(temp.path().to_path_buf()),
            models_dir: Some(temp.path().join("weights")),
            ..settings(temp.path())
        };
        let config = fresh_config_for(&installed).unwrap();
        assert!(!config.llm.enabled);
        assert_eq!(
            config.model("primeline").path,
            installed.models_dir().join("primeline")
        );
        assert!(
            config
                .asr("flix")
                .script
                .starts_with(installed.runtime_dir().unwrap())
        );
    }

    #[test]
    fn repair_discovers_only_locally_pinned_role_revision() {
        let temp = project_temp();
        let installed = settings(temp.path());
        let config = fresh_config_for(&installed).unwrap();
        let spec = config.model("primeline");
        let revision_dir = installed.models_dir().join("primeline/abc123");
        fs::create_dir_all(&revision_dir).unwrap();
        fs::write(revision_dir.join("model.bin"), b"weights").unwrap();
        let digest = hex::encode(Sha256::digest(b"weights"));
        let manifest = serde_json::json!({
            "repo": spec.repo,
            "revision": "abc123",
            "files": {"model.bin": digest},
        });
        fs::write(
            revision_dir.join(".shum-manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let choices = local_pin_candidates(&installed, spec).unwrap();
        assert_eq!(choices.len(), 1);
        models::verify_local(&choices[0]).unwrap();
        fs::write(revision_dir.join("model.bin"), b"changed").unwrap();
        assert!(models::verify_local(&choices[0]).is_err());
    }
}
