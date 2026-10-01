use std::env;
use std::fs::{self, DirBuilder, OpenOptions};
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::config::Config;

const SETTINGS_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub version: u32,
    pub storage_dir: PathBuf,
}

impl Settings {
    pub fn projects_dir(&self) -> PathBuf {
        self.storage_dir.join("projects")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.storage_dir.join("models")
    }
}

pub fn repo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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
    Ok(config_dir()?.join("settings.toml"))
}

pub fn config_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("config.toml"))
}

pub fn load() -> Result<Settings> {
    let path = settings_path()?;
    ensure_outside_repo(&canonical_target(&path)?)?;
    let text = fs::read_to_string(&path).with_context(|| {
        format!(
            "Shum is not configured at {}; run `shum --onboarding`",
            path.display()
        )
    })?;
    let settings: Settings =
        toml::from_str(&text).with_context(|| format!("invalid settings {}", path.display()))?;
    ensure!(
        settings.version == SETTINGS_VERSION,
        "unsupported Shum settings version {}",
        settings.version
    );
    ensure!(
        settings.storage_dir.is_absolute(),
        "storage_dir must be absolute"
    );
    let root = settings.storage_dir.canonicalize().with_context(|| {
        format!(
            "storage directory {} is missing",
            settings.storage_dir.display()
        )
    })?;
    ensure!(
        root == settings.storage_dir && root.is_dir(),
        "storage directory {} must be a real directory",
        settings.storage_dir.display()
    );
    ensure_outside_repo(&root)?;
    ensure!(
        settings.projects_dir().is_dir(),
        "projects directory {} is missing; run `shum --onboarding`",
        settings.projects_dir().display()
    );
    Ok(settings)
}

pub fn ensure_outside_repo(path: &Path) -> Result<()> {
    let repo = repo_dir()
        .canonicalize()
        .context("source directory is missing")?;
    ensure!(
        !path.starts_with(&repo),
        "{} is inside the Shum source tree; choose a directory outside it",
        path.display()
    );
    Ok(())
}

fn expand_home(path: &Path) -> Result<PathBuf> {
    if path == Path::new("~") || path.starts_with("~/") {
        let home = env::var_os("HOME").context("HOME is unset; cannot expand ~")?;
        let suffix = path.strip_prefix("~")?;
        return Ok(PathBuf::from(home).join(suffix));
    }
    Ok(path.to_path_buf())
}

fn absolute(path: &Path) -> Result<PathBuf> {
    let path = expand_home(path)?;
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

fn existing_ancestor(path: &Path) -> Result<PathBuf> {
    let mut missing = Vec::new();
    let mut current = path;
    while !current.exists() {
        let component = current
            .file_name()
            .context("path has no existing ancestor")?;
        missing.push(component.to_os_string());
        current = current.parent().context("path has no parent")?;
    }
    let mut resolved = current.canonicalize()?;
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn canonical_target(path: &Path) -> Result<PathBuf> {
    let path = absolute(path)?;
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir | Component::Prefix(_) | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                ensure!(normalized.pop(), "path escapes the filesystem root");
            }
        }
    }
    existing_ancestor(&normalized)
}

fn private_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder
            .create(path)
            .with_context(|| format!("cannot create {}", path.display()))?;
    }
    #[cfg(not(unix))]
    fs::create_dir_all(path).with_context(|| format!("cannot create {}", path.display()))?;
    Ok(())
}

fn atomic_private_write(path: &Path, text: &str) -> Result<()> {
    private_dir(path.parent().context("settings path has no parent")?)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .with_context(|| format!("cannot create {}", temporary.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    fs::rename(&temporary, path)
        .with_context(|| format!("cannot save settings to {}", path.display()))?;
    Ok(())
}

pub fn save_config(path: &Path, config: &Config) -> Result<()> {
    atomic_private_write(path, &toml::to_string_pretty(config)?)
}

fn migrate_directory(source: &Path, destination: &Path) -> Result<()> {
    if source == destination || !source.exists() {
        return Ok(());
    }
    ensure!(
        !destination.exists(),
        "cannot move {}: {} already exists; no files were overwritten",
        source.display(),
        destination.display()
    );
    fs::rename(source, destination).with_context(|| {
        format!(
            "cannot move {} to {} (if the locations are on different filesystems, move the directory manually and rerun onboarding)",
            source.display(),
            destination.display()
        )
    })
}

fn rebase(path: &Path, old: &Path, new: &Path) -> PathBuf {
    path.strip_prefix(old)
        .map_or_else(|_| path.to_path_buf(), |suffix| new.join(suffix))
}

fn relocate_config(config: &mut Config, old_models: &Path, old_projects: &Path, target: &Settings) {
    for model in &mut config.models {
        model.path = rebase(&model.path, old_models, &target.models_dir());
    }
    if let Some(path) = &mut config.resources.offload_dir {
        *path = rebase(path, old_projects, &target.projects_dir());
    }
    if let Some(path) = &mut config.review_rules {
        *path = rebase(path, old_projects, &target.projects_dir());
    }
}

fn check_storage_dir(path: &Path) -> Result<PathBuf> {
    let target = canonical_target(path)?;
    ensure_outside_repo(&target)?;
    private_dir(&target)
        .with_context(|| format!("cannot create storage directory {}", target.display()))?;
    let target = target.canonicalize()?;
    ensure_outside_repo(&target)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&target, fs::Permissions::from_mode(0o700))?;
    }
    Ok(target)
}

pub fn onboard(chosen: Option<&Path>) -> Result<()> {
    let previous = if settings_path()?.exists() {
        Some(load()?)
    } else {
        None
    };
    let target = if let Some(path) = chosen {
        path.to_path_buf()
    } else {
        ensure!(
            io::stdin().is_terminal(),
            "onboarding requires a terminal; use `shum --onboarding --storage-dir PATH`"
        );
        let fallback = if let Some(settings) = &previous {
            settings.storage_dir.clone()
        } else {
            PathBuf::from(env::var_os("HOME").context("HOME is unset; use --storage-dir PATH")?)
                .join("shum")
        };
        print!(
            "Directory for Shum data (projects and models) [{}]: ",
            fallback.display()
        );
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().lock().read_line(&mut answer)?;
        if answer.trim().is_empty() {
            fallback
        } else {
            PathBuf::from(answer.trim())
        }
    };
    let repo = repo_dir().canonicalize()?;
    let app_dir = config_dir()?;
    let settings = onboard_at(&target, &repo, &app_dir, previous.as_ref())?;
    println!(
        "Shum projects: {}\nModels: {}\nSettings: {}",
        settings.projects_dir().display(),
        settings.models_dir().display(),
        app_dir.join("settings.toml").display()
    );
    Ok(())
}

fn onboard_at(
    chosen: &Path,
    repo: &Path,
    app_dir: &Path,
    previous: Option<&Settings>,
) -> Result<Settings> {
    ensure_outside_repo(&canonical_target(app_dir)?)?;
    let target = check_storage_dir(chosen)?;
    let settings = Settings {
        version: SETTINGS_VERSION,
        storage_dir: target,
    };
    let old_root = previous.map_or(repo, |previous| previous.storage_dir.as_path());
    let old_projects = if previous.is_some() {
        old_root.join("projects")
    } else {
        old_root.join("jobs")
    };
    let old_models = old_root.join("models");
    let new_projects = settings.projects_dir();
    let new_models = settings.models_dir();
    for (source, destination) in [(&old_projects, &new_projects), (&old_models, &new_models)] {
        ensure!(
            source == destination || !source.exists() || !destination.exists(),
            "cannot move {}: {} already exists; no files were overwritten",
            source.display(),
            destination.display()
        );
    }
    let private_config = app_dir.join("config.toml");
    let legacy_config = repo.join("shum.toml");
    ensure!(
        !legacy_config.exists() || !private_config.exists() || previous.is_some(),
        "both {} and {} exist; resolve the config conflict before onboarding",
        legacy_config.display(),
        private_config.display()
    );
    let source_config = if private_config.exists() {
        &private_config
    } else if legacy_config.exists() {
        &legacy_config
    } else {
        &repo.join("shum.example.toml")
    };
    let mut config = Config::load(source_config)?;
    relocate_config(&mut config, &old_models, &old_projects, &settings);
    if source_config.ends_with("shum.example.toml") {
        for model in &mut config.models {
            model.path = new_models.join(&model.role);
        }
    }
    if let Some(rules) = &config.review_rules {
        ensure_outside_repo(&canonical_target(rules)?).with_context(|| {
            format!(
                "review_rules {} remains inside the source tree; move this user file outside the repository and update the config before onboarding",
                rules.display()
            )
        })?;
    }

    migrate_directory(&old_projects, &new_projects)?;
    migrate_directory(&old_models, &new_models)?;
    private_dir(&new_projects)?;
    private_dir(&new_models)?;
    save_config(&private_config, &config)?;
    atomic_private_write(
        &app_dir.join("settings.toml"),
        &toml::to_string_pretty(&settings)?,
    )?;
    if legacy_config.exists() {
        fs::remove_file(&legacy_config)
            .with_context(|| format!("cannot remove old config {}", legacy_config.display()))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&new_projects, fs::Permissions::from_mode(0o700))?;
        fs::set_permissions(&new_models, fs::Permissions::from_mode(0o700))?;
        fs::set_permissions(app_dir, fs::Permissions::from_mode(0o700))?;
    }
    Ok(settings)
}

pub fn output_path(settings: &Settings, supplied: Option<&Path>, name: &str) -> Result<PathBuf> {
    let root = settings.projects_dir().canonicalize().with_context(|| {
        format!(
            "projects directory {} is missing",
            settings.projects_dir().display()
        )
    })?;
    ensure_outside_repo(&root)?;
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
    ensure_outside_repo(&canonical_target(path)?)?;
    for model in &config.models {
        ensure_outside_repo(&canonical_target(&model.path)?)?;
    }
    if let Some(offload) = &config.resources.offload_dir {
        ensure_outside_repo(&canonical_target(offload)?)?;
    }
    if let Some(rules) = &config.review_rules {
        ensure_outside_repo(&canonical_target(rules)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_paths_stay_under_projects_even_with_symlinks() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        fs::create_dir(root.join("projects")).unwrap();
        let settings = Settings {
            version: SETTINGS_VERSION,
            storage_dir: root.clone(),
        };
        assert_eq!(
            output_path(&settings, None, "recording").unwrap(),
            root.join("projects/recording")
        );
        assert_eq!(
            output_path(&settings, Some(Path::new("review/run")), "").unwrap(),
            root.join("projects/review/run")
        );
        assert!(output_path(&settings, Some(Path::new("../escape")), "").is_err());
        assert!(output_path(&settings, Some(temp.path()), "").is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(temp.path(), root.join("projects/link")).unwrap();
            assert!(output_path(&settings, Some(Path::new("link/escape")), "").is_err());
        }
    }

    #[test]
    fn migration_never_overwrites_existing_projects() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("jobs");
        let destination = temp.path().join("projects");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&destination).unwrap();
        fs::write(source.join("keep"), "original").unwrap();
        assert!(migrate_directory(&source, &destination).is_err());
        assert_eq!(fs::read_to_string(source.join("keep")).unwrap(), "original");
    }

    #[test]
    fn migration_preserves_nested_project_data() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("jobs");
        let destination = temp.path().join("projects");
        fs::create_dir_all(source.join("run/clips")).unwrap();
        fs::write(source.join("run/clips/00000.wav"), b"audio").unwrap();
        migrate_directory(&source, &destination).unwrap();
        assert!(!source.exists());
        assert_eq!(
            fs::read(destination.join("run/clips/00000.wav")).unwrap(),
            b"audio"
        );
    }

    #[test]
    fn onboarding_moves_data_and_rewrites_config_outside_source() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("source");
        let app = temp.path().join("settings");
        let storage = temp.path().join("storage");
        fs::create_dir_all(repo.join("jobs/run")).unwrap();
        fs::create_dir_all(repo.join("models/primeline/revision")).unwrap();
        fs::write(repo.join("jobs/run/transcript.json"), b"private transcript").unwrap();
        fs::write(
            repo.join("models/primeline/revision/model.bin"),
            b"model weights",
        )
        .unwrap();
        fs::write(
            repo.join("shum.toml"),
            include_str!("../shum.example.toml").replace(
                "revision = \"\"\npath = \"models/primeline\"",
                "revision = \"revision\"\npath = \"models/primeline/revision\"",
            ),
        )
        .unwrap();

        let settings = onboard_at(&storage, &repo, &app, None).unwrap();
        assert_eq!(
            fs::read(settings.projects_dir().join("run/transcript.json")).unwrap(),
            b"private transcript"
        );
        assert_eq!(
            fs::read(settings.models_dir().join("primeline/revision/model.bin")).unwrap(),
            b"model weights"
        );
        assert!(!repo.join("jobs").exists());
        assert!(!repo.join("models").exists());
        assert!(!repo.join("shum.toml").exists());
        let config = Config::load(&app.join("config.toml")).unwrap();
        assert_eq!(
            config.model("primeline").path,
            settings.models_dir().join("primeline/revision")
        );
        assert_eq!(config.asr("flix").script, repo.join("adapters/asr/main.py"));
        assert!(onboard_at(&storage, &repo, &app, Some(&settings)).is_ok());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(app.join("settings.toml"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(settings.projects_dir())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(settings.storage_dir)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
    }

    #[test]
    fn onboarding_checks_all_collisions_before_moving_any_data() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("source");
        let storage = temp.path().join("storage");
        fs::create_dir_all(repo.join("jobs")).unwrap();
        fs::create_dir_all(repo.join("models")).unwrap();
        fs::create_dir_all(storage.join("models")).unwrap();
        let error = onboard_at(&storage, &repo, &temp.path().join("settings"), None).unwrap_err();
        assert!(error.to_string().contains("already exists"));
        assert!(repo.join("jobs").exists());
        assert!(!storage.join("projects").exists());
    }

    #[test]
    fn rerunning_onboarding_moves_project_relative_settings() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("source");
        let app = temp.path().join("settings");
        let initial = temp.path().join("first");
        let next = temp.path().join("second");
        fs::create_dir_all(repo.join("jobs/run")).unwrap();
        fs::write(repo.join("jobs/run/result.json"), b"result").unwrap();
        fs::write(
            repo.join("shum.toml"),
            format!(
                "review_rules = \"jobs/rules.txt\"\n{}",
                include_str!("../shum.example.toml").replace(
                    "# offload_dir = \"/absolute/shum-data/projects/.asr-offload\"",
                    "offload_dir = \"jobs/.asr-offload\""
                )
            ),
        )
        .unwrap();
        fs::write(repo.join("jobs/rules.txt"), b"private vocabulary").unwrap();
        let previous = onboard_at(&initial, &repo, &app, None).unwrap();
        let current = onboard_at(&next, &repo, &app, Some(&previous)).unwrap();
        let config = Config::load(&app.join("config.toml")).unwrap();
        assert_eq!(
            config.resources.offload_dir.unwrap(),
            current.projects_dir().join(".asr-offload")
        );
        assert_eq!(
            config.review_rules.unwrap(),
            current.projects_dir().join("rules.txt")
        );
        assert_eq!(
            fs::read(current.projects_dir().join("run/result.json")).unwrap(),
            b"result"
        );
        assert!(!initial.join("projects").exists());
    }
}
