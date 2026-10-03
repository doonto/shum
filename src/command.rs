use std::env;
use std::ffi::OsStr;
use std::fs::{self, DirBuilder};
use std::os::unix::fs::{DirBuilderExt, PermissionsExt, symlink};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::onboarding;

pub struct Registration {
    pub launcher: PathBuf,
    pub available: bool,
    pub profiles: Vec<PathBuf>,
}

pub fn available(binary: &Path, path: &OsStr) -> bool {
    let Ok(target) = binary.canonicalize() else {
        return false;
    };
    for directory in env::split_paths(path) {
        let candidate = directory.join("shum");
        if fs::metadata(&candidate)
            .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        {
            return candidate.canonicalize().is_ok_and(|p| p == target);
        }
    }
    false
}

pub fn default_directory(binary: &Path, path: &OsStr) -> Result<PathBuf> {
    if let Some(parent) = binary.parent()
        && env::split_paths(path).any(|p| p.is_absolute() && p == parent)
    {
        return Ok(parent.to_path_buf());
    }
    let home = PathBuf::from(env::var_os("HOME").context("HOME is unset")?);
    ensure!(home.is_absolute(), "HOME must be absolute");
    let local = home.join(".local/bin");
    let bin = home.join("bin");
    if env::split_paths(path).any(|p| p == bin) && !env::split_paths(path).any(|p| p == local) {
        Ok(bin)
    } else {
        Ok(local)
    }
}

pub fn setup(binary: &Path, directory: Option<&Path>, setup_shell: bool) -> Result<Registration> {
    let path = env::var_os("PATH").unwrap_or_default();
    let directory = match directory {
        Some(directory) => {
            ensure!(
                directory.is_absolute(),
                "command directory must be absolute"
            );
            directory.to_path_buf()
        }
        None => default_directory(binary, &path)?,
    };
    let mut registration = register(binary, &directory, &path)?;
    if setup_shell {
        let home = PathBuf::from(env::var_os("HOME").context("HOME is unset")?);
        let shell = PathBuf::from(env::var_os("SHELL").context("SHELL is unset")?);
        let zdotdir = env::var_os("ZDOTDIR").map(PathBuf::from);
        registration.profiles = configure_profiles(
            registration
                .launcher
                .parent()
                .context("command has no directory")?,
            &home,
            &shell,
            zdotdir.as_deref(),
        )
        .with_context(|| {
            format!(
                "command registered at {}, but shell PATH setup failed",
                registration.launcher.display()
            )
        })?;
    }
    Ok(registration)
}

fn register(binary: &Path, directory: &Path, path: &OsStr) -> Result<Registration> {
    let metadata = fs::symlink_metadata(binary)
        .with_context(|| format!("installed binary is missing: {}", binary.display()))?;
    ensure!(
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0,
        "installed binary must be a regular executable: {}",
        binary.display()
    );
    let binary = binary.canonicalize()?;
    let directory = onboarding::canonical_target(directory)?;
    onboarding::ensure_not_source_tree(&directory)?;
    let text = directory
        .to_str()
        .context("command directory must be UTF-8")?;
    ensure!(
        !text.contains(':'),
        "command directory cannot contain the PATH separator ':'"
    );
    let launcher = directory.join("shum");
    if launcher != binary {
        match fs::symlink_metadata(&launcher) {
            Ok(metadata) => ensure!(
                metadata.file_type().is_symlink()
                    && launcher.canonicalize().is_ok_and(|p| p == binary),
                "command {} already exists and points elsewhere; refusing to replace it. Choose another --command-dir or explicitly remove the conflicting command",
                launcher.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                DirBuilder::new()
                    .recursive(true)
                    .mode(0o700)
                    .create(&directory)?;
                symlink(&binary, &launcher)
                    .with_context(|| format!("cannot register command {}", launcher.display()))?;
            }
            Err(error) => return Err(error).context("cannot inspect existing command"),
        }
    }
    Ok(Registration {
        available: available(&binary, path),
        launcher,
        profiles: Vec::new(),
    })
}

fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

pub fn export_line(directory: &Path) -> Result<String> {
    Ok(format!(
        "export PATH={}:\"$PATH\"",
        quote(
            directory
                .to_str()
                .context("command directory must be UTF-8")?
        )
    ))
}

fn configure_profiles(
    directory: &Path,
    home: &Path,
    shell: &Path,
    zdotdir: Option<&Path>,
) -> Result<Vec<PathBuf>> {
    ensure!(home.is_absolute(), "HOME must be absolute");
    let profiles = match shell.file_name().and_then(OsStr::to_str) {
        Some("bash") => {
            let login = [".bash_profile", ".bash_login"]
                .iter()
                .map(|name| home.join(name))
                .find(|path| fs::symlink_metadata(path).is_ok())
                .unwrap_or_else(|| home.join(".profile"));
            vec![home.join(".bashrc"), login]
        }
        Some("zsh") => {
            let root = zdotdir.unwrap_or(home);
            ensure!(root.is_absolute(), "ZDOTDIR must be absolute");
            vec![root.join(".zshrc"), root.join(".zprofile")]
        }
        _ => anyhow::bail!(
            "automatic PATH setup supports Bash and Zsh only; configure your shell manually using {}",
            export_line(directory)?
        ),
    };
    let directory = directory
        .to_str()
        .context("command directory must be UTF-8")?;
    let line = format!(
        "case \"$PATH\" in {}*|{}) ;; *) export PATH={}:\"$PATH\" ;; esac",
        quote(&format!("{directory}:")),
        quote(directory),
        quote(directory),
    );
    let mut updates = Vec::new();
    for profile in &profiles {
        let existing = match fs::symlink_metadata(profile) {
            Ok(metadata) => {
                ensure!(
                    metadata.is_file(),
                    "shell profile {} is not a regular file; configure PATH manually without replacing it",
                    profile.display()
                );
                Some(
                    fs::read_to_string(profile).with_context(|| {
                        format!("cannot read shell profile {}", profile.display())
                    })?,
                )
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error).context("cannot inspect shell profile"),
        };
        if existing
            .as_ref()
            .is_some_and(|s| s.lines().any(|l| l == line))
        {
            continue;
        }
        let mut text = existing.clone().unwrap_or_default();
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str("\n# Shum command PATH (managed)\n");
        text.push_str(&line);
        text.push('\n');
        updates.push((profile, text, existing.is_some()));
    }
    for (profile, text, existed) in updates {
        if existed {
            let backup = onboarding::backup_file(profile)?;
            eprintln!("Shell profile backup: {}", backup.display());
        }
        onboarding::atomic_private_write(profile, &text)?;
    }
    Ok(profiles)
}

pub fn print_registration(registration: &Registration) -> Result<()> {
    println!("Command: {}", registration.launcher.display());
    for profile in &registration.profiles {
        println!("Shell PATH configured: {}", profile.display());
    }
    if registration.available {
        println!("Terminal command `shum` is available on PATH.");
    } else {
        let directory = registration
            .launcher
            .parent()
            .context("command has no directory")?;
        println!(
            "Command registration is saved, but this terminal's PATH does not select it.\nRun in your current terminal:\n  {}\nThen run `shum --help`.",
            export_line(directory)?
        );
        if registration.profiles.is_empty() {
            println!(
                "For future Bash/Zsh terminals, run:\n  {} setup-command --command-dir {} --setup-shell",
                quote(
                    registration
                        .launcher
                        .to_str()
                        .context("command path must be UTF-8")?
                ),
                quote(
                    directory
                        .to_str()
                        .context("command directory must be UTF-8")?
                )
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = env::temp_dir().join(format!(
                "shum-command-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn command_registration_is_idempotent_and_resolves_by_name() {
        let temp = Temp::new();
        let directory = temp.0.join("commands with spaces");
        let binary = env::current_exe().unwrap();
        let path = directory.as_os_str();
        let first = register(&binary, &directory, path).unwrap();
        assert!(first.available);
        let second = register(&binary, &directory, path).unwrap();
        assert_eq!(first.launcher, second.launcher);
        assert_eq!(first.launcher.canonicalize().unwrap(), binary);
        assert!(
            register(
                &binary,
                binary.parent().unwrap(),
                binary.parent().unwrap().as_os_str()
            )
            .is_err()
        );
    }

    #[test]
    fn foreign_regular_dangling_and_other_links_are_never_replaced() {
        let temp = Temp::new();
        let entry = temp.0.join("shum");
        let binary = env::current_exe().unwrap();
        fs::write(&entry, "foreign command").unwrap();
        assert!(register(&binary, &temp.0, temp.0.as_os_str()).is_err());
        assert_eq!(fs::read_to_string(&entry).unwrap(), "foreign command");
        fs::remove_file(&entry).unwrap();
        symlink(temp.0.join("missing"), &entry).unwrap();
        assert!(register(&binary, &temp.0, temp.0.as_os_str()).is_err());
        assert_eq!(fs::read_link(&entry).unwrap(), temp.0.join("missing"));
        fs::remove_file(&entry).unwrap();
        symlink("/bin/true", &entry).unwrap();
        assert!(register(&binary, &temp.0, temp.0.as_os_str()).is_err());
        assert_eq!(fs::read_link(&entry).unwrap(), Path::new("/bin/true"));
    }

    #[test]
    fn earlier_foreign_command_is_reported_as_not_available() {
        let temp = Temp::new();
        let foreign = temp.0.join("foreign");
        fs::create_dir(&foreign).unwrap();
        symlink("/bin/true", foreign.join("shum")).unwrap();
        let own = temp.0.join("own");
        let path = env::join_paths([&foreign, &own]).unwrap();
        let registration = register(&env::current_exe().unwrap(), &own, &path).unwrap();
        assert!(!registration.available);
    }

    #[test]
    fn profiles_preserve_content_quote_paths_and_do_not_duplicate_path() {
        let temp = Temp::new();
        let directory = temp.0.join("commands with space ' $ ü");
        let binary = env::current_exe().unwrap();
        register(&binary, &directory, OsStr::new("/usr/bin:/bin")).unwrap();
        let rc = temp.0.join(".bashrc");
        fs::write(&rc, "# existing user settings\nexport KEEP_SHUM_TEST=yes").unwrap();
        configure_profiles(&directory, &temp.0, Path::new("/bin/bash"), None).unwrap();
        let text = fs::read_to_string(&rc).unwrap();
        assert!(text.starts_with("# existing user settings\nexport KEEP_SHUM_TEST=yes\n"));
        let backups: Vec<_> = fs::read_dir(&temp.0)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .contains(".bak-")
            })
            .collect();
        assert_eq!(backups.len(), 1);
        assert_eq!(
            fs::metadata(&backups[0]).unwrap().permissions().mode() & 0o077,
            0
        );
        configure_profiles(&directory, &temp.0, Path::new("/bin/bash"), None).unwrap();
        assert_eq!(text, fs::read_to_string(&rc).unwrap());
        let result = Command::new("/bin/bash")
            .args([
                "--noprofile",
                "--norc",
                "-c",
                ". \"$1\"; . \"$1\"; command -v shum; printf '%s\\n' \"$KEEP_SHUM_TEST\" \"$PATH\"",
                "shum-test",
            ])
            .arg(&rc)
            .env("PATH", "/usr/bin:/bin")
            .output()
            .unwrap();
        assert!(result.status.success());
        let output = String::from_utf8(result.stdout).unwrap();
        assert_eq!(
            output.lines().next().unwrap(),
            directory.join("shum").to_str().unwrap()
        );
        assert!(output.lines().nth(1).is_some_and(|s| s == "yes"));
        assert_eq!(
            output.lines().nth(2).unwrap(),
            format!("{}:/usr/bin:/bin", directory.display())
        );
    }

    #[test]
    fn shell_profiles_reject_symlinks_before_any_changes() {
        let temp = Temp::new();
        let rc = temp.0.join(".bashrc");
        fs::write(&rc, "# keep\n").unwrap();
        let foreign = temp.0.join("foreign");
        fs::write(&foreign, "# do not change\n").unwrap();
        symlink(&foreign, temp.0.join(".profile")).unwrap();
        assert!(configure_profiles(&temp.0, &temp.0, Path::new("/bin/bash"), None).is_err());
        assert_eq!(fs::read_to_string(&rc).unwrap(), "# keep\n");
        assert_eq!(fs::read_to_string(&foreign).unwrap(), "# do not change\n");
    }

    #[test]
    fn bash_login_precedence_zdotdir_and_unsupported_shell_are_explicit() {
        let temp = Temp::new();
        fs::write(temp.0.join(".bash_profile"), "# login\n").unwrap();
        let paths = configure_profiles(&temp.0, &temp.0, Path::new("/bin/bash"), None).unwrap();
        assert_eq!(paths[1], temp.0.join(".bash_profile"));
        assert!(!temp.0.join(".profile").exists());
        let zdir = temp.0.join("zsh");
        let paths =
            configure_profiles(&temp.0, &temp.0, Path::new("/bin/zsh"), Some(&zdir)).unwrap();
        assert_eq!(paths, vec![zdir.join(".zshrc"), zdir.join(".zprofile")]);
        assert!(configure_profiles(&temp.0, &temp.0, Path::new("/bin/fish"), None).is_err());
        assert!(
            configure_profiles(
                &temp.0,
                &temp.0,
                Path::new("/bin/zsh"),
                Some(Path::new("relative"))
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_command_directories_fail_without_writing() {
        let temp = Temp::new();
        let binary = env::current_exe().unwrap();
        assert!(setup(&binary, Some(Path::new("relative")), false).is_err());
        let invalid = temp.0.join("colon:directory");
        assert!(register(&binary, &invalid, invalid.as_os_str()).is_err());
        assert!(!invalid.exists());
    }

    #[test]
    fn installed_binary_already_on_path_needs_no_symlink() {
        let temp = Temp::new();
        let binary = temp.0.join("shum");
        fs::copy(env::current_exe().unwrap(), &binary).unwrap();
        let registration = register(&binary, &temp.0, temp.0.as_os_str()).unwrap();
        assert!(registration.available);
        assert_eq!(registration.launcher, binary);
        assert!(
            !fs::symlink_metadata(&binary)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }
}
