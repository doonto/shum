use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

use crate::config::Config;

const FILES: &[(&str, &str)] = &[
    ("README.md", include_str!("../README.md")),
    (
        "adapters/asr/main.py",
        include_str!("../adapters/asr/main.py"),
    ),
    (
        "adapters/asr/flix_fast_common.py",
        include_str!("../adapters/asr/flix_fast_common.py"),
    ),
    (
        "adapters/asr/flix_fast_convert.py",
        include_str!("../adapters/asr/flix_fast_convert.py"),
    ),
    (
        "adapters/asr/flix_fast_worker.py",
        include_str!("../adapters/asr/flix_fast_worker.py"),
    ),
    (
        "adapters/asr/flix_onnx_probe.py",
        include_str!("../adapters/asr/flix_onnx_probe.py"),
    ),
    (
        "adapters/asr/flix_onnx_pilot.py",
        include_str!("../adapters/asr/flix_onnx_pilot.py"),
    ),
    (
        "adapters/asr/flix_onnx_audit.py",
        include_str!("../adapters/asr/flix_onnx_audit.py"),
    ),
    (
        "adapters/asr/flix_onnx_README.md",
        include_str!("../adapters/asr/flix_onnx_README.md"),
    ),
    (
        "scripts/flix_study.py",
        include_str!("../scripts/flix_study.py"),
    ),
    (
        "scripts/flix_matrix.py",
        include_str!("../scripts/flix_matrix.py"),
    ),
    (
        "scripts/asr_engine_study.py",
        include_str!("../scripts/asr_engine_study.py"),
    ),
    (
        "scripts/shum_headless.py",
        include_str!("../scripts/shum_headless.py"),
    ),
    (
        "scripts/onboarding-smoke.sh",
        include_str!("../scripts/onboarding-smoke.sh"),
    ),
    ("docs/USAGE.ru.md", include_str!("../docs/USAGE.ru.md")),
    ("docs/API.md", include_str!("../docs/API.md")),
    (
        "adapters/asr/qwen_gguf_worker.py",
        include_str!("../adapters/asr/qwen_gguf_worker.py"),
    ),
    (
        "adapters/asr/primeline_onnx_worker.py",
        include_str!("../adapters/asr/primeline_onnx_worker.py"),
    ),
    (
        "adapters/asr/primeline_onnx_README.md",
        include_str!("../adapters/asr/primeline_onnx_README.md"),
    ),
    (
        "adapters/asr/primeline_onnx_requirements.txt",
        include_str!("../adapters/asr/primeline_onnx_requirements.txt"),
    ),
    (
        "adapters/asr/turbo_worker.py",
        include_str!("../adapters/asr/turbo_worker.py"),
    ),
    (
        "adapters/asr/turbo_ct2_worker.py",
        include_str!("../adapters/asr/turbo_ct2_worker.py"),
    ),
    (
        "adapters/asr/turbo_onnx_worker.py",
        include_str!("../adapters/asr/turbo_onnx_worker.py"),
    ),
    (
        "adapters/asr/turbo_README.md",
        include_str!("../adapters/asr/turbo_README.md"),
    ),
    (
        "adapters/asr/turbo_requirements.txt",
        include_str!("../adapters/asr/turbo_requirements.txt"),
    ),
    (
        "adapters/asr/qwen_native_worker.py",
        include_str!("../adapters/asr/qwen_native_worker.py"),
    ),
    (
        "adapters/asr/pyproject.toml",
        include_str!("../adapters/asr/pyproject.toml"),
    ),
    (
        "adapters/asr/uv.lock",
        include_str!("../adapters/asr/uv.lock"),
    ),
    (
        "adapters/asr/README.md",
        include_str!("../adapters/asr/README.md"),
    ),
    (
        "adapters/asr/flix_fast_README.md",
        include_str!("../adapters/asr/flix_fast_README.md"),
    ),
    (
        "adapters/asr/flix_fast_requirements.txt",
        include_str!("../adapters/asr/flix_fast_requirements.txt"),
    ),
    (
        "adapters/diarization/main.py",
        include_str!("../adapters/diarization/main.py"),
    ),
    (
        "adapters/diarization/onnx_worker.py",
        include_str!("../adapters/diarization/onnx_worker.py"),
    ),
    (
        "adapters/diarization/requirements.txt",
        include_str!("../adapters/diarization/requirements.txt"),
    ),
    (
        "adapters/diarization/README.md",
        include_str!("../adapters/diarization/README.md"),
    ),
];

pub fn bundled_file(path: &str) -> Option<&'static str> {
    FILES
        .iter()
        .find(|(name, _)| *name == path)
        .map(|(_, contents)| *contents)
}

pub const TEMPLATE: &str = include_str!("../shum.example.toml");

pub fn runtime_dir(root: &Path) -> Result<PathBuf> {
    let mut digest = Sha256::new();
    for (name, contents) in FILES {
        digest.update(name.as_bytes());
        digest.update(contents.as_bytes());
    }
    Ok(root.join("runtime").join(format!(
        "{}-{}",
        env!("CARGO_PKG_VERSION"),
        &hex::encode(digest.finalize())[..12]
    )))
}

pub fn runtime_is_complete(destination: &Path) -> Result<bool> {
    if !destination.is_dir() {
        return Ok(false);
    }
    if fs::symlink_metadata(destination)?.file_type().is_symlink()
        || !destination.join(".complete").is_file()
    {
        return Ok(false);
    }
    let root = destination.canonicalize()?;
    for (name, contents) in FILES {
        let path = destination.join(name);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("cannot verify runtime {}", path.display()));
            }
        };
        if bytes != contents.as_bytes() || !path.canonicalize()?.starts_with(&root) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn publish_runtime(root: &Path) -> Result<PathBuf> {
    let destination = runtime_dir(root)?;
    if destination.exists() {
        ensure!(
            runtime_is_complete(&destination)?,
            "incomplete runtime at {}; repair it manually before proceeding",
            destination.display()
        );
        return Ok(destination);
    }
    if root.join("runtime").exists() {
        ensure!(
            !fs::symlink_metadata(root.join("runtime"))?
                .file_type()
                .is_symlink(),
            "runtime parent must be a real directory, not a symlink"
        );
    }
    fs::create_dir_all(root.join("runtime"))?;
    let staging = destination.with_extension(format!("staging-{}", std::process::id()));
    fs::create_dir(&staging).with_context(|| format!("cannot stage {}", staging.display()))?;
    let result = (|| -> Result<()> {
        for (name, contents) in FILES {
            let path = staging.join(name);
            fs::create_dir_all(path.parent().context("runtime file without parent")?)?;
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            file.write_all(contents.as_bytes())?;
            file.sync_all()?;
        }
        fs::write(staging.join(".complete"), env!("CARGO_PKG_VERSION"))?;
        fs::rename(&staging, &destination)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result?;
    Ok(destination)
}

pub fn repair_runtime(root: &Path, consent: bool) -> Result<PathBuf> {
    ensure!(consent, "runtime repair requires explicit consent");
    if root.join("runtime").exists() {
        ensure!(
            !fs::symlink_metadata(root.join("runtime"))?
                .file_type()
                .is_symlink(),
            "runtime parent must be a real directory, not a symlink"
        );
    }
    let destination = runtime_dir(root)?;
    if !destination.exists() || runtime_is_complete(&destination)? {
        return publish_runtime(root);
    }
    let backup = repair_backup_path(&destination);
    ensure!(
        !backup.exists(),
        "runtime backup already exists: {}",
        backup.display()
    );
    fs::rename(&destination, &backup)?;
    match publish_runtime(root) {
        Ok(runtime) => Ok(runtime),
        Err(error) => {
            if !destination.exists() {
                fs::rename(&backup, &destination).with_context(|| {
                    format!(
                        "runtime repair failed ({error:#}) and cannot restore {}",
                        destination.display()
                    )
                })?;
            }

            Err(error).context("runtime repair failed; previous resources were preserved")
        }
    }
}

pub fn repair_backup_path(destination: &Path) -> PathBuf {
    destination.with_extension(format!("damaged-{}", std::process::id()))
}

pub fn environment_name<'a>(role: &'a str, script: &Path) -> &'a str {
    match script.file_name().and_then(|name| name.to_str()) {
        Some("flix_fast_worker.py") => "flix-fast",
        Some("turbo_worker.py") => "turbo",
        Some("turbo_ct2_worker.py") => "turbo-ct2",
        Some("turbo_onnx_worker.py") => "turbo-onnx",
        Some("primeline_onnx_worker.py") => "primeline-onnx",
        Some("qwen_gguf_worker.py") => "qwen-gguf",
        Some("qwen_native_worker.py") => "qwen-native",
        Some("onnx_worker.py") if role == "diarization" => "diarization-onnx",
        _ => role,
    }
}

pub fn asr_environment_name<'a>(role: &'a str, script: &Path, engine: &'a str) -> &'a str {
    if script.ends_with("adapters/asr/main.py") && !matches!(role, "primeline" | "flix" | "qwen") {
        engine
    } else {
        environment_name(role, script)
    }
}

pub fn prepare_environments(root: &Path, runtime: &Path, config: &Config) -> Result<()> {
    let asr = runtime.join("adapters/asr");
    let environments = root.join("runtime/environments");
    fs::create_dir_all(&environments)?;
    for role in config.asr_roles() {
        let adapter = config.asr(&role);
        let engine = config.asr_engine(&role)?;
        let venv = environments.join(asr_environment_name(&role, &adapter.script, engine));
        if adapter.script != asr.join("main.py") || adapter.program != venv.join("bin/python") {
            eprintln!(
                "{role}: custom/experimental adapter environment is not changed; prepare {} using the bundled adapter instructions",
                adapter.program.display()
            );
            continue;
        }
        let marker = venv.join(".shum-runtime");
        if !venv.join("bin/python").exists() {
            run_uv(
                [
                    "venv",
                    "--python",
                    "3.11",
                    venv.to_str().context("non-UTF8 environment path")?,
                ],
                None,
            )?;
        }
        if fs::read_to_string(&marker).ok().as_deref()
            != Some(runtime.to_str().context("non-UTF8 runtime path")?)
        {
            run_uv(
                [
                    "sync", "--frozen", "--no-dev", "--extra", engine, "--active",
                ],
                Some((&asr, &venv)),
            )?;
            fs::write(marker, runtime.to_str().context("non-UTF8 runtime path")?)?;
        }
    }
    let venv = environments.join("diarization");
    if config.diarization.script != runtime.join("adapters/diarization/main.py")
        || config.diarization.program != venv.join("bin/python")
    {
        eprintln!(
            "diarization: custom/experimental adapter environment is not changed; prepare {} using the bundled adapter instructions",
            config.diarization.program.display()
        );
        return Ok(());
    }
    let marker = venv.join(".shum-runtime");
    if !venv.join("bin/python").exists() {
        run_uv(
            [
                "venv",
                "--python",
                "3.11",
                venv.to_str().context("non-UTF8 environment path")?,
            ],
            None,
        )?;
    }
    if fs::read_to_string(&marker).ok().as_deref()
        != Some(runtime.to_str().context("non-UTF8 runtime path")?)
    {
        run_uv(
            [
                "pip",
                "install",
                "--torch-backend",
                "cpu",
                "--python",
                venv.join("bin/python")
                    .to_str()
                    .context("non-UTF8 python path")?,
                "-r",
                runtime
                    .join("adapters/diarization/requirements.txt")
                    .to_str()
                    .context("non-UTF8 requirements path")?,
            ],
            None,
        )?;
        fs::write(marker, runtime.to_str().context("non-UTF8 runtime path")?)?;
    }
    Ok(())
}

fn run_uv<const N: usize>(args: [&str; N], context: Option<(&Path, &Path)>) -> Result<()> {
    let mut cmd = Command::new("uv");
    cmd.args(args);
    if let Some((dir, venv)) = context {
        cmd.current_dir(dir).env("VIRTUAL_ENV", venv);
    }
    let status = cmd
        .status()
        .context("uv is required to prepare Python environments")?;
    ensure!(status.success(), "uv failed with status {status}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_publishes_atomically_and_reuses_existing_runtime() {
        let temp = tempfile::Builder::new()
            .prefix(".shum-test-")
            .tempdir_in(std::env::current_dir().unwrap())
            .unwrap();
        let first = publish_runtime(temp.path()).unwrap();
        assert!(first.join("adapters/asr/uv.lock").is_file());
        assert!(first.join("adapters/diarization/onnx_worker.py").is_file());
        assert!(first.join("adapters/asr/flix_onnx_probe.py").is_file());
        assert!(
            first
                .join("adapters/asr/primeline_onnx_worker.py")
                .is_file()
        );
        assert!(first.join("adapters/asr/turbo_worker.py").is_file());
        assert!(first.join("adapters/asr/turbo_requirements.txt").is_file());
        assert!(first.join("scripts/flix_study.py").is_file());
        assert_eq!(publish_runtime(temp.path()).unwrap(), first);
        fs::remove_file(first.join(".complete")).unwrap();
        assert!(publish_runtime(temp.path()).is_err());
    }

    #[test]
    fn damaged_runtime_repair_preserves_previous_resources() {
        let temp = tempfile::tempdir().unwrap();
        let runtime = publish_runtime(temp.path()).unwrap();
        fs::write(runtime.join("adapters/asr/main.py"), "damaged").unwrap();
        assert!(!runtime_is_complete(&runtime).unwrap());
        assert!(repair_runtime(temp.path(), false).is_err());
        assert_eq!(repair_runtime(temp.path(), true).unwrap(), runtime);
        assert!(runtime_is_complete(&runtime).unwrap());
        let backup = runtime.with_extension(format!("damaged-{}", std::process::id()));
        assert_eq!(
            fs::read_to_string(backup.join("adapters/asr/main.py")).unwrap(),
            "damaged"
        );
    }

    #[cfg(unix)]
    #[test]
    fn runtime_repair_rejects_a_symlinked_parent() {
        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), temp.path().join("runtime")).unwrap();
        assert!(publish_runtime(temp.path()).is_err());
        assert!(repair_runtime(temp.path(), true).is_err());
        assert!(fs::read_dir(outside.path()).unwrap().next().is_none());
    }

    #[test]
    fn experimental_engines_keep_separate_environment_profiles() {
        assert_eq!(
            environment_name("primeline", Path::new("primeline_onnx_worker.py")),
            "primeline-onnx"
        );
        assert_eq!(
            environment_name("flix", Path::new("turbo_worker.py")),
            "turbo"
        );
        assert_eq!(environment_name("flix", Path::new("main.py")), "flix");
        assert_eq!(
            environment_name("meeting", Path::new("turbo_worker.py")),
            "turbo"
        );
        assert_eq!(
            environment_name("flix", Path::new("flix_fast_worker.py")),
            "flix-fast"
        );
        assert_eq!(
            environment_name("qwen", Path::new("qwen_gguf_worker.py")),
            "qwen-gguf"
        );
        assert_eq!(
            environment_name("diarization", Path::new("onnx_worker.py")),
            "diarization-onnx"
        );
        assert_eq!(
            asr_environment_name("meeting", Path::new("adapters/asr/main.py"), "whisper"),
            "whisper"
        );
        assert_eq!(
            asr_environment_name("flix", Path::new("adapters/asr/main.py"), "whisper"),
            "flix"
        );
        assert_eq!(
            asr_environment_name(
                "meeting",
                Path::new("adapters/asr/turbo_worker.py"),
                "whisper"
            ),
            "turbo"
        );
    }
}
