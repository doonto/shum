use std::fs::{self, File};
use std::io::{IsTerminal, Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use wait_timeout::ChildExt;

use crate::config::{Llm, LlmProvider};
use crate::process::{self, OwnedProcess, TaskContext};

const PROBE_PROMPT: &str = "Reply with exactly SHUM_READY. Do not call tools or read files.";
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);
const REVIEW_TIMEOUT: Duration = Duration::from_secs(300);
static READINESS_CACHE: OnceLock<Mutex<std::collections::HashMap<Vec<u8>, Readiness>>> =
    OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readiness {
    Ready,
    Disabled,
    Unavailable,
    NeedsLogin,
    Network,
    Quota,
    UnsupportedModel,
    InvalidResponse,
    Unverified,
    TimedOut,
    Failed,
}

impl Readiness {
    pub fn help(self, provider: LlmProvider) -> &'static str {
        match self {
            Self::Ready => "LLM is ready.",
            Self::Disabled => "LLM is disabled; review can be resumed after setup.",
            Self::Unavailable => "LLM CLI was not found. Install it or choose another provider.",
            Self::NeedsLogin => match provider {
                LlmProvider::Codex => "Sign in using `codex login` in a terminal, then retry.",
                LlmProvider::Copilot => "Sign in using `copilot login` in a terminal, then retry.",
                _ => "Sign in using the selected provider CLI in a terminal, then retry.",
            },
            Self::Network | Self::TimedOut => "Check network connectivity and retry.",
            Self::Quota => "Provider quota is exhausted; retry later or change provider.",
            Self::UnsupportedModel => "Selected model is not available; choose another model.",
            Self::InvalidResponse => "The CLI did not return a compatible response.",
            Self::Unverified => {
                "Custom CLI has no availability probe; access is unverified. Configure probe_args to check it."
            }
            Self::Failed => "The provider failed; inspect its own CLI diagnostics.",
        }
    }
}

fn provider(config: &Llm) -> LlmProvider {
    match config.provider {
        LlmProvider::Auto => LlmProvider::Custom,
        explicit => explicit,
    }
}

fn cache_key(config: &Llm) -> Option<Vec<u8>> {
    serde_json::to_vec(config).ok()
}

fn readiness_cache() -> &'static Mutex<std::collections::HashMap<Vec<u8>, Readiness>> {
    READINESS_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn program_available(program: &Path) -> bool {
    fn executable(path: &Path) -> bool {
        let Ok(metadata) = path.metadata() else {
            return false;
        };
        if !metadata.is_file() {
            return false;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            metadata.permissions().mode() & 0o111 != 0
        }
        #[cfg(not(unix))]
        {
            true
        }
    }
    if program.components().count() != 1 || program.is_absolute() {
        return executable(program);
    }
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|entry| executable(&entry.join(program)))
    })
}

fn build_command(config: &Llm, output: Option<&Path>, cwd: Option<&Path>) -> Command {
    let mut cmd = Command::new(&config.program);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    match provider(config) {
        LlmProvider::Codex => {
            cmd.args([
                "exec",
                "--json",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--disable",
                "shell_tool",
                "--disable",
                "unified_exec",
                "--disable",
                "code_mode_host",
                "--disable",
                "browser_use",
                "--disable",
                "computer_use",
            ]);
            if let Some(model) = &config.model {
                cmd.arg("--model").arg(model);
            }
            cmd.arg("-");
        }
        LlmProvider::Copilot => {
            cmd.args([
                "--silent",
                "--disable-builtin-mcps",
                "--no-custom-instructions",
                "--disallow-temp-dir",
                "--available-tools=none",
                "--deny-tool=shell",
                "--deny-tool=write",
                "--deny-tool=url",
            ]);
            if let Some(model) = &config.model {
                cmd.arg("--model").arg(model);
            }
            // Copilot's -p flag takes the prompt as an argument. It does not
            // accept stdin as a replacement for -p in noninteractive mode.
        }
        LlmProvider::Auto | LlmProvider::Custom => {
            for arg in &config.args {
                let value = match output {
                    Some(path) => arg.replace("{output}", &path.to_string_lossy()),
                    None => arg.clone(),
                };
                cmd.arg(match &config.model {
                    Some(model) => value.replace("{model}", model),
                    None => value,
                });
            }
        }
    }
    cmd
}

fn codex_message(stdout: &[u8]) -> Option<String> {
    stdout
        .split(|byte| *byte == b'\n')
        .filter_map(|line| serde_json::from_slice::<Value>(line).ok())
        .filter_map(|event| {
            if event.get("type")?.as_str()? != "item.completed" {
                return None;
            }
            let item = event.get("item")?;
            (item.get("type")?.as_str()? == "agent_message")
                .then(|| item.get("text")?.as_str().map(str::to_owned))
                .flatten()
        })
        .next_back()
}

fn classify_error(stderr: &[u8], stdout: &[u8]) -> Readiness {
    let text = String::from_utf8_lossy(stderr).to_ascii_lowercase()
        + &String::from_utf8_lossy(stdout).to_ascii_lowercase();
    if [
        "not logged in",
        "not authenticated",
        "authentication required",
        "please log in",
        "please login",
        "sign in",
        "unauthorized",
        "invalid api key",
        "401",
    ]
    .iter()
    .any(|s| text.contains(s))
    {
        Readiness::NeedsLogin
    } else if [
        "quota",
        "rate limit",
        "too many requests",
        "429",
        "credit balance",
    ]
    .iter()
    .any(|s| text.contains(s))
    {
        Readiness::Quota
    } else if [
        "unsupported model",
        "model not found",
        "model is not available",
        "invalid model",
        "model does not exist",
        "model access",
        "model_not_found",
    ]
    .iter()
    .any(|s| text.contains(s))
    {
        Readiness::UnsupportedModel
    } else if [
        "connection",
        "dns",
        "network",
        "timed out",
        "timeout",
        "tls",
        "offline",
        "503",
        "502",
    ]
    .iter()
    .any(|s| text.contains(s))
    {
        Readiness::Network
    } else if [
        "unknown option",
        "unrecognized option",
        "unexpected argument",
        "unknown flag",
    ]
    .iter()
    .any(|s| text.contains(s))
    {
        Readiness::InvalidResponse
    } else {
        Readiness::Failed
    }
}

fn capture(pipe: impl Read + Send + 'static) -> Receiver<std::io::Result<Vec<u8>>> {
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = (|| {
            let mut result = Vec::new();
            let mut pipe = pipe;
            let mut buffer = [0_u8; 8192];
            loop {
                let count = pipe.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                let allowed = 65536usize.saturating_sub(result.len()).min(count);
                result.extend_from_slice(&buffer[..allowed]);
            }
            Ok(result)
        })();
        let _ = sender.send(result);
    });
    receiver
}

pub fn cached_readiness(config: &Llm) -> Option<Readiness> {
    let key = cache_key(config)?;
    readiness_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key)
        .copied()
}

/// Performs one bounded, real inference using only a fixed technical prompt.
/// The caller should explain that this may consume provider quota before invoking it.
pub fn readiness(config: &Llm) -> Readiness {
    let state = readiness_with_timeout(config, PROBE_TIMEOUT);
    if let Some(key) = cache_key(config) {
        readiness_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key, state);
    }
    state
}

fn readiness_with_timeout(config: &Llm, timeout: Duration) -> Readiness {
    if !config.enabled {
        return Readiness::Disabled;
    }
    if config.program.as_os_str().is_empty() {
        return Readiness::Unavailable;
    }
    if provider(config) == LlmProvider::Custom && config.probe_args.is_none() {
        return if program_available(&config.program) {
            Readiness::Unverified
        } else {
            Readiness::Unavailable
        };
    }
    if config
        .probe_args
        .as_ref()
        .is_some_and(|args| args.iter().any(|arg| arg.contains("{output}")))
    {
        return Readiness::InvalidResponse;
    }
    let mut probe = config.clone();
    if let Some(args) = &config.probe_args {
        probe.args = args.clone();
    }
    let mut command = build_command(&probe, None, None);
    #[cfg(unix)]
    command.current_dir("/");
    if provider(config) == LlmProvider::Copilot {
        command.arg("--prompt").arg(PROBE_PROMPT);
    }
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Readiness::Unavailable;
        }
        Err(_) => return Readiness::Failed,
    };
    let stdout = capture(child.stdout.take().expect("piped stdout"));
    let stderr = capture(child.stderr.take().expect("piped stderr"));
    let mut stdin = child.stdin.take().expect("piped stdin");
    let send_stdin = provider(config) != LlmProvider::Copilot;
    // A separate writer ensures a CLI that never reads stdin cannot block the timeout.
    let (send, written) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = if send_stdin {
            stdin.write_all(PROBE_PROMPT.as_bytes())
        } else {
            Ok(())
        };
        let _ = send.send(result);
    });
    let status = match child.wait_timeout(timeout) {
        Ok(Some(status)) => status,
        Ok(None) => {
            let _ = child.kill();
            let _ = child.wait();
            return Readiness::TimedOut;
        }
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            return Readiness::Failed;
        }
    };
    let stdout = match stdout.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok(stdout)) => stdout,
        Ok(Err(_)) => return Readiness::Failed,
        Err(_) => return Readiness::TimedOut,
    };
    let stderr = match stderr.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok(stderr)) => stderr,
        Ok(Err(_)) => return Readiness::Failed,
        Err(_) => return Readiness::TimedOut,
    };
    if !status.success() {
        return classify_error(&stderr, &stdout);
    }
    if !matches!(written.recv_timeout(Duration::from_secs(2)), Ok(Ok(()))) {
        return Readiness::Failed;
    }
    let response = if provider(config) == LlmProvider::Codex {
        codex_message(&stdout)
    } else {
        String::from_utf8(stdout).ok()
    };
    match response {
        Some(text) if text.trim() == "SHUM_READY" => Readiness::Ready,
        _ => Readiness::InvalidResponse,
    }
}

/// Start the provider's own interactive login, only with explicit consent.
/// Shum never accepts, reads, or stores credentials.
pub fn login(config: &Llm, consent: bool) -> Result<()> {
    ensure!(consent, "provider login requires explicit user consent");
    ensure!(
        std::io::stdin().is_terminal() && std::io::stdout().is_terminal(),
        "provider login requires an interactive terminal"
    );
    ensure!(
        !config.program.as_os_str().is_empty(),
        "LLM CLI is not configured"
    );
    let action = match provider(config) {
        LlmProvider::Codex | LlmProvider::Copilot => "login",
        _ => bail!("custom CLI login must be performed through the provider's own instructions"),
    };
    let status = Command::new(&config.program)
        .arg(action)
        .status()
        .with_context(|| format!("cannot start LLM CLI {}", config.program.display()))?;
    ensure!(status.success(), "provider login did not complete");
    Ok(())
}

/// Writes the raw model response to `output`. The caller validates its
/// Decisions schema before publishing or checkpointing it.
#[cfg(test)]
pub fn review(config: &Llm, prompt: &str, output: &Path, stderr: &Path) -> Result<()> {
    review_observed(config, prompt, output, stderr, None)
}

fn private_file(path: &Path) -> Result<File> {
    let file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    file.set_permissions(fs::Permissions::from_mode(0o600))?;
    Ok(file)
}

pub fn review_observed(
    config: &Llm,
    prompt: &str,
    output: &Path,
    stderr: &Path,
    context: Option<&TaskContext>,
) -> Result<()> {
    ensure!(config.enabled, "LLM review is disabled");
    ensure!(
        !config.program.as_os_str().is_empty(),
        "LLM CLI is not configured"
    );
    if let Some(key) = cache_key(config)
        && let Some(state) = readiness_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .copied()
    {
        ensure!(
            matches!(state, Readiness::Ready | Readiness::Unverified),
            "LLM review unavailable: {}",
            state.help(config.provider)
        );
    }
    if let Some(context) = context {
        context.check_cancelled()?;
        context.report_operation("review transcript")?;
    }
    let cwd = output
        .parent()
        .context("review path has no parent")?
        .canonicalize()?;
    let output = if output.is_absolute() {
        output.to_path_buf()
    } else {
        cwd.join(output.file_name().context("review path has no filename")?)
    };
    let stdout_path = output.with_extension("stdout");
    if output.exists() {
        fs::remove_file(&output)?;
    }
    let mut command = build_command(config, Some(&output), Some(&cwd));
    if provider(config) == LlmProvider::Copilot {
        command.arg("--prompt").arg(prompt);
    }
    process::configure_owned_command(&mut command);
    // SAFETY: umask changes only the child before exec, keeping provider-created replies private.
    unsafe {
        command.pre_exec(|| {
            libc::umask(0o077);
            Ok(())
        });
    }
    let spawned = command
        .stdin(Stdio::piped())
        .stdout(private_file(&stdout_path)?)
        .stderr(private_file(stderr)?)
        .spawn();
    let child = match spawned {
        Ok(child) => child,
        Err(error) => {
            let _ = fs::remove_file(&stdout_path);
            return Err(error)
                .with_context(|| format!("cannot start LLM CLI {}", config.program.display()));
        }
    };
    let mut owned = OwnedProcess::new(child, None);
    let mut stdin = owned
        .child_mut()
        .stdin
        .take()
        .context("LLM stdin missing")?;
    let send_stdin = provider(config) != LlmProvider::Copilot;
    let prompt = prompt.to_owned();
    let (send, written) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = if send_stdin {
            stdin.write_all(prompt.as_bytes())
        } else {
            Ok(())
        };
        let _ = send.send(result);
    });
    let started = std::time::Instant::now();
    let status = (|| -> Result<_> {
        loop {
            if let Some(context) = context {
                context.check_cancelled()?;
            }
            if owned.try_exited()? {
                owned.stop()?;
                let status = owned.child_mut().wait()?;
                break Ok(status);
            }
            if started.elapsed() >= REVIEW_TIMEOUT {
                owned.stop()?;
                bail!("LLM timed out; see {}", stderr.display());
            }
            thread::sleep(Duration::from_millis(40));
        }
    })();
    let status = match status {
        Ok(status) => status,
        Err(error) => {
            let _ = fs::remove_file(&stdout_path);
            return Err(error);
        }
    };
    if !matches!(written.recv_timeout(Duration::from_secs(2)), Ok(Ok(()))) {
        let _ = fs::remove_file(&stdout_path);
        bail!("LLM did not consume the prompt; see {}", stderr.display());
    }
    if !status.success() {
        let _ = fs::remove_file(&stdout_path);
        bail!("LLM exited with {status}; see {}", stderr.display());
    }
    if provider(config) == LlmProvider::Codex {
        let result = (|| {
            let events = fs::read(&stdout_path)?;
            let response = codex_message(&events).context("Codex returned no final message")?;
            private_file(&output)?.write_all(response.as_bytes())?;
            Ok::<(), anyhow::Error>(())
        })();
        let _ = fs::remove_file(&stdout_path);
        result?;
    } else if output.exists() {
        fs::remove_file(&stdout_path)?;
    } else {
        fs::rename(&stdout_path, &output).context("LLM did not produce a response")?;
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn probe_capture_reports_stream_read_errors() {
        struct BrokenPipe;
        impl Read for BrokenPipe {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("read failed"))
            }
        }
        assert!(
            capture(BrokenPipe)
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .is_err()
        );
    }

    fn cli(script: &str) -> (tempfile::TempDir, Llm) {
        let dir = tempfile::Builder::new()
            .prefix("shum-llm-test-")
            .tempdir_in(".")
            .unwrap();
        let path = dir.path().canonicalize().unwrap().join("mock-cli");
        fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        thread::sleep(Duration::from_millis(10));
        let mut config = Llm::preset(LlmProvider::Custom);
        config.program = path;
        config.probe_args = Some(vec![]);
        (dir, config)
    }

    #[test]
    fn disabled_or_missing_cli_never_probes() {
        let (_dir, mut config) = cli("exit 99");
        config.enabled = false;
        assert_eq!(readiness(&config), Readiness::Disabled);
        config.enabled = true;
        config.program = "a-cli-that-does-not-exist".into();
        assert_eq!(readiness(&config), Readiness::Unavailable);
        config.probe_args = None;
        assert_eq!(readiness(&config), Readiness::Unavailable);
        config.program = _dir.path().canonicalize().unwrap().join("mock-cli");
        assert_eq!(readiness(&config), Readiness::Unverified);
        assert!(login(&config, false).is_err());
    }

    #[test]
    fn readiness_sends_only_technical_probe_and_classifies_failures() {
        let (dir, mut config) = cli("cat > \"$1\"; printf SHUM_READY");
        let recorded = dir.path().canonicalize().unwrap().join("probe");
        config.probe_args = Some(vec![recorded.to_string_lossy().to_string()]);
        assert_eq!(readiness(&config), Readiness::Ready);
        assert_eq!(fs::read_to_string(&recorded).unwrap(), PROBE_PROMPT);
        for (diagnostic, expected) in [
            ("authentication required", Readiness::NeedsLogin),
            ("connection refused", Readiness::Network),
            ("rate limit exceeded", Readiness::Quota),
            ("unsupported model", Readiness::UnsupportedModel),
            ("unknown option", Readiness::InvalidResponse),
            ("unknown failure", Readiness::Failed),
        ] {
            let (_dir, config) = cli(&format!("echo '{diagnostic}' >&2; exit 1"));
            assert_eq!(readiness(&config), expected, "{diagnostic}");
        }
    }

    #[test]
    fn timeout_and_invalid_response() {
        let (_dir, config) = cli("sleep 1");
        assert_eq!(
            readiness_with_timeout(&config, Duration::from_millis(40)),
            Readiness::TimedOut
        );
        let (_dir, config) = cli("printf 'not the expected response'");
        assert_eq!(readiness(&config), Readiness::InvalidResponse);
    }

    #[test]
    fn custom_review_preserves_output_substitution_and_stdout_fallback() {
        let (dir, mut config) = cli("cat >/dev/null; printf '{\"ok\":true}' > \"$1\"");
        config.args = vec!["{output}".into()];
        let output = dir.path().join("review.partial.json");
        let stderr = dir.path().join("review.stderr");
        review(&config, "project prompt", &output, &stderr).unwrap();
        assert_eq!(fs::read_to_string(&output).unwrap(), "{\"ok\":true}");

        let (_fallback_dir, mut config) = cli("cat >/dev/null; printf '{\"ok\":true}'");
        config.args.clear();
        review(&config, "project prompt", &output, &stderr).unwrap();
        assert_eq!(fs::read_to_string(&output).unwrap(), "{\"ok\":true}");
        config.enabled = false;
        assert!(review(&config, "prompt", &output, &stderr).is_err());
    }

    #[test]
    fn cancelled_review_does_not_publish_partial_response() {
        let (dir, mut config) = cli("cat >/dev/null; sleep 30");
        config.probe_args = None;
        let output = dir.path().join("review.json");
        let stderr = dir.path().join("review.stderr");
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = counter.clone();
        let context = TaskContext::new(
            Some(std::sync::Arc::new(move || {
                Ok(seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 2)
            })),
            None,
            None,
        );
        let error =
            review_observed(&config, "prompt", &output, &stderr, Some(&context)).unwrap_err();
        assert!(error.is::<process::TaskCancelled>(), "{error:#}");
        assert!(!output.exists());
        assert!(!output.with_extension("stdout").exists());
    }

    #[test]
    fn preset_commands_avoid_tool_permissions_and_extract_final_message() {
        let (dir, mut config) = cli(
            "cat >/dev/null; printf '%s\\n' '{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"SHUM_READY\"}}'",
        );
        config.provider = LlmProvider::Codex;
        assert_eq!(readiness(&config), Readiness::Ready);
        let output = dir.path().join("review.json");
        review(&config, "review this", &output, &dir.path().join("stderr")).unwrap();
        assert_eq!(fs::read_to_string(&output).unwrap(), "SHUM_READY");
        let (_copilot_dir, mut config) = cli("printf SHUM_READY");
        config.provider = LlmProvider::Copilot;
        assert_eq!(readiness(&config), Readiness::Ready);
    }

    #[test]
    fn custom_probe_uses_selected_model_without_review_arguments() {
        let (_dir, mut config) = cli(
            "[ \"$1\" = selected-model ] || exit 8; [ \"$2\" = probe ] || exit 9; cat >/dev/null; printf SHUM_READY",
        );
        config.args = vec!["review-only-argument".into()];
        config.probe_args = Some(vec!["{model}".into(), "probe".into()]);
        config.model = Some("selected-model".into());
        assert_eq!(readiness(&config), Readiness::Ready);
    }

    #[test]
    fn failed_probe_blocks_review_until_a_successful_refresh() {
        let (dir, mut config) = cli(
            "echo \"$1\" >> \"$2\"\nif [ \"$1\" = probe ]; then\n  if [ ! -f \"$3\" ]; then echo 'connection refused' >&2; exit 1; fi\n  cat >/dev/null; printf SHUM_READY\nelse\n  cat >/dev/null; printf '{\"ok\":true}'\nfi",
        );
        let marker = dir.path().canonicalize().unwrap().join("calls");
        let ready = dir.path().canonicalize().unwrap().join("ready");
        config.probe_args = Some(vec![
            "probe".into(),
            marker.to_string_lossy().into_owned(),
            ready.to_string_lossy().into_owned(),
        ]);
        config.args = vec!["review".into(), marker.to_string_lossy().into_owned()];
        let output = dir.path().join("review.json");
        let stderr = dir.path().join("review.stderr");
        assert_eq!(readiness(&config), Readiness::Network);
        assert!(review(&config, "private project prompt", &output, &stderr).is_err());
        assert_eq!(fs::read_to_string(&marker).unwrap(), "probe\n");
        assert!(!output.exists());

        fs::write(&ready, "").unwrap();
        assert_eq!(readiness(&config), Readiness::Ready);
        review(&config, "private project prompt", &output, &stderr).unwrap();
        assert_eq!(
            fs::read_to_string(&marker).unwrap(),
            "probe\nprobe\nreview\n"
        );
        assert_eq!(fs::read_to_string(&output).unwrap(), "{\"ok\":true}");
    }

    #[test]
    fn unverified_custom_cli_can_run_real_review() {
        let (dir, mut config) = cli("cat >/dev/null; printf '{\"segments\":[]}'");
        config.probe_args = None;
        assert_eq!(readiness(&config), Readiness::Unverified);
        let output = dir.path().join("response.json");
        review(
            &config,
            "review prompt",
            &output,
            &dir.path().join("stderr"),
        )
        .unwrap();
        assert_eq!(fs::read_to_string(output).unwrap(), "{\"segments\":[]}");
    }
}
