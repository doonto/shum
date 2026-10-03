use std::fmt;
use std::fs::File;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

pub type CancellationCallback = Arc<dyn Fn() -> Result<bool> + Send + Sync>;
pub type ProgressCallback = Arc<dyn Fn(u64) -> Result<()> + Send + Sync>;
pub type OperationCallback = Arc<dyn Fn(&str) -> Result<()> + Send + Sync>;
pub type MeasurementCallback = Arc<dyn Fn(&str, u64, u64) -> Result<()> + Send + Sync>;

#[derive(Clone, Default)]
pub struct TaskContext {
    pub cancel: Option<CancellationCallback>,
    pub progress_ms: Option<ProgressCallback>,
    pub operation: Option<OperationCallback>,
    pub measurements: Option<MeasurementCallback>,
}

impl TaskContext {
    pub fn new(
        cancel: Option<CancellationCallback>,
        progress_ms: Option<ProgressCallback>,
        operation: Option<OperationCallback>,
    ) -> Self {
        Self {
            cancel,
            progress_ms,
            operation,
            measurements: None,
        }
    }

    pub fn with_measurements(mut self, callback: MeasurementCallback) -> Self {
        self.measurements = Some(callback);
        self
    }

    pub fn check_cancelled(&self) -> Result<()> {
        if interrupted() {
            return Err(TaskCancelled.into());
        }
        if let Some(callback) = &self.cancel
            && callback()?
        {
            return Err(TaskCancelled.into());
        }
        Ok(())
    }

    pub fn report_progress_ms(&self, elapsed_ms: u64) -> Result<()> {
        if let Some(callback) = &self.progress_ms {
            callback(elapsed_ms)?;
        }
        Ok(())
    }

    pub fn report_operation(&self, name: &str) -> Result<()> {
        if let Some(callback) = &self.operation {
            callback(name)?;
        }
        Ok(())
    }

    pub fn report_measurement(&self, phase: &str, completed: u64, total: u64) -> Result<()> {
        if let Some(callback) = &self.measurements {
            callback(phase, completed, total)?;
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasuredProgress {
    event: String,
    version: u64,
    phase: String,
    completed: u64,
    total: u64,
}

fn parse_measured_progress(line: &str) -> Result<Option<MeasuredProgress>> {
    let line = line.trim();
    if !line.starts_with('{') {
        return Ok(None);
    }
    let value: serde_json::Value = match serde_json::from_str(line) {
        Ok(value) => value,
        Err(error) if line.contains("shum.progress") => {
            return Err(error).context("invalid adapter progress record");
        }
        Err(_) => return Ok(None),
    };
    if value.get("event").and_then(serde_json::Value::as_str) != Some("shum.progress") {
        return Ok(None);
    }
    let measurement: MeasuredProgress =
        serde_json::from_value(value).context("invalid adapter progress record")?;
    ensure!(
        measurement.event == "shum.progress" && measurement.version == 1,
        "unsupported adapter progress event/version"
    );
    ensure!(
        measurement.total > 0 && measurement.completed <= measurement.total,
        "invalid adapter progress counters"
    );
    ensure!(
        matches!(measurement.phase.as_str(), "embeddings" | "segmentation"),
        "unsupported adapter progress phase"
    );
    Ok(Some(measurement))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskCancelled;

impl fmt::Display for TaskCancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("task cancelled")
    }
}

impl std::error::Error for TaskCancelled {}

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static HANDLER_STATE: OnceLock<Mutex<HandlerState>> = OnceLock::new();

struct HandlerState {
    users: usize,
    previous: Option<libc::sigaction>,
}

fn handler_state() -> &'static Mutex<HandlerState> {
    HANDLER_STATE.get_or_init(|| {
        Mutex::new(HandlerState {
            users: 0,
            previous: None,
        })
    })
}

extern "C" fn interrupt_handler(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

/// Request cancellation of the foreground project, including its owned tasks.
pub fn interrupt_project() {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// Installs a process-wide SIGINT flag for a single foreground workflow.
/// Readiness probes and provider login do not install this guard.
pub struct InterruptGuard;

impl InterruptGuard {
    pub fn install() -> Result<Self> {
        let mut state = handler_state()
            .lock()
            .map_err(|_| anyhow::anyhow!("SIGINT handler state mutex poisoned"))?;
        if state.users > 0 {
            state.users += 1;
            return Ok(Self);
        }
        // SAFETY: sigaction is initialized before use; the handler only writes an atomic.
        let previous = unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = interrupt_handler as *const () as usize;
            libc::sigemptyset(&mut action.sa_mask);
            action.sa_flags = 0;
            let mut previous = std::mem::zeroed();
            if libc::sigaction(libc::SIGINT, &action, &mut previous) == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(previous)
            }
        }
        .context("cannot install SIGINT handler")?;
        INTERRUPTED.store(false, Ordering::SeqCst);
        state.previous = Some(previous);
        state.users = 1;
        Ok(Self)
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        let Ok(mut state) = handler_state().lock() else {
            eprintln!("cannot restore SIGINT handler: state mutex poisoned");
            return;
        };
        state.users -= 1;
        if state.users == 0
            && let Some(previous) = state.previous.take()
        {
            // SAFETY: previous was returned by a successful sigaction call.
            if unsafe { libc::sigaction(libc::SIGINT, &previous, std::ptr::null_mut()) } == -1 {
                eprintln!(
                    "cannot restore SIGINT handler: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    }
}

/// Give a task subprocess a new session before spawning it.
pub fn configure_owned_command(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: setsid only changes the child's session before exec, without touching Rust state.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

fn signal_group(pgid: u32, signal: libc::c_int) -> Result<()> {
    // A group created by setsid has its child's PID as its group ID. The child
    // stays unreaped until cleanup finishes, preventing that ID from being reused.
    let result = unsafe { libc::kill(-(pgid as libc::pid_t), signal) };
    if result == -1 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            return Err(error).with_context(|| format!("cannot signal owned process group {pgid}"));
        }
    }
    Ok(())
}

/// Stop an unreaped child and its process group, recording reaping even if service cleanup fails.
/// Do not call `wait` or `try_wait` on the child before this function.
fn stop_owned_child(
    child: &mut Child,
    reaped: &mut bool,
    stop_service: impl FnOnce() -> Result<()>,
) -> Result<()> {
    // Never target a PGID whose leader has already been reaped: its PID may
    // have been recycled by the kernel.
    child_exited(child)?;
    let service_error = stop_service().err();
    let pgid = child.id();
    signal_group(pgid, libc::SIGTERM)?;
    let deadline = Instant::now() + Duration::from_millis(50);
    while Instant::now() < deadline {
        let alive = unsafe { libc::kill(-(pgid as libc::pid_t), 0) } == 0;
        if !alive {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    signal_group(pgid, libc::SIGKILL)?;
    child.wait().context("cannot reap owned child")?;
    *reaped = true;
    if let Some(error) = service_error {
        return Err(error).context("cannot stop owned systemd service");
    }
    Ok(())
}

fn child_exited(child: &Child) -> Result<bool> {
    // WNOWAIT observes the exit without reaping the group leader. Its PID
    // remains reserved while the entire owned group is being stopped.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    let result = unsafe {
        libc::waitid(
            libc::P_PID,
            child.id(),
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    if result == -1 {
        return Err(std::io::Error::last_os_error()).context("cannot observe owned child");
    }
    Ok(unsafe { info.si_pid() } != 0)
}

pub struct OwnedProcess {
    child: Child,
    systemd_unit: Option<String>,
    finished: bool,
}

impl OwnedProcess {
    /// The child must have been spawned with `configure_owned_command`.
    pub fn new(child: Child, systemd_unit: Option<String>) -> Self {
        Self {
            child,
            systemd_unit,
            finished: false,
        }
    }

    /// Do not reap through this reference; use `try_exited` or `wait_timeout`.
    pub fn child_mut(&mut self) -> &mut Child {
        &mut self.child
    }

    pub fn try_exited(&self) -> Result<bool> {
        if self.finished {
            return Ok(true);
        }
        child_exited(&self.child)
    }

    pub fn stop(&mut self) -> Result<()> {
        if !self.finished {
            let unit = self.systemd_unit.as_deref();
            stop_owned_child(&mut self.child, &mut self.finished, || {
                if let Some(unit) = unit {
                    crate::resources::stop_service(unit)
                } else {
                    Ok(())
                }
            })?;
        }
        Ok(())
    }

    pub fn wait_timeout(&mut self, timeout: Duration) -> Result<Option<ExitStatus>> {
        let start = Instant::now();
        loop {
            if self.try_exited()? {
                self.stop()?;
                return self
                    .child
                    .wait()
                    .map(Some)
                    .context("cannot reap owned child");
            }
            if start.elapsed() >= timeout {
                return Ok(None);
            }
            std::thread::sleep(
                Duration::from_millis(40).min(timeout.saturating_sub(start.elapsed())),
            );
        }
    }
}

impl Drop for OwnedProcess {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            eprintln!("cannot clean up owned process: {error:#}");
        }
    }
}

pub fn wait_child_observed(
    owned: &mut OwnedProcess,
    timeout: Duration,
    context: Option<&TaskContext>,
) -> Result<Option<ExitStatus>> {
    let started = Instant::now();
    loop {
        if let Some(context) = context
            && let Err(error) = context.check_cancelled()
        {
            owned.stop()?;
            return Err(error);
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if let Some(status) = owned.wait_timeout(remaining.min(Duration::from_millis(40)))? {
            return Ok(Some(status));
        }
        if started.elapsed() >= timeout {
            return Ok(None);
        }
    }
}

fn execute(
    command: &mut Command,
    stdout: Stdio,
    stderr_stream: Stdio,
    stderr: &Path,
    timeout: Duration,
    context: Option<&TaskContext>,
    mut on_tick: impl FnMut(&mut Child) -> Result<()>,
) -> Result<()> {
    if let Some(context) = context {
        context.check_cancelled()?;
    }
    configure_owned_command(command);
    let child = command
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr_stream)
        .spawn()
        .with_context(|| format!("cannot start {command:?}"))?;
    let mut owned = OwnedProcess::new(child, None);
    let start = Instant::now();
    loop {
        if let Some(context) = context {
            context.check_cancelled()?;
        }
        on_tick(owned.child_mut())?;
        if owned.try_exited()? {
            owned.stop()?;
            let status = owned.child_mut().wait().context("cannot reap adapter")?;
            if !status.success() {
                bail!("adapter exited with {status}; stderr: {}", stderr.display());
            }
            return Ok(());
        }
        if start.elapsed() >= timeout {
            owned.stop()?;
            bail!(
                "adapter timed out after {} seconds; stderr: {}",
                timeout.as_secs(),
                stderr.display()
            );
        }
        std::thread::sleep(Duration::from_millis(40));
    }
}

pub fn run_observed(
    command: &mut Command,
    stdout: &Path,
    stderr: &Path,
    timeout: Duration,
    context: Option<&TaskContext>,
) -> Result<()> {
    run_observed_stderr_lines(command, stdout, stderr, timeout, context, |line| {
        if let Some(measurement) = parse_measured_progress(line)?
            && let Some(context) = context
        {
            context.report_measurement(
                &measurement.phase,
                measurement.completed,
                measurement.total,
            )?;
        }
        Ok(())
    })
}

pub fn run(command: &mut Command, stdout: &Path, stderr: &Path, timeout: Duration) -> Result<()> {
    run_observed(command, stdout, stderr, timeout, None)
}

pub fn run_observed_lines(
    command: &mut Command,
    stderr: &Path,
    timeout: Duration,
    context: Option<&TaskContext>,
    mut line: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    use std::io::{BufRead, BufReader};

    let err =
        File::create(stderr).with_context(|| format!("cannot create {}", stderr.display()))?;
    let (send, recv) = mpsc::channel::<std::io::Result<String>>();
    let mut reader = None;
    let run = execute(
        command,
        Stdio::piped(),
        Stdio::from(err),
        stderr,
        timeout,
        context,
        |child| {
            if reader.is_none() {
                let stdout = child.stdout.take().context("adapter stdout missing")?;
                let send = send.clone();
                reader = Some(std::thread::spawn(move || {
                    for line in BufReader::new(stdout).lines() {
                        if send.send(line).is_err() {
                            break;
                        }
                    }
                }));
            }
            while let Ok(next) = recv.try_recv() {
                line(&next.context("cannot read adapter progress")?)?;
            }
            Ok(())
        },
    );
    if let Some(reader) = reader {
        reader
            .join()
            .map_err(|_| anyhow::anyhow!("adapter progress reader panicked"))?;
    }
    run?;
    while let Ok(next) = recv.try_recv() {
        line(&next.context("cannot read adapter progress")?)?;
    }
    Ok(())
}

pub fn run_observed_stderr_lines(
    command: &mut Command,
    stdout: &Path,
    stderr: &Path,
    timeout: Duration,
    context: Option<&TaskContext>,
    mut line: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    use std::io::{BufRead, BufReader, Write};

    let out =
        File::create(stdout).with_context(|| format!("cannot create {}", stdout.display()))?;
    let mut err =
        Some(File::create(stderr).with_context(|| format!("cannot create {}", stderr.display()))?);
    let (send, recv) = mpsc::channel();
    let mut reader = None;
    let run = execute(
        command,
        Stdio::from(out),
        Stdio::piped(),
        stderr,
        timeout,
        context,
        |child| {
            if reader.is_none() {
                let pipe = child.stderr.take().context("adapter stderr missing")?;
                let send = send.clone();
                let mut log = err.take().context("adapter stderr log missing")?;
                reader = Some(std::thread::spawn(move || {
                    let result = (|| -> std::io::Result<()> {
                        let mut pipe = BufReader::new(pipe);
                        let mut buffer = Vec::new();
                        loop {
                            buffer.clear();
                            if pipe.read_until(b'\n', &mut buffer)? == 0 {
                                return Ok(());
                            }
                            log.write_all(&buffer)?;
                            if send
                                .send(Ok(String::from_utf8_lossy(&buffer).into_owned()))
                                .is_err()
                            {
                                return Ok(());
                            }
                        }
                    })();
                    if let Err(error) = &result {
                        let _ =
                            send.send(Err(std::io::Error::new(error.kind(), error.to_string())));
                    }
                    result
                }));
            }
            while let Ok(next) = recv.try_recv() {
                line(
                    next.context("cannot read adapter stderr")?
                        .trim_end_matches(['\n', '\r']),
                )?;
            }
            Ok(())
        },
    );
    if let Some(reader) = reader {
        reader
            .join()
            .map_err(|_| anyhow::anyhow!("adapter stderr reader panicked"))?
            .context("cannot record adapter stderr")?;
    }
    run?;
    while let Ok(next) = recv.try_recv() {
        line(
            next.context("cannot read adapter stderr")?
                .trim_end_matches(['\n', '\r']),
        )?;
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn service_cleanup_error_does_not_reap_or_signal_the_child_twice() {
        let mut command = Command::new("sleep");
        command.arg("30");
        configure_owned_command(&mut command);
        let mut owned = OwnedProcess::new(command.spawn().unwrap(), None);
        let error = stop_owned_child(&mut owned.child, &mut owned.finished, || {
            anyhow::bail!("injected service cleanup failure")
        })
        .unwrap_err();
        assert!(format!("{error:#}").contains("injected service cleanup failure"));
        assert!(owned.finished);
        assert!(owned.try_exited().unwrap());
        owned.stop().unwrap();
        owned.stop().unwrap();
        assert!(owned.child.wait().unwrap().code().is_none());
    }

    #[test]
    fn cancellation_stops_descendants_without_touching_other_processes() {
        let dir = tempfile::tempdir().unwrap();
        let descendant = dir.path().join("descendant.pid");
        let mut unrelated_command = Command::new("sleep");
        unrelated_command.arg("30");
        configure_owned_command(&mut unrelated_command);
        let unrelated = unrelated_command.spawn().unwrap();
        let mut unrelated = OwnedProcess::new(unrelated, None);

        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = calls.clone();
        let context = TaskContext::new(
            Some(Arc::new(move || {
                Ok(callback_calls.fetch_add(1, Ordering::SeqCst) > 2)
            })),
            None,
            None,
        );
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg("sleep 30 & echo $! > \"$1\"; wait")
            .arg("sh")
            .arg(&descendant);
        let error = run_observed(
            &mut command,
            &dir.path().join("stdout"),
            &dir.path().join("stderr"),
            Duration::from_secs(5),
            Some(&context),
        )
        .unwrap_err();
        assert!(error.is::<TaskCancelled>(), "{error:#}");
        assert!(unrelated.child_mut().try_wait().unwrap().is_none());
        let pid: u32 = std::fs::read_to_string(&descendant)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"));
        assert!(
            stat.is_err() || stat.unwrap().split_whitespace().nth(2) == Some("Z"),
            "descendant is still running"
        );
        unrelated.stop().unwrap();
    }

    #[test]
    fn timeout_and_callback_errors_reap_owned_children() {
        let dir = tempfile::tempdir().unwrap();
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 30 & wait"]);
        let error = run_observed(
            &mut command,
            &dir.path().join("stdout"),
            &dir.path().join("stderr"),
            Duration::from_millis(50),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("timed out"), "{error:#}");

        let context = TaskContext::new(
            Some(Arc::new(|| Err(anyhow::anyhow!("callback failed")))),
            None,
            None,
        );
        let error = run_observed(
            &mut Command::new("true"),
            &dir.path().join("stdout"),
            &dir.path().join("stderr"),
            Duration::from_secs(1),
            Some(&context),
        )
        .unwrap_err();
        assert!(error.to_string().contains("callback failed"));
        assert!(!error.is::<TaskCancelled>());
    }

    #[test]
    fn observed_worker_shutdown_stops_group_on_cancellation() {
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 30 & wait"]);
        configure_owned_command(&mut command);
        let child = command.spawn().unwrap();
        let mut owned = OwnedProcess::new(child, None);
        let context = TaskContext::new(Some(Arc::new(|| Ok(true))), None, None);
        let error =
            wait_child_observed(&mut owned, Duration::from_secs(2), Some(&context)).unwrap_err();
        assert!(error.is::<TaskCancelled>(), "{error:#}");
        assert!(owned.child_mut().try_wait().unwrap().is_some());
    }

    #[test]
    fn observed_lines_do_not_write_to_parent_stdout() {
        let dir = tempfile::tempdir().unwrap();
        let mut command = Command::new("sh");
        command.args(["-c", "printf 'out_time_us=1234567\\n'; echo warning >&2"]);
        let mut observed = Vec::new();
        run_observed_lines(
            &mut command,
            &dir.path().join("stderr"),
            Duration::from_secs(2),
            None,
            |line| {
                observed.push(line.to_string());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(observed, ["out_time_us=1234567"]);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("stderr")).unwrap(),
            "warning\n"
        );
    }

    #[test]
    fn progress_callback_failure_stops_owned_group() {
        let dir = tempfile::tempdir().unwrap();
        let mut command = Command::new("sh");
        command.args(["-c", "echo out_time_us=1000; sleep 30 & wait"]);
        let error = run_observed_lines(
            &mut command,
            &dir.path().join("stderr"),
            Duration::from_secs(2),
            None,
            |_| bail!("progress rejected"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("progress rejected"), "{error:#}");
    }

    #[test]
    fn stderr_progress_is_tee_logged_without_becoming_parent_stdout() {
        let dir = tempfile::tempdir().unwrap();
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "printf 'result\\n'; printf '{\"event\":\"shum.progress\",\"version\":1,\"phase\":\"embeddings\",\"completed\":1,\"total\":4}\\nwarning\\n' >&2",
        ]);
        let stdout = dir.path().join("stdout");
        let stderr = dir.path().join("stderr");
        let mut seen = Vec::new();
        run_observed_stderr_lines(
            &mut command,
            &stdout,
            &stderr,
            Duration::from_secs(2),
            None,
            |line| {
                seen.push(line.to_owned());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(std::fs::read_to_string(stdout).unwrap(), "result\n");
        assert_eq!(
            seen,
            [
                "{\"event\":\"shum.progress\",\"version\":1,\"phase\":\"embeddings\",\"completed\":1,\"total\":4}",
                "warning"
            ]
        );
        assert_eq!(
            std::fs::read_to_string(stderr).unwrap(),
            seen.join("\n") + "\n"
        );
    }

    #[test]
    fn stderr_progress_callback_failure_stops_child() {
        let dir = tempfile::tempdir().unwrap();
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "echo '{\"event\":\"shum.progress\",\"version\":1}' >&2; sleep 30 & wait",
        ]);
        let error = run_observed_stderr_lines(
            &mut command,
            &dir.path().join("stdout"),
            &dir.path().join("stderr"),
            Duration::from_secs(3),
            None,
            |_| bail!("invalid progress record"),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("invalid progress record"),
            "{error:#}"
        );
    }

    #[test]
    fn run_observed_reports_only_measured_counts_and_keeps_logs() {
        let dir = tempfile::tempdir().unwrap();
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "echo diagnostic >&2; echo '{\"event\":\"shum.progress\",\"version\":1,\"phase\":\"embeddings\",\"completed\":2,\"total\":7}' >&2",
        ]);
        let measurements = Arc::new(std::sync::Mutex::new(Vec::new()));
        let received = measurements.clone();
        let context = TaskContext::new(None, None, None).with_measurements(Arc::new(
            move |phase, completed, total| {
                received
                    .lock()
                    .unwrap()
                    .push((phase.to_owned(), completed, total));
                Ok(())
            },
        ));
        run_observed(
            &mut command,
            &dir.path().join("stdout"),
            &dir.path().join("stderr"),
            Duration::from_secs(2),
            Some(&context),
        )
        .unwrap();
        assert_eq!(*measurements.lock().unwrap(), [("embeddings".into(), 2, 7)]);
        assert!(
            std::fs::read_to_string(dir.path().join("stderr"))
                .unwrap()
                .starts_with("diagnostic\n")
        );
        assert!(parse_measured_progress("{\"event\":\"shum.progress\",\"version\":2,\"phase\":\"embeddings\",\"completed\":2,\"total\":7}").is_err());
        assert!(parse_measured_progress("{\"event\":\"shum.progress\",\"version\":1,\"phase\":\"embeddings\",\"completed\":8,\"total\":7}").is_err());
        let reordered = parse_measured_progress(
            r#" { "completed":2, "total":7, "phase":"embeddings", "version":1, "event":"shum.progress" } "#,
        )
        .unwrap()
        .unwrap();
        assert_eq!((reordered.completed, reordered.total), (2, 7));
    }
}
