use std::collections::BTreeMap;
use std::io::{self, IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{self, Event as InputEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    style::{Color, ResetColor, SetForegroundColor},
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use unicode_width::UnicodeWidthChar;

use crate::{
    execution::{self, ControlAction, ControlRequest, Handle, Initiator, Snapshot},
    workflow::{self, State},
};

const POLL: Duration = Duration::from_millis(250);

fn colors_enabled(tty: bool, no_color: bool, term: &str) -> bool {
    tty && !no_color && term != "dumb"
}

pub struct Guard {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<()>>>,
    interactive: bool,
}

pub struct Terminal;

impl Terminal {
    pub fn start(handle: Handle) -> Result<Guard> {
        let interactive = io::stdin().is_terminal()
            && io::stderr().is_terminal()
            && std::env::var("TERM").is_ok_and(|term| term != "dumb");
        let snapshot = handle.snapshot()?;
        let source = workflow::status(handle.project())?
            .input
            .display()
            .to_string();
        for line in lines(
            &snapshot,
            &handle.project().display().to_string(),
            &source,
            "ETA: insufficient data",
            120,
            &BTreeMap::new(),
            &[],
        ) {
            eprintln!("{line}");
        }
        if interactive {
            terminal::enable_raw_mode().context("could not enable terminal input")?;
            if let Err(error) = execute!(io::stderr(), EnterAlternateScreen, Hide) {
                if let Err(restore_error) = execute!(io::stderr(), Show, LeaveAlternateScreen) {
                    eprintln!("could not restore terminal screen: {restore_error}");
                }
                terminal::disable_raw_mode().context("could not restore terminal input")?;
                return Err(error).context("could not initialize terminal view");
            }
        }
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let worker = thread::spawn(move || run(handle, interactive, &stopping));
        Ok(Guard {
            stop,
            worker: Some(worker),
            interactive,
        })
    }
}

pub fn start(handle: Handle) -> Result<Guard> {
    Terminal::start(handle)
}

impl Guard {
    pub fn finish(mut self) -> Result<()> {
        self.stop.store(true, Ordering::Release);
        let result = self.worker.take().expect("terminal worker exists").join();
        let restore = if self.interactive {
            let screen = execute!(io::stderr(), Show, LeaveAlternateScreen);
            let input = terminal::disable_raw_mode();
            screen.context("could not restore terminal screen")?;
            input.context("could not restore terminal input")
        } else {
            Ok(())
        };
        restore?;
        result.map_err(|panic| anyhow!("terminal worker panicked: {panic:?}"))?
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.worker.is_some() {
            self.stop.store(true, Ordering::Release);
            if let Some(worker) = self.worker.take() {
                match worker.join() {
                    Ok(Err(error)) => eprintln!("terminal view failed: {error:#}"),
                    Err(error) => eprintln!("terminal worker panicked: {error:?}"),
                    Ok(Ok(())) => {}
                }
            }
            if self.interactive {
                if let Err(error) = execute!(io::stderr(), Show, LeaveAlternateScreen) {
                    eprintln!("could not restore terminal screen: {error}");
                }
                if let Err(error) = terminal::disable_raw_mode() {
                    eprintln!("could not restore terminal input: {error}");
                }
            }
        }
    }
}

#[derive(Default)]
struct Throughput {
    last_total: Option<usize>,
    last_done: usize,
    last_time: Option<Instant>,
    samples: Vec<f64>,
    graph_changed: bool,
}

impl Throughput {
    fn observe(&mut self, snapshot: &Snapshot, now: Instant) {
        let total = snapshot
            .attempts
            .values()
            .filter(|attempt| attempt.step.starts_with("asr:"))
            .filter(|attempt| {
                matches!(
                    attempt.status,
                    State::Ready | State::Running | State::NotStarted
                )
            })
            .map(|attempt| attempt.fragments.len())
            .sum();
        let done = snapshot
            .attempts
            .values()
            .filter(|attempt| attempt.step.starts_with("asr:"))
            .filter(|attempt| {
                matches!(
                    attempt.status,
                    State::Ready | State::Running | State::NotStarted
                )
            })
            .map(|attempt| attempt.done.len())
            .sum::<usize>();
        if self.last_total.is_some_and(|previous| total > previous) {
            self.samples.clear();
            self.graph_changed = true;
            self.last_time = None;
        }
        if done < self.last_done {
            self.samples.clear();
            self.last_time = None;
        } else if done > self.last_done {
            if let Some(last) = self.last_time {
                let elapsed = now.duration_since(last).as_secs_f64();
                if elapsed > 0.0 {
                    self.samples.push((done - self.last_done) as f64 / elapsed);
                    if self.samples.len() > 8 {
                        self.samples.remove(0);
                    }
                }
            }
            self.last_time = Some(now);
        }
        self.last_done = done;
        self.last_total = Some(total);
    }

    fn label(&self) -> String {
        let Some(total) = self.last_total else {
            return "ETA: insufficient data".into();
        };
        let remaining = total.saturating_sub(self.last_done);
        if total == 0
            || self.samples.len() < 2
            || remaining == 0
            || self
                .last_time
                .is_some_and(|last| last.elapsed() > Duration::from_secs(60))
        {
            return if self.graph_changed {
                "ETA: insufficient data (plan changed; collecting new observations)".into()
            } else {
                "ETA: insufficient data".into()
            };
        }
        let low = self.samples.iter().copied().fold(f64::INFINITY, f64::min);
        let high = self.samples.iter().copied().fold(0.0_f64, f64::max);
        format!(
            "Known-fragment ETA: {}–{} (observed throughput; parallel work included{}); project ETA insufficient data",
            duration_label((remaining as f64 / high).ceil() as u64),
            duration_label((remaining as f64 / low).ceil() as u64),
            if self.graph_changed {
                "; plan extended"
            } else {
                ""
            },
        )
    }
}

fn duration_label(seconds: u64) -> String {
    if seconds < 60 {
        format!("{seconds}s")
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}

fn clipped(text: &str, width: usize) -> String {
    let mut result = String::new();
    let mut used = 0;
    for ch in text
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
    {
        let len = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + len > width {
            if width > 1 {
                while used >= width && !result.is_empty() {
                    let removed = result.pop().unwrap();
                    used -= UnicodeWidthChar::width(removed).unwrap_or(0);
                }
                result.push('…');
            }
            return result;
        }
        result.push(ch);
        used += len;
    }
    result
}

fn status_label(state: State) -> &'static str {
    match state {
        State::Ready => "✓ ready",
        State::Running => "▶ running",
        State::Stale => "↻ requires recalculation",
        State::Skipped => "– skipped by user",
        State::Cancelled => "– attempt cancelled",
        State::Blocked => "! blocked",
        State::Failed => "× error",
        State::PendingExternal => "○ awaiting external review",
        State::NotStarted => "○ waiting",
    }
}

fn unit_progress(done: usize, total: usize, ready: bool) -> String {
    if total == 0 {
        return "activity; measurable total unavailable".into();
    }
    let percent = done
        .saturating_mul(100)
        .checked_div(total)
        .unwrap_or(0)
        .min(if ready { 100 } else { 99 });
    let filled = done
        .saturating_mul(10)
        .checked_div(total)
        .unwrap_or(0)
        .min(10);
    format!(
        "{done}/{total} {percent}% [{}{}]",
        "█".repeat(filled),
        "░".repeat(10 - filled)
    )
}

fn lines(
    snapshot: &Snapshot,
    project: &str,
    source: &str,
    eta: &str,
    width: usize,
    operations: &BTreeMap<String, (String, Instant)>,
    history: &[execution::Event],
) -> Vec<String> {
    let mut lines = vec![
        clipped(&format!("SHUM  ·  {project}"), width),
        clipped(
            &format!(
                "Source: {source} · duration: {}",
                snapshot
                    .duration
                    .map(|s| duration_label(s.ceil() as u64))
                    .unwrap_or_else(|| "unknown".into())
            ),
            width,
        ),
        clipped(&format!("Revision {} · {}", snapshot.revision, eta), width),
        clipped(
            if width < 20 {
                "Plan: now"
            } else {
                "Plan: known so far"
            },
            width,
        ),
        clipped(
            if width < 20 {
                "Acc unvrf"
            } else {
                "accuracy is unverified"
            },
            width,
        ),
    ];
    let roles = snapshot
        .stages
        .keys()
        .filter(|step| snapshot.mode_steps.contains(*step))
        .filter_map(|step| step.strip_prefix("asr:"))
        .collect::<Vec<_>>();
    lines.push(clipped(&format!("ASR roles: {}", roles.join(", ")), width));
    if let Some(models) = snapshot
        .metadata
        .get("models")
        .and_then(|value| value.as_object())
    {
        for (role, model) in models {
            lines.push(clipped(
                &format!(
                    "  {role}: engine {} · threads {} · in-flight {}",
                    model["engine"].as_str().unwrap_or("unknown"),
                    model["threads"],
                    model["max_in_flight"]
                ),
                width,
            ));
        }
    }
    lines.push(clipped(
        &format!(
            "Concurrency: workers {} · simultaneous models {}",
            snapshot.metadata["max_asr_workers"], snapshot.metadata["max_parallel_asr_models"]
        ),
        width,
    ));
    lines.push(clipped(
        &format!("External review: {}", snapshot.metadata["external_review"]),
        width,
    ));
    lines.push(clipped(
        "Quality checks may add attempts; this is not a final step count.",
        width,
    ));
    let mut counts = [0_usize; 3];
    for (step, stage) in &snapshot.stages {
        if snapshot.mode_steps.contains(step) {
            counts[if stage.status == State::Ready {
                0
            } else if stage.status == State::Running {
                1
            } else {
                2
            }] += 1;
        }
    }
    lines.push(clipped(
        &format!(
            "Tasks: {} completed · {} active · {} waiting/blocked · {} repeats",
            counts[0],
            counts[1],
            counts[2],
            snapshot
                .attempts
                .values()
                .filter(|attempt| attempt.kind == "retry")
                .count()
        ),
        width,
    ));
    lines.push(clipped(
        &format!(
            "Automatic budget: {}/{} attempts · {}/{}s used · {}s remaining",
            snapshot
                .attempts
                .values()
                .filter(|attempt| attempt.auto_retry)
                .count(),
            snapshot.budget.max_auto_repeats,
            snapshot.auto_runtime_seconds,
            snapshot.budget.max_auto_runtime_seconds,
            snapshot
                .budget
                .max_auto_runtime_seconds
                .saturating_sub(snapshot.auto_runtime_seconds)
        ),
        width,
    ));
    let mut stage_lines = Vec::new();
    for (step, stage) in &snapshot.stages {
        let count = if stage.units.is_empty() {
            stage
                .numeric_total
                .map(|total| format!("{}/{total} {}", stage.numeric_done, stage.unit))
                .unwrap_or_default()
        } else {
            format!(
                "{} {}",
                unit_progress(
                    stage.done.len(),
                    stage.units.len(),
                    stage.status == State::Ready
                ),
                stage.unit
            )
        };
        let selected = stage.selected_attempt.as_deref().unwrap_or("none");
        let active = stage.active_attempt.as_deref().unwrap_or("none");
        let status = if snapshot.mode_steps.contains(step) {
            status_label(stage.status)
        } else {
            "excluded from this run"
        };
        stage_lines.push(clipped(
            &format!("{step} · {status} · {count} · active {active} · selected {selected}",),
            width,
        ));
        let dependencies = stage
            .active_attempt
            .as_ref()
            .or(stage.selected_attempt.as_ref())
            .and_then(|id| snapshot.attempts.get(id))
            .map(|attempt| attempt.dependencies.clone());
        stage_lines.push(clipped(
            &format!(
                "  depends on: {}",
                dependencies
                    .map(|ids| format!("{ids:?}"))
                    .unwrap_or_else(|| format!(
                        "{:?}",
                        workflow::parents(
                            step,
                            &roles.iter().map(|role| (*role).into()).collect::<Vec<_>>()
                        )
                    ))
            ),
            width,
        ));
        if stage.status == State::PendingExternal {
            stage_lines.push(clipped("  awaiting external review", width));
        }
        if let Some(id) = &stage.selected_attempt
            && let Some(attempt) = snapshot.attempts.get(id)
        {
            stage_lines.push(clipped(
                &format!(
                    "  selected attempt {} · {:?} · checkpoint {}",
                    id,
                    attempt.status,
                    if attempt.artifact.is_some() {
                        "saved"
                    } else {
                        "not recorded"
                    },
                ),
                width,
            ));
        }
        if let Some(operation) = &stage.operation {
            let elapsed = operations
                .get(step)
                .map(|(_, started)| duration_label(started.elapsed().as_secs()))
                .unwrap_or_default();
            stage_lines.push(clipped(
                &format!("  {operation} · {elapsed} elapsed since observed"),
                width,
            ));
        }
    }
    let active = snapshot
        .attempts
        .values()
        .filter(|attempt| attempt.status == State::Running);
    for attempt in active {
        let stage = &snapshot.stages[&attempt.step];
        let total = attempt.fragments.len();
        let done = attempt
            .fragments
            .iter()
            .filter(|id| attempt.done.contains(id))
            .count();
        let scope = if total == 0 {
            "whole stage".into()
        } else {
            format!("{total} fragments")
        };
        let progress = if total == 0 {
            if stage.active_attempt.as_deref() == Some(&attempt.id) && !stage.units.is_empty() {
                format!("{}/{} {}", stage.done.len(), stage.units.len(), stage.unit)
            } else {
                stage
                    .numeric_total
                    .map(|total| format!("{}/{total} {}", stage.numeric_done, stage.unit))
                    .unwrap_or_else(|| "count unknown".into())
            }
        } else {
            format!("{} fragments", unit_progress(done, total, false))
        };
        let elapsed = attempt
            .started_at_seconds
            .and_then(|started| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .ok()
                    .map(|now| duration_label(now.as_secs().saturating_sub(started)))
            })
            .unwrap_or_else(|| "unknown".into());
        lines.push(clipped(
            &format!(
                "Attempt {} · {} ({}) · {} elapsed",
                attempt.id, attempt.step, attempt.kind, elapsed
            ),
            width,
        ));
        lines.push(clipped(
            &format!(
                "  {} · {} · dependencies {:?} · threads {:?} · in-flight {:?}",
                progress,
                scope,
                attempt.dependencies,
                attempt.parameters.threads,
                attempt.parameters.max_in_flight,
            ),
            width,
        ));
    }
    for attempt in snapshot
        .attempts
        .values()
        .filter(|attempt| attempt.status == State::NotStarted)
    {
        let ready = attempt
            .dependencies
            .iter()
            .all(|id| snapshot.attempts[id].status == State::Ready);
        lines.push(clipped(
            &format!(
                "○ {} · {} ({}) · {} assigned fragments · {}",
                attempt.id,
                attempt.step,
                attempt.kind,
                attempt.fragments.len(),
                if ready {
                    "waiting for executor checkpoint/resources"
                } else {
                    "waiting for input artifacts"
                }
            ),
            width,
        ));
    }
    lines.extend(stage_lines);
    if !snapshot.queued.is_empty() {
        lines.push(clipped(
            &format!("Controls: {} requests", snapshot.queued.len()),
            width,
        ));
    }
    if !history.is_empty() {
        lines.push(clipped("Recent events:", width));
        for event in history.iter().rev().take(5).rev() {
            lines.push(clipped(
                &format!(
                    "  {} {} {}: {}",
                    event.sequence,
                    event.kind,
                    event.step.as_deref().unwrap_or(""),
                    event.message.as_deref().unwrap_or("")
                ),
                width,
            ));
            if matches!(event.kind.as_str(), "retry_planned" | "check_planned") {
                let fragments = event.payload["attempt"]["fragments"]
                    .as_array()
                    .map_or(0, Vec::len);
                lines.push(clipped(
                    &format!("  Plan extended: {fragments} assigned fragment operations"),
                    width,
                ));
            }
        }
    }
    lines.push(clipped(
        "↑↓ scroll · s select active attempt · Ctrl+C interrupt project",
        width,
    ));
    lines
}

struct Viewport {
    width: usize,
    color: bool,
    scroll: usize,
}

fn render(
    snapshot: &Snapshot,
    project: &str,
    source: &str,
    eta: &str,
    viewport: Viewport,
    operations: &BTreeMap<String, (String, Instant)>,
    history: &[execution::Event],
) -> Result<()> {
    let Viewport {
        width,
        color,
        scroll,
    } = viewport;
    let mut stderr = io::stderr().lock();
    execute!(stderr, MoveTo(0, 0), Clear(ClearType::All))?;
    let height = terminal::size()
        .map(|(_, h)| h as usize)
        .unwrap_or(24)
        .max(2);
    let mut display = lines(snapshot, project, source, eta, width, operations, history);
    let controls = display.pop().unwrap();
    let offset = scroll.min(display.len().saturating_sub(height - 1));
    display = display.into_iter().skip(offset).take(height - 1).collect();
    display.push(controls);
    for line in &display {
        if color && line.starts_with("SHUM") {
            execute!(stderr, SetForegroundColor(Color::Cyan))?;
        } else if color && line.contains("✓ ready") {
            execute!(stderr, SetForegroundColor(Color::Green))?;
        } else if color && line.contains("▶ running") {
            execute!(stderr, SetForegroundColor(Color::Blue))?;
        } else if color && (line.contains("× error") || line.contains("! blocked")) {
            execute!(stderr, SetForegroundColor(Color::Red))?;
        } else if color && line.contains("unverified") {
            execute!(stderr, SetForegroundColor(Color::Yellow))?;
        }
        writeln!(stderr, "{line}")?;
        if color {
            execute!(stderr, ResetColor)?;
        }
    }
    stderr.flush()?;
    Ok(())
}

fn read_key(stop: &AtomicBool) -> Result<Option<KeyEvent>> {
    loop {
        if stop.load(Ordering::Acquire) {
            return Ok(None);
        }
        if !event::poll(POLL)? {
            continue;
        }
        if let InputEvent::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            return Ok(Some(key));
        }
    }
}

fn prompt(message: &str, stop: &AtomicBool) -> Result<String> {
    let mut stderr = io::stderr().lock();
    execute!(stderr, MoveTo(0, 0), Clear(ClearType::All))?;
    writeln!(stderr, "{message}")?;
    stderr.flush()?;
    drop(stderr);
    let mut answer = String::new();
    loop {
        match read_key(stop)? {
            None => return Ok(String::new()),
            Some(key)
                if key.code == KeyCode::Char('c')
                    && key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                crate::process::interrupt_project();
                return Ok(String::new());
            }
            Some(KeyEvent {
                code: KeyCode::Esc, ..
            }) => return Ok(String::new()),
            Some(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) => return Ok(answer),
            Some(KeyEvent {
                code: KeyCode::Backspace,
                ..
            }) => {
                answer.pop();
            }
            Some(KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            }) if !ch.is_control() && !modifiers.contains(KeyModifiers::CONTROL) => answer.push(ch),
            _ => {}
        }
        let mut stderr = io::stderr().lock();
        execute!(stderr, MoveTo(0, 0), Clear(ClearType::All))?;
        write!(stderr, "{message}\n{answer}")?;
        stderr.flush()?;
    }
}

fn confirmed_request(mut request: ControlRequest, answer: &str) -> Option<ControlRequest> {
    if answer != "y" {
        return None;
    }
    request.confirmed = true;
    Some(request)
}

fn select_attempt(handle: &Handle, snapshot: &Snapshot, stop: &AtomicBool) -> Result<()> {
    let active: Vec<_> = snapshot
        .attempts
        .values()
        .filter(|attempt| attempt.status == State::Running)
        .collect();
    if active.is_empty() {
        return Ok(());
    }
    let mut message = String::from("Choose an active attempt (number, Enter; Esc cancels):\n");
    for (index, attempt) in active.iter().enumerate() {
        message.push_str(&format!(
            "{}. {} · {} · {} fragments\n",
            index + 1,
            attempt.id,
            attempt.step,
            attempt.fragments.len()
        ));
    }
    let choice = prompt(&message, stop)?;
    let Ok(index) = choice.parse::<usize>() else {
        return Ok(());
    };
    let Some(attempt) = index.checked_sub(1).and_then(|index| active.get(index)) else {
        return Ok(());
    };
    let choice = prompt(
        &format!(
            "{} · {}. Type c to cancel this attempt, or s to skip the logical stage; Enter; Esc aborts.",
            attempt.id, attempt.step
        ),
        stop,
    )?;
    let action = match choice.as_str() {
        "c" => ControlAction::CancelAttempt {
            attempt_id: attempt.id.clone(),
        },
        "s" => ControlAction::Skip {
            step: attempt.step.clone(),
        },
        _ => return Ok(()),
    };
    let request = ControlRequest {
        version: 1,
        request_id: format!(
            "terminal-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ),
        expected_revision: Some(handle.snapshot()?.revision),
        initiator: Initiator::User,
        reason: format!("Interactive control for {}", attempt.id),
        evidence: Vec::new(),
        confirmed: false,
        action,
    };
    let mut preview_request = request.clone();
    preview_request.confirmed = true;
    let preview = execution::preview(handle.project(), &preview_request)?;
    let affected = preview
        .affected_steps
        .iter()
        .filter_map(|step| snapshot.stages.get(step).map(|stage| (step, stage)));
    let selected_outputs = affected
        .filter_map(|(step, stage)| {
            stage
                .selected_attempt
                .as_ref()
                .map(|id| format!("{step}: {id}"))
        })
        .collect::<Vec<_>>();
    let consequences = match &request.action {
        ControlAction::CancelAttempt { .. } => format!(
            "Current attempt {} stops; {} completed units remain in its history.\nSelected output on {}: {}.\nDependent stages: none newly invalidated by this action.",
            attempt.id,
            attempt.done.len(),
            attempt.step,
            snapshot.stages[&attempt.step]
                .selected_attempt
                .as_deref()
                .unwrap_or("none"),
        ),
        ControlAction::Skip { .. } => format!(
            "Stage {} is skipped; dependent stages {:?} are affected.\nExisting selected outputs: {:?}. Dependent selections may become stale; saved files are not deleted.\nDependencies of current attempt: {:?}.",
            attempt.step, preview.affected_steps, selected_outputs, attempt.dependencies,
        ),
        _ => unreachable!(),
    };
    let consequences = format!(
        "Action: {:?}\nAffected stages: {:?}; fragments: {:?}\n{consequences}\nType y then Enter to confirm; anything else cancels.",
        request.action, preview.affected_steps, preview.fragments,
    );
    if let Some(mut request) = confirmed_request(request, &prompt(&consequences, stop)?)
        && !stop.load(Ordering::Acquire)
        && !crate::process::interrupted()
    {
        request.expected_revision = Some(handle.snapshot()?.revision);
        execution::enqueue(handle.project(), &request)?;
    }
    Ok(())
}

fn significant(event: &execution::Event) -> bool {
    if event.kind == "operation" {
        return event.message.as_deref().is_some_and(|message| {
            let message = message.to_ascii_lowercase();
            message.contains("wait") || message.contains("load")
        });
    }
    !matches!(
        event.kind.as_str(),
        "progress" | "unit_done" | "retry_unit_done" | "runtime_observed"
    )
}

fn run(handle: Handle, interactive: bool, stop: &AtomicBool) -> Result<()> {
    let project = handle.project().display().to_string();
    let source = workflow::status(handle.project())?
        .input
        .display()
        .to_string();
    let mut after = 0;
    let mut last_revision = None;
    let mut throughput = Throughput::default();
    let color = colors_enabled(
        interactive,
        std::env::var_os("NO_COLOR").is_some(),
        &std::env::var("TERM").unwrap_or_default(),
    );
    let mut operations: BTreeMap<String, (String, Instant)> = BTreeMap::new();
    let mut last_draw = Instant::now();
    let mut scroll = 0;
    let mut journal_started = true;
    loop {
        if stop.load(Ordering::Acquire) || crate::process::interrupted() {
            return Ok(());
        }
        let snapshot = handle.snapshot()?;
        if snapshot.interrupted {
            return Ok(());
        }
        if interactive {
            throughput.observe(&snapshot, Instant::now());
            for (step, stage) in &snapshot.stages {
                match &stage.operation {
                    Some(operation)
                        if operations.get(step).is_none_or(|(old, _)| old != operation) =>
                    {
                        operations.insert(step.clone(), (operation.clone(), Instant::now()));
                    }
                    None => {
                        operations.remove(step);
                    }
                    _ => {}
                }
            }
            if last_revision != Some(snapshot.revision)
                || last_draw.elapsed() >= Duration::from_secs(1)
            {
                let page = execution::events(
                    handle.project(),
                    snapshot.revision.saturating_sub(1000),
                    1000,
                )?;
                let history = page.into_iter().filter(significant).collect::<Vec<_>>();
                let width = terminal::size()
                    .map(|(w, _)| w as usize)
                    .unwrap_or(80)
                    .max(1);
                render(
                    &snapshot,
                    &project,
                    &source,
                    &if snapshot
                        .metadata
                        .get("models")
                        .and_then(|models| models.as_object())
                        .is_some_and(|models| models.len() == 1)
                    {
                        throughput.label()
                    } else {
                        "ETA: insufficient data across different model speeds".into()
                    },
                    Viewport {
                        width,
                        color,
                        scroll,
                    },
                    &operations,
                    &history,
                )?;
                last_revision = Some(snapshot.revision);
                last_draw = Instant::now();
            }
            if event::poll(POLL)? {
                match event::read()? {
                    InputEvent::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            crate::process::interrupt_project();
                            return Ok(());
                        }
                        KeyCode::Char('s') => {
                            if let Err(error) = select_attempt(&handle, &handle.snapshot()?, stop) {
                                prompt(
                                    &format!("Control failed: {error:#}\nPress Enter to continue."),
                                    stop,
                                )?;
                            }
                            last_revision = None;
                        }
                        KeyCode::Up => {
                            scroll = scroll.saturating_sub(1);
                            last_revision = None;
                        }
                        KeyCode::Down => {
                            scroll = scroll.saturating_add(1);
                            last_revision = None;
                        }
                        KeyCode::PageUp => {
                            scroll = scroll.saturating_sub(10);
                            last_revision = None;
                        }
                        KeyCode::PageDown => {
                            scroll = scroll.saturating_add(10);
                            last_revision = None;
                        }
                        KeyCode::Home => {
                            scroll = 0;
                            last_revision = None;
                        }
                        _ => {}
                    },
                    InputEvent::Resize(_, _) => last_revision = None,
                    _ => {}
                }
            }
        } else {
            if !journal_started {
                for line in lines(
                    &snapshot,
                    &project,
                    &source,
                    "ETA: insufficient data",
                    120,
                    &BTreeMap::new(),
                    &[],
                )
                .into_iter()
                {
                    eprintln!("{line}");
                }
                journal_started = true;
            }
            let page = if snapshot.revision > after {
                execution::events(handle.project(), after, 100)?
            } else {
                Vec::new()
            };
            for item in &page {
                after = item.sequence;
                if significant(item) {
                    if item.kind == "duration" {
                        eprintln!(
                            "[{}] duration: {}",
                            item.sequence,
                            snapshot
                                .duration
                                .map(|s| duration_label(s.ceil() as u64))
                                .unwrap_or_else(|| "unknown".into())
                        );
                        continue;
                    }
                    eprintln!(
                        "{}",
                        clipped(
                            &format!(
                                "[{}] {}{}{}",
                                item.sequence,
                                item.kind,
                                item.step
                                    .as_ref()
                                    .map_or(String::new(), |s| format!(" {s}")),
                                item.message
                                    .as_ref()
                                    .map_or(String::new(), |s| format!(": {s}"))
                            ),
                            usize::MAX
                        )
                    );
                }
            }
            thread::sleep(POLL);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_is_opt_in_and_plain_lines_have_no_escape() {
        let snapshot = Snapshot {
            schema_version: 1,
            revision: 0,
            duration: None,
            interrupted: false,
            metadata: serde_json::Value::Null,
            project_root: None,
            auto_runtime_seconds: 0,
            stages: Default::default(),
            attempts: Default::default(),
            decisions: Vec::new(),
            skipped: Default::default(),
            mode_steps: Default::default(),
            budget: Default::default(),
            queued: Vec::new(),
        };
        let plain = lines(
            &snapshot,
            "project",
            "audio.wav",
            "ETA: insufficient data",
            80,
            &BTreeMap::new(),
            &[],
        )
        .join("\n");
        assert!(!plain.contains('\x1b'));
        assert!(plain.contains("accuracy is unverified"));
        assert!(plain.contains("insufficient data"));
        assert!(!colors_enabled(true, true, "xterm"));
        assert!(!colors_enabled(true, false, "dumb"));
        assert!(!colors_enabled(false, false, "xterm"));
        assert!(colors_enabled(true, false, "xterm"));
    }

    #[test]
    fn eta_requires_observed_samples_and_resets_on_plan_extension() {
        let mut samples = Throughput {
            last_total: Some(10),
            last_done: 3,
            ..Default::default()
        };
        assert!(samples.label().contains("insufficient data"));
        samples.samples = vec![1.0, 2.0];
        assert!(samples.label().contains("4s–7s"));
        samples.samples.clear();
        samples.graph_changed = true;
        assert!(samples.label().contains("plan changed"));
    }

    #[test]
    fn narrow_width_truncates_unicode_without_splitting_characters() {
        assert!(
            unicode_width::UnicodeWidthStr::width(clipped("очень длинное имя", 9).as_str()) <= 9
        );
    }

    #[test]
    fn control_requires_explicit_y_after_preview() {
        let request = ControlRequest {
            version: 1,
            request_id: "terminal-test".into(),
            expected_revision: Some(3),
            initiator: Initiator::User,
            reason: "test".into(),
            evidence: Vec::new(),
            confirmed: false,
            action: ControlAction::Skip {
                step: "review".into(),
            },
        };
        assert!(confirmed_request(request.clone(), "").is_none());
        assert!(confirmed_request(request.clone(), "yes").is_none());
        assert!(confirmed_request(request.clone(), "Y").is_none());
        let confirmed = confirmed_request(request, "y").unwrap();
        assert!(confirmed.confirmed);
        assert_eq!(confirmed.expected_revision, Some(3));
    }

    #[test]
    fn plain_journal_keeps_attempts_and_waits_but_not_unit_chatter() {
        let mut event = execution::Event {
            version: 1,
            sequence: 1,
            kind: "attempt_started".into(),
            step: Some("asr:flix".into()),
            attempt_id: Some("a1".into()),
            message: None,
            payload: serde_json::Value::Null,
        };
        assert!(significant(&event));
        event.kind = "retry_unit_done".into();
        assert!(!significant(&event));
        event.kind = "operation".into();
        event.message = Some("Waiting for available memory".into());
        assert!(significant(&event));
    }
}
