use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::config::Resources;

const MIB: u64 = 1024 * 1024;
static EVENT_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Copy)]
pub struct Memory {
    pub available_mib: u64,
    pub swap_free_mib: u64,
}

fn meminfo_value(text: &str, key: &str) -> Result<u64> {
    let line = text
        .lines()
        .find(|line| line.starts_with(key) && line.as_bytes().get(key.len()) == Some(&b':'))
        .with_context(|| format!("/proc/meminfo lacks {key}"))?;
    let kb: u64 = line[key.len() + 1..]
        .split_whitespace()
        .next()
        .context("missing memory value")?
        .parse()
        .with_context(|| format!("invalid {key} in /proc/meminfo"))?;
    Ok(kb / 1024)
}

fn cgroup_headroom() -> Result<Option<u64>> {
    let membership = fs::read_to_string("/proc/self/cgroup").context("cannot read cgroup")?;
    let relative = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .context("cgroup v2 membership is unavailable")?;
    let base = Path::new("/sys/fs/cgroup");
    let mut headroom = None;
    let mut dir = base.to_path_buf();
    let root_limit = base.join("memory.max");
    if root_limit.exists() {
        let limit =
            fs::read_to_string(&root_limit).context("cannot read root cgroup memory.max")?;
        if limit.trim() != "max" {
            let maximum: u64 = limit
                .trim()
                .parse()
                .context("invalid root cgroup memory.max")?;
            let current: u64 = fs::read_to_string(base.join("memory.current"))
                .context("cannot read root cgroup memory.current")?
                .trim()
                .parse()
                .context("invalid root cgroup memory.current")?;
            headroom = Some(maximum.saturating_sub(current) / MIB);
        }
    }
    for part in relative.split('/').filter(|part| !part.is_empty()) {
        ensure!(part != "." && part != "..", "invalid cgroup path");
        dir.push(part);
        let limit = fs::read_to_string(dir.join("memory.max"))
            .with_context(|| format!("cannot read {}", dir.join("memory.max").display()))?;
        if let Ok(max) = limit.trim().parse::<u64>() {
            let current: u64 = fs::read_to_string(dir.join("memory.current"))
                .with_context(|| format!("cannot read {}", dir.join("memory.current").display()))?
                .trim()
                .parse()
                .context("invalid cgroup memory.current")?;
            let available = max.saturating_sub(current) / MIB;
            headroom = Some(headroom.map_or(available, |prior: u64| prior.min(available)));
        } else {
            ensure!(limit.trim() == "max", "invalid cgroup memory.max");
        }
    }
    Ok(headroom)
}

pub fn memory() -> Result<Memory> {
    let meminfo = fs::read_to_string("/proc/meminfo").context("cannot read /proc/meminfo")?;
    let available = meminfo_value(&meminfo, "MemAvailable")?;
    Ok(Memory {
        available_mib: cgroup_headroom()?.map_or(available, |limit| available.min(limit)),
        swap_free_mib: meminfo_value(&meminfo, "SwapFree")?,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkSample {
    pub offset_ms: u128,
    pub tracked_process_count: usize,
    pub aggregate_rss_mib: u64,
    pub own_vmswap_kib: u64,
    pub mem_available_mib: u64,
    pub psi_memory_full_avg10_percent: f64,
    pub system_pswpin_pages: Option<u64>,
    pub system_pswpout_pages: Option<u64>,
    pub load_average_1m: Option<f64>,
    pub cpu_frequency_mhz: Option<u64>,
    pub temperature_millidegrees: Option<i64>,
    pub throttled_periods: Option<u64>,
}

#[cfg(target_os = "linux")]
fn vmstat_counter(text: &str, name: &str) -> Result<u64> {
    let value = text
        .lines()
        .find_map(|line| line.strip_prefix(name)?.strip_prefix(' '))
        .with_context(|| format!("/proc/vmstat lacks {name}"))?;
    value
        .trim()
        .parse()
        .with_context(|| format!("invalid {name} in /proc/vmstat"))
}

pub fn system_swap_pages() -> Result<(Option<u64>, Option<u64>)> {
    #[cfg(target_os = "linux")]
    {
        let text = fs::read_to_string("/proc/vmstat").context("cannot read /proc/vmstat")?;
        Ok((
            Some(vmstat_counter(&text, "pswpin")?),
            Some(vmstat_counter(&text, "pswpout")?),
        ))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok((None, None))
    }
}

pub fn new_system_swap_out(baseline: Option<u64>, current: Option<u64>) -> Result<bool> {
    match (baseline, current) {
        (Some(before), Some(now)) => {
            ensure!(
                now >= before,
                "system pswpout counter reset during benchmark"
            );
            Ok(now > before)
        }
        (None, None) => Ok(false),
        _ => anyhow::bail!("system pswpout counter became unavailable during benchmark"),
    }
}

pub const MEMORY_FULL_AVG10_THRESHOLD_PERCENT: f64 = 1.0;

pub fn memory_full_avg10(text: &str) -> Result<f64> {
    let full = text
        .lines()
        .find_map(|line| line.strip_prefix("full "))
        .context("memory PSI lacks full line")?;
    let value = full
        .split_whitespace()
        .find_map(|field| field.strip_prefix("avg10="))
        .context("memory PSI full lacks avg10")?
        .parse::<f64>()
        .context("invalid memory PSI full avg10")?;
    ensure!(
        value.is_finite() && value >= 0.0,
        "invalid memory PSI full avg10"
    );
    Ok(value)
}

fn parse_proc_memory(text: &str, pid: u32) -> Result<(u64, u64)> {
    if text
        .lines()
        .find_map(|line| line.strip_prefix("State:"))
        .is_some_and(|state| state.trim_start().starts_with(['Z', 'X']))
    {
        return Ok((0, 0));
    }
    let field = |key: &str| -> Result<u64> {
        text.lines()
            .find_map(|line| line.strip_prefix(key))
            .and_then(|value| value.split_whitespace().next())
            .with_context(|| format!("worker {pid} status lacks {key}"))?
            .parse()
            .with_context(|| format!("invalid {key} for worker {pid}"))
    };
    Ok((field("VmRSS:")?, field("VmSwap:")?))
}

fn proc_memory(pid: u32) -> Result<(u64, u64)> {
    let text = fs::read_to_string(format!("/proc/{pid}/status"))
        .with_context(|| format!("cannot read memory for worker {pid}"))?;
    parse_proc_memory(&text, pid)
}

fn descendants(pid: u32, found: &mut std::collections::BTreeSet<u32>) -> Result<()> {
    if !found.insert(pid) {
        return Ok(());
    }
    let children = match fs::read_to_string(format!("/proc/{pid}/task/{pid}/children")) {
        Ok(children) => children,
        Err(_) if !Path::new(&format!("/proc/{pid}")).exists() => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("cannot list worker {pid} descendants"));
        }
    };
    for child in children.split_whitespace() {
        descendants(child.parse().context("invalid descendant PID")?, found)?;
    }
    Ok(())
}

pub fn benchmark_sample(offset_ms: u128, pids: &[u32]) -> Result<BenchmarkSample> {
    let (system_pswpin_pages, system_pswpout_pages) = system_swap_pages()?;
    let psi_memory_full_avg10_percent = memory_full_avg10(
        &fs::read_to_string("/proc/pressure/memory").context("cannot read memory PSI")?,
    )?;
    let mut rss_kib = 0_u64;
    let mut own_vmswap_kib = 0_u64;
    let mut tracked = std::collections::BTreeSet::new();
    for pid in pids {
        descendants(*pid, &mut tracked)?;
    }
    let tracked_process_count = tracked.len();
    for pid in tracked {
        match proc_memory(pid) {
            Ok((rss, swap)) => {
                rss_kib = rss_kib.saturating_add(rss);
                own_vmswap_kib = own_vmswap_kib.saturating_add(swap);
            }
            Err(error) if !Path::new(&format!("/proc/{pid}")).exists() => {
                let _ = error;
            }
            Err(error) => return Err(error),
        }
    }
    let load_average_1m = fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|text| text.split_whitespace().next()?.parse().ok());
    let cpu_frequency_mhz =
        fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .map(|khz| khz / 1000);
    let temperature_millidegrees = fs::read_to_string("/sys/class/thermal/thermal_zone0/temp")
        .ok()
        .and_then(|text| text.trim().parse().ok());
    let throttled_periods = fs::read_to_string("/sys/fs/cgroup/cpu.stat")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("nr_throttled "))?
                .parse()
                .ok()
        });
    Ok(BenchmarkSample {
        offset_ms,
        tracked_process_count,
        aggregate_rss_mib: rss_kib / 1024,
        own_vmswap_kib,
        mem_available_mib: memory()?.available_mib,
        psi_memory_full_avg10_percent,
        system_pswpin_pages,
        system_pswpout_pages,
        load_average_1m,
        cpu_frequency_mhz,
        temperature_millidegrees,
        throttled_periods,
    })
}

pub fn prepare_offload_dir(path: &Path, min_free_mib: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        if !path.exists() {
            let mut builder = fs::DirBuilder::new();
            builder.recursive(true).mode(0o700);
            builder
                .create(path)
                .with_context(|| format!("cannot create offload directory {}", path.display()))?;
        }
        let metadata = fs::symlink_metadata(path)?;
        ensure!(
            metadata.is_dir()
                && !metadata.file_type().is_symlink()
                && metadata.permissions().mode() & 0o077 == 0,
            "offload directory must be a private, non-symlink directory (mode 0700): {}",
            path.display()
        );
    }
    #[cfg(not(unix))]
    {
        ensure!(
            path.is_dir(),
            "offload directory must already exist: {}",
            path.display()
        );
    }
    let free = fs2::available_space(path)
        .with_context(|| format!("cannot check free disk space at {}", path.display()))?
        / MIB;
    ensure!(
        free >= min_free_mib,
        "offload directory {} has {free} MiB free; requires at least {min_free_mib} MiB",
        path.display()
    );
    Ok(())
}

pub fn single_worker_peak_mib(role: &str, settings: &Resources) -> u64 {
    if settings.asr_worker_counts.get(role).copied().unwrap_or(1) > 1 {
        settings.asr_worker_peak_mib[role]
    } else {
        settings.estimated_peak_mib[role]
    }
}

pub fn choose_batch<'a>(
    pending: &[&'a str],
    budget: Memory,
    settings: &Resources,
    forced_sequential: bool,
) -> Result<Vec<(&'a str, usize)>> {
    let mut batch = Vec::new();
    let mut remaining = budget.available_mib.saturating_sub(settings.reserve_mib);
    let max_workers = if forced_sequential {
        1
    } else {
        settings.max_asr_workers
    };
    let mut slots = max_workers;
    for role in pending {
        let _ = settings
            .estimated_peak_mib
            .get(*role)
            .with_context(|| format!("no memory estimate for {role}"))?;
        let estimate = single_worker_peak_mib(role, settings);
        let requested = if forced_sequential {
            1
        } else {
            *settings.asr_worker_counts.get(*role).unwrap_or(&1)
        };
        let parallel_peak = if requested > 1 {
            let per_worker = settings
                .asr_worker_peak_mib
                .get(*role)
                .with_context(|| format!("no per-worker memory estimate for {role}"))?;
            per_worker.saturating_mul(requested as u64)
        } else {
            estimate
        };
        let (count, cost) = if requested <= slots && parallel_peak <= remaining {
            (requested, parallel_peak)
        } else if batch.is_empty() && slots > 0 && estimate <= remaining {
            (1, estimate)
        } else {
            continue;
        };
        batch.push((*role, count));
        slots -= count;
        remaining -= cost;
        if slots == 0
            || settings.sequential_asr_roles
            || settings
                .max_parallel_asr_models
                .is_some_and(|limit| batch.len() >= limit)
        {
            break;
        }
    }
    if batch.is_empty() && !pending.is_empty() {
        batch.push((pending[0], 1));
    }
    Ok(batch)
}

#[derive(Serialize)]
struct Admission<'a> {
    roles: Vec<&'a str>,
    worker_counts: &'a [(&'a str, usize)],
    offload_roles: &'a [&'a str],
    available_mib: u64,
    swap_free_mib: u64,
    reserve_mib: u64,
    forced_sequential: bool,
    sequential_asr_roles: bool,
    using_swap_as_safety_net: bool,
}

pub fn record_admission(
    output: &Path,
    batch: &[(&str, usize)],
    offload_roles: &[&str],
    budget: Memory,
    settings: &Resources,
    forced_sequential: bool,
) -> Result<()> {
    let estimated = batch.iter().fold(0_u64, |total, (role, count)| {
        let peak = if *count > 1 {
            settings.asr_worker_peak_mib[*role].saturating_mul(*count as u64)
        } else if offload_roles.contains(role) {
            settings.offload_estimated_peak_mib[*role]
        } else {
            single_worker_peak_mib(role, settings)
        };
        total.saturating_add(peak)
    });
    let using_swap_as_safety_net =
        estimated.saturating_add(settings.reserve_mib) > budget.available_mib;
    let event = Admission {
        roles: batch.iter().map(|(role, _)| *role).collect(),
        worker_counts: batch,
        offload_roles,
        available_mib: budget.available_mib,
        swap_free_mib: budget.swap_free_mib,
        reserve_mib: settings.reserve_mib,
        forced_sequential,
        sequential_asr_roles: settings.sequential_asr_roles,
        using_swap_as_safety_net,
    };
    append_event(output, &event)?;
    eprintln!(
        "ASR admission: {:?}; disk offload {:?}; available {} MiB, reserve {} MiB, swap free {} MiB{}",
        batch,
        offload_roles,
        budget.available_mib,
        settings.reserve_mib,
        budget.swap_free_mib,
        if using_swap_as_safety_net {
            "; RAM estimate exceeds available memory, relying on system swap (OOM remains possible)"
        } else {
            ""
        }
    );
    Ok(())
}

pub fn systemd_result(unit: &str) -> Result<String> {
    systemd_property(unit, "Result")
}

fn systemd_property(unit: &str, property: &str) -> Result<String> {
    let output = Command::new("systemctl")
        .args([
            "--user",
            "show",
            unit,
            &format!("--property={property}"),
            "--value",
            "--no-pager",
        ])
        .output()
        .context("cannot query ASR systemd unit")?;
    ensure!(
        output.status.success(),
        "cannot query ASR systemd unit {unit}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result = String::from_utf8(output.stdout)?.trim().to_owned();
    ensure!(
        !result.is_empty(),
        "ASR systemd unit {unit} has no {property}"
    );
    Ok(result)
}

pub fn record_worker(
    output: &Path,
    role: &str,
    elapsed_ms: u128,
    memory_peak_mib: Option<u64>,
    mode: &str,
    result: &str,
) -> Result<()> {
    let event = serde_json::json!({
        "type": "worker",
        "role": role,
        "elapsed_ms": elapsed_ms,
        "memory_peak_mib": memory_peak_mib,
        "mode": mode,
        "result": result,
    });
    append_event(output, &event)
}

pub struct WindowEvent<'a> {
    pub role: &'a str,
    pub index: usize,
    pub audio_seconds: f64,
    pub elapsed_ms: u128,
    pub mode: &'a str,
    pub characters: usize,
    pub repeated_fourgram_fraction: f64,
}

pub fn record_window(output: &Path, window: WindowEvent<'_>) -> Result<()> {
    let event = serde_json::json!({
        "type": "window",
        "role": window.role,
        "index": window.index,
        "audio_seconds": window.audio_seconds,
        "elapsed_ms": window.elapsed_ms,
        "mode": window.mode,
        "characters": window.characters,
        "repeated_fourgram_fraction": window.repeated_fourgram_fraction,
    });
    append_event(output, &event)
}

fn append_event(output: &Path, event: &impl Serialize) -> Result<()> {
    let _guard = EVENT_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("resource event log mutex poisoned"))?;
    let mut log = OpenOptions::new()
        .append(true)
        .create(true)
        .open(output.join("resource-events.jsonl"))
        .context("cannot open resource event log")?;
    serde_json::to_writer(&mut log, event)?;
    writeln!(log)?;
    log.sync_data()?;
    Ok(())
}

pub fn systemd_memory_peak(unit: &str) -> Result<Option<u64>> {
    let value = systemd_property(unit, "MemoryPeak")?;
    if value == "[not set]" {
        return Ok(None);
    }
    value
        .parse::<u64>()
        .map(|bytes| Some(bytes / MIB))
        .with_context(|| format!("invalid MemoryPeak for {unit}: {value}"))
}

pub fn stop_service(unit: &str) -> Result<()> {
    let output = Command::new("systemctl")
        .args(["--user", "stop", unit])
        .output()
        .context("cannot stop ASR systemd unit")?;
    validate_service_stop(unit, &output, || systemd_property(unit, "LoadState"))
}

fn validate_service_stop(
    unit: &str,
    output: &std::process::Output,
    load_state: impl FnOnce() -> Result<String>,
) -> Result<()> {
    if output.status.success() {
        return Ok(());
    }
    let diagnostic = format!(
        "cannot stop ASR systemd unit {unit}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let state = load_state().with_context(|| diagnostic.clone())?;
    ensure!(state == "not-found", "{diagnostic}; LoadState={state}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_stop_accepts_only_verified_absent_units() {
        use std::os::unix::process::ExitStatusExt;
        let failed = std::process::Output {
            status: std::process::ExitStatus::from_raw(1 << 8),
            stdout: Vec::new(),
            stderr: b"stop failed".to_vec(),
        };
        assert!(validate_service_stop("owned.service", &failed, || Ok("not-found".into())).is_ok());
        for state in ["loaded", "masked", "error", ""] {
            let error =
                validate_service_stop("owned.service", &failed, || Ok(state.into())).unwrap_err();
            assert!(error.to_string().contains("stop failed"));
        }
        let error = validate_service_stop("owned.service", &failed, || {
            anyhow::bail!("manager unavailable")
        })
        .unwrap_err();
        assert!(format!("{error:#}").contains("manager unavailable"));
        assert!(format!("{error:#}").contains("stop failed"));
        let success = std::process::Output {
            status: std::process::ExitStatus::from_raw(0),
            stdout: Vec::new(),
            stderr: Vec::new(),
        };
        validate_service_stop("owned.service", &success, || panic!("unnecessary query")).unwrap();
    }

    #[test]
    fn swap_pressure_uses_only_new_system_swap_out() {
        assert!(!new_system_swap_out(Some(183), Some(183)).unwrap());
        assert!(new_system_swap_out(Some(183), Some(184)).unwrap());
        assert!(!new_system_swap_out(None, None).unwrap());
        assert!(new_system_swap_out(Some(183), Some(182)).is_err());
        assert!(new_system_swap_out(Some(183), None).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn swap_counters_require_both_valid_vmstat_fields() {
        let baseline = "pswpin 200\npswpout 183\n";
        assert_eq!(vmstat_counter(baseline, "pswpout").unwrap(), 183);
        assert_eq!(vmstat_counter(baseline, "pswpin").unwrap(), 200);
        assert!(vmstat_counter("pswpin 200\n", "pswpout").is_err());
        assert!(vmstat_counter("pswpout wrong\n", "pswpout").is_err());
    }

    #[test]
    fn psi_and_worker_counters_fail_closed_on_invalid_data() {
        assert_eq!(
            memory_full_avg10("some avg10=90\nfull avg10=0.99 total=42").unwrap(),
            0.99
        );
        assert_eq!(memory_full_avg10("full avg10=1.00 total=42").unwrap(), 1.0);
        for text in [
            "some avg10=1.0",
            "full avg60=1.0",
            "full avg10=NaN",
            "full avg10=-1",
        ] {
            assert!(memory_full_avg10(text).is_err());
        }
        assert_eq!(
            parse_proc_memory("VmRSS:\t123 kB\nVmSwap:\t4 kB", 1).unwrap(),
            (123, 4)
        );
        assert!(parse_proc_memory("VmRSS:\t123 kB", 1).is_err());
        assert_eq!(parse_proc_memory("State:\tZ (zombie)", 1).unwrap(), (0, 0));
    }

    #[test]
    fn chooses_workers_with_reserve_and_never_silently_drops_a_role() {
        let settings = Resources::default();
        let roles = ["primeline", "flix", "qwen"];
        let budget = Memory {
            available_mib: 22_000,
            swap_free_mib: 0,
        };
        assert_eq!(
            choose_batch(&roles, budget, &settings, false).unwrap(),
            [("primeline", 1), ("flix", 1)]
        );
        assert_eq!(
            choose_batch(&roles, budget, &settings, true).unwrap(),
            [("primeline", 1)]
        );
        let low = Memory {
            available_mib: 2000,
            swap_free_mib: 8000,
        };
        assert_eq!(
            choose_batch(&roles, low, &settings, false).unwrap(),
            [("primeline", 1)]
        );
    }

    #[test]
    fn plans_parallel_replicas_without_exceeding_cpu_slots_or_memory() {
        let mut settings = Resources::default();
        settings.asr_worker_counts.insert("flix".into(), 3);
        settings.asr_worker_peak_mib.insert("flix".into(), 2600);
        let roles = ["flix", "primeline"];
        let budget = Memory {
            available_mib: 12_000,
            swap_free_mib: 0,
        };
        assert_eq!(
            choose_batch(&roles, budget, &settings, false).unwrap(),
            [("flix", 3)]
        );
        assert_eq!(
            choose_batch(&roles, budget, &settings, true).unwrap(),
            [("flix", 1)]
        );
        let lower = Memory {
            available_mib: 10_000,
            swap_free_mib: 0,
        };
        assert_eq!(
            choose_batch(&roles, lower, &settings, false).unwrap(),
            [("flix", 1)]
        );
        let fits_prime_but_not_all_replicas = Memory {
            available_mib: 18_000,
            swap_free_mib: 0,
        };
        assert_eq!(
            choose_batch(
                &["primeline", "flix"],
                fits_prime_but_not_all_replicas,
                &settings,
                false
            )
            .unwrap(),
            [("primeline", 1)]
        );
    }

    #[test]
    fn serializes_roles_without_disabling_measured_replicas() {
        let mut settings = Resources {
            sequential_asr_roles: true,
            ..Resources::default()
        };
        settings.asr_worker_counts.insert("flix".into(), 3);
        settings.asr_worker_peak_mib.insert("flix".into(), 2600);
        let budget = Memory {
            available_mib: 30_000,
            swap_free_mib: 0,
        };
        assert_eq!(
            choose_batch(&["primeline", "flix", "qwen"], budget, &settings, false).unwrap(),
            [("primeline", 1)]
        );
        assert_eq!(
            choose_batch(&["flix", "qwen"], budget, &settings, false).unwrap(),
            [("flix", 3)]
        );
        assert_eq!(
            choose_batch(&["flix", "qwen"], budget, &settings, true).unwrap(),
            [("flix", 1)]
        );
    }

    #[test]
    fn independent_model_and_worker_limits_admit_dynamic_roles() {
        let roles: Vec<String> = (0..5).map(|index| format!("custom_{index}")).collect();
        let pending: Vec<&str> = roles.iter().map(String::as_str).collect();
        let mut settings = Resources {
            max_asr_workers: 5,
            ..Resources::default()
        };
        for role in &roles {
            settings.estimated_peak_mib.insert(role.clone(), 100);
        }
        let budget = Memory {
            available_mib: 20_000,
            swap_free_mib: 0,
        };
        assert_eq!(
            choose_batch(&pending, budget, &settings, false)
                .unwrap()
                .len(),
            5
        );
        settings.max_parallel_asr_models = Some(4);
        assert_eq!(
            choose_batch(&pending, budget, &settings, false)
                .unwrap()
                .len(),
            4
        );
        settings.max_parallel_asr_models = Some(2);
        assert_eq!(
            choose_batch(&pending, budget, &settings, false)
                .unwrap()
                .len(),
            2
        );
        settings.max_parallel_asr_models = None;
        settings.max_asr_workers = 3;
        assert_eq!(
            choose_batch(&pending, budget, &settings, false)
                .unwrap()
                .len(),
            3
        );
        settings.max_parallel_asr_models = Some(5);
        settings.asr_worker_counts.insert(roles[0].clone(), 2);
        settings.asr_worker_peak_mib.insert(roles[0].clone(), 100);
        assert_eq!(
            choose_batch(&pending, budget, &settings, false).unwrap(),
            [(roles[0].as_str(), 2), (roles[1].as_str(), 1)]
        );
        assert_eq!(
            choose_batch(&pending, budget, &settings, true).unwrap(),
            [(roles[0].as_str(), 1)]
        );
        settings.sequential_asr_roles = true;
        assert_eq!(
            choose_batch(&pending, budget, &settings, false).unwrap(),
            [(roles[0].as_str(), 2)]
        );
        let temp = tempfile::tempdir().unwrap();
        record_admission(
            temp.path(),
            &[(roles[0].as_str(), 1)],
            &[],
            budget,
            &settings,
            false,
        )
        .unwrap();
        let event: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(temp.path().join("resource-events.jsonl")).unwrap(),
        )
        .unwrap();
        assert_eq!(event["roles"], serde_json::json!(["custom_0"]));
    }

    #[test]
    fn records_pressure_in_private_job_log() {
        let temp = tempfile::tempdir().unwrap();
        record_admission(
            temp.path(),
            &[("qwen", 1)],
            &[],
            Memory {
                available_mib: 100,
                swap_free_mib: 100,
            },
            &Resources::default(),
            false,
        )
        .unwrap();
        let event: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(temp.path().join("resource-events.jsonl")).unwrap(),
        )
        .unwrap();
        assert_eq!(event["using_swap_as_safety_net"], true);
        assert_eq!(event["roles"], serde_json::json!(["qwen"]));
    }

    #[test]
    fn prepares_private_offload_directory_and_rejects_insufficient_disk() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("offload");
        prepare_offload_dir(&path, 1).unwrap();
        assert!(prepare_offload_dir(&path, u64::MAX).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
    }
}
