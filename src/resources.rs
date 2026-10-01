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

pub fn choose_batch(
    pending: &[&'static str],
    budget: Memory,
    settings: &Resources,
    forced_sequential: bool,
) -> Result<Vec<(&'static str, usize)>> {
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
        if slots == 0 {
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
    roles: Vec<&'static str>,
    worker_counts: &'a [(&'static str, usize)],
    offload_roles: &'a [&'static str],
    available_mib: u64,
    swap_free_mib: u64,
    reserve_mib: u64,
    forced_sequential: bool,
    using_swap_as_safety_net: bool,
}

pub fn record_admission(
    output: &Path,
    batch: &[(&'static str, usize)],
    offload_roles: &[&'static str],
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
    ensure!(
        output.status.success(),
        "cannot stop ASR systemd unit {unit}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
