use std::fs::File;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use wait_timeout::ChildExt;

pub fn run(command: &mut Command, stdout: &Path, stderr: &Path, timeout: Duration) -> Result<()> {
    let out =
        File::create(stdout).with_context(|| format!("cannot create {}", stdout.display()))?;
    let err =
        File::create(stderr).with_context(|| format!("cannot create {}", stderr.display()))?;
    let mut child = command
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .spawn()
        .with_context(|| format!("cannot start {command:?}"))?;
    let status = match child
        .wait_timeout(timeout)
        .context("cannot wait for adapter")?
    {
        Some(status) => status,
        None => {
            child.kill().context("cannot stop timed-out adapter")?;
            child.wait().context("cannot reap timed-out adapter")?;
            bail!(
                "adapter timed out after {} seconds; stderr: {}",
                timeout.as_secs(),
                stderr.display()
            );
        }
    };
    if !status.success() {
        bail!("adapter exited with {status}; stderr: {}", stderr.display());
    }
    Ok(())
}
