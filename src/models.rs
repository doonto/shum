//! Model provisioning and integrity checking for `shum`.
//!
//! A [`ModelSpec`] pins a role (e.g. "primeline", "flix", "qwen",
//! "diarization") to a Hugging Face `repo` + `revision` (a commit sha) and a
//! local `path` where the snapshot is materialized on disk. `revision` may
//! be an empty string for a model that has not yet been installed.
//!
//! Three operations are provided:
//!
//! - [`check`] never modifies anything on disk or on the network target. It
//!   verifies that a pinned model is present and intact locally, and
//!   *reports* (without acting on) whether a newer revision exists upstream.
//!   It errors on an unpinned (empty-revision) spec, telling the caller to
//!   run [`update`] first. Models never auto-update from `check` alone.
//! - [`update`] is the only provisioning entry point the CLI exposes. It
//!   **always** resolves the latest revision Hugging Face currently reports
//!   for the repo's default branch — regardless of whether `spec.revision`
//!   was already pinned — downloads it and validates its integrity into its
//!   own immutable directory. This is what lets a pinned model actually
//!   move to a newer revision that `check` reported as available:
//!   re-running `update` on an already-pinned spec used to just re-download
//!   the same pinned sha and silently do nothing.
//! - [`install_pinned`] downloads and validates the *exact* `spec.revision`
//!   already recorded on the spec (no network lookup for "latest"), for
//!   callers that need reproducible installs of a known revision. It is not
//!   currently wired into the CLI, which only exposes `update`.
//!
//! Both [`update`] and [`install_pinned`] materialize a revision into its
//! own immutable, never-reused directory `<base>/<revision>` (see
//! [`base_dir_for`]) rather than mutating the directory named by
//! `spec.path`: they stage the download, hash and manifest it, and only
//! once fully validated rename the staging directory into that final
//! location. `spec.path` itself is never read, modified, or removed by
//! either function — a currently active model (which some other process
//! may be reading right now) is never disturbed by an update in progress,
//! and a failed or partial download never leaves anything at the final
//! path for a later `check` to mistake as valid. The returned spec's `path`
//! points at the new directory; *activating* it — making it what future
//! runs use — is entirely up to the caller, typically by persisting the
//! returned spec into `shum.toml`.
//!
//! # Local manifest
//!
//! Every managed model directory contains a `.shum-manifest.json` file
//! recording the `repo`, `revision`, and a sha256 hash for every file in the
//! snapshot. [`check`] recomputes these hashes to detect corruption or drift
//! between the on-disk content and the spec (integrity), independently of
//! whatever Hugging Face reports upstream.
//!
//! The manifest is written atomically: content is written to a sibling
//! `.tmp` file first, then renamed into place, so a crash mid-write can
//! never leave a half-written manifest behind.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Name of the manifest file stored inside every managed model directory.
const MANIFEST_FILENAME: &str = ".shum-manifest.json";
/// Temporary name used while writing the manifest atomically.
const MANIFEST_TMP_FILENAME: &str = ".shum-manifest.json.tmp";

/// Network timeout applied to Hugging Face API lookups performed by [`check`].
const HF_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// A pinned model: which role it fills, where it comes from, at which
/// revision, and where it lives on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Logical role this model fills in the pipeline: "primeline", "flix",
    /// "qwen", or "diarization".
    pub role: String,
    /// Hugging Face repo id, e.g. "primeline/whisper-large-v3-german".
    pub repo: String,
    /// Fixed revision (commit sha, tag, or branch) this spec is pinned to.
    pub revision: String,
    /// Local directory the model snapshot is (or should be) materialized in.
    pub path: PathBuf,
}

/// Whether a newer revision is known to be available upstream, and whether
/// that could even be determined right now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    /// The pinned revision matches the latest revision reported upstream.
    UpToDate,
    /// Upstream reports a different (newer) revision than the one pinned.
    UpdateAvailable {
        /// The revision currently reported by Hugging Face for this repo.
        remote_revision: String,
    },
    /// Upstream could not be reached (offline, DNS failure, HTTP error,
    /// unparsable response, ...). The local model is still usable; this is
    /// purely informational and never causes [`check`] to fail as long as
    /// the local model itself is present.
    Unavailable,
}

/// Result of checking one [`ModelSpec`] against its local manifest and
/// (best-effort) against the Hugging Face API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStatus {
    pub role: String,
    pub repo: String,
    pub revision: String,
    pub path: PathBuf,
    /// `true` if every file recorded in the local manifest still hashes to
    /// the recorded value *and* the manifest's revision matches the spec's
    /// pinned revision. `false` indicates corruption, tampering, or a spec
    /// whose revision was edited without running [`update`] afterward.
    pub integrity_ok: bool,
    /// Whether a newer revision is available, as far as could be determined.
    pub availability: Availability,
}

/// On-disk manifest recording exactly what [`update`] wrote for a model, so
/// [`check`] can later verify it offline.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    repo: String,
    revision: String,
    /// Relative (forward-slash normalized) file path -> hex sha256.
    files: BTreeMap<String, String>,
}

/// Check every spec: verify local presence + integrity, and best-effort
/// report whether a newer revision exists upstream.
///
/// # Errors
///
/// Returns `Err` immediately (without checking the remaining specs) if a
/// spec's local model is missing (no manifest found at `path`). A model that
/// is present but corrupt, or a repo that is unreachable, is *not* an error:
/// it is reported via `integrity_ok` / [`Availability::Unavailable`] in the
/// corresponding [`ModelStatus`] instead.
pub fn check(specs: &[ModelSpec]) -> anyhow::Result<Vec<ModelStatus>> {
    let client = build_hf_client()?;

    specs
        .iter()
        .map(|spec| check_one_with(spec, |repo| fetch_remote_revision(&client, repo)))
        .collect()
}

/// Build the shared blocking HTTP client used for Hugging Face API lookups.
fn build_hf_client() -> anyhow::Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .timeout(HF_REQUEST_TIMEOUT)
        .user_agent(concat!("shum/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("failed to build Hugging Face HTTP client")
}

/// Core of [`check`] for a single spec, with the remote-lookup step
/// injected as a closure so it can be exercised offline in tests.
fn check_one_with<F>(spec: &ModelSpec, fetch_remote_revision: F) -> anyhow::Result<ModelStatus>
where
    F: FnOnce(&str) -> anyhow::Result<String>,
{
    if spec.revision.is_empty() {
        bail!(
            "model role '{}' ({}) has no pinned revision yet — run `shum models update --role {}` to bootstrap it",
            spec.role,
            spec.repo,
            spec.role
        );
    }

    let manifest = load_manifest(&spec.path).with_context(|| {
        format!(
            "local model for role '{}' ({}) is missing at {} — run `shum models update --role {}` first",
            spec.role,
            spec.repo,
            spec.path.display(),
            spec.role
        )
    })?;

    let integrity_ok = manifest.repo == spec.repo
        && manifest.revision == spec.revision
        && compute_dir_hashes(&spec.path)
            .map(|current| current == manifest.files)
            .unwrap_or(false);

    let availability = match fetch_remote_revision(&spec.repo) {
        Ok(remote_revision) if remote_revision == spec.revision => Availability::UpToDate,
        Ok(remote_revision) => Availability::UpdateAvailable { remote_revision },
        Err(_) => Availability::Unavailable,
    };

    Ok(ModelStatus {
        role: spec.role.clone(),
        repo: spec.repo.clone(),
        revision: spec.revision.clone(),
        path: spec.path.clone(),
        integrity_ok,
        availability,
    })
}

/// Query the Hugging Face model API for the current default-branch revision
/// of `repo`.
fn fetch_remote_revision(client: &reqwest::blocking::Client, repo: &str) -> anyhow::Result<String> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    let body = client
        .get(&url)
        .send()
        .and_then(|resp| resp.error_for_status())
        .with_context(|| format!("request to {url} failed"))?
        .text()
        .context("failed to read Hugging Face API response body")?;
    parse_remote_revision(&body)
}

/// Parse the `sha` field out of a Hugging Face `/api/models/{repo}` JSON
/// response. Split out from [`fetch_remote_revision`] so parsing can be unit
/// tested without any network access.
fn parse_remote_revision(body: &str) -> anyhow::Result<String> {
    #[derive(Deserialize)]
    struct HfModelInfo {
        sha: Option<String>,
    }

    let info: HfModelInfo =
        serde_json::from_str(body).context("failed to parse Hugging Face API response as JSON")?;
    info.sha
        .filter(|sha| !sha.is_empty())
        .ok_or_else(|| anyhow!("Hugging Face API response is missing a 'sha' field"))
}

/// Suffix appended to a revision directory's name for the staging area used
/// while downloading a candidate revision, before it is renamed into its
/// final immutable location.
const STAGING_SUFFIX: &str = ".shum-staging";

/// The directory under which per-revision immutable snapshots for `spec`
/// live. If `spec.path` already names its own currently-pinned revision
/// (i.e. it came from a previous call to [`update`]/[`install_pinned`],
/// which always return a path ending in the revision they installed), the
/// base is `spec.path`'s parent, so a newly resolved revision becomes a
/// *sibling* directory rather than nesting under the old one. Otherwise —
/// notably the fresh `models/<role>` path from the example config, before
/// anything has ever been installed — `spec.path` itself is the base.
fn base_dir_for(spec: &ModelSpec) -> &Path {
    let path_names_current_revision = !spec.revision.is_empty()
        && spec.path.file_name() == Some(std::ffi::OsStr::new(spec.revision.as_str()));
    if path_names_current_revision {
        spec.path.parent().unwrap_or(spec.path.as_path())
    } else {
        spec.path.as_path()
    }
}

/// Always resolve the latest revision Hugging Face reports for `spec.repo`'s
/// default branch, download it, and validate it into its own immutable
/// directory — even if `spec.revision` was already pinned to something
/// else. This is the only way an already-pinned model can move to a newer
/// revision; re-downloading the same pinned sha in place would be a no-op.
///
/// The returned spec's `path` points at the freshly materialized
/// `<base>/<revision>` directory (see [`base_dir_for`]); `spec.path` (the
/// previously active directory, if different) is never read, modified, or
/// removed by this function. Making the returned spec "active" — e.g. by
/// persisting it into `shum.toml` — is the caller's responsibility;
/// `update` only produces a validated, ready directory.
///
/// If Hugging Face cannot be reached, the download fails, or integrity
/// hashing fails, nothing already on disk changes: no partial directory is
/// left behind that could later be mistaken for a valid install, and the
/// previously installed revision keeps working untouched.
///
/// # Errors
///
/// Fails if the latest revision cannot be resolved (e.g. offline), if the
/// `hf` CLI is not installed/on `PATH`, if the download command exits
/// non-zero, or if the manifest cannot be written afterward.
pub fn update(spec: &ModelSpec) -> anyhow::Result<ModelSpec> {
    let client = build_hf_client()?;
    update_with(
        spec,
        |repo| fetch_remote_revision(&client, repo),
        run_hf_download,
    )
}

/// Core of [`update`], with both the remote "latest revision" lookup and the
/// actual download step injected as closures so it is fully unit-testable
/// without any network access or the real `hf` CLI.
fn update_with<Fetch, Download>(
    spec: &ModelSpec,
    fetch_remote_revision: Fetch,
    download: Download,
) -> anyhow::Result<ModelSpec>
where
    Fetch: FnOnce(&str) -> anyhow::Result<String>,
    Download: FnOnce(&str, &str, &Path) -> anyhow::Result<()>,
{
    let latest = fetch_remote_revision(&spec.repo).with_context(|| {
        format!(
            "role '{}' ({}): failed to resolve the latest revision from Hugging Face — the previously installed revision (if any) is unchanged",
            spec.role, spec.repo
        )
    })?;
    let repo = spec.repo.clone();
    materialize_revision(spec, &latest, |revision, staging| {
        download(&repo, revision, staging)
    })
}

/// Download and validate the *exact* `spec.revision` already recorded on the
/// spec into its own immutable directory, without consulting Hugging Face
/// for a "latest" revision. Useful for reproducible installs of a known
/// revision; not wired into the CLI, which only exposes [`update`].
///
/// # Errors
///
/// Fails if `spec.revision` is empty (nothing to pin to — use [`update`] to
/// bootstrap), if the `hf` CLI is not installed/on `PATH`, if the download
/// command exits non-zero, or if the manifest cannot be written afterward.
#[allow(dead_code)] // Intentionally-provided API not yet wired into the CLI (which only exposes `update`).
pub fn install_pinned(spec: &ModelSpec) -> anyhow::Result<ModelSpec> {
    install_pinned_with(spec, run_hf_download)
}

/// Core of [`install_pinned`], with the download step injected as a closure
/// so it is unit-testable without the real `hf` CLI.
#[allow(dead_code)] // Only reachable through `install_pinned`, see above.
fn install_pinned_with<Download>(spec: &ModelSpec, download: Download) -> anyhow::Result<ModelSpec>
where
    Download: FnOnce(&str, &str, &Path) -> anyhow::Result<()>,
{
    if spec.revision.is_empty() {
        bail!(
            "role '{}' ({}) has no pinned revision to install — use `update` to fetch and pin the latest one",
            spec.role,
            spec.repo
        );
    }
    let revision = spec.revision.clone();
    let repo = spec.repo.clone();
    materialize_revision(spec, &revision, |revision, staging| {
        download(&repo, revision, staging)
    })
}

/// Download `revision` via the injected `download` closure into its own
/// immutable directory `<base>/<revision>` (see [`base_dir_for`]), hash and
/// manifest the result, and only then rename it into that final location.
/// Returns `spec` with `path` set to the new directory and `revision` set
/// to `revision`. The directory named by `spec.path` itself is never
/// touched by this function.
///
/// If `<base>/<revision>` already exists and its manifest and file hashes
/// are already valid for `revision` — e.g. a repeat `update` that resolved
/// to the same already-materialized revision — this is a fast, idempotent
/// no-op: nothing is re-downloaded. If it exists but is invalid (e.g. a
/// previous attempt was interrupted), it is discarded and redone from
/// scratch.
///
/// The `download` step is injected so this (and [`update`]/
/// [`install_pinned`]) can be unit-tested offline: production code passes a
/// closure that shells out to the real `hf` CLI; tests pass one that just
/// writes fixture files into the staging directory.
fn materialize_revision<D>(
    spec: &ModelSpec,
    revision: &str,
    download: D,
) -> anyhow::Result<ModelSpec>
where
    D: FnOnce(&str, &Path) -> anyhow::Result<()>,
{
    let target = base_dir_for(spec).join(revision);

    if target.exists() {
        let already_valid = load_manifest(&target)
            .ok()
            .filter(|existing| existing.repo == spec.repo && existing.revision == revision)
            .and_then(|existing| {
                compute_dir_hashes(&target)
                    .ok()
                    .map(|files| files == existing.files)
            })
            .unwrap_or(false);
        if already_valid {
            return Ok(ModelSpec {
                revision: revision.to_string(),
                path: target,
                ..spec.clone()
            });
        }
        // Present but not (or no longer) valid for this exact revision —
        // most likely an interrupted previous attempt, or a bare leftover
        // directory with no manifest at all. Clear it so the download below
        // starts from a clean slate.
        fs::remove_dir_all(&target).with_context(|| {
            format!(
                "failed to clear invalid existing directory {}",
                target.display()
            )
        })?;
    }

    let staging = sibling_with_suffix(&target, STAGING_SUFFIX)?;
    if staging.exists() {
        fs::remove_dir_all(&staging).with_context(|| {
            format!(
                "failed to clear stale staging directory {}",
                staging.display()
            )
        })?;
    }
    fs::create_dir_all(&staging)
        .with_context(|| format!("failed to create staging directory {}", staging.display()))?;

    let outcome = download(revision, &staging).and_then(|()| {
        let files = compute_dir_hashes(&staging)
            .context("failed to hash downloaded model files for the integrity manifest")?;
        if files.is_empty() {
            bail!(
                "Hugging Face model {}/{} downloaded no files",
                spec.repo,
                revision
            );
        }
        let manifest = Manifest {
            repo: spec.repo.clone(),
            revision: revision.to_string(),
            files,
        };
        save_manifest(&staging, &manifest)
    });

    if let Err(err) = outcome {
        // Never leave a half-downloaded candidate lying around — and since
        // `target` is a brand-new path distinct from `spec.path`, nothing
        // previously installed has been touched at all.
        let _ = fs::remove_dir_all(&staging);
        return Err(err);
    }

    fs::rename(&staging, &target).with_context(|| {
        format!(
            "downloaded and validated revision {revision} but failed to finalize it at {}",
            target.display()
        )
    })?;

    Ok(ModelSpec {
        revision: revision.to_string(),
        path: target,
        ..spec.clone()
    })
}

/// Shell out to the real `hf` CLI to download `revision` of `repo` into
/// `staging`. Split out from [`materialize_revision`] so the latter can be
/// unit-tested with a fake `download` closure instead.
fn run_hf_download(repo: &str, revision: &str, staging: &Path) -> anyhow::Result<()> {
    let status = Command::new("hf")
        .arg("download")
        .arg(repo)
        .arg("--revision")
        .arg(revision)
        .arg("--local-dir")
        .arg(staging)
        .status()
        .context(
            "failed to run `hf` CLI — install it with `pip install -U \"huggingface_hub[cli]\"`",
        )?;

    if !status.success() {
        bail!(
            "`hf download {repo} --revision {revision} --local-dir {}` exited with {status}",
            staging.display()
        );
    }
    Ok(())
}

/// Build a sibling path of `path` with `suffix` appended to its file name,
/// e.g. `models/primeline` + `.shum-staging` -> `models/primeline.shum-staging`.
fn sibling_with_suffix(path: &Path, suffix: &str) -> anyhow::Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("model path {} has no parent directory", path.display()))?;
    let name = path
        .file_name()
        .ok_or_else(|| anyhow!("model path {} has no file name", path.display()))?;
    let mut staged_name = name.to_os_string();
    staged_name.push(suffix);
    Ok(parent.join(staged_name))
}

/// Load and parse the manifest for a model directory. `Err` means "no valid
/// local model here", which callers treat as "must run `update` first".
fn load_manifest(root: &Path) -> anyhow::Result<Manifest> {
    let manifest_path = root.join(MANIFEST_FILENAME);
    let data = fs::read_to_string(&manifest_path)
        .with_context(|| format!("no manifest at {}", manifest_path.display()))?;
    serde_json::from_str(&data)
        .with_context(|| format!("manifest at {} is not valid JSON", manifest_path.display()))
}

/// Write the manifest atomically: write to a temp file, then rename into
/// place, so a crash mid-write never leaves a corrupt/partial manifest.
fn save_manifest(root: &Path, manifest: &Manifest) -> anyhow::Result<()> {
    let final_path = root.join(MANIFEST_FILENAME);
    let tmp_path = root.join(MANIFEST_TMP_FILENAME);
    let data = serde_json::to_vec_pretty(manifest).context("failed to serialize manifest")?;
    fs::write(&tmp_path, &data)
        .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("failed to finalize {}", final_path.display()))?;
    Ok(())
}

/// Recursively hash every file under `root` (skipping the manifest itself),
/// keyed by its forward-slash-normalized path relative to `root`.
fn compute_dir_hashes(root: &Path) -> anyhow::Result<BTreeMap<String, String>> {
    let mut hashes = BTreeMap::new();
    let mut dirs = vec![PathBuf::new()];

    while let Some(rel_dir) = dirs.pop() {
        let abs_dir = root.join(&rel_dir);
        for entry in fs::read_dir(&abs_dir)
            .with_context(|| format!("failed to read directory {}", abs_dir.display()))?
        {
            let entry = entry?;
            let name = entry.file_name();
            if name == MANIFEST_FILENAME || name == MANIFEST_TMP_FILENAME {
                continue;
            }
            let rel_path = rel_dir.join(&name);
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                dirs.push(rel_path);
            } else if file_type.is_file() {
                let hash = compute_file_sha256(&entry.path())?;
                let key = rel_path.to_string_lossy().replace('\\', "/");
                hashes.insert(key, hash);
            } else {
                bail!("unsupported model entry: {}", entry.path().display());
            }
        }
    }

    Ok(hashes)
}

/// Compute the hex-encoded sha256 digest of a single file's contents.
fn compute_file_sha256(path: &Path) -> anyhow::Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A directory under the OS temp dir, unique per test, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "shum-models-test-{label}-{}-{n}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            TempDir(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn spec(role: &str, repo: &str, revision: &str, path: &Path) -> ModelSpec {
        ModelSpec {
            role: role.to_string(),
            repo: repo.to_string(),
            revision: revision.to_string(),
            path: path.to_path_buf(),
        }
    }

    #[test]
    fn model_spec_round_trips_through_json() {
        let s = spec("asr", "org/repo", "deadbeef", Path::new("/models/asr"));
        let json = serde_json::to_string(&s).unwrap();
        let back: ModelSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn parse_remote_revision_extracts_sha() {
        let body = r#"{"id":"org/repo","sha":"abc123","lastModified":"2024-01-01T00:00:00.000Z"}"#;
        assert_eq!(parse_remote_revision(body).unwrap(), "abc123");
    }

    #[test]
    fn parse_remote_revision_rejects_missing_sha() {
        let body = r#"{"id":"org/repo"}"#;
        assert!(parse_remote_revision(body).is_err());
    }

    #[test]
    fn parse_remote_revision_rejects_invalid_json() {
        assert!(parse_remote_revision("not json").is_err());
    }

    #[test]
    fn sha256_matches_known_test_vector() {
        let dir = TempDir::new("hash-vector");
        let file = dir.path().join("hello.txt");
        fs::write(&file, b"hello world").unwrap();
        // Well-known sha256("hello world").
        assert_eq!(
            compute_file_sha256(&file).unwrap(),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn compute_dir_hashes_walks_nested_dirs_and_skips_manifest() {
        let dir = TempDir::new("walk");
        fs::write(dir.path().join("a.bin"), b"aaa").unwrap();
        fs::create_dir_all(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/b.bin"), b"bbb").unwrap();
        fs::write(dir.path().join(MANIFEST_FILENAME), b"{}").unwrap();

        let hashes = compute_dir_hashes(dir.path()).unwrap();

        assert_eq!(hashes.len(), 2);
        assert!(hashes.contains_key("a.bin"));
        assert!(hashes.contains_key("sub/b.bin"));
        assert!(!hashes.contains_key(MANIFEST_FILENAME));
    }

    #[test]
    fn manifest_round_trips_atomically() {
        let dir = TempDir::new("manifest-roundtrip");
        let manifest = Manifest {
            repo: "org/repo".to_string(),
            revision: "deadbeef".to_string(),
            files: BTreeMap::from([("weights.bin".to_string(), "abc".to_string())]),
        };

        save_manifest(dir.path(), &manifest).unwrap();
        // No leftover temp file after a successful save.
        assert!(!dir.path().join(MANIFEST_TMP_FILENAME).exists());

        let loaded = load_manifest(dir.path()).unwrap();
        assert_eq!(loaded.repo, manifest.repo);
        assert_eq!(loaded.revision, manifest.revision);
        assert_eq!(loaded.files, manifest.files);

        // Overwriting (as update() would on a re-download) still round-trips.
        let manifest2 = Manifest {
            files: BTreeMap::from([("weights.bin".to_string(), "def".to_string())]),
            ..manifest
        };
        save_manifest(dir.path(), &manifest2).unwrap();
        assert_eq!(load_manifest(dir.path()).unwrap().files, manifest2.files);
    }

    #[test]
    fn check_errors_when_local_model_missing() {
        let dir = TempDir::new("missing-local");
        let missing_path = dir.path().join("does-not-exist");
        let s = spec("asr", "org/repo", "deadbeef", &missing_path);

        let err = check_one_with(&s, |_repo| Ok("deadbeef".to_string())).unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn check_errors_when_revision_not_yet_pinned() {
        let dir = TempDir::new("unbootstrapped");
        let s = spec("primeline", "org/repo", "", dir.path());

        // Must bail before touching the filesystem or the network closure.
        let err =
            check_one_with(&s, |_repo| panic!("must not fetch for an unpinned spec")).unwrap_err();
        assert!(err.to_string().contains("no pinned revision"));
    }

    #[test]
    fn base_dir_for_uses_parent_when_path_names_current_revision() {
        let s = spec(
            "primeline",
            "org/repo",
            "old-sha",
            Path::new("/models/primeline/old-sha"),
        );
        assert_eq!(base_dir_for(&s), Path::new("/models/primeline"));
    }

    #[test]
    fn base_dir_for_uses_path_itself_for_a_fresh_role_directory() {
        // The example config's starting state: `path = "models/<role>"` with
        // no sha suffix at all, and (initially) an empty revision.
        let s = spec("primeline", "org/repo", "", Path::new("/models/primeline"));
        assert_eq!(base_dir_for(&s), Path::new("/models/primeline"));

        // Also applies once pinned, as long as the path itself doesn't name
        // that revision (e.g. a flat, pre-per-revision-layout install).
        let s = spec(
            "primeline",
            "org/repo",
            "old-sha",
            Path::new("/models/primeline"),
        );
        assert_eq!(base_dir_for(&s), Path::new("/models/primeline"));
    }

    #[test]
    fn update_materializes_new_revision_as_sibling_directory_and_leaves_old_pinned_path_untouched()
    {
        let dir = TempDir::new("update-pinned-to-new");
        let old_path = dir.path().join("old-sha");
        fs::create_dir_all(&old_path).unwrap();
        fs::write(old_path.join("weights.bin"), b"old-weights").unwrap();
        let original_files = compute_dir_hashes(&old_path).unwrap();
        save_manifest(
            &old_path,
            &Manifest {
                repo: "org/repo".to_string(),
                revision: "old-sha".to_string(),
                files: original_files.clone(),
            },
        )
        .unwrap();
        let s = spec("primeline", "org/repo", "old-sha", &old_path);

        let updated = update_with(
            &s,
            |_repo| Ok("new-sha".to_string()),
            |_repo, _revision, staging| {
                fs::write(staging.join("new-weights.bin"), b"new-weights")?;
                Ok(())
            },
        )
        .unwrap();

        // The new revision lives in a brand-new sibling directory, not in
        // place of the old one.
        let new_path = dir.path().join("new-sha");
        assert_eq!(updated.revision, "new-sha");
        assert_eq!(updated.role, "primeline");
        assert_eq!(updated.path, new_path);
        assert!(new_path.join("new-weights.bin").exists());
        assert_eq!(load_manifest(&new_path).unwrap().revision, "new-sha");

        // The previously pinned directory and its manifest are completely
        // untouched: same content, same revision, still there.
        assert!(old_path.join("weights.bin").exists());
        let old_manifest = load_manifest(&old_path).unwrap();
        assert_eq!(old_manifest.revision, "old-sha");
        assert_eq!(old_manifest.files, original_files);

        // No staging directory left behind next to the new revision.
        assert!(
            !sibling_with_suffix(&new_path, STAGING_SUFFIX)
                .unwrap()
                .exists()
        );
    }

    #[test]
    fn update_bootstraps_a_never_installed_model() {
        let dir = TempDir::new("update-bootstrap");
        // Model directory does not exist yet at all (first-ever install);
        // this is the example config's starting state: `models/<role>` with
        // an empty revision.
        let role_dir = dir.path().join("primeline");
        let s = spec("primeline", "org/repo", "", &role_dir);

        let updated = update_with(
            &s,
            |_repo| Ok("first-sha".to_string()),
            |_repo, _revision, staging| {
                fs::write(staging.join("new-weights.bin"), b"new-weights")?;
                Ok(())
            },
        )
        .unwrap();

        let target = role_dir.join("first-sha");
        assert_eq!(updated.revision, "first-sha");
        assert_eq!(updated.path, target);
        assert!(target.join("new-weights.bin").exists());
        assert_eq!(load_manifest(&target).unwrap().revision, "first-sha");
    }

    #[test]
    fn update_is_idempotent_when_the_resolved_revision_is_already_materialized() {
        let dir = TempDir::new("update-idempotent");
        let target = dir.path().join("same-sha");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("weights.bin"), b"weights").unwrap();
        let files = compute_dir_hashes(&target).unwrap();
        save_manifest(
            &target,
            &Manifest {
                repo: "org/repo".to_string(),
                revision: "same-sha".to_string(),
                files: files.clone(),
            },
        )
        .unwrap();
        let s = spec("primeline", "org/repo", "same-sha", &target);

        // "Latest" resolves to the same sha that's already materialized;
        // the download closure must never run.
        let updated = update_with(
            &s,
            |_repo| Ok("same-sha".to_string()),
            |_repo, _revision, _staging| panic!("must not re-download an already-valid revision"),
        )
        .unwrap();

        assert_eq!(updated.revision, "same-sha");
        assert_eq!(updated.path, target);
        assert_eq!(load_manifest(&target).unwrap().files, files);
    }

    #[test]
    fn update_redownloads_when_an_existing_target_directory_is_invalid() {
        let dir = TempDir::new("update-invalid-target");
        let target = dir.path().join("same-sha");
        // A bare leftover directory with no manifest at all — e.g. from an
        // interrupted previous attempt — must not be mistaken for valid.
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("partial.bin"), b"incomplete").unwrap();
        let s = spec("primeline", "org/repo", "", dir.path());

        let updated = update_with(
            &s,
            |_repo| Ok("same-sha".to_string()),
            |_repo, _revision, staging| {
                fs::write(staging.join("weights.bin"), b"fresh-weights")?;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(updated.path, target);
        assert!(!target.join("partial.bin").exists());
        assert!(target.join("weights.bin").exists());
        assert_eq!(load_manifest(&target).unwrap().revision, "same-sha");
    }

    #[test]
    fn update_retains_old_spec_and_manifest_when_remote_lookup_fails_offline() {
        let dir = TempDir::new("update-offline-fetch-failure");
        let old_path = dir.path().join("old-sha");
        fs::create_dir_all(&old_path).unwrap();
        fs::write(old_path.join("weights.bin"), b"old-weights").unwrap();
        let original_files = compute_dir_hashes(&old_path).unwrap();
        save_manifest(
            &old_path,
            &Manifest {
                repo: "org/repo".to_string(),
                revision: "old-sha".to_string(),
                files: original_files.clone(),
            },
        )
        .unwrap();
        let s = spec("primeline", "org/repo", "old-sha", &old_path);

        let err = update_with(
            &s,
            |_repo| Err(anyhow!("simulated offline failure")),
            |_repo, _revision, _staging| {
                panic!("download must not be attempted when the latest revision can't be resolved")
            },
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("failed to resolve the latest revision")
        );

        // Nothing on disk changed: same manifest, same files, and no new
        // sibling directory was ever created for a revision that was never
        // even resolved.
        let manifest = load_manifest(&old_path).unwrap();
        assert_eq!(manifest.revision, "old-sha");
        assert_eq!(manifest.files, original_files);
        assert!(old_path.join("weights.bin").exists());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn update_retains_old_spec_and_manifest_when_download_itself_fails() {
        let dir = TempDir::new("update-download-failure");
        let old_path = dir.path().join("old-sha");
        fs::create_dir_all(&old_path).unwrap();
        fs::write(old_path.join("weights.bin"), b"old-weights").unwrap();
        let original_files = compute_dir_hashes(&old_path).unwrap();
        save_manifest(
            &old_path,
            &Manifest {
                repo: "org/repo".to_string(),
                revision: "old-sha".to_string(),
                files: original_files.clone(),
            },
        )
        .unwrap();
        let s = spec("primeline", "org/repo", "old-sha", &old_path);

        // Latest resolves fine, but the (simulated) download itself fails
        // partway through — e.g. connection dropped mid-transfer.
        let err = materialize_revision(&s, "new-sha", |_revision, _staging| {
            Err(anyhow!("simulated mid-download failure"))
        })
        .unwrap_err();
        assert!(err.to_string().contains("mid-download"));

        // The old pinned directory is untouched, and no partial "new-sha"
        // directory (or its staging leftover) was left behind to later be
        // mistaken for a valid install.
        let manifest = load_manifest(&old_path).unwrap();
        assert_eq!(manifest.revision, "old-sha");
        assert_eq!(manifest.files, original_files);
        assert!(old_path.join("weights.bin").exists());
        let new_path = dir.path().join("new-sha");
        assert!(!new_path.exists());
        assert!(
            !sibling_with_suffix(&new_path, STAGING_SUFFIX)
                .unwrap()
                .exists()
        );
    }

    #[test]
    fn install_pinned_downloads_exact_revision_without_fetching_latest() {
        let dir = TempDir::new("install-pinned");
        let role_dir = dir.path().join("primeline");
        let s = spec("primeline", "org/repo", "exact-sha", &role_dir);

        let installed = install_pinned_with(&s, |_repo, _revision, staging| {
            fs::write(staging.join("new-weights.bin"), b"pinned-weights")?;
            Ok(())
        })
        .unwrap();

        let target = role_dir.join("exact-sha");
        assert_eq!(installed.revision, "exact-sha");
        assert_eq!(installed.path, target);
        assert_eq!(load_manifest(&target).unwrap().revision, "exact-sha");
    }

    #[test]
    fn install_pinned_errors_on_empty_revision() {
        let s = spec("primeline", "org/repo", "", Path::new("/models/primeline"));
        let err = install_pinned(&s).unwrap_err();
        assert!(err.to_string().contains("no pinned revision to install"));
    }

    #[test]
    fn check_reports_up_to_date_when_remote_matches() {
        let dir = TempDir::new("up-to-date");
        fs::write(dir.path().join("weights.bin"), b"weights").unwrap();
        let files = compute_dir_hashes(dir.path()).unwrap();
        save_manifest(
            dir.path(),
            &Manifest {
                repo: "org/repo".to_string(),
                revision: "deadbeef".to_string(),
                files,
            },
        )
        .unwrap();
        let s = spec("asr", "org/repo", "deadbeef", dir.path());

        let status = check_one_with(&s, |_repo| Ok("deadbeef".to_string())).unwrap();
        assert!(status.integrity_ok);
        assert_eq!(status.availability, Availability::UpToDate);
    }

    #[test]
    fn check_reports_update_available_when_remote_differs() {
        let dir = TempDir::new("update-available");
        fs::write(dir.path().join("weights.bin"), b"weights").unwrap();
        let files = compute_dir_hashes(dir.path()).unwrap();
        save_manifest(
            dir.path(),
            &Manifest {
                repo: "org/repo".to_string(),
                revision: "old-sha".to_string(),
                files,
            },
        )
        .unwrap();
        let s = spec("asr", "org/repo", "old-sha", dir.path());

        let status = check_one_with(&s, |_repo| Ok("new-sha".to_string())).unwrap();
        assert!(status.integrity_ok);
        assert_eq!(
            status.availability,
            Availability::UpdateAvailable {
                remote_revision: "new-sha".to_string()
            }
        );
    }

    #[test]
    fn check_reports_unavailable_on_network_failure_but_still_succeeds() {
        let dir = TempDir::new("network-failure");
        fs::write(dir.path().join("weights.bin"), b"weights").unwrap();
        let files = compute_dir_hashes(dir.path()).unwrap();
        save_manifest(
            dir.path(),
            &Manifest {
                repo: "org/repo".to_string(),
                revision: "deadbeef".to_string(),
                files,
            },
        )
        .unwrap();
        let s = spec("asr", "org/repo", "deadbeef", dir.path());

        let status = check_one_with(&s, |_repo| Err(anyhow!("simulated offline failure"))).unwrap();
        assert!(status.integrity_ok);
        assert_eq!(status.availability, Availability::Unavailable);
    }

    #[test]
    fn check_flags_integrity_failure_on_file_corruption() {
        let dir = TempDir::new("corruption");
        fs::write(dir.path().join("weights.bin"), b"weights").unwrap();
        let files = compute_dir_hashes(dir.path()).unwrap();
        save_manifest(
            dir.path(),
            &Manifest {
                repo: "org/repo".to_string(),
                revision: "deadbeef".to_string(),
                files,
            },
        )
        .unwrap();
        // Corrupt the file after the manifest was written.
        fs::write(dir.path().join("weights.bin"), b"tampered").unwrap();
        let s = spec("asr", "org/repo", "deadbeef", dir.path());

        let status = check_one_with(&s, |_repo| Ok("deadbeef".to_string())).unwrap();
        assert!(!status.integrity_ok);
    }

    #[test]
    fn check_flags_integrity_failure_on_revision_drift() {
        let dir = TempDir::new("drift");
        fs::write(dir.path().join("weights.bin"), b"weights").unwrap();
        let files = compute_dir_hashes(dir.path()).unwrap();
        save_manifest(
            dir.path(),
            &Manifest {
                repo: "org/repo".to_string(),
                revision: "old-sha".to_string(),
                files,
            },
        )
        .unwrap();
        // Spec was edited to a new revision without calling update() yet.
        let s = spec("asr", "org/repo", "new-sha", dir.path());

        let status = check_one_with(&s, |_repo| Ok("new-sha".to_string())).unwrap();
        assert!(!status.integrity_ok);
        assert_eq!(status.availability, Availability::UpToDate);
    }
}
