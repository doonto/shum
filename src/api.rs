use std::fs::{DirBuilder, File, OpenOptions};
use std::io::{self, BufRead, Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{config::Config, execution, models, onboarding, workflow};

const MAX_REQUEST: usize = 65536;
const MAX_LOG: u64 = 65536;
static REQUEST_NUMBER: AtomicU64 = AtomicU64::new(0);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    version: u32,
    id: String,
    method: String,
    #[serde(default = "empty_params")]
    params: Value,
}

fn empty_params() -> Value {
    json!({})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Setup {
    installation: PathBuf,
    bin: PathBuf,
    settings: PathBuf,
    projects: PathBuf,
    models: PathBuf,
    command_dir: Option<PathBuf>,
    #[serde(default)]
    setup_shell: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Project {
    project: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Events {
    project: PathBuf,
    #[serde(default)]
    after: u64,
    #[serde(default = "event_limit")]
    limit: usize,
}

fn event_limit() -> usize {
    100
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Control {
    project: PathBuf,
    request: execution::ControlRequest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Start {
    input: PathBuf,
    project: PathBuf,
    #[serde(default)]
    sequential_asr: bool,
    #[serde(default)]
    no_quality: bool,
    asr: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Resume {
    project: PathBuf,
    input: Option<PathBuf>,
    #[serde(default)]
    sequential_asr: bool,
    #[serde(default)]
    no_quality: bool,
    asr: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Stage {
    project: PathBuf,
    step: String,
    input: Option<PathBuf>,
    #[serde(default)]
    sequential_asr: bool,
    #[serde(default)]
    no_quality: bool,
    asr: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityPage {
    project: PathBuf,
    #[serde(default)]
    offset: usize,
    #[serde(default = "quality_page_limit")]
    limit: usize,
}

fn quality_page_limit() -> usize {
    10
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Verify {
    role: Option<String>,
}

struct Failure {
    code: &'static str,
    message: String,
    data: Option<Value>,
}

type ApiResult = std::result::Result<Value, Failure>;

fn failure(code: &'static str, error: impl std::fmt::Display) -> Failure {
    Failure {
        code,
        message: error.to_string(),
        data: None,
    }
}

fn params<T: serde::de::DeserializeOwned>(value: &Value) -> std::result::Result<T, Failure> {
    serde_json::from_value(value.clone()).map_err(|e| failure("invalid_params", e))
}

fn context(
    installation: Option<&Path>,
    config: Option<&Path>,
) -> Result<(onboarding::Settings, Config)> {
    let settings = onboarding::load_from(installation, config)?;
    let path = match config {
        Some(path) => path.to_path_buf(),
        None => onboarding::config_path_for(&settings)?,
    };
    let config = Config::load(&path)?;
    onboarding::validate_config_location(&path, &config)?;
    Ok((settings, config))
}

fn command(installation: Option<&Path>, config: Option<&Path>) -> Result<Command> {
    let mut command = Command::new(std::env::current_exe()?);
    if let Some(path) = installation {
        command.arg("--installation").arg(path);
    }
    if let Some(path) = config {
        command.arg("--config").arg(path);
    }
    Ok(command)
}

fn logs(command: &mut Command) -> Result<(i32, PathBuf, String)> {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = std::env::temp_dir().join(format!(
        "shum-api-{}-{stamp}-{}",
        std::process::id(),
        REQUEST_NUMBER.fetch_add(1, Ordering::Relaxed)
    ));
    DirBuilder::new().mode(0o700).create(&path)?;
    let stdout = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path.join("stdout.log"))?;
    let stderr = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path.join("stderr.log"))?;
    let status = command
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .status()
        .with_context(|| format!("could not launch workflow; logs: {}", path.display()))?;
    let mut tail = File::open(path.join("stderr.log"))?;
    let length = tail.metadata()?.len();
    use std::io::{Seek, SeekFrom};
    tail.seek(SeekFrom::Start(length.saturating_sub(MAX_LOG)))?;
    let mut bytes = Vec::new();
    tail.take(MAX_LOG).read_to_end(&mut bytes)?;
    let prefix = if length > MAX_LOG {
        "[stderr truncated to final 64 KiB]\n"
    } else {
        ""
    };
    Ok((
        status.code().unwrap_or(1),
        path,
        format!("{prefix}{}", String::from_utf8_lossy(&bytes)),
    ))
}

fn workflow_outcome(exit_code: i32, step: Option<&str>) -> &'static str {
    match exit_code {
        2 => "pending_review",
        3 => "blocked",
        130 => "interrupted",
        _ if step.is_some() => "stage_complete",
        _ => "complete",
    }
}

fn run(command: &mut Command, project: Option<&Path>, step: Option<&str>) -> ApiResult {
    let (exit_code, log_directory, stderr) =
        logs(command).map_err(|e| failure("execution_failed", format!("{e:#}")))?;
    if !matches!(exit_code, 0 | 2 | 3 | 130) {
        return Err(Failure {
            code: "workflow_failed",
            message: stderr,
            data: Some(
                json!({"exit_code":exit_code,"log_directory":log_directory,"project":project}),
            ),
        });
    }
    let status = project
        .map(workflow::status)
        .transpose()
        .map_err(|e| failure("invalid_project", format!("{e:#}")))?;
    let outcome = workflow_outcome(exit_code, step);
    let exports = status
        .as_ref()
        .zip(project)
        .map(|(status, project)| status.export_paths(project))
        .transpose()
        .map_err(|e| failure("invalid_project", format!("{e:#}")))?;
    let quality = project.and_then(|project| match crate::quality::saved_summary(project) {
        Ok(summary) => summary,
        Err(error) => Some(json!({
            "execution_status": "failed",
            "accuracy_status": "unverified",
            "error": format!("{error:#}")
        })),
    });
    Ok(
        json!({"outcome":outcome,"exit_code":exit_code,"project":project,"step":step,"status":status,"exports":exports,"quality":quality,"log_directory":log_directory}),
    )
}

fn dispatch(request: &Request, installation: Option<&Path>, config: Option<&Path>) -> ApiResult {
    match request.method.as_str() {
        "installation.status" => {
            let _: Empty = params(&request.params)?;
            match context(installation, config) {
                Ok((settings, config)) => Ok(json!({
                    "state":"configured", "installation":settings, "projects_dir":settings.projects_dir(),
                    "steps":workflow::steps_for_roles(&config.asr_roles()), "model_verification":"not_checked",
                    "llm":{"enabled":config.llm.enabled,"provider":config.llm.provider,"access_checked":false},
                    "notice":"Configuration exists; models, environments and external CLI access require separate validation."
                })),
                Err(error) => Ok(
                    json!({"state":"needs_setup_or_repair","message":format!("{error:#}"),
                    "repair":"Use installation.setup for a fresh explicit layout, or interactive shum --onboarding / --setup-models / --setup-llm for repair.",
                    "external_access_checked":false}),
                ),
            }
        }
        "installation.setup" => {
            let p: Setup = params(&request.params)?;
            if installation.is_some() || config.is_some() {
                return Err(failure(
                    "invalid_params",
                    "installation.setup uses its explicit layout; omit global config/installation selectors",
                ));
            }
            if [&p.installation, &p.bin, &p.settings, &p.projects, &p.models]
                .iter()
                .any(|p| !p.is_absolute())
            {
                return Err(failure(
                    "invalid_params",
                    "all setup directories must be absolute",
                ));
            }
            if p.command_dir.as_ref().is_some_and(|p| !p.is_absolute())
                || (p.setup_shell && p.command_dir.is_none())
            {
                return Err(failure(
                    "invalid_params",
                    "command_dir must be absolute; setup_shell requires command_dir",
                ));
            }
            let mut child = command(None, None).map_err(|e| failure("execution_failed", e))?;
            child.env_remove("SHUM_SETTINGS");
            child
                .arg("--onboarding")
                .arg("--storage-dir")
                .arg(&p.installation)
                .arg("--bin-dir")
                .arg(&p.bin)
                .arg("--settings-dir")
                .arg(&p.settings)
                .arg("--projects-dir")
                .arg(&p.projects)
                .arg("--models-dir")
                .arg(&p.models);
            if let Some(directory) = &p.command_dir {
                child.arg("--command-dir").arg(directory);
            }
            if p.setup_shell {
                child.arg("--setup-shell");
            }
            let result = run(&mut child, None, None)?;
            let command_path = p
                .command_dir
                .as_deref()
                .map(|directory| onboarding::canonical_target(directory).map(|p| p.join("shum")))
                .transpose()
                .map_err(|e| failure("execution_failed", e))?;
            Ok(
                json!({"state":"layout_configured","layout":{"installation":p.installation,"settings":p.settings,
                "bin":p.bin,"projects":p.projects,"models":p.models},
                "command_path":command_path,"execution":result,
                "notice":"No model downloads or provider authentication were requested; configure environments, models and optional LLM separately."}),
            )
        }
        "models.verify" => {
            let p: Verify = params(&request.params)?;
            let (_, config) = context(installation, config)
                .map_err(|e| failure("setup_required", format!("{e:#}")))?;
            if let Some(role) = &p.role
                && !config.models.iter().any(|m| &m.role == role)
            {
                return Err(failure("invalid_params", "unknown configured model role"));
            }
            let checks: Vec<_> = config.models.iter().filter(|m| p.role.as_ref().is_none_or(|r| r == &m.role))
                .map(|m| match models::verify_local(m) {
                    Ok(()) => json!({"role":m.role,"revision":m.revision,"verified":true}),
                    Err(e) => json!({"role":m.role,"revision":m.revision,"verified":false,"message":format!("{e:#}")}),
                }).collect();
            Ok(
                json!({"all_verified":checks.iter().all(|m| m["verified"] == true),"models":checks,"network_access":false}),
            )
        }
        "project.status" => {
            let p: Project = params(&request.params)?;
            let (settings, _) = context(installation, config)
                .map_err(|e| failure("setup_required", format!("{e:#}")))?;
            let path = onboarding::output_path(&settings, Some(&p.project), "project")
                .map_err(|e| failure("invalid_params", format!("{e:#}")))?;
            let status = workflow::status(&path)
                .map_err(|e| failure("invalid_project", format!("{e:#}")))?;
            let exports = status
                .export_paths(&path)
                .map_err(|e| failure("invalid_project", format!("{e:#}")))?;
            let quality = crate::quality::saved_summary(&path)
                .map_err(|e| failure("invalid_quality_report", format!("{e:#}")))?;
            Ok(json!({"project":path,"status":status,"exports":exports,"quality":quality}))
        }
        "project.plan" => {
            let p: Project = params(&request.params)?;
            let (settings, _) = context(installation, config)
                .map_err(|e| failure("setup_required", format!("{e:#}")))?;
            let project = onboarding::output_path(&settings, Some(&p.project), "project")
                .map_err(|e| failure("invalid_params", format!("{e:#}")))?;
            let plan = execution::snapshot(&project)
                .map_err(|e| failure("invalid_project", format!("{e:#}")))?;
            Ok(json!({"project":project,"plan":plan}))
        }
        "project.events" => {
            let p: Events = params(&request.params)?;
            if !(1..=500).contains(&p.limit) {
                return Err(failure(
                    "invalid_params",
                    "event limit must be between 1 and 500",
                ));
            }
            let (settings, _) = context(installation, config)
                .map_err(|e| failure("setup_required", format!("{e:#}")))?;
            let project = onboarding::output_path(&settings, Some(&p.project), "project")
                .map_err(|e| failure("invalid_params", format!("{e:#}")))?;
            let page = execution::events(&project, p.after, p.limit)
                .map_err(|e| failure("invalid_project", format!("{e:#}")))?;
            let next_after = page.last().map_or(p.after, |event| event.sequence);
            Ok(json!({"project":project,"after":p.after,"next_after":next_after,"events":page}))
        }
        "workflow.control_preview" | "workflow.control" => {
            let p: Control = params(&request.params)?;
            let (settings, _) = context(installation, config)
                .map_err(|e| failure("setup_required", format!("{e:#}")))?;
            let project = onboarding::output_path(&settings, Some(&p.project), "project")
                .map_err(|e| failure("invalid_params", format!("{e:#}")))?;
            if request.method == "workflow.control_preview" {
                let mut preview_request = p.request;
                preview_request.confirmed = true;
                let preview = execution::preview(&project, &preview_request)
                    .map_err(|e| failure("control_rejected", format!("{e:#}")))?;
                Ok(json!({"project":project,"preview":preview}))
            } else {
                let receipt = execution::enqueue(&project, &p.request)
                    .map_err(|e| failure("control_rejected", format!("{e:#}")))?;
                Ok(json!({"project":project,"receipt":receipt}))
            }
        }
        "project.check_quality" | "project.quality_report" => {
            let (settings, config) = context(installation, config)
                .map_err(|e| failure("setup_required", format!("{e:#}")))?;
            let (project, pagination) = if request.method == "project.quality_report" {
                let p: QualityPage = params(&request.params)?;
                if !(1..=50).contains(&p.limit) {
                    return Err(failure(
                        "invalid_params",
                        "quality page limit must be between 1 and 50",
                    ));
                }
                (p.project, Some((p.offset, p.limit)))
            } else {
                (params::<Project>(&request.params)?.project, None)
            };
            let path = onboarding::output_path(&settings, Some(&project), "project")
                .map_err(|e| failure("invalid_params", format!("{e:#}")))?;
            match pagination {
                Some((offset, limit)) => crate::quality::read_page(&path, offset, limit)
                    .map_err(|e| failure("invalid_quality_report", format!("{e:#}"))),
                None => {
                    let report = crate::quality::inspect(&path, &config.quality)
                        .map_err(|e| failure("diagnostics_failed", format!("{e:#}")))?;
                    let summary = crate::quality::summary(&report, &path);
                    if report.execution_status == "failed" {
                        Err(Failure {
                            code: "diagnostics_failed",
                            message: "Quality diagnostics failed; transcription unchanged".into(),
                            data: Some(summary),
                        })
                    } else {
                        Ok(summary)
                    }
                }
            }
        }
        method @ ("workflow.start" | "workflow.resume" | "workflow.rerun" | "workflow.step") => {
            let (settings, configured) = context(installation, config)
                .map_err(|e| failure("setup_required", format!("{e:#}")))?;
            let (project, input, step, sequential, no_quality, asr) = match method {
                "workflow.start" => {
                    let p: Start = params(&request.params)?;
                    (
                        p.project,
                        Some(p.input),
                        None,
                        p.sequential_asr,
                        p.no_quality,
                        p.asr,
                    )
                }
                "workflow.resume" => {
                    let p: Resume = params(&request.params)?;
                    (
                        p.project,
                        p.input,
                        None,
                        p.sequential_asr,
                        p.no_quality,
                        p.asr,
                    )
                }
                _ => {
                    let p: Stage = params(&request.params)?;
                    let parsed = p
                        .step
                        .parse::<workflow::Step>()
                        .map_err(|e| failure("invalid_params", e))?;
                    if let Some(role) = parsed.as_str().strip_prefix("asr:")
                        && !configured.asr.contains_key(role)
                    {
                        return Err(failure(
                            "invalid_params",
                            format!("ASR role {role:?} is not configured"),
                        ));
                    }
                    if method == "workflow.rerun" && p.input.is_some() {
                        return Err(failure(
                            "invalid_params",
                            "rerun does not accept input; resume to rebind first",
                        ));
                    }
                    (
                        p.project,
                        p.input,
                        Some(p.step),
                        p.sequential_asr,
                        p.no_quality,
                        p.asr,
                    )
                }
            };
            let project = onboarding::output_path(&settings, Some(&project), "project")
                .map_err(|e| failure("invalid_params", format!("{e:#}")))?;
            let mut child =
                command(installation, config).map_err(|e| failure("execution_failed", e))?;
            if no_quality {
                child.arg("--no-quality");
            }
            if let Some(asr) = asr {
                let roles = configured
                    .select_asr_roles(Some(&asr))
                    .map_err(|e| failure("invalid_params", format!("{e:#}")))?;
                child.arg("--asr").arg(roles.join(","));
            }
            if method == "workflow.start" {
                child
                    .arg("transcribe")
                    .arg(
                        input
                            .context("missing input")
                            .map_err(|e| failure("invalid_params", e))?,
                    )
                    .arg("--output")
                    .arg(&project);
            } else {
                child
                    .arg("project")
                    .arg(match method {
                        "workflow.resume" => "resume",
                        "workflow.rerun" => "rerun",
                        _ => "run",
                    })
                    .arg(&project);
                if let Some(step) = &step {
                    child.arg("--step").arg(step);
                }
                if let Some(input) = input {
                    child.arg("--input").arg(input);
                }
            }
            if sequential {
                child.arg("--sequential-asr");
            }
            run(&mut child, Some(&project), step.as_deref())
        }
        _ => Err(failure("method_not_found", "unknown API method")),
    }
}

fn response(line: &[u8], installation: Option<&Path>, config: Option<&Path>) -> Value {
    let raw: Value = match serde_json::from_slice(line) {
        Ok(value) => value,
        Err(error) => {
            return json!({"version":1,"id":null,"error":{"code":"invalid_request","message":error.to_string()}});
        }
    };
    let id = raw
        .get("id")
        .filter(|id| id.is_string())
        .cloned()
        .unwrap_or(Value::Null);
    let parsed = serde_json::from_slice::<Request>(line);
    let result = match parsed {
        Err(error) => Err(failure("invalid_request", error)),
        Ok(request) if request.version != 1 => {
            Err(failure("unsupported_version", "API version must be 1"))
        }
        Ok(request) if request.id.is_empty() || request.id.len() > 128 => Err(failure(
            "invalid_request",
            "id must contain 1-128 UTF-8 bytes",
        )),
        Ok(request) => dispatch(&request, installation, config),
    };
    match result {
        Ok(result) => json!({"version":1,"id":id,"result":result}),
        Err(error) => {
            eprintln!("API {}: {}", error.code, error.message);
            json!({"version":1,"id":id,"error":{"code":error.code,"message":error.message,"data":error.data}})
        }
    }
}

pub fn serve(installation: Option<&Path>, config: Option<&Path>) -> Result<i32> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut failed = false;
    loop {
        let mut line = Vec::new();
        let mut oversized = false;
        loop {
            let buffer = input.fill_buf()?;
            if buffer.is_empty() {
                break;
            }
            let size = buffer
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(buffer.len(), |i| i + 1);
            let newline = buffer[size - 1] == b'\n';
            if line.len() + size > MAX_REQUEST {
                oversized = true;
            }
            if !oversized {
                line.extend_from_slice(&buffer[..size]);
            }
            input.consume(size);
            if newline {
                break;
            }
        }
        if line.is_empty() && !oversized {
            break;
        }
        let result = if oversized {
            json!({"version":1,"id":null,"error":{"code":"request_too_large","message":"request exceeds 64 KiB"}})
        } else {
            response(&line, installation, config)
        };
        failed |= result.get("error").is_some();
        let mut bytes = serde_json::to_vec(&result)?;
        if bytes.len() > 1024 * 1024 {
            failed = true;
            bytes = serde_json::to_vec(&json!({"version":1,"id":result.get("id"),
                "error":{"code":"response_too_large","message":"response exceeds 1 MiB; inspect project checkpoint files locally"}}))?;
        }
        output.write_all(&bytes)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(i32::from(failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_requests_are_explicit_and_correlated() {
        for (input, code) in [
            (
                r#"{"version":2,"id":"a","method":"installation.status"}"#,
                "unsupported_version",
            ),
            (
                r#"{"version":1,"id":"a","method":"wrong"}"#,
                "method_not_found",
            ),
            (
                r#"{"version":1,"id":"a","method":"installation.status","params":{"unexpected":1}}"#,
                "invalid_params",
            ),
            (
                r#"{"version":1,"id":"a","method":"installation.status","secret":true}"#,
                "invalid_request",
            ),
        ] {
            let result = response(input.as_bytes(), None, None);
            assert_eq!(result["id"], "a");
            assert_eq!(result["error"]["code"], code);
            assert!(result.get("result").is_none());
        }
        assert_eq!(
            response(b"{", None, None)["error"]["code"],
            "invalid_request"
        );
    }

    #[test]
    fn workflow_params_reject_unknown_flags_and_invalid_step_names() {
        assert!(params::<Stage>(&json!({"project":"p","step":"audio","shell":"rm"})).is_err());
        assert!(
            params::<Start>(&json!({"project":"p","input":"x","sequential_asr":"yes"})).is_err()
        );
        assert!("arbitrary".parse::<workflow::Step>().is_err());
    }

    #[test]
    fn execution_queries_validate_params_before_accessing_project() {
        for method in [
            "project.plan",
            "project.events",
            "workflow.control_preview",
            "workflow.control",
        ] {
            let malformed = json!({"version":1,"id":"query","method":method,"params":{"project":"missing","extra":true}});
            let response = response(
                serde_json::to_string(&malformed).unwrap().as_bytes(),
                None,
                None,
            );
            assert_eq!(response["error"]["code"], "invalid_params", "{method}");
        }
        let invalid = json!({"version":1,"id":"events","method":"project.events","params":{"project":"missing","limit":0}});
        let response = response(
            serde_json::to_string(&invalid).unwrap().as_bytes(),
            None,
            None,
        );
        assert_eq!(response["error"]["code"], "invalid_params");
    }

    #[test]
    fn blocked_and_interrupted_workflows_are_not_reported_complete() {
        assert_eq!(workflow_outcome(3, None), "blocked");
        assert_eq!(workflow_outcome(130, Some("asr:flix")), "interrupted");
        assert_eq!(workflow_outcome(2, None), "pending_review");
        assert_eq!(workflow_outcome(0, Some("audio")), "stage_complete");
        assert_eq!(workflow_outcome(0, None), "complete");
    }
}
