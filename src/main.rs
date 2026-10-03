mod api;
mod audio;
mod command;
mod config;
mod execution;
mod llm;
mod models;
mod onboarding;
mod pipeline;
mod process;
mod quality;
mod resources;
mod terminal;
mod workflow;

use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use config::{Config, Llm, LlmProvider};

#[derive(Parser)]
#[command(
    version,
    about = "Local CPU speech transcription with configured ASR models"
)]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    installation: Option<PathBuf>,
    /// Disable advisory quality diagnostics for this invocation.
    #[arg(long, global = true)]
    no_quality: bool,
    /// Override configured ASR roles for this invocation (comma-separated).
    #[arg(long, global = true, value_delimiter = ',')]
    asr: Option<Vec<String>>,
    #[arg(long)]
    onboarding: bool,
    #[arg(long, conflicts_with = "onboarding")]
    setup_llm: bool,
    #[arg(long, conflicts_with_all = ["onboarding", "setup_llm"])]
    setup_models: bool,
    #[arg(long, alias = "install-dir", requires = "onboarding")]
    storage_dir: Option<PathBuf>,
    #[arg(long, requires = "storage_dir")]
    bin_dir: Option<PathBuf>,
    #[arg(long, requires = "storage_dir")]
    settings_dir: Option<PathBuf>,
    #[arg(long, requires = "storage_dir")]
    projects_dir: Option<PathBuf>,
    #[arg(long, requires = "storage_dir")]
    models_dir: Option<PathBuf>,
    /// Register shum in this command directory (explicit opt-in without a TTY).
    #[arg(long, requires = "onboarding")]
    command_dir: Option<PathBuf>,
    /// Add the command directory to Bash/Zsh startup files.
    #[arg(long, requires = "command_dir")]
    setup_shell: bool,
    #[command(subcommand)]
    command: Option<Action>,
}

#[derive(Subcommand)]
enum Action {
    /// Versioned JSON-lines API for agents; never prompts.
    Api,
    /// Repair terminal command registration only; no model or provider operations.
    SetupCommand {
        #[arg(long)]
        command_dir: Option<PathBuf>,
        #[arg(long)]
        setup_shell: bool,
    },
    Doctor,
    Project {
        #[command(subcommand)]
        action: ProjectAction,
    },
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },
    Transcribe {
        input: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        sequential_asr: bool,
    },
    BenchmarkAsr {
        input: PathBuf,
        #[arg(
            long,
            help = "Use verified prepared WAV clips from a v1 JSON corpus manifest (benchmark only); output must be new"
        )]
        corpus_manifest: Option<PathBuf>,
        #[arg(long)]
        role: String,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        offload: bool,
        #[arg(long, requires = "script")]
        program: Option<PathBuf>,
        #[arg(long, requires = "program")]
        script: Option<PathBuf>,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long)]
        in_flight: Option<usize>,
        #[arg(long, default_value_t = 1)]
        workers: usize,
        #[arg(long = "adapter-arg")]
        adapter_args: Vec<String>,
        #[arg(long)]
        worker_peak_mib: Option<u64>,
    },
    BenchmarkDiarization {
        input: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, requires = "script")]
        program: Option<PathBuf>,
        #[arg(long, requires = "program")]
        script: Option<PathBuf>,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long = "adapter-arg")]
        adapter_args: Vec<String>,
    },
}

#[derive(Subcommand)]
enum ProjectAction {
    /// Show the canonical execution plan and active attempts.
    Plan {
        project: PathBuf,
    },
    /// Read the durable execution event journal.
    Events {
        project: PathBuf,
        #[arg(long, default_value_t = 0)]
        after: u64,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
    /// Preview a versioned control request from a JSON file without applying it.
    ControlPreview {
        project: PathBuf,
        #[arg(long)]
        request: PathBuf,
    },
    /// Queue an explicitly confirmed, versioned control request from a JSON file.
    Control {
        project: PathBuf,
        #[arg(long)]
        request: PathBuf,
    },
    /// Queue a restore request from a JSON file.
    Restore {
        project: PathBuf,
        #[arg(long)]
        request: PathBuf,
    },
    /// Diagnose existing project artifacts offline, without ASR or LLM calls.
    CheckQuality {
        project: PathBuf,
    },
    /// Run one stage and missing prerequisites, without running descendants.
    Run {
        project: PathBuf,
        #[arg(long)]
        step: workflow::Step,
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long)]
        sequential_asr: bool,
    },
    Status {
        project: PathBuf,
    },
    Resume {
        project: PathBuf,
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long)]
        sequential_asr: bool,
    },
    Rerun {
        project: PathBuf,
        #[arg(long)]
        step: workflow::Step,
        #[arg(long)]
        sequential_asr: bool,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    Check,
    InstallPinned {
        #[arg(long)]
        role: String,
    },
    Update {
        #[arg(long)]
        role: String,
    },
}

fn main() {
    match execute() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(1);
        }
    }
}

fn execute() -> Result<i32> {
    let cli = Cli::parse();
    if matches!(cli.command, Some(Action::Api)) {
        ensure!(
            !cli.onboarding && !cli.setup_llm && !cli.setup_models,
            "api cannot be combined with interactive setup flags"
        );
        return api::serve(cli.installation.as_deref(), cli.config.as_deref());
    }
    if let Some(Action::SetupCommand {
        command_dir,
        setup_shell,
    }) = &cli.command
    {
        ensure!(
            !cli.onboarding && !cli.setup_llm && !cli.setup_models,
            "setup-command cannot be combined with other setup flags"
        );
        let binary =
            onboarding::installed_binary_from(cli.installation.as_deref(), cli.config.as_deref())?;
        let registration = command::setup(&binary, command_dir.as_deref(), *setup_shell)?;
        command::print_registration(&registration)?;
        return Ok(0);
    }
    let options = onboarding::OnboardOptions {
        install_dir: cli.storage_dir.clone(),
        bin_dir: cli.bin_dir.clone(),
        settings_dir: cli.settings_dir.clone(),
        projects_dir: cli.projects_dir.clone(),
        models_dir: cli.models_dir.clone(),
        installation_locator: cli.installation.clone(),
        command_dir: cli.command_dir.clone(),
        setup_shell: cli.setup_shell,
    };
    if cli.onboarding {
        ensure!(
            cli.command.is_none(),
            "--onboarding cannot be combined with a command"
        );
        ensure!(
            cli.config.is_none(),
            "--onboarding imports the existing config automatically"
        );
        let settings = onboarding::onboard_with(&options)?;
        if io::stdin().is_terminal() {
            configure_llm_on_disk(&settings)?;
        }
        return Ok(0);
    }
    let settings = match onboarding::load_from(cli.installation.as_deref(), cli.config.as_deref()) {
        Ok(settings) => settings,
        Err(error) if io::stdin().is_terminal() => {
            eprintln!("Installation needs setup: {error:#}");
            let settings = onboarding::onboard_with(&options)?;
            configure_llm_on_disk(&settings)?;
            settings
        }
        Err(error) => {
            return Err(error).context(
                "No interactive terminal is available. Run `shum --onboarding --storage-dir PATH` to configure the installation.",
            );
        }
    };
    let config_path = cli
        .config
        .unwrap_or(onboarding::config_path_for(&settings)?);
    let mut config = match Config::load(&config_path) {
        Ok(config) => config,
        Err(error) if io::stdin().is_terminal() => {
            eprintln!("Configuration needs repair: {error:#}");
            let mut config = onboarding::repair_config(&settings, &config_path)?;
            onboarding::configure_models_at(&mut config, &config_path)?;
            configure_llm(&mut config.llm)?;
            onboarding::save_config(&config_path, &config)?;
            config
        }
        Err(error) => return Err(error).context(
            "Configuration repair requires a terminal; existing settings and project data were not replaced.",
        ),
    };
    onboarding::validate_config_location(&config_path, &config)?;
    if cli.setup_llm {
        ensure!(
            cli.command.is_none(),
            "--setup-llm cannot be combined with a command"
        );
        configure_llm(&mut config.llm)?;
        onboarding::save_config(&config_path, &config)?;
        return Ok(0);
    }
    if cli.setup_models {
        ensure!(
            cli.command.is_none(),
            "--setup-models cannot be combined with a command"
        );
        ensure!(io::stdin().is_terminal(), "model setup requires a terminal");
        onboarding::configure_models_at(&mut config, &config_path)?;
        onboarding::save_config(&config_path, &config)?;
        return Ok(0);
    }
    if cli.no_quality {
        config.quality.enabled = false;
    }
    config.asr_selection = cli
        .asr
        .as_deref()
        .map(|roles| config.select_asr_roles(Some(roles)))
        .transpose()?;
    if requires_early_llm_probe(cli.command.as_ref()) {
        check_llm_at_startup(&config.llm)?;
    }
    let Some(command) = cli.command else {
        println!(
            "Shum is configured. Projects: {}\nRun `shum --help` for commands.",
            settings.projects_dir().display()
        );
        return Ok(0);
    };
    match command {
        Action::Doctor => {
            for (binary, arg) in [("ffmpeg", "-version"), ("ffprobe", "-version")] {
                let status = Command::new(binary)
                    .arg(arg)
                    .output()
                    .with_context(|| format!("{binary} not found"))?;
                ensure!(status.status.success(), "{binary} {arg} failed");
            }
            for adapter in config
                .asr
                .values()
                .chain(std::iter::once(&config.diarization))
            {
                ensure!(
                    adapter.script.is_file(),
                    "adapter missing: {}",
                    adapter.script.display()
                );
                let status = Command::new(&adapter.program)
                    .arg("--version")
                    .status()
                    .with_context(|| {
                        format!("adapter interpreter missing: {}", adapter.program.display())
                    })?;
                ensure!(
                    status.success(),
                    "adapter interpreter failed: {}",
                    adapter.program.display()
                );
            }
            for model in &config.models {
                models::verify_local(model)
                    .with_context(|| format!("model {} failed local verification", model.role))?;
            }
            println!(
                "Configured local tools and model integrity verified. LLM access and audio inference have not been tested."
            );
        }
        Action::Models {
            action: ModelAction::Check,
        } => {
            report_models(&config)?;
        }
        Action::Models {
            action: ModelAction::Update { role },
        } => {
            install_model(&mut config, &config_path, &role, false)?;
        }
        Action::Models {
            action: ModelAction::InstallPinned { role },
        } => {
            install_model(&mut config, &config_path, &role, true)?;
        }
        Action::Project { action } => {
            let project = match &action {
                ProjectAction::Status { project }
                | ProjectAction::Plan { project }
                | ProjectAction::Events { project, .. }
                | ProjectAction::ControlPreview { project, .. }
                | ProjectAction::Control { project, .. }
                | ProjectAction::Restore { project, .. }
                | ProjectAction::CheckQuality { project }
                | ProjectAction::Run { project, .. }
                | ProjectAction::Resume { project, .. }
                | ProjectAction::Rerun { project, .. } => project,
            };
            let project = onboarding::output_path(&settings, Some(project), "project")?;
            match action {
                ProjectAction::Plan { .. } => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&execution::snapshot(&project)?)?
                    );
                }
                ProjectAction::Events { after, limit, .. } => {
                    ensure!(
                        (1..=500).contains(&limit),
                        "event limit must be between 1 and 500"
                    );
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&execution::events(&project, after, limit)?)?
                    );
                }
                ProjectAction::ControlPreview { request, .. } => println!(
                    "{}",
                    serde_json::to_string_pretty(&preview_control_request(&project, &request)?)?
                ),
                ProjectAction::Control { request, .. } => println!(
                    "{}",
                    serde_json::to_string_pretty(&execution::enqueue(
                        &project,
                        &read_control_request(&request)?
                    )?)?
                ),
                ProjectAction::Restore { request, .. } => {
                    let control = read_control_request(&request)?;
                    ensure!(
                        matches!(control.action, execution::ControlAction::Restore { .. }),
                        "project restore requires a restore action"
                    );
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&execution::enqueue(&project, &control)?)?
                    );
                }
                ProjectAction::CheckQuality { .. } => {
                    let report = quality::inspect(&project, &config.quality)?;
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&quality::summary(&report, &project))?
                    );
                    return Ok(if report.execution_status == "failed" {
                        1
                    } else {
                        0
                    });
                }
                ProjectAction::Run {
                    step,
                    input,
                    sequential_asr,
                    ..
                } => {
                    return Ok(print_project_outcome(
                        &project,
                        workflow::run_stage(
                            &config,
                            &project,
                            step,
                            input.as_deref(),
                            sequential_asr,
                        )?,
                    ));
                }
                ProjectAction::Status { .. } => {
                    let mut status = serde_json::to_value(workflow::status(&project)?)?;
                    status["quality"] = serde_json::to_value(quality::saved_summary(&project)?)?;
                    println!("{}", serde_json::to_string_pretty(&status)?);
                }
                ProjectAction::Resume {
                    input,
                    sequential_asr,
                    ..
                } => {
                    let outcome = match input {
                        Some(input) => {
                            workflow::resume_with_input(&config, &project, &input, sequential_asr)?
                        }
                        None => workflow::resume(&config, &project, sequential_asr)?,
                    };
                    return Ok(print_project_outcome(&project, outcome));
                }
                ProjectAction::Rerun {
                    step,
                    sequential_asr,
                    ..
                } => {
                    return Ok(print_project_outcome(
                        &project,
                        workflow::rerun(&config, &project, &step, sequential_asr)?,
                    ));
                }
            }
        }
        Action::Transcribe {
            input,
            output,
            sequential_asr,
        } => {
            let name = if output.is_some() {
                "project".to_owned()
            } else {
                project_name(&input, "transcript")?
            };
            let output = onboarding::output_path(&settings, output.as_deref(), &name)?;
            return Ok(print_project_outcome(
                &output,
                pipeline::transcribe_outcome(&config, &input, &output, sequential_asr)?,
            ));
        }
        Action::BenchmarkAsr {
            input,
            corpus_manifest,
            role,
            output,
            offload,
            program,
            script,
            threads,
            in_flight,
            workers,
            adapter_args,
            worker_peak_mib,
        } => {
            let name = project_name(&input, &format!("bench-{role}"))?;
            let output = onboarding::output_path(&settings, output.as_deref(), &name)?;
            if let (Some(program), Some(script)) = (program, script) {
                let adapter = config.asr.get_mut(&role).context("unknown ASR role")?;
                adapter.program = program;
                adapter.script = script;
            }
            if let Some(threads) = threads {
                ensure!(threads > 0, "--threads must be positive");
                config
                    .asr
                    .get_mut(&role)
                    .context("unknown ASR role")?
                    .threads = threads;
            }
            if let Some(in_flight) = in_flight {
                ensure!((1..=4).contains(&in_flight), "--in-flight must be 1-4");
                config
                    .asr
                    .get_mut(&role)
                    .context("unknown ASR role")?
                    .max_in_flight = in_flight;
            }
            config
                .asr
                .get_mut(&role)
                .context("unknown ASR role")?
                .args
                .extend(adapter_args);
            pipeline::benchmark_asr(
                &config,
                &input,
                &output,
                &role,
                pipeline::BenchmarkOptions {
                    offload,
                    workers,
                    worker_peak_mib,
                    corpus_manifest: corpus_manifest.as_deref(),
                },
            )?;
        }
        Action::BenchmarkDiarization {
            input,
            output,
            program,
            script,
            threads,
            adapter_args,
        } => {
            let name = project_name(&input, "bench-diarization")?;
            let output = onboarding::output_path(&settings, output.as_deref(), &name)?;
            if let (Some(program), Some(script)) = (program, script) {
                config.diarization.program = program;
                config.diarization.script = script;
            }
            if let Some(threads) = threads {
                ensure!(threads > 0, "--threads must be positive");
                config.diarization.threads = threads;
            }
            config.diarization.args.extend(adapter_args);
            pipeline::benchmark_diarization(&config, &input, &output)?;
        }
        Action::Api | Action::SetupCommand { .. } => {
            unreachable!("non-interactive utility is dispatched before configuration")
        }
    }
    Ok(0)
}

fn print_project_outcome(project: &Path, outcome: workflow::Outcome) -> i32 {
    match outcome {
        workflow::Outcome::StageComplete => {
            println!(
                "Selected stage completed; project checkpoints: {}",
                project.display()
            );
            0
        }
        workflow::Outcome::Complete => match workflow::export_paths(project) {
            Ok(exports) => {
                println!("{}", exports.txt.display());
                0
            }
            Err(error) => {
                eprintln!("error: could not read final export paths: {error:#}");
                1
            }
        },
        workflow::Outcome::PendingReview => {
            let quoted_project = project.to_string_lossy().replace('\'', "'\\''");
            eprintln!(
                "Project is waiting for LLM review. Completed local results are saved at {}.\nResume with: shum project resume '{}'\nStart a fresh review with: shum project rerun '{}' --step review",
                project.display(),
                quoted_project,
                quoted_project,
            );
            2
        }
        workflow::Outcome::Blocked => {
            eprintln!(
                "Project is blocked; inspect the execution plan and events at {}.",
                project.display()
            );
            3
        }
        workflow::Outcome::Interrupted => {
            eprintln!(
                "Project interrupted; checkpoints are saved at {}.",
                project.display()
            );
            130
        }
    }
}

fn read_control_request(path: &Path) -> Result<execution::ControlRequest> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("could not read {}", path.display()))?,
    )
    .with_context(|| format!("invalid control request in {}", path.display()))
}

fn preview_control_request(project: &Path, path: &Path) -> Result<execution::Preview> {
    let mut request = read_control_request(path)?;
    request.confirmed = true;
    execution::preview(project, &request)
}

fn answer(question: &str) -> Result<String> {
    ensure!(
        io::stdin().is_terminal(),
        "setup requires a terminal; edit the private config for non-interactive use"
    );
    print!("{question} ");
    io::stdout().flush()?;
    let mut value = String::new();
    ensure!(
        io::stdin().read_line(&mut value)? > 0,
        "setup input ended before completion; existing settings were not changed"
    );
    Ok(value.trim().to_owned())
}

fn configure_llm_on_disk(settings: &onboarding::Settings) -> Result<()> {
    let path = onboarding::config_path_for(settings)?;
    let mut config = Config::load(&path)?;
    configure_llm(&mut config.llm)?;
    onboarding::save_config(&path, &config)
}

fn configure_llm(existing: &mut Llm) -> Result<()> {
    println!(
        "LLM review sends transcript text to the selected provider, never audio.\nThe access check sends only a fixed technical request and may consume quota."
    );
    loop {
        let choice = answer("LLM provider: [codex/copilot/custom/keep/skip] (default: keep):")?;
        let mut candidate = match choice.as_str() {
            "" | "keep" => existing.clone(),
            "skip" => {
                existing.enabled = false;
                println!("LLM skipped. Projects can run local stages and wait for review.");
                return Ok(());
            }
            "codex" => Llm::preset(LlmProvider::Codex),
            "copilot" => Llm::preset(LlmProvider::Copilot),
            "custom" => {
                let program = answer("Custom CLI executable (name in PATH or absolute path):")?;
                ensure!(!program.is_empty(), "custom CLI executable cannot be empty");
                let program_path = PathBuf::from(&program);
                ensure!(
                    program_path.is_absolute() || program_path.components().count() == 1,
                    "custom CLI must be a name in PATH or an absolute path"
                );
                let args = answer(
                    "Review arguments as a JSON string array; prompt arrives on stdin, {output} and {model} are placeholders:",
                )?;
                let probe_args = answer(
                    "Probe arguments as a JSON string array; technical prompt arrives on stdin, return exactly SHUM_READY on stdout:",
                )?;
                Llm {
                    program: program.into(),
                    args: serde_json::from_str(&args).context("invalid review arguments JSON")?,
                    enabled: true,
                    provider: LlmProvider::Custom,
                    model: None,
                    probe_args: Some(
                        serde_json::from_str(&probe_args)
                            .context("invalid probe arguments JSON")?,
                    ),
                }
            }
            _ => {
                println!("Choose codex, copilot, custom, keep or skip.");
                continue;
            }
        };
        if !candidate.enabled {
            *existing = candidate;
            return Ok(());
        }
        if !matches!(choice.as_str(), "" | "keep") {
            let model = answer("LLM model (empty: use the CLI default):")?;
            candidate.model = (!model.is_empty()).then_some(model);
        }
        let mut readiness = llm::readiness(&candidate);
        if readiness == llm::Readiness::NeedsLogin
            && matches!(
                candidate.provider,
                LlmProvider::Codex | LlmProvider::Copilot
            )
            && answer("Open the selected CLI's login now? [y/N]:")?.eq_ignore_ascii_case("y")
        {
            match llm::login(&candidate, true) {
                Ok(()) => readiness = llm::readiness(&candidate),
                Err(error) => eprintln!("Login did not complete: {error:#}"),
            }
        }
        println!("{}", readiness.help(candidate.provider));
        if readiness == llm::Readiness::Ready {
            *existing = candidate;
            return Ok(());
        }
        match answer("Access is not ready: [save/retry/skip] (default: save):")?.as_str() {
            "" | "save" => {
                *existing = candidate;
                println!(
                    "Configuration saved with unavailable review. Local stages remain usable."
                );
                return Ok(());
            }
            "skip" => {
                candidate.enabled = false;
                *existing = candidate;
                return Ok(());
            }
            "retry" => {}
            _ => println!("Choose save, retry or skip."),
        }
    }
}

fn requires_early_llm_probe(command: Option<&Action>) -> bool {
    !matches!(
        command,
        Some(Action::Doctor | Action::Project { .. } | Action::Transcribe { .. })
    )
}

fn check_llm_at_startup(config: &Llm) -> Result<()> {
    let mut readiness = llm::cached_readiness(config).unwrap_or_else(|| llm::readiness(config));
    if readiness == llm::Readiness::NeedsLogin
        && io::stdin().is_terminal()
        && matches!(config.provider, LlmProvider::Codex | LlmProvider::Copilot)
        && answer("LLM login needs repair. Open login for this CLI only? [y/N]:")?
            .eq_ignore_ascii_case("y")
    {
        match llm::login(config, true) {
            Ok(()) => readiness = llm::readiness(config),
            Err(error) => eprintln!("Login did not complete: {error:#}"),
        }
    }
    if readiness != llm::Readiness::Ready {
        eprintln!(
            "LLM warning: {} Local stages remain usable. Run `shum --setup-llm` to change review settings.",
            readiness.help(config.provider)
        );
    }
    Ok(())
}

fn project_name(input: &Path, kind: &str) -> Result<String> {
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty() && *value != "." && *value != "..")
        .context("input filename needs a nonempty UTF-8 stem")?;
    Ok(format!("{stem}-{kind}"))
}

fn install_model(config: &mut Config, path: &Path, role: &str, pinned: bool) -> Result<()> {
    let index = config
        .models
        .iter()
        .position(|model| model.role == role)
        .context("unknown model role")?;
    config.models[index] = if pinned {
        models::install_pinned(&config.models[index])?
    } else {
        models::update(&config.models[index])?
    };
    onboarding::save_config(path, config)
        .with_context(|| format!("cannot activate updated config {}", path.display()))?;
    println!(
        "Installed {role} at revision {}",
        config.models[index].revision
    );
    Ok(())
}

fn report_models(config: &Config) -> Result<()> {
    for status in models::check(&config.models)? {
        ensure!(
            status.integrity_ok,
            "model {} failed local checksum/revision verification at {}; run `shum models update --role {}`",
            status.role,
            status.path.display(),
            status.role
        );
        match status.availability {
            models::Availability::UpToDate => {
                eprintln!("{}: pinned revision is current", status.role)
            }
            models::Availability::UpdateAvailable { remote_revision } => {
                eprintln!(
                    "{}: new revision {remote_revision} available; run `shum models update --role {}`",
                    status.role, status.role
                );
            }
            models::Availability::Unavailable => {
                eprintln!(
                    "{}: warning: Hugging Face unavailable; using verified pinned revision",
                    status.role
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn prepared_corpus_option_is_only_on_asr_benchmark() {
        let cli = Cli::try_parse_from([
            "shum",
            "benchmark-asr",
            "input.mp3",
            "--role",
            "flix",
            "--corpus-manifest",
            "corpus.json",
            "--output",
            "fresh-run",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Action::BenchmarkAsr { corpus_manifest: Some(ref path), .. })
                if path == Path::new("corpus.json")
        ));
        assert!(
            Cli::try_parse_from([
                "shum",
                "transcribe",
                "input.mp3",
                "--corpus-manifest",
                "corpus.json",
            ])
            .is_err()
        );
    }

    #[test]
    fn project_commands_parse_without_an_input_recording() {
        let cli = Cli::try_parse_from(["shum", "project", "status", "example"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Action::Project {
                action: ProjectAction::Status { .. }
            })
        ));
        let cli = Cli::try_parse_from([
            "shum",
            "project",
            "rerun",
            "example",
            "--step",
            "asr:flix",
            "--sequential-asr",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Action::Project {
                action: ProjectAction::Rerun {
                    step,
                    sequential_asr: true,
                    ..
                }
            }) if step.as_str() == "asr:flix"
        ));
    }

    #[test]
    fn execution_project_commands_require_explicit_requests() {
        let cli =
            Cli::try_parse_from(["shum", "project", "events", "example", "--after", "12"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Action::Project {
                action: ProjectAction::Events { after: 12, .. }
            })
        ));
        for command in ["control", "control-preview", "restore"] {
            assert!(Cli::try_parse_from(["shum", "project", command, "example"]).is_err());
            assert!(
                Cli::try_parse_from([
                    "shum",
                    "project",
                    command,
                    "example",
                    "--request",
                    "control.json"
                ])
                .is_ok()
            );
        }
        assert!(Cli::try_parse_from(["shum", "project", "plan", "example"]).is_ok());
    }

    #[test]
    fn workflows_plan_before_remote_readiness_probes() {
        for args in [
            vec!["shum", "transcribe", "input.wav"],
            vec!["shum", "project", "resume", "example"],
            vec!["shum", "project", "rerun", "example", "--step", "review"],
            vec!["shum", "project", "run", "example", "--step", "review"],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert!(!requires_early_llm_probe(cli.command.as_ref()));
        }
        let doctor = Cli::try_parse_from(["shum", "doctor"]).unwrap();
        assert!(!requires_early_llm_probe(doctor.command.as_ref()));
    }

    #[test]
    fn rerun_requires_an_explicit_stage() {
        assert!(Cli::try_parse_from(["shum", "project", "rerun", "example"]).is_err());
        assert!(
            Cli::try_parse_from([
                "shum",
                "project",
                "rerun",
                "example",
                "--step",
                "invalid-stage",
            ])
            .is_err()
        );
    }

    #[test]
    fn no_command_is_valid_for_first_launch() {
        assert!(Cli::try_parse_from(["shum"]).unwrap().command.is_none());
    }

    #[test]
    fn independent_paths_require_an_explicit_installation_root() {
        assert!(
            Cli::try_parse_from(["shum", "--onboarding", "--models-dir", "/tmp/models"]).is_err()
        );
        let cli = Cli::try_parse_from([
            "shum",
            "--onboarding",
            "--install-dir",
            "/tmp/install",
            "--models-dir",
            "/tmp/models",
            "--settings-dir",
            "/tmp/settings",
        ])
        .unwrap();
        assert_eq!(cli.storage_dir, Some("/tmp/install".into()));
        assert_eq!(cli.models_dir, Some("/tmp/models".into()));
    }

    #[test]
    fn partial_workflow_has_a_distinct_exit_code() {
        assert_eq!(
            print_project_outcome(Path::new("example"), workflow::Outcome::PendingReview),
            2
        );
    }
}
