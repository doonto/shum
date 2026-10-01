mod audio;
mod config;
mod models;
mod onboarding;
mod pipeline;
mod process;
mod resources;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use config::Config;

#[derive(Parser)]
#[command(
    version,
    about = "Local CPU speech transcription with independent ASR models"
)]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long)]
    onboarding: bool,
    #[arg(long, requires = "onboarding")]
    storage_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Action>,
}

#[derive(Subcommand)]
enum Action {
    Doctor,
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
    if let Err(error) = execute() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn execute() -> Result<()> {
    let cli = Cli::parse();
    if cli.onboarding {
        ensure!(
            cli.command.is_none(),
            "--onboarding cannot be combined with a command"
        );
        ensure!(
            cli.config.is_none(),
            "--onboarding imports the existing config automatically"
        );
        return onboarding::onboard(cli.storage_dir.as_deref());
    }
    let command = cli
        .command
        .context("choose a command, or run `shum --onboarding`")?;
    let settings = onboarding::load()?;
    let config_path = cli.config.unwrap_or(onboarding::config_path()?);
    let mut config = Config::load(&config_path)?;
    onboarding::validate_config_location(&config_path, &config)?;
    match command {
        Action::Doctor => {
            for (binary, arg) in [
                ("ffmpeg", "-version"),
                ("ffprobe", "-version"),
                ("hf", "--help"),
            ] {
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
            let status = Command::new(&config.llm.program)
                .arg("--version")
                .output()
                .with_context(|| format!("LLM CLI missing: {}", config.llm.program.display()))?;
            ensure!(status.status.success(), "LLM CLI --version failed");
            for model in &config.models {
                ensure!(
                    model.path.exists(),
                    "model {} missing: {}",
                    model.role,
                    model.path.display()
                );
            }
            println!(
                "Tools and model paths found. Authentication and inference have not been tested."
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
        Action::Transcribe {
            input,
            output,
            sequential_asr,
        } => {
            let name = project_name(&input, "transcript")?;
            let output = onboarding::output_path(&settings, output.as_deref(), &name)?;
            report_models(&config)?;
            pipeline::transcribe(&config, &input, &output, sequential_asr)?;
            println!("{}", output.join("transcript.txt").display());
        }
        Action::BenchmarkAsr {
            input,
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
            report_models(&config)?;
            pipeline::benchmark_asr(
                &config,
                &input,
                &output,
                &role,
                offload,
                workers,
                worker_peak_mib,
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
            report_models(&config)?;
            pipeline::benchmark_diarization(&config, &input, &output)?;
        }
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
