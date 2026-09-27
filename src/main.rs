mod audio;
mod config;
mod models;
mod pipeline;
mod process;

use std::path::PathBuf;
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
    #[arg(long, default_value = "shum.toml", global = true)]
    config: PathBuf,
    #[command(subcommand)]
    command: Action,
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
        output: PathBuf,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    Check,
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
    let mut config = Config::load(&cli.config)?;
    match cli.command {
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
            let index = config
                .models
                .iter()
                .position(|model| model.role == role)
                .context("unknown model role")?;
            config.models[index] = models::update(&config.models[index])?;
            let partial = cli.config.with_extension("partial.toml");
            std::fs::write(&partial, toml::to_string_pretty(&config)?)
                .with_context(|| format!("cannot update {}", cli.config.display()))?;
            std::fs::rename(&partial, &cli.config).with_context(|| {
                format!("cannot activate updated config {}", cli.config.display())
            })?;
            println!(
                "Updated {role} to revision {}",
                config.models[index].revision
            );
        }
        Action::Transcribe { input, output } => {
            report_models(&config)?;
            pipeline::transcribe(&config, &input, &output)?;
            println!("{}", output.join("transcript.txt").display());
        }
    }
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
