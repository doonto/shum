use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use wait_timeout::ChildExt;

use crate::{
    audio::{self, Interval},
    config::{Adapter, Config},
    process,
};

const ROLES: [(&str, &str); 3] = [("primeline", "nemo"), ("flix", "whisper"), ("qwen", "qwen")];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordSpan {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub text: String,
    pub language: Option<String>,
    #[serde(default)]
    pub segments: Vec<WordSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diarization {
    pub turns: Vec<Turn>,
    pub exclusive_turns: Vec<Turn>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FinalSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: Option<String>,
    pub text: String,
    pub language: String,
    pub gap: bool,
    pub hypotheses: Vec<(String, Hypothesis)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Transcript {
    pub schema_version: u32,
    pub source_sha256: String,
    pub duration: f64,
    pub quality_status: String,
    pub models: Vec<crate::models::ModelSpec>,
    pub segments: Vec<FinalSegment>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct JobManifest {
    schema_version: u32,
    input_sha256: String,
    models: Vec<crate::models::ModelSpec>,
    asr: BTreeMap<String, Adapter>,
    asr_script_sha256: BTreeMap<String, String>,
    diarization: Adapter,
    diarization_script_sha256: String,
    llm: crate::config::Llm,
    review_rules_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Decision {
    index: usize,
    text: String,
    language: String,
    gap: bool,
}

#[derive(Debug, Deserialize)]
struct Decisions {
    segments: Vec<Decision>,
}

pub fn transcribe(config: &Config, input: &Path, output: &Path) -> Result<()> {
    ensure!(
        input
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("mp3")),
        "input must be MP3"
    );
    let input = input.canonicalize().context("input MP3 is unavailable")?;
    let hash = hash_file(&input)?;
    fs::create_dir_all(output).with_context(|| format!("cannot create {}", output.display()))?;
    let output = output
        .canonicalize()
        .context("job directory is unavailable")?;
    let meta = output.join("manifest.json");
    let asr_script_sha256 = config
        .asr
        .iter()
        .map(|(role, adapter)| Ok((role.clone(), hash_file(&adapter.script)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let manifest = JobManifest {
        schema_version: 1,
        input_sha256: hash.clone(),
        models: config.models.clone(),
        asr: config.asr.clone(),
        asr_script_sha256,
        diarization: config.diarization.clone(),
        diarization_script_sha256: hash_file(&config.diarization.script)?,
        llm: config.llm.clone(),
        review_rules_sha256: config.review_rules.as_deref().map(hash_file).transpose()?,
    };
    if meta.exists() {
        let existing: JobManifest = read_json(&meta)?;
        ensure!(
            existing == manifest,
            "job input, adapter, LLM or pinned model revision changed; use another output directory"
        );
    } else {
        write_json(&meta, &manifest)?;
    }
    let duration = audio::duration(&input)?;
    let wav = output.join("audio.wav");
    if !wav.exists() {
        let partial = output.join("audio.partial.wav");
        audio::normalize(&input, &partial)?;
        fs::rename(partial, &wav)?;
    }
    let diarization_path = output.join("diarization.json");
    if !diarization_path.exists() {
        let partial = output.join("diarization.partial.json");
        let model = config.model("diarization");
        let mut command = Command::new(&config.diarization.program);
        command
            .arg(&config.diarization.script)
            .arg("--model")
            .arg(&model.path)
            .arg("--input")
            .arg(&wav)
            .arg("--output")
            .arg(&partial)
            .arg("--threads")
            .arg(config.diarization.threads.to_string());
        process::run(
            &mut command,
            &output.join("diarization.stdout"),
            &output.join("diarization.stderr"),
            Duration::from_secs(4 * 3600),
        )?;
        let _: Diarization = read_json(&partial)?;
        fs::rename(partial, &diarization_path)?;
    }
    let diarization: Diarization = read_json(&diarization_path)?;
    validate_diarization(&diarization, duration)?;
    let chunks_path = output.join("chunks.json");
    let chunks: Vec<Interval> = if chunks_path.exists() {
        read_json(&chunks_path)?
    } else {
        let speech = audio::speech_intervals(&wav, duration)?;
        ensure!(!speech.is_empty(), "no speech detected");
        let turns: Vec<Interval> = diarization
            .exclusive_turns
            .iter()
            .map(|t| Interval {
                start: t.start,
                end: t.end,
            })
            .collect();
        let chunks = audio::chunks(&audio::split_on_turns(&speech, &turns), 28.0, 0.8)?;
        write_json(&chunks_path, &chunks)?;
        chunks
    };
    let clips = output.join("clips");
    fs::create_dir_all(&clips)?;
    for (i, interval) in chunks.iter().enumerate() {
        let clip = clips.join(format!("{i:05}.wav"));
        if !clip.exists() {
            audio::extract(&wav, interval, &clips, i)?;
        }
    }
    thread::scope(|scope| -> Result<()> {
        let intervals = &chunks;
        let clips_ref = clips.as_path();
        let output_ref = output.as_path();
        let handles: Vec<_> = ROLES
            .iter()
            .map(|(role, engine)| {
                scope.spawn(move || run_asr(config, role, engine, clips_ref, output_ref, intervals))
            })
            .collect();
        let mut errors = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => errors.push(format!("{err:#}")),
                Err(_) => errors.push("ASR worker panicked".to_string()),
            }
        }
        ensure!(errors.is_empty(), "ASR failed: {}", errors.join("; "));
        Ok(())
    })?;
    let mut hypotheses = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let mut group = Vec::new();
        for (role, _) in ROLES {
            let path = output.join(role).join(format!("{i:05}.json"));
            let hypothesis: Hypothesis = read_json(&path)?;
            validate_hypothesis(&hypothesis, chunk.end - chunk.start)?;
            group.push((role.to_owned(), hypothesis));
        }
        hypotheses.push(group);
    }
    let mut segments = Vec::new();
    let review_rules = config
        .review_rules
        .as_ref()
        .map(|path| {
            fs::read_to_string(path)
                .with_context(|| format!("cannot read review rules {}", path.display()))
        })
        .transpose()?;
    ensure!(
        review_rules.as_ref().is_none_or(|text| text.len() <= 65536),
        "review rules exceed 64 KiB"
    );
    let review_dir = output.join("reviews");
    fs::create_dir_all(&review_dir)?;
    for start in (0..chunks.len()).step_by(8) {
        let end = (start + 8).min(chunks.len());
        let decisions_path = review_dir.join(format!("{start:05}.json"));
        let decisions: Decisions = if decisions_path.exists() {
            read_json(&decisions_path)?
        } else {
            let prompt = serde_json::json!({
                "instruction": "Do not use tools or read files. Compare three ASR hypotheses for each audio interval; use only their text, never invent words. For German including Swiss dialect, return standard German without changing meaning. Keep Russian in Russian. If speech cannot be recovered, set gap=true and put [пропуск] in text. Return strictly JSON: {\"segments\":[{\"index\":integer,\"text\":string,\"language\":\"de|ru|mixed|unknown\",\"gap\":boolean}]}. One entry per index. Never claim to have heard the audio.",
                "review_rules": review_rules,
                "intervals": (start..end).map(|i| serde_json::json!({"index": i, "start": chunks[i].start, "end": chunks[i].end, "hypotheses": hypotheses[i]})).collect::<Vec<_>>()
            });
            llm(
                config,
                &prompt.to_string(),
                &decisions_path,
                &review_dir.join(format!("{start:05}.stderr")),
            )?;
            read_json(&decisions_path)?
        };
        ensure!(
            decisions.segments.len() == end - start,
            "LLM returned wrong number of segments in batch {start}"
        );
        for i in start..end {
            let decision = decisions
                .segments
                .iter()
                .find(|d| d.index == i)
                .context("LLM omitted segment")?;
            ensure!(
                decisions.segments.iter().filter(|d| d.index == i).count() == 1,
                "LLM duplicated segment {i}"
            );
            ensure!(
                ["de", "ru", "mixed", "unknown"].contains(&decision.language.as_str()),
                "LLM returned invalid language"
            );
            ensure!(
                !decision.text.trim().is_empty(),
                "LLM returned empty text for segment {i}"
            );
            let missing = hypotheses[i].iter().all(|(_, h)| h.text.trim().is_empty());
            ensure!(
                !missing || decision.gap,
                "all ASR outputs empty for speech interval {i}, but LLM did not mark a gap"
            );
            ensure!(
                !decision.gap || decision.text.contains("[пропуск]"),
                "gap must be explicit in text"
            );
            let midpoint = (chunks[i].start + chunks[i].end) / 2.0;
            let speaker = diarization
                .exclusive_turns
                .iter()
                .find(|t| t.start <= midpoint && midpoint < t.end)
                .map(|t| t.speaker.clone());
            let mut text = decision.text.clone();
            if let Some(previous) = segments.last() {
                let previous: &FinalSegment = previous;
                if previous.end > chunks[i].start && previous.speaker == speaker {
                    text = trim_repeated_prefix(&previous.text, &text);
                }
            }
            if text.is_empty() && !decision.gap {
                continue;
            }
            segments.push(FinalSegment {
                start: chunks[i].start,
                end: chunks[i].end,
                speaker,
                text,
                language: decision.language.clone(),
                gap: decision.gap,
                hypotheses: hypotheses[i].clone(),
            });
        }
    }
    let status = if segments.iter().any(|s| s.gap) {
        "needs_audio_review"
    } else {
        "text_reviewed_audio_not_verified"
    };
    let transcript = Transcript {
        schema_version: 1,
        source_sha256: hash,
        duration,
        quality_status: status.to_owned(),
        models: config.models.clone(),
        segments,
    };
    write_json(&output.join("transcript.json"), &transcript)?;
    let mut text = String::new();
    for s in &transcript.segments {
        text.push_str(&format!(
            "[{:02}:{:02}:{:02}] {}: {}\n",
            (s.start as u64) / 3600,
            ((s.start as u64) / 60) % 60,
            (s.start as u64) % 60,
            s.speaker.as_deref().unwrap_or("неизвестный"),
            s.text
        ));
    }
    fs::write(output.join("transcript.txt"), text)?;
    Ok(())
}

fn run_asr(
    config: &Config,
    role: &str,
    engine: &str,
    clips: &Path,
    output: &Path,
    chunks: &[Interval],
) -> Result<()> {
    let model = config.model(role);
    let adapter = config.asr(role);
    let folder = output.join(role);
    fs::create_dir_all(&folder)?;
    let mut pending = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let path = folder.join(format!("{i:05}.json"));
        if path.exists() {
            let value: Hypothesis = read_json(&path)?;
            validate_hypothesis(&value, chunk.end - chunk.start)?;
            continue;
        }
        pending.push((i, chunk));
    }
    if pending.is_empty() {
        return Ok(());
    }
    let stderr = folder.join("worker.stderr");
    let mut child = Command::new(&adapter.program)
        .arg(&adapter.script)
        .arg("--engine")
        .arg(engine)
        .arg("--model")
        .arg(&model.path)
        .arg("--threads")
        .arg(adapter.threads.to_string())
        .arg("--serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(File::create(&stderr)?)
        .spawn()
        .with_context(|| format!("cannot start {role} worker"))?;
    let mut stdin = child.stdin.take().context("ASR worker stdin missing")?;
    let stdout = child.stdout.take().context("ASR worker stdout missing")?;
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    let run = (|| -> Result<()> {
        for (i, chunk) in pending {
            let clip = clips.join(format!("{i:05}.wav")).canonicalize()?;
            writeln!(stdin, "{}", serde_json::json!({"input": clip}))?;
            stdin.flush()?;
            let line = receiver
                .recv_timeout(Duration::from_secs(1800))
                .with_context(|| {
                    format!(
                        "{role} chunk {i}: worker timed out or exited; see {}",
                        stderr.display()
                    )
                })?
                .with_context(|| format!("{role} chunk {i}: worker stdout failed"))?;
            let response: serde_json::Value = serde_json::from_str(&line).with_context(|| {
                format!(
                    "{role} chunk {i}: invalid worker JSON; see {}",
                    stderr.display()
                )
            })?;
            if let Some(error) = response.get("error") {
                bail!("{role} chunk {i}: {error}; see {}", stderr.display());
            }
            let value: Hypothesis = serde_json::from_value(response)
                .with_context(|| format!("{role} chunk {i}: invalid hypothesis"))?;
            validate_hypothesis(&value, chunk.end - chunk.start)?;
            write_json(&folder.join(format!("{i:05}.json")), &value)?;
        }
        Ok(())
    })();
    drop(stdin);
    if run.is_err() && child.try_wait()?.is_none() {
        child.kill().context("cannot stop failed ASR worker")?;
    }
    let status = child.wait_timeout(Duration::from_secs(60))?;
    let exit = match status {
        Some(exit) => exit,
        None => {
            child
                .kill()
                .context("cannot stop ASR worker after exit timeout")?;
            child.wait()?
        }
    };
    reader
        .join()
        .map_err(|_| anyhow::anyhow!("ASR reader thread panicked"))?;
    run?;
    ensure!(
        exit.success(),
        "{role} worker failed with {exit}; see {}",
        stderr.display()
    );
    Ok(())
}

fn llm(config: &Config, prompt: &str, output: &Path, stderr: &Path) -> Result<()> {
    let stdout = output.with_extension("stdout");
    let partial = output.with_extension("partial.json");
    if partial.exists() {
        fs::remove_file(&partial)?;
    }
    let mut command = Command::new(&config.llm.program);
    let review_dir = output
        .parent()
        .context("review path has no parent")?
        .canonicalize()?;
    let absolute_output = review_dir.join(
        partial
            .file_name()
            .context("review path has no file name")?,
    );
    command.current_dir(review_dir);
    for arg in &config.llm.args {
        command.arg(arg.replace("{output}", &absolute_output.to_string_lossy()));
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(File::create(&stdout)?)
        .stderr(File::create(stderr)?)
        .spawn()
        .with_context(|| format!("cannot start LLM CLI {:?}", config.llm.program))?;
    child
        .stdin
        .take()
        .context("LLM stdin missing")?
        .write_all(prompt.as_bytes())?;
    match child.wait_timeout(Duration::from_secs(300))? {
        Some(status) if status.success() => {}
        Some(status) => bail!("LLM exited with {status}; see {}", stderr.display()),
        None => {
            child.kill()?;
            child.wait()?;
            bail!("LLM timed out; see {}", stderr.display());
        }
    }
    if !partial.exists() {
        fs::rename(&stdout, &partial).context("LLM did not produce a response")?;
    }
    let _: Decisions = read_json(&partial)?;
    fs::rename(partial, output)?;
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0_u8; 65536];
    loop {
        let len = file.read(&mut buf)?;
        if len == 0 {
            break;
        }
        hash.update(&buf[..len]);
    }
    Ok(hex::encode(hash.finalize()))
}

fn trim_repeated_prefix(previous: &str, current: &str) -> String {
    let left: Vec<_> = previous.split_whitespace().collect();
    let right: Vec<_> = current.split_whitespace().collect();
    for count in (1..=left.len().min(right.len()).min(12)).rev() {
        if left[left.len() - count..]
            .iter()
            .zip(&right[..count])
            .all(|(a, b)| {
                a.trim_matches(|c: char| !c.is_alphanumeric())
                    .eq_ignore_ascii_case(b.trim_matches(|c: char| !c.is_alphanumeric()))
            })
        {
            return right[count..].join(" ");
        }
    }
    current.to_owned()
}

fn validate_hypothesis(value: &Hypothesis, duration: f64) -> Result<()> {
    for segment in &value.segments {
        ensure!(
            segment.start.is_finite()
                && segment.end.is_finite()
                && segment.start >= 0.0
                && segment.end >= segment.start
                && segment.end <= duration + 0.5,
            "ASR returned invalid clip-relative timestamp"
        );
    }
    Ok(())
}

fn validate_diarization(value: &Diarization, duration: f64) -> Result<()> {
    for turn in value.turns.iter().chain(&value.exclusive_turns) {
        ensure!(
            turn.start.is_finite()
                && turn.end.is_finite()
                && turn.start >= 0.0
                && turn.end > turn.start
                && turn.end <= duration + 0.5
                && !turn.speaker.trim().is_empty(),
            "diarization returned invalid speaker turn"
        );
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_reader(
        File::open(path).with_context(|| format!("missing {}", path.display()))?,
    )
    .with_context(|| format!("invalid JSON in {}", path.display()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let temp = path.with_extension("partial");
    serde_json::to_writer_pretty(File::create(&temp)?, value)?;
    fs::rename(temp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Hypothesis, Transcript, WordSpan, transcribe, trim_repeated_prefix, validate_hypothesis,
    };
    use crate::config::{Adapter, Config, Llm};
    use crate::models::ModelSpec;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    fn strips_only_shared_boundary() {
        assert_eq!(
            trim_repeated_prefix("Ich gehe heute", "heute nach Hause"),
            "nach Hause"
        );
        assert_eq!(
            trim_repeated_prefix("Ich gehe heute", "morgen nach Hause"),
            "morgen nach Hause"
        );
    }

    #[test]
    fn rejects_out_of_clip_timestamps() {
        let value = Hypothesis {
            text: "test".into(),
            language: None,
            segments: vec![WordSpan {
                start: 0.0,
                end: 3.0,
                text: "test".into(),
            }],
        };
        assert!(validate_hypothesis(&value, 1.0).is_err());
    }

    #[test]
    fn runs_three_asr_processes_in_parallel_and_resumes() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let asr = root.join("asr.py");
        fs::write(
            &asr,
            r#"
import argparse, json, pathlib, sys, time
p = argparse.ArgumentParser()
for name in ("engine", "model", "threads"):
    p.add_argument("--" + name)
p.add_argument("--serve", action="store_true")
a = p.parse_args()
assert a.serve
marker = pathlib.Path(a.model).parent / (a.engine + ".started")
with marker.open("a") as stream:
    stream.write("started\n")
deadline = time.monotonic() + 5
while len(list(marker.parent.glob("*.started"))) != 3:
    if time.monotonic() > deadline:
        raise RuntimeError("ASR processes were not running in parallel")
    time.sleep(0.02)
for request in sys.stdin:
    clip = pathlib.Path(json.loads(request)["input"])
    assert clip.is_file()
    print(json.dumps({"text":"Guten Tag " + clip.stem, "language":"de", "segments":[]}), flush=True)
"#,
        )
        .unwrap();
        let dia = root.join("dia.py");
        fs::write(&dia, r#"
import argparse, json, pathlib
p = argparse.ArgumentParser()
for name in ("model", "input", "output", "threads"):
    p.add_argument("--" + name)
a = p.parse_args()
pathlib.Path(a.output).write_text(json.dumps({"turns":[{"start":0.0,"end":70.0,"speaker":"SPEAKER_00"}],"exclusive_turns":[{"start":0.0,"end":70.0,"speaker":"SPEAKER_00"}]}))
"#).unwrap();
        let llm = root.join("llm.py");
        fs::write(&llm, r#"
import json, pathlib, sys
prompt = json.load(sys.stdin)
assert len(prompt["intervals"][0]["hypotheses"]) == 3
pathlib.Path(sys.argv[1]).write_text(json.dumps({"segments":[{"index":s["index"],"text":"Guten Tag " + str(s["index"]),"language":"de","gap":False} for s in prompt["intervals"]]}))
"#).unwrap();
        let input = root.join("sample.mp3");
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=70",
                "-y",
            ])
            .arg(&input)
            .status()
            .unwrap();
        assert!(status.success());
        let config = Config {
            models: ["primeline", "flix", "qwen", "diarization"]
                .iter()
                .map(|role| ModelSpec {
                    role: (*role).to_owned(),
                    repo: role.to_string(),
                    revision: "test-revision".to_owned(),
                    path: root.join(role),
                })
                .collect(),
            asr: ["primeline", "flix", "qwen"]
                .iter()
                .map(|role| {
                    (
                        (*role).to_owned(),
                        Adapter {
                            program: PathBuf::from("python3"),
                            script: asr.clone(),
                            threads: 1,
                        },
                    )
                })
                .collect(),
            diarization: Adapter {
                program: PathBuf::from("python3"),
                script: dia,
                threads: 1,
            },
            llm: Llm {
                program: PathBuf::from("python3"),
                args: vec![llm.to_string_lossy().into_owned(), "{output}".into()],
            },
            review_rules: None,
        };
        let output = root.join("job");
        transcribe(&config, &input, &output).unwrap();
        transcribe(&config, &input, &output).unwrap();
        let transcript: Transcript =
            serde_json::from_slice(&fs::read(output.join("transcript.json")).unwrap()).unwrap();
        assert!(transcript.segments.len() >= 2);
        assert_eq!(
            transcript.segments[0].speaker.as_deref(),
            Some("SPEAKER_00")
        );
        assert_eq!(transcript.segments[0].hypotheses.len(), 3);
        assert_eq!(transcript.segments[0].text, "Guten Tag 0");
        for engine in ["nemo", "whisper", "qwen"] {
            assert_eq!(
                fs::read_to_string(root.join(format!("{engine}.started"))).unwrap(),
                "started\n"
            );
        }
    }
}
