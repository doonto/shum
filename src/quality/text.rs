use std::collections::HashMap;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::pipeline::{Hypothesis, repeated_fourgram_fraction};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub min_words_per_second: f64,
    pub max_words_per_second: f64,
    pub min_characters_per_second: f64,
    pub max_characters_per_second: f64,
    pub max_repeated_fourgram_fraction: f64,
    pub min_consecutive_word_run: usize,
    pub min_pairwise_similarity: f64,
    pub max_hypotheses: usize,
    pub max_text_characters: usize,
    pub max_pairwise_cells: usize,
    pub max_pairwise_pairs: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            min_words_per_second: 0.5,
            max_words_per_second: 5.0,
            min_characters_per_second: 2.0,
            max_characters_per_second: 35.0,
            max_repeated_fourgram_fraction: 0.5,
            min_consecutive_word_run: 4,
            min_pairwise_similarity: 0.5,
            max_hypotheses: 8,
            max_text_characters: 100_000,
            max_pairwise_cells: 250_000,
            max_pairwise_pairs: 28,
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.min_words_per_second.is_finite()
                && self.min_words_per_second >= 0.0
                && self.max_words_per_second.is_finite()
                && self.max_words_per_second > self.min_words_per_second,
            "invalid word density thresholds"
        );
        ensure!(
            self.min_characters_per_second.is_finite()
                && self.min_characters_per_second >= 0.0
                && self.max_characters_per_second.is_finite()
                && self.max_characters_per_second > self.min_characters_per_second,
            "invalid character density thresholds"
        );
        ensure!(
            self.max_repeated_fourgram_fraction.is_finite()
                && (0.0..=1.0).contains(&self.max_repeated_fourgram_fraction),
            "invalid fourgram repetition threshold"
        );
        ensure!(
            self.min_consecutive_word_run >= 2,
            "invalid word run threshold"
        );
        ensure!(
            self.min_pairwise_similarity.is_finite()
                && (0.0..=1.0).contains(&self.min_pairwise_similarity),
            "invalid pairwise similarity threshold"
        );
        ensure!(
            self.max_hypotheses > 0
                && self.max_text_characters > 0
                && self.max_pairwise_cells > 0
                && self.max_pairwise_pairs > 0,
            "computation budgets must be positive"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
}

#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub code: &'static str,
    pub severity: Severity,
    pub role: Option<String>,
    pub evidence: String,
    pub recommendation: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoleMetrics {
    pub role: String,
    pub words: usize,
    pub characters: usize,
    pub words_per_second: f64,
    pub characters_per_second: f64,
    pub repeated_fourgram_fraction: f64,
    pub longest_consecutive_word_run: usize,
    pub invalid_timing_segments: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "availability", rename_all = "snake_case")]
pub enum ComparisonMetrics {
    Available {
        token_edit_similarity: f64,
        token_overlap: f64,
    },
    Unavailable {
        reason: &'static str,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct PairwiseMetrics {
    pub left_role: String,
    pub right_role: String,
    #[serde(flatten)]
    pub comparison: ComparisonMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub accuracy: &'static str,
    pub source_status: &'static str,
    pub eos: &'static str,
    pub roles: Vec<RoleMetrics>,
    pub pairwise_omitted_roles: Vec<String>,
    pub pairwise: Vec<PairwiseMetrics>,
    pub findings: Vec<Finding>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourceReport {
    pub accuracy: &'static str,
    pub worker_events: usize,
    pub findings: Vec<Finding>,
}

fn finding(
    code: &'static str,
    severity: Severity,
    role: Option<&str>,
    evidence: impl Into<String>,
    recommendation: &'static str,
) -> Finding {
    Finding {
        code,
        severity,
        role: role.map(str::to_owned),
        evidence: evidence.into(),
        recommendation,
    }
}

fn normalized_tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|word| {
            let word = word.trim_matches(|c: char| {
                matches!(
                    c,
                    '.' | ','
                        | '!'
                        | '?'
                        | ':'
                        | ';'
                        | '"'
                        | '\''
                        | '('
                        | ')'
                        | '['
                        | ']'
                        | '{'
                        | '}'
                        | '«'
                        | '»'
                        | '„'
                        | '“'
                        | '”'
                        | '…'
                )
            });
            (!word.is_empty()).then(|| word.to_lowercase())
        })
        .collect()
}

fn longest_run(tokens: &[String]) -> usize {
    let (mut longest, mut current) = (0, 0);
    for (index, token) in tokens.iter().enumerate() {
        current = if index > 0 && token == &tokens[index - 1] {
            current + 1
        } else {
            1
        };
        longest = longest.max(current);
    }
    longest
}

fn token_overlap(left: &[String], right: &[String]) -> f64 {
    let mut counts = HashMap::<&str, (usize, usize)>::new();
    for token in left {
        counts.entry(token).or_default().0 += 1;
    }
    for token in right {
        counts.entry(token).or_default().1 += 1;
    }
    let intersection: usize = counts.values().map(|(a, b)| a.min(b)).sum();
    let union = left.len() + right.len() - intersection;
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

fn token_edit_similarity(left: &[String], right: &[String]) -> f64 {
    let (short, long) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    if long.is_empty() {
        return 1.0;
    }
    let mut previous: Vec<usize> = (0..=short.len()).collect();
    let mut current = vec![0; short.len() + 1];
    for (row, token) in long.iter().enumerate() {
        current[0] = row + 1;
        for (column, other) in short.iter().enumerate() {
            current[column + 1] = (previous[column + 1] + 1)
                .min(current[column] + 1)
                .min(previous[column] + usize::from(token != other));
        }
        std::mem::swap(&mut previous, &mut current);
    }
    1.0 - previous[short.len()] as f64 / long.len() as f64
}

pub fn analyze(
    hypotheses: &[(String, Hypothesis)],
    duration: f64,
    settings: &Settings,
) -> Result<Report> {
    settings.validate()?;
    ensure!(
        duration.is_finite() && duration > 0.0,
        "duration must be positive and finite"
    );
    let mut report = Report {
        accuracy: "unverified",
        source_status: match hypotheses.len() {
            0 => "no_sources",
            1 => "single_asr_source",
            _ => "multiple_asr_sources_unverified",
        },
        eos: "unavailable",
        roles: Vec::with_capacity(hypotheses.len()),
        pairwise_omitted_roles: hypotheses
            .iter()
            .skip(settings.max_hypotheses)
            .map(|(role, _)| role.clone())
            .collect(),
        pairwise: Vec::new(),
        findings: Vec::new(),
    };
    let mut tokens = Vec::with_capacity(hypotheses.len().min(settings.max_hypotheses));
    if !report.pairwise_omitted_roles.is_empty() {
        report.findings.push(finding(
            "pairwise_hypothesis_budget",
            Severity::Info,
            None,
            format!(
                "pairwise analysis limited to {} of {} ASR sources; per-role metrics cover all sources",
                settings.max_hypotheses,
                hypotheses.len()
            ),
            "Increase quality.text.max_hypotheses for wider pairwise coverage; agreement does not establish accuracy.",
        ));
    }
    if hypotheses.is_empty() {
        report.findings.push(finding(
            "no_hypotheses",
            Severity::Warning,
            None,
            "no ASR hypotheses were provided",
            "Inspect the ASR worker results; no transcription accuracy is implied.",
        ));
    }
    for (index, (role, hypothesis)) in hypotheses.iter().enumerate() {
        let characters = hypothesis.text.chars().count();
        ensure!(
            characters <= settings.max_text_characters,
            "text for role {role} exceeds configured character budget"
        );
        let words = hypothesis.text.split_whitespace().count();
        let normalized = normalized_tokens(&hypothesis.text);
        let words_per_second = words as f64 / duration;
        let characters_per_second = characters as f64 / duration;
        ensure!(
            words_per_second.is_finite() && characters_per_second.is_finite(),
            "density for role {role} is not finite"
        );
        let repetition = repeated_fourgram_fraction(&hypothesis.text);
        let run = longest_run(&normalized);
        let invalid_timing_segments = hypothesis
            .segments
            .iter()
            .filter(|segment| {
                !segment.start.is_finite()
                    || !segment.end.is_finite()
                    || segment.start < -1e-6
                    || segment.end > duration + 1e-6
                    || segment.end + 1e-6 < segment.start
            })
            .count()
            + hypothesis.out_of_bounds_segments.len();
        report.roles.push(RoleMetrics {
            role: role.clone(),
            words,
            characters,
            words_per_second,
            characters_per_second,
            repeated_fourgram_fraction: repetition,
            longest_consecutive_word_run: run,
            invalid_timing_segments,
        });
        if hypothesis.text.trim().is_empty() {
            report.findings.push(finding(
                "empty_hypothesis",
                Severity::Warning,
                Some(role),
                "ASR text is empty",
                "Inspect the corresponding worker output and audio manually.",
            ));
        } else {
            if words_per_second < settings.min_words_per_second
                || words_per_second > settings.max_words_per_second
            {
                report.findings.push(finding(
                    "word_density",
                    Severity::Warning,
                    Some(role),
                    format!("{words_per_second:.3} words/s outside configured range"),
                    "Review the text against the audio; density alone cannot establish accuracy.",
                ));
            }
            if characters_per_second < settings.min_characters_per_second
                || characters_per_second > settings.max_characters_per_second
            {
                report.findings.push(finding(
                    "character_density",
                    Severity::Warning,
                    Some(role),
                    format!("{characters_per_second:.3} characters/s outside configured range"),
                    "Review the text against the audio; density alone cannot establish accuracy.",
                ));
            }
        }
        if repetition > settings.max_repeated_fourgram_fraction {
            report.findings.push(finding(
                "repeated_fourgrams",
                Severity::Warning,
                Some(role),
                format!("{repetition:.3} repeated fourgram fraction"),
                "Check for an ASR repetition loop against the audio.",
            ));
        }
        if run >= settings.min_consecutive_word_run {
            report.findings.push(finding(
                "consecutive_word_run",
                Severity::Warning,
                Some(role),
                format!("{run} consecutive identical normalized words"),
                "Check the repeated phrase against the audio.",
            ));
        }
        if invalid_timing_segments > 0 {
            report.findings.push(finding(
                "invalid_timing",
                Severity::Warning,
                Some(role),
                format!("{invalid_timing_segments} invalid or reported out-of-bounds segments"),
                "Inspect word timestamps; coarse or rounded timings need manual review.",
            ));
        }
        if index < settings.max_hypotheses {
            tokens.push(normalized);
        }
    }

    for left in 0..tokens.len() {
        for right in left + 1..tokens.len() {
            let (a, b) = (&tokens[left], &tokens[right]);
            let pair = PairwiseMetrics {
                left_role: hypotheses[left].0.clone(),
                right_role: hypotheses[right].0.clone(),
                comparison: if report.pairwise.len() >= settings.max_pairwise_pairs {
                    ComparisonMetrics::Unavailable {
                        reason: "pair_budget_exceeded",
                    }
                } else if a
                    .len()
                    .checked_mul(b.len())
                    .is_none_or(|cells| cells > settings.max_pairwise_cells)
                {
                    ComparisonMetrics::Unavailable {
                        reason: "edit_cell_budget_exceeded",
                    }
                } else {
                    ComparisonMetrics::Available {
                        token_edit_similarity: token_edit_similarity(a, b),
                        token_overlap: token_overlap(a, b),
                    }
                },
            };
            match &pair.comparison {
                ComparisonMetrics::Available {
                    token_edit_similarity,
                    ..
                } if *token_edit_similarity < settings.min_pairwise_similarity => {
                    report.findings.push(finding(
                        "hypothesis_disagreement",
                        Severity::Warning,
                        None,
                        format!(
                            "{} vs {}: normalized token edit similarity {token_edit_similarity:.3}",
                            pair.left_role, pair.right_role
                        ),
                        "Compare both ASR outputs with the audio; do not choose one automatically.",
                    ));
                }
                ComparisonMetrics::Unavailable { reason } => {
                    report.findings.push(finding(
                        "pairwise_unavailable",
                        Severity::Info,
                        None,
                        format!("{} vs {}: {reason}", pair.left_role, pair.right_role),
                        "Review the ASR outputs manually; no similarity score was computed.",
                    ));
                }
                _ => {}
            }
            report.pairwise.push(pair);
        }
    }
    Ok(report)
}

pub fn analyze_resources(events: &[Value]) -> Result<ResourceReport> {
    let mut report = ResourceReport {
        accuracy: "unverified",
        worker_events: 0,
        findings: Vec::new(),
    };
    let mut saw_admission = false;
    for (index, event) in events.iter().enumerate() {
        if let Some(safety_net) = event.get("using_swap_as_safety_net") {
            let enabled = safety_net.as_bool().ok_or_else(|| {
                anyhow::anyhow!("resource event {index} has invalid swap safety-net flag")
            })?;
            saw_admission = true;
            if enabled {
                report.findings.push(finding(
                    "swap_safety_net",
                    Severity::Warning,
                    None,
                    "recorded admission estimate relied on system swap; OOM remained possible",
                    "Inspect recorded memory headroom and swap-pressure samples before retrying.",
                ));
            }
        }
        ensure!(
            event.get("type").is_none() || event.get("type").and_then(Value::as_str).is_some(),
            "resource event {index} has invalid type"
        );
        if event.get("type").and_then(Value::as_str) != Some("worker") {
            ensure!(
                event.get("type").is_some()
                    || (event.get("role").is_none() && event.get("result").is_none()),
                "resource event {index} has worker fields but no type"
            );
            continue;
        }
        report.worker_events += 1;
        let role = event
            .get("role")
            .and_then(Value::as_str)
            .filter(|role| !role.is_empty())
            .ok_or_else(|| anyhow::anyhow!("worker event {index} has no valid role"))?;
        let result = event.get("result").and_then(Value::as_str).ok_or_else(|| {
            anyhow::anyhow!("worker event {index} for {role} has no valid result")
        })?;
        let (code, recommendation) = match result {
            "success" => ("", ""),
            "oom-kill" | "oom" => (
                "worker_oom",
                "Inspect recorded worker memory limits and process logs before retrying.",
            ),
            "timeout" => (
                "worker_timeout",
                "Inspect recorded worker runtime and deadline logs.",
            ),
            "error" | "worker_error" => (
                "worker_error",
                "Inspect the recorded worker exit status and error logs.",
            ),
            _ => (
                "unknown_worker_result",
                "Inspect the recorded worker result and logs.",
            ),
        };
        if !code.is_empty() {
            report.findings.push(finding(
                code,
                Severity::Warning,
                Some(role),
                format!("recorded worker result (historical event): {result}"),
                recommendation,
            ));
        }
        if event.get("elapsed_ms").and_then(Value::as_u64).is_none() {
            report.findings.push(finding(
                "elapsed_unavailable",
                Severity::Info,
                Some(role),
                "recorded worker elapsed_ms is missing or invalid",
                "Check recorded worker runtime telemetry.",
            ));
        }
        if event
            .get("memory_peak_mib")
            .and_then(Value::as_u64)
            .is_none()
        {
            report.findings.push(finding(
                "memory_peak_unavailable",
                Severity::Info,
                Some(role),
                "recorded worker memory_peak_mib is missing or invalid",
                "Check recorded worker memory telemetry; no memory safety claim is possible.",
            ));
        }
    }
    if report.worker_events == 0 {
        report.findings.push(finding(
            "worker_telemetry_unavailable",
            Severity::Info,
            None,
            "no worker resource events were provided",
            "Inspect worker telemetry before assessing runtime health.",
        ));
    }
    if !saw_admission {
        report.findings.push(finding(
            "swap_safety_net_unavailable",
            Severity::Info,
            None,
            "no admission swap safety-net flag was provided",
            "Inspect admission telemetry; swap safety-net usage is unknown.",
        ));
    }
    report.findings.push(finding(
        "swap_telemetry_unavailable",
        Severity::Info,
        None,
        "worker events do not contain swap safety-net counters",
        "Inspect separately recorded swap and memory-pressure samples.",
    ));
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::WordSpan;
    use serde_json::json;

    fn hypothesis(text: &str) -> Hypothesis {
        Hypothesis {
            text: text.into(),
            language: None,
            segments: vec![],
            raw_text: None,
            out_of_bounds_segments: vec![],
            processing_mode: None,
            model_revision: None,
        }
    }

    fn has(report: &Report, code: &str) -> bool {
        report.findings.iter().any(|item| item.code == code)
    }

    #[test]
    fn settings_defaults_and_invalid_boundaries() {
        let defaults = Settings::default();
        assert_eq!(
            serde_json::from_value::<Settings>(json!({}))
                .unwrap()
                .max_pairwise_pairs,
            28
        );
        defaults.validate().unwrap();
        let mut invalid = defaults.clone();
        invalid.max_words_per_second = f64::INFINITY;
        assert!(invalid.validate().is_err());
        invalid = defaults.clone();
        invalid.max_words_per_second = invalid.min_words_per_second;
        assert!(invalid.validate().is_err());
        invalid = defaults.clone();
        invalid.max_repeated_fourgram_fraction = 1.01;
        assert!(invalid.validate().is_err());
        invalid = defaults.clone();
        invalid.min_pairwise_similarity = f64::NAN;
        assert!(invalid.validate().is_err());
        invalid = defaults.clone();
        invalid.max_pairwise_cells = 0;
        assert!(invalid.validate().is_err());
        invalid = defaults.clone();
        invalid.min_consecutive_word_run = 1;
        assert!(invalid.validate().is_err());
        assert!(analyze(&[], 0.0, &defaults).is_err());
        assert!(analyze(&[], f64::INFINITY, &defaults).is_err());
    }

    #[test]
    fn natural_repetition_and_empty_conflict() {
        let settings = Settings::default();
        let clean = analyze(
            &[("a".into(), hypothesis("eins zwei drei vier fünf sechs"))],
            3.0,
            &settings,
        )
        .unwrap();
        assert_eq!(clean.source_status, "single_asr_source");
        assert_eq!(clean.accuracy, "unverified");
        assert!(clean.pairwise.is_empty());
        assert!(!has(&clean, "repeated_fourgrams"));
        let looped = analyze(
            &[("a".into(), hypothesis("ja ja ja ja ja ja ja ja ja"))],
            3.0,
            &settings,
        )
        .unwrap();
        assert!(has(&looped, "repeated_fourgrams"));
        assert!(has(&looped, "consecutive_word_run"));
        let empty = analyze(&[], 3.0, &settings).unwrap();
        assert!(has(&empty, "no_hypotheses"));
        let conflict = analyze(
            &[
                ("empty".into(), hypothesis(" ")),
                ("full".into(), hypothesis("hallo Welt")),
            ],
            2.0,
            &settings,
        )
        .unwrap();
        assert!(has(&conflict, "empty_hypothesis"));
        assert!(has(&conflict, "hypothesis_disagreement"));
    }

    #[test]
    fn comparison_preserves_numbers_and_unicode_without_rewriting_source() {
        let hypotheses = [
            ("de".into(), hypothesis("Straße 1,5 ÄPFEL")),
            ("ru".into(), hypothesis("straße 1,5 äpfel")),
            ("different".into(), hypothesis("улица 15 яблок")),
        ];
        let report = analyze(&hypotheses, 2.0, &Settings::default()).unwrap();
        assert_eq!(hypotheses[0].1.text, "Straße 1,5 ÄPFEL");
        match report.pairwise[0].comparison {
            ComparisonMetrics::Available {
                token_edit_similarity,
                token_overlap,
            } => {
                assert_eq!(token_edit_similarity, 1.0);
                assert_eq!(token_overlap, 1.0);
            }
            _ => panic!("identical normalized text must be comparable"),
        }
        assert!(has(&report, "hypothesis_disagreement"));
        assert_eq!(report.accuracy, "unverified");
        assert_eq!(report.eos, "unavailable");
        assert_eq!(
            normalized_tokens("1,5 15 -3 3.14"),
            ["1,5", "15", "-3", "3.14"]
        );
    }

    #[test]
    fn bounded_comparisons_are_explicitly_unavailable() {
        let mut settings = Settings {
            max_pairwise_cells: 1,
            max_pairwise_pairs: 1,
            ..Settings::default()
        };
        let hypotheses = [
            ("one".into(), hypothesis("eins zwei drei")),
            ("two".into(), hypothesis("eins zwei drei")),
            ("three".into(), hypothesis("eins zwei drei")),
        ];
        let report = analyze(&hypotheses, 2.0, &settings).unwrap();
        assert_eq!(report.pairwise.len(), 3);
        assert!(
            report
                .pairwise
                .iter()
                .all(|pair| matches!(pair.comparison, ComparisonMetrics::Unavailable { .. }))
        );
        assert!(has(&report, "pairwise_unavailable"));
        settings.max_pairwise_cells = 9;
        let report = analyze(&hypotheses, 2.0, &settings).unwrap();
        assert!(matches!(
            report.pairwise[0].comparison,
            ComparisonMetrics::Available { .. }
        ));
        assert!(matches!(
            report.pairwise[1].comparison,
            ComparisonMetrics::Unavailable {
                reason: "pair_budget_exceeded"
            }
        ));
        settings.max_text_characters = 3;
        assert!(analyze(&hypotheses, 2.0, &settings).is_err());
    }

    #[test]
    fn comparison_budget_does_not_limit_per_role_diagnostics() {
        let hypotheses: Vec<_> = (0..12)
            .map(|index| (format!("model{index:02}"), hypothesis("eins zwei drei")))
            .collect();
        let report = analyze(&hypotheses, 2.0, &Settings::default()).unwrap();
        assert_eq!(report.roles.len(), 12);
        assert_eq!(report.pairwise.len(), 28);
        assert_eq!(
            report.pairwise_omitted_roles,
            ["model08", "model09", "model10", "model11"]
        );
        assert_eq!(report.accuracy, "unverified");
        assert!(has(&report, "pairwise_hypothesis_budget"));
    }

    #[test]
    fn density_and_timing_respect_roundoff_and_coarse_words() {
        let mut value = hypothesis("eins zwei drei");
        value.segments = vec![
            WordSpan {
                start: -0.0000005,
                end: 0.0,
                text: "eins".into(),
            },
            WordSpan {
                start: 1.0,
                end: 1.0,
                text: "zwei".into(),
            },
            WordSpan {
                start: 2.0,
                end: 2.000002,
                text: "drei".into(),
            },
        ];
        let report = analyze(&[("a".into(), value.clone())], 2.0, &Settings::default()).unwrap();
        assert_eq!(report.roles[0].invalid_timing_segments, 1);
        assert!(has(&report, "invalid_timing"));
        value.segments.pop();
        let report = analyze(&[("a".into(), value)], 100.0, &Settings::default()).unwrap();
        assert_eq!(report.roles[0].invalid_timing_segments, 0);
        assert!(has(&report, "word_density"));
        assert!(has(&report, "character_density"));
        let serialized = serde_json::to_value(&report).unwrap();
        assert_eq!(serialized["accuracy"], "unverified");
        assert!(serialized["roles"][0]["words_per_second"].is_number());
    }

    #[test]
    fn resource_results_and_missing_counters_never_imply_all_clear() {
        let empty = analyze_resources(&[]).unwrap();
        assert!(
            empty
                .findings
                .iter()
                .any(|f| f.code == "worker_telemetry_unavailable")
        );
        assert!(
            empty
                .findings
                .iter()
                .any(|f| f.code == "swap_telemetry_unavailable")
        );
        assert!(
            empty
                .findings
                .iter()
                .any(|f| f.code == "swap_safety_net_unavailable")
        );
        let events = [
            json!({"type": "window", "result": "error"}),
            json!({"roles": ["de"], "using_swap_as_safety_net": true}),
            json!({"type": "worker", "role": "de", "result": "oom-kill", "elapsed_ms": 0, "memory_peak_mib": null}),
            json!({"type": "worker", "role": "ru", "result": "timeout", "elapsed_ms": 120}),
            json!({"type": "worker", "role": "alt", "result": "error"}),
            json!({"type": "worker", "role": "legacy", "result": "worker_error"}),
        ];
        let report = analyze_resources(&events).unwrap();
        assert_eq!(report.worker_events, 4);
        for code in [
            "worker_oom",
            "worker_timeout",
            "worker_error",
            "memory_peak_unavailable",
            "elapsed_unavailable",
            "swap_safety_net",
        ] {
            assert!(report.findings.iter().any(|f| f.code == code), "{code}");
        }
        assert!(
            report
                .findings
                .iter()
                .filter(|f| f.code == "worker_error")
                .all(|f| f.evidence.contains("historical event"))
        );
        assert!(analyze_resources(&[json!({"type": "worker", "role": "de"})]).is_err());
        assert!(analyze_resources(&[json!({"type": "worker", "result": "success"})]).is_err());
        assert!(analyze_resources(&[json!({"role": "de", "result": "success"})]).is_err());
        assert!(analyze_resources(&[json!({"using_swap_as_safety_net": "yes"})]).is_err());
        assert!(analyze_resources(&[json!({"type": 123, "role": "de"})]).is_err());
    }
}
