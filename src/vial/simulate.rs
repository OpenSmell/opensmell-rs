//! Fixture simulation for a Vial's declared anomaly profile.
//!
//! This is not the live dual-Kalman engine. It is the profile's **state
//! machine**: baseline, per-channel response, min-channel corroboration,
//! persistence, hysteresis, and the escalation/reset timers. Fixtures prove
//! that the profile an author declared produces the states they claim on a
//! deterministic sample sequence. The desktop host remains the runtime; the
//! simulator is the authoring gate, and the same fixture runs in `noses test`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureSample {
    #[serde(default)]
    pub t: Option<f64>,
    #[serde(default)]
    pub channels: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureAlert {
    pub t: f64,
    pub level: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureExpected {
    #[serde(rename = "final")]
    pub final_state: String,
    #[serde(default)]
    pub alerts: Vec<FixtureAlert>,
    /// Allowed timing error for an expected alert. Defaults to one sample.
    #[serde(rename = "toleranceS", default)]
    pub tolerance_s: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Fixture {
    #[serde(default)]
    pub schema: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "dtS", default = "default_dt")]
    pub dt_s: f64,
    pub samples: Vec<FixtureSample>,
    pub expected: FixtureExpected,
}

fn default_dt() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize)]
pub struct SimAlert {
    pub t: f64,
    pub level: u8,
    pub state: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimResult {
    pub final_state: &'static str,
    pub final_level: u8,
    pub alerts: Vec<SimAlert>,
    /// Every level transition, for debugging a fixture.
    pub timeline: Vec<SimAlert>,
}

#[derive(Debug, Clone)]
pub struct FixtureOutcome {
    pub passed: bool,
    pub problems: Vec<String>,
    pub result: SimResult,
}

fn state_name(level: u8) -> &'static str {
    match level {
        0 => "normal",
        1 => "warning",
        2 => "critical",
        _ => "emergency",
    }
}

fn state_level(name: &str) -> Option<u8> {
    match name {
        "normal" => Some(0),
        "warning" => Some(1),
        "critical" => Some(2),
        "emergency" => Some(3),
        _ => None,
    }
}

fn number(v: &Value, path: &[&str]) -> Option<f64> {
    let mut current = v;
    for key in path {
        current = current.get(key)?;
    }
    current.as_f64()
}

fn string<'a>(v: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut current = v;
    for key in path {
        current = current.get(key)?;
    }
    current.as_str()
}

fn limits(v: &Value) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    if let Some(map) = v
        .get("anomaly")
        .and_then(|a| a.get("detection"))
        .and_then(|d| d.get("perChannelLimits"))
        .and_then(Value::as_object)
    {
        for (key, raw) in map {
            if let Some(value) = raw.as_f64() {
                if value > 0.0 {
                    out.insert(key.clone(), value);
                }
            }
        }
    }
    out
}

fn response(mode: &str, value: f64, baseline: f64) -> f64 {
    match mode {
        "relative" => {
            if baseline.abs() < 1e-12 { 0.0 } else { value / baseline }
        }
        "delta" => value - baseline,
        _ => {
            // delta-over-r0
            if baseline.abs() < 1e-12 { 0.0 } else { (value - baseline) / baseline }
        }
    }
}

/// Simulate one fixture against a manifest. The function returns a structured
/// result and a list of human-readable mismatches; it never panics on bad input.
pub fn simulate(manifest: &Value, fixture: &Fixture) -> FixtureOutcome {
    let dt = if fixture.dt_s > 0.0 { fixture.dt_s } else { 1.0 };
    let warmup = number(manifest, &["anomaly", "baseline", "warmupS"]).unwrap_or(6.0);
    let mode = string(manifest, &["anomaly", "transforms", "response"]).unwrap_or("delta-over-r0");
    let min_channels = number(manifest, &["anomaly", "detection", "minChannels"]).unwrap_or(1.0).max(1.0) as usize;
    let warning_s = number(manifest, &["anomaly", "detection", "escalation", "warningS"]).unwrap_or(0.2);
    let critical_s = number(manifest, &["anomaly", "detection", "escalation", "criticalS"]).unwrap_or(0.5);
    let emergency_s = number(manifest, &["anomaly", "detection", "escalation", "emergencyS"]).unwrap_or(1.0);
    let reset_s = number(manifest, &["anomaly", "detection", "escalation", "resetNormalS"]).unwrap_or(2.0);
    let hysteresis = number(manifest, &["anomaly", "detection", "hysteresis"]).unwrap_or(0.0).clamp(0.0, 0.95);
    let declared_limits = limits(manifest);
    let mut problems: Vec<String> = Vec::new();

    if fixture.samples.is_empty() {
        problems.push("fixture has no samples".into());
        return FixtureOutcome {
            passed: false,
            problems,
            result: SimResult { final_state: "normal", final_level: 0, alerts: vec![], timeline: vec![] },
        };
    }
    if declared_limits.is_empty() {
        problems.push("profile declares no perChannelLimits; the simulator has no per-channel trigger to evaluate".into());
    }

    // Union of declared channels and channels present in the fixture.
    let mut channels: BTreeSet<String> = declared_limits.keys().cloned().collect();
    for sample in &fixture.samples {
        channels.extend(sample.channels.keys().cloned());
    }
    let channels: Vec<String> = channels.into_iter().collect();

    let t0 = fixture.samples.first().and_then(|s| s.t).unwrap_or(0.0);
    let mut sums: BTreeMap<String, f64> = BTreeMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut baseline_end = 0usize;
    for (index, sample) in fixture.samples.iter().enumerate() {
        let t = sample.t.unwrap_or(t0 + index as f64 * dt);
        if t - t0 >= warmup && baseline_end > 0 {
            break;
        }
        for (channel, value) in &sample.channels {
            if value.is_finite() {
                *sums.entry(channel.clone()).or_insert(0.0) += value;
                *counts.entry(channel.clone()).or_insert(0) += 1;
            }
        }
        baseline_end = index + 1;
    }
    if baseline_end == 0 {
        problems.push("fixture has no finite baseline samples".into());
    }
    let baseline: BTreeMap<String, f64> = sums
        .into_iter()
        .filter_map(|(channel, sum)| {
            let n = counts.get(&channel).copied().unwrap_or(0);
            (n > 0).then_some((channel, sum / n as f64))
        })
        .collect();

    let mut level: u8 = 0;
    let mut anomalous_s = 0.0f64;
    let mut normal_s = 0.0f64;
    let mut peak_response = 0.0f64;
    let mut alerts: Vec<SimAlert> = Vec::new();
    let mut timeline: Vec<SimAlert> = Vec::new();
    let mut previous_level = 0u8;

    for (index, sample) in fixture.samples.iter().enumerate().skip(baseline_end) {
        let t = sample.t.unwrap_or(t0 + index as f64 * dt);
        let mut triggered = 0usize;
        let mut strongest = 0.0f64;
        for channel in &channels {
            let Some(value) = sample.channels.get(channel).copied() else { continue };
            if !value.is_finite() {
                continue;
            }
            let base = baseline.get(channel).copied().unwrap_or(0.0);
            let value_response = response(mode, value, base);
            // Hysteresis needs the actual response even when it is below the
            // trigger, so track the strongest declared-channel response first.
            if declared_limits.contains_key(channel) {
                strongest = strongest.max(value_response.abs());
            }
            let Some(limit) = declared_limits.get(channel).copied() else { continue };
            if value_response.abs() >= limit {
                triggered += 1;
            }
        }
        let anomaly = triggered >= min_channels;
        if anomaly {
            anomalous_s += dt;
            normal_s = 0.0;
            peak_response = peak_response.max(strongest);
            level = if anomalous_s + 1e-9 >= emergency_s {
                3
            } else if anomalous_s + 1e-9 >= critical_s {
                2
            } else if anomalous_s + 1e-9 >= warning_s {
                1
            } else {
                level
            };
        } else {
            normal_s += dt;
            anomalous_s = 0.0;
            let below_peak = hysteresis <= 0.0
                || peak_response <= 0.0
                || strongest <= (1.0 - hysteresis) * peak_response;
            if normal_s + 1e-9 >= reset_s && below_peak {
                level = 0;
                peak_response = 0.0;
            }
        }
        if level != previous_level {
            let event = SimAlert { t, level, state: state_name(level) };
            timeline.push(event.clone());
            if level > previous_level {
                alerts.push(event);
            }
            previous_level = level;
        }
    }

    let final_state = state_name(level);
    if let Some(expected_level) = state_level(&fixture.expected.final_state) {
        if level != expected_level {
            problems.push(format!(
                "expected final state `{}` (level {}), simulator ended at `{}` (level {})",
                fixture.expected.final_state, expected_level, final_state, level
            ));
        }
    } else {
        problems.push(format!("fixture expected.final is not a known state: {}", fixture.expected.final_state));
    }

    let tolerance = fixture.expected.tolerance_s.unwrap_or(dt.max(0.001));
    for expected in &fixture.expected.alerts {
        let hit = alerts.iter().find(|actual| {
            actual.level == expected.level && (actual.t - expected.t).abs() <= tolerance + 1e-9
        });
        if hit.is_none() {
            let near: Vec<String> = alerts
                .iter()
                .filter(|actual| actual.level == expected.level)
                .map(|actual| format!("{}s", actual.t))
                .collect();
            problems.push(format!(
                "expected a level {} alert at {}s (±{}s); actual level-{} alerts: {}",
                expected.level,
                expected.t,
                tolerance,
                expected.level,
                if near.is_empty() { "none".into() } else { near.join(", ") }
            ));
        }
    }

    FixtureOutcome {
        passed: problems.is_empty(),
        problems,
        result: SimResult { final_state, final_level: level, alerts, timeline },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifest(hysteresis: f64) -> Value {
        json!({
            "anomaly": {
                "baseline": { "warmupS": 2.0 },
                "transforms": { "response": "delta-over-r0" },
                "detection": {
                    "minChannels": 1,
                    "perChannelLimits": { "ch1": 0.2 },
                    "hysteresis": hysteresis,
                    "escalation": {
                        "warningS": 1.0,
                        "criticalS": 2.0,
                        "emergencyS": 3.0,
                        "resetNormalS": 1.0
                    }
                }
            }
        })
    }

    fn fixture(values: &[(f64, f64)]) -> Fixture {
        Fixture {
            schema: Some("osmograph.vial-fixture/1".into()),
            description: None,
            dt_s: 1.0,
            samples: values.iter().map(|(t, value)| FixtureSample {
                t: Some(*t),
                channels: [("ch1".to_string(), *value)].into_iter().collect(),
            }).collect(),
            expected: FixtureExpected {
                final_state: "emergency".into(),
                alerts: vec![
                    FixtureAlert { t: 2.0, level: 1 },
                    FixtureAlert { t: 3.0, level: 2 },
                    FixtureAlert { t: 4.0, level: 3 },
                ],
                tolerance_s: None,
            },
        }
    }

    #[test]
    fn a_step_escalates_on_the_declared_timers() {
        let values: Vec<(f64, f64)> = (0..8)
            .map(|i| (i as f64, if i < 2 { 1.0 } else { 1.5 }))
            .collect();
        let outcome = simulate(&manifest(0.0), &fixture(&values));
        assert!(outcome.passed, "{:?}", outcome.problems);
        assert_eq!(outcome.result.final_state, "emergency");
    }

    #[test]
    fn hysteresis_holds_the_alert_until_the_signal_falls_below_the_peak_fraction() {
        // Peak response is 0.5. With 0.9 hysteresis the level cannot clear until
        // the response is at or below 0.05, not merely below the 0.2 trigger.
        let mut values: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, if i < 2 { 1.0 } else { 1.5 })).collect();
        values.push((5.0, 1.1)); // response 0.1: below the trigger, above the clear band
        values.push((6.0, 1.0)); // response 0.0: clears
        let outcome = simulate(&manifest(0.9), &fixture(&values));
        assert_eq!(outcome.result.final_state, "normal", "{:?}", outcome.problems);
        // It did not clear at t=5, where the signal was below the trigger but
        // still above the hysteresis band. It cleared only at t=6.
        assert!(!outcome.result.timeline.iter().any(|event| event.t == 5.0 && event.level == 0));
        assert!(outcome.result.timeline.iter().any(|event| event.t == 6.0 && event.level == 0));
    }
}

/// Load and simulate a fixture file beside a manifest. The path is resolved
/// relative to the manifest directory; the caller has already checked that the
/// path matches `fixtures/<name>.json`.
pub fn run_fixture(manifest: &Value, manifest_dir: &Path, fixture_file: &str) -> Result<FixtureOutcome, String> {
    if !fixture_file.starts_with("fixtures/") || fixture_file.contains("..") {
        return Err(format!("invalid fixture path `{fixture_file}`"));
    }
    let path = manifest_dir.join(fixture_file);
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let fixture: Fixture = serde_json::from_str(&raw)
        .map_err(|e| format!("{} is not a valid fixture: {e}", path.display()))?;
    if let Some(schema) = &fixture.schema {
        if schema != "osmograph.vial-fixture/1" {
            return Err(format!("{} has schema `{schema}`; expected osmograph.vial-fixture/1", path.display()));
        }
    }
    Ok(simulate(manifest, &fixture))
}
