//! Quality scoring — 7-factor MOX scorer, port of
//! `opensmell/opensmell/mox/quality.py` + `mox/normalize.py` (OSMELL_FORMAT_SPEC.md §7).
//!
//! Factors and weights:
//!
//! | factor                  | weight |
//! |-------------------------|--------|
//! | continuity              | 0.15   |
//! | dynamicRange            | 0.10   |
//! | saturationFree          | 0.10   |
//! | baselineStability       | 0.20   |
//! | signalStrength          | 0.20   |
//! | recoveryCompleteness    | 0.15   |
//! | durationAdequacy        | 0.10   |
//!
//! G and R are excluded from the total for any role other than `exposure`. When
//! `baselineSource == "auto"`, B is capped at 50. When `adcMax` is undeclared,
//! upper-rail clipping is not detectable and only the lower rail (`<= 0`) counts
//! toward saturation. When `samplingRateHz` is undeclared, continuity uses the
//! median gap as the nominal schedule.
//!
//! # PROVISIONAL WEIGHTS
//!
//! The values in [`WEIGHTS`] are **provisional and deliberately not calibrated**.
//! Three calibration attempts failed for reasons documented in
//! `opensmell/docs/quality-weight-calibration.md`. Do not cite them as a measured
//! optimum.
//!
//! Four defects were found during that study, all now fixed. They come in two
//! shapes. Noise, saturation, and a dead channel each **raised** the score:
//! worse data scored better. The mislabelled time column did the opposite and is
//! the more instructive of the two kinds -- it did not reward bad data, it made a
//! clean recording score identically to a broken one, so a caller reading
//! continuity 0 could not tell a mislabelled column from genuine packet loss.
//! A score that improves for worse data is wrong, and a score that reports the
//! same number for two opposite faults is not measuring anything.
//!
//! Three of those are handled here:
//!
//! - Dynamic-range span and the signal-strength peak are both measured over
//!   *unclipped* samples only. A sample pinned at the converter rail carries no
//!   amplitude information -- the true peak is unknown and lies above full scale --
//!   so counting it as span or as peak made a saturated channel score a wider range
//!   and a stronger signal than the same channel read in range. `saturationFree`
//!   still reports the clipping. This requires `adcMax` to be declared.
//! - Span is a robust 5th-95th percentile range minus three times the channel's
//!   noise floor, the floor estimated from successive differences, so interference
//!   cannot inflate the score.
//! - Each dead channel costs [`DEAD_SENSOR_PENALTY`] off the total.
//!
//! One defect remains open and is the blocker for calibration: the subscores are not
//! on a common scale. Compare subscores within a recording, not across devices.

use std::collections::BTreeMap;

use serde::Serialize;

// Constants (shared with the web lib: `opensmell/types.py`).
pub const DEFAULT_ADC_MAX: f64 = 4095.0;
pub const DEFAULT_R0_SAMPLES: usize = 15;
pub const DEAD_CV_THRESHOLD: f64 = 0.001;
pub const NOISE_CV_LIMIT: f64 = 0.05;
pub const SNR_TARGET: f64 = 10.0;
pub const FULL_SCORE_DURATION_S: f64 = 60.0;
pub const MIN_SPAN_FRACTION: f64 = 0.1;
pub const GAP_TOLERANCE: f64 = 0.1;

// The time column is assumed to be milliseconds. When `sampling_rate_hz` is
// declared, the observed median gap should sit within these multiples of the
// expected period. A ratio far from 1 means the column is probably in seconds
// or microseconds, which is a unit fault rather than packet loss.
pub const MIN_TIME_UNIT_RATIO: f64 = 0.5;
pub const MAX_TIME_UNIT_RATIO: f64 = 2.0;

// Dynamic range uses a robust 5th-95th percentile span minus this multiple of
// the channel's noise floor, so an interference burst cannot raise the score.
pub const NOISE_SPAN_TOLERANCE: f64 = 3.0;

// Each dead sensing element costs this much of the total, because dead channels
// are excluded from the live-channel means and exclusion would otherwise read
// as an improvement.
pub const DEAD_SENSOR_PENALTY: f64 = 12.5;

pub const WEIGHTS: &[(&str, f64)] = &[
    ("continuity", 0.15),
    ("dynamicRange", 0.10),
    ("saturationFree", 0.10),
    ("baselineStability", 0.20),
    ("signalStrength", 0.20),
    ("recoveryCompleteness", 0.15),
    ("durationAdequacy", 0.10),
];

fn weight_of(key: &str) -> f64 {
    WEIGHTS
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, w)| *w)
        .unwrap_or(0.0)
}

/// One channel's recorded values.
#[derive(Debug, Clone)]
pub struct ChannelSeries {
    pub id: String,
    pub values: Vec<f64>,
}

impl ChannelSeries {
    pub fn new(id: impl Into<String>, values: Vec<f64>) -> Self {
        Self { id: id.into(), values }
    }
}

/// Manifest-derived parameters mirrored from `SensorDescriptor` / `BaselineDescriptor`.
#[derive(Debug, Clone)]
pub struct QualityParams {
    pub adc_max: Option<f64>,
    pub sampling_rate_hz: Option<f64>,
    pub guess_sampling_rate_hz: f64,
    pub role: String,
    pub baseline_source: String,
    pub r0_samples: Option<usize>,
    pub unsorted_rows: bool,
    pub non_finite_samples: usize,
}

impl Default for QualityParams {
    fn default() -> Self {
        Self {
            adc_max: None,
            sampling_rate_hz: None,
            guess_sampling_rate_hz: 10.0,
            role: "single".to_string(),
            baseline_source: "none".to_string(),
            r0_samples: None,
            unsorted_rows: false,
            non_finite_samples: 0,
        }
    }
}

/// One scored factor: `value` is None when the factor does not apply.
#[derive(Debug, Clone, Serialize)]
pub struct SubScore {
    pub value: Option<f64>,
    pub reason: String,
}

/// Quality flags (serialized camelCase, matching the app's `quality_report` dict).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityFlags {
    pub dead_sensors: Vec<String>,
    pub unsorted_rows: bool,
    pub non_finite_samples: usize,
    pub used_default_adc_max: bool,
    pub used_median_sampling_rate: bool,
    pub no_baseline: bool,
    pub empty_recording: bool,
    pub time_unit_mismatch: bool,
}

/// Full quality report; serializes to the same shape as the Python app's
/// `SessionRecord.quality_report` (`_quality_to_dict`), so the frontend
/// `QualityReportPanel` renders it unchanged.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityReport {
    pub format: String,
    pub version: String,
    pub computed_at: String,
    pub total: Option<f64>,
    pub badge: String,
    pub subscores: BTreeMap<String, SubScore>,
    pub flags: QualityFlags,
    pub reasons: BTreeMap<String, String>,
    pub notes: Vec<String>,
}

// ---- statistical helpers (opensmell/normalize.py) ----

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n.is_multiple_of(2) {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population standard deviation (ddof=0), matching the reference.
fn std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let m = mean(values);
    (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

fn is_finite(v: f64) -> bool {
    v.is_finite()
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v.is_nan() {
        return lo;
    }
    v.max(lo).min(hi)
}

/// Python `round()` — round-half-to-even, so totals land on the same integer
/// as the reference implementation.
fn py_round(x: f64) -> f64 {
    let f = x.floor();
    let frac = x - f;
    if frac < 0.5 {
        f
    } else if frac > 0.5 {
        f + 1.0
    } else if f % 2.0 == 0.0 {
        f
    } else {
        f + 1.0
    }
}

/// Map a possibly non-finite float to `None` so serde never hits a NaN.
fn finite_opt(v: f64) -> Option<f64> {
    if v.is_finite() {
        Some(v)
    } else {
        None
    }
}

// ---- baseline / normalization (mox/normalize.py) ----

fn r0_from_samples(values: &[f64], n: usize) -> f64 {
    let window: Vec<f64> = values.iter().take(n).copied().collect();
    if window.is_empty() {
        return f64::NAN;
    }
    let mut sorted = window.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let r0 = if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    if r0 > 0.0 {
        return r0;
    }
    let positive: Vec<f64> = window.iter().copied().filter(|v| *v > 0.0).collect();
    if positive.is_empty() {
        1.0
    } else {
        mean(&positive)
    }
}

/// Returns `(r0, window_values, cv)` for a channel, mirroring `baseline_for_channel`.
fn baseline_for_channel(
    source: &str,
    r0_samples: Option<usize>,
    values: &[f64],
) -> (f64, Vec<f64>, f64) {
    if source == "explicit" {
        let r0 = r0_from_samples(values, values.len());
        let cv = if r0 > 0.0 { std(values) / r0 } else { f64::INFINITY };
        return (r0, values.to_vec(), cv);
    }
    let n = r0_samples.unwrap_or(DEFAULT_R0_SAMPLES);
    let valid: Vec<f64> = values
        .iter()
        .take(n)
        .copied()
        .filter(|v| is_finite(*v))
        .collect();
    let r0 = r0_from_samples(&valid, n);
    let cv = if r0 > 0.0 { std(&valid) / r0 } else { f64::INFINITY };
    (r0, valid, cv)
}

fn normalized_series(values: &[f64], r0: f64) -> Vec<f64> {
    if !r0.is_finite() || r0 <= 0.0 {
        return vec![f64::NAN; values.len()];
    }
    values.iter().map(|v| (v - r0) / r0).collect()
}

struct ChannelStats {
    id: String,
    dead: bool,
}

fn channel_stats(values: &[f64], r0: f64) -> ChannelStats {
    let finite: Vec<f64> = values.iter().copied().filter(|v| is_finite(*v)).collect();
    let sd = std(&finite);
    let cv = if r0 > 0.0 { sd / r0 } else { f64::INFINITY };
    // Raw min-max span is deliberately not retained. Dynamic range now uses a
    // noise-corrected percentile span computed in `net_span`, because a raw
    // span is inflated by interference and rewards a noisier recording.
    ChannelStats {
        id: String::new(),
        dead: cv < DEAD_CV_THRESHOLD,
    }
}

// ---- scorer (mox/quality.py) ----

/// Compute the 7-factor quality report for a recording.
///
/// `time` is the series of timestamps in ms; `channels` the per-channel ADC
/// values. Mirrors `compute_quality` for the `mox` / `unknown` sensor families
/// (the only registered families; `miris`/`electrochemical` have no scorer).
pub fn compute_quality(
    time: &[f64],
    channels: &[ChannelSeries],
    params: &QualityParams,
) -> QualityReport {
    let sample_count = time.len();
    let adc_declared = params.adc_max.is_some();
    let adc_max = params.adc_max.unwrap_or(DEFAULT_ADC_MAX);
    let rate_declared = params.sampling_rate_hz.is_some();
    let sampling_rate_hz = params.sampling_rate_hz.unwrap_or(params.guess_sampling_rate_hz);
    let role = params.role.as_str();
    let baseline_source = params.baseline_source.as_str();

    let mut flags = QualityFlags {
        dead_sensors: Vec::new(),
        unsorted_rows: params.unsorted_rows,
        non_finite_samples: params.non_finite_samples,
        used_default_adc_max: !adc_declared,
        used_median_sampling_rate: !rate_declared,
        no_baseline: baseline_source == "none",
        empty_recording: sample_count == 0,
        time_unit_mismatch: false,
    };
    let mut notes: Vec<String> = Vec::new();
    let mut reasons: BTreeMap<String, String> = BTreeMap::new();

    // --- Continuity (spec 7.1.1) ---
    let gaps: Vec<f64> = time.windows(2).map(|w| w[1] - w[0]).collect();
    let positive_gaps: Vec<f64> = gaps.iter().copied().filter(|g| *g > 0.0).collect();
    let (mut continuity_value, mut continuity_reason) = (100.0_f64, "ok");
    let mut continuity_withheld = false;

    // The time column is assumed to be milliseconds. If a caller supplies
    // seconds the nominal period below would be wrong by 1000x and continuity
    // would collapse to 0 with reason "irregular_gaps", which looks identical
    // to real packet loss. Withhold continuity and say so instead.
    let observed_median = median(&positive_gaps);
    let observed_median = if observed_median.is_finite() { Some(observed_median) } else { None };
    if let (Some(observed), true) = (observed_median, rate_declared) {
        if sampling_rate_hz > 0.0 {
            let expected = 1000.0 / sampling_rate_hz;
            let ratio = observed / expected;
            if ratio < MIN_TIME_UNIT_RATIO || ratio > MAX_TIME_UNIT_RATIO {
                continuity_withheld = true;
                flags.time_unit_mismatch = true;
                notes.push(format!(
                    "median gap {:.6} ms is {:.6}x the expected {:.6} ms for {:.6} Hz; the time column is probably in seconds or microseconds rather than milliseconds. Continuity is not reported.",
                    observed, ratio, expected, sampling_rate_hz
                ));
            }
        }
    }

    if sample_count >= 2 && continuity_withheld {
        continuity_value = f64::NAN;
        continuity_reason = "time_unit_mismatch";
    } else if sample_count >= 2 {
        let nominal = if rate_declared {
            if sampling_rate_hz > 0.0 {
                Some(1000.0 / sampling_rate_hz)
            } else {
                None
            }
        } else {
            let m = observed_median;
            if m.is_some() {
                notes.push("samplingRateHz not declared; nominal period taken as the median gap.".to_string());
            }
            flags.used_median_sampling_rate = true;
            m
        };
        match nominal {
            Some(nom) if nom > 0.0 => {
                let tol = GAP_TOLERANCE * nom;
                let regular = gaps.iter().copied().filter(|g| (g - nom).abs() <= tol).count();
                let total = gaps.len();
                continuity_value = if total == 0 {
                    100.0
                } else {
                    regular as f64 / total as f64 * 100.0
                };
                if regular < total {
                    continuity_reason = "irregular_gaps";
                }
            }
            _ => {
                continuity_value = 50.0;
                continuity_reason = "irregular_gaps";
            }
        }
    }

    // --- Per-channel stats with R0 ---
    let mut stats: Vec<ChannelStats> = Vec::new();
    for c in channels {
        let (r0, _, _) = baseline_for_channel(baseline_source, params.r0_samples, &c.values);
        let mut st = channel_stats(&c.values, r0);
        st.id = c.id.clone();
        if st.dead {
            flags.dead_sensors.push(c.id.clone());
        }
        stats.push(st);
    }

    // --- Saturation-free (spec 7.1.3) ---
    let mut sat_scores: Vec<f64> = Vec::new();
    for s in &mut stats {
        let values = channels.iter().find(|c| c.id == s.id).map(|c| &c.values).cloned().unwrap_or_default();
        let clipped = if adc_declared {
            values.iter().filter(|v| **v >= adc_max || **v <= 0.0).count()
        } else {
            values.iter().filter(|v| **v <= 0.0).count()
        };
        sat_scores.push(if values.is_empty() {
            100.0
        } else {
            100.0 * (1.0 - clipped as f64 / values.len() as f64)
        });
    }
    let saturation_value = mean(&sat_scores);

    let live: Vec<&ChannelStats> = stats.iter().filter(|s| !s.dead).collect();

    // --- Dynamic range (spec 7.1.2) ---
    // Span is a robust 5th-95th percentile range with the channel's noise floor
    // subtracted, so an interference burst cannot inflate the score. The noise
    // floor is estimated from sample-to-sample differences rather than the
    // overall standard deviation, because the overall std includes the
    // exposure we are trying to measure.
    let net_span = |s: &ChannelStats| -> f64 {
        let values: Vec<f64> = channels
            .iter()
            .find(|c| c.id == s.id)
            .map(|c| c.values.iter().copied().filter(|x| x.is_finite()).collect())
            .unwrap_or_default();
        // Samples sitting on the converter rail are not measurements of the
        // chemistry. Including them inflates the span, so a channel driven into
        // saturation scores a *wider* dynamic range than the same channel read
        // below the rail -- the same reward-for-worse-data failure the noise
        // correction fixes. Span is measured over the unclipped samples only;
        // saturation itself is scored separately by the saturation subscore, so
        // discarding these samples here loses no signal.
        let unclipped: Vec<f64> = values
            .iter()
            .copied()
            .filter(|x| !(adc_declared && (*x >= adc_max || *x <= 0.0)))
            .collect();
        if unclipped.len() < 5 {
            return 0.0;
        }
        let mut ordered = unclipped.clone();
        ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = ((ordered.len() - 1) as f64 * 0.05) as usize;
        let k = k.max(1);
        let robust = ordered[ordered.len() - 1 - k] - ordered[k];
        let diffs: Vec<f64> = unclipped.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let noise = if diffs.is_empty() { 0.0 } else { median(&diffs) / 1.4142135623730951 };
        (robust - NOISE_SPAN_TOLERANCE * noise).max(0.0)
    };
    let dynamic_value = if live.is_empty() {
        0.0
    } else {
        let scores: Vec<f64> = live
            .iter()
            .map(|s| clamp((net_span(s) / adc_max) * (1.0 / MIN_SPAN_FRACTION), 0.0, 1.0))
            .collect();
        100.0 * mean(&scores)
    };
    let dynamic_reason = if dynamic_value < 50.0 { "low_span" } else { "ok" };
    if dynamic_reason == "low_span" {
        reasons.insert(
            "dynamicRange".to_string(),
            "channel_span_below_10_percent_of_adc_range_after_noise_correction".to_string(),
        );
    }

    // --- Baseline stability (spec 7.1.4) ---
    let (mut baseline_value, mut baseline_reason) = (0.0_f64, "no_baseline");
    if baseline_source != "none" {
        let cvs: Vec<f64> = channels
            .iter()
            .map(|c| baseline_for_channel(baseline_source, params.r0_samples, &c.values).2)
            .collect();
        let finite_cvs: Vec<f64> = cvs.iter().copied().filter(|v| is_finite(*v)).collect();
        let cv_window = if finite_cvs.is_empty() {
            f64::NAN
        } else {
            mean(&finite_cvs)
        };
        let raw_b = 100.0 * clamp(1.0 - cv_window / NOISE_CV_LIMIT, 0.0, 1.0);
        if baseline_source == "auto" {
            baseline_value = raw_b.min(50.0);
            baseline_reason = "auto_r0";
        } else {
            baseline_value = raw_b;
            baseline_reason = if cv_window >= NOISE_CV_LIMIT {
                "r0_window_cv_too_high"
            } else {
                "ok"
            };
        }
    }

    // --- Signal strength + recovery (spec 7.1.5 / 7.1.6) ---
    let exposure_with_r0 = role == "exposure" && baseline_source != "none";
    let (mut signal_value, mut signal_reason, mut recovery_value, mut recovery_reason) =
        (None, "no_exposure_signal", None, "no_exposure_signal");
    if exposure_with_r0 {
        let mut best_g: Vec<f64> = Vec::new();
        let mut recovery_scores: Vec<f64> = Vec::new();
        for s in live {
            let c = channels.iter().find(|c| c.id == s.id).expect("channel present");
            let (r0, _, base_cv) = baseline_for_channel(baseline_source, params.r0_samples, &c.values);
            // A sample pinned at the converter rail carries no amplitude information: the
            // true peak is unknown and lies somewhere above full scale. Counting
            // it as the peak would let saturation *raise* the SNR score, so rail
            // samples are excluded from the normalised series used for peak and
            // recovery. The saturation subscore reports the clipping itself.
            let usable: Vec<f64> = if adc_declared {
                c.values.iter().copied().filter(|v| !(*v >= adc_max || *v <= 0.0)).collect()
            } else {
                c.values.clone()
            };
            let norm: Vec<f64> = normalized_series(&usable, r0)
                .into_iter()
                .filter(|v| is_finite(*v))
                .collect();
            let noise = base_cv.max(1e-6);
            if norm.is_empty() {
                best_g.push(0.0);
                recovery_scores.push(0.0);
                continue;
            }
            let peak = norm.iter().copied().fold(0.0f64, |a, v| a.max(v.abs()));
            best_g.push(clamp(peak / noise / SNR_TARGET, 0.0, 1.0) * 100.0);
            let start = norm.len().saturating_sub(15);
            let final_win = median(&norm[start..]);
            let recovered = 1.0 - clamp(final_win.abs() / peak.max(1e-6), 0.0, 1.0);
            recovery_scores.push(100.0 * recovered);
        }
        signal_value = finite_opt(if best_g.is_empty() {
            0.0
        } else {
            best_g.iter().copied().fold(0.0f64, f64::max)
        });
        signal_reason = "ok";
        recovery_value = finite_opt(if recovery_scores.is_empty() {
            0.0
        } else {
            mean(&recovery_scores)
        });
        recovery_reason = "ok";
    }

    // --- Duration adequacy (spec 7.1.7) ---
    let t_seconds = if sampling_rate_hz > 0.0 {
        (sample_count as f64 - 1.0) / sampling_rate_hz
    } else {
        0.0
    };
    let duration_value = 100.0 * clamp(t_seconds / FULL_SCORE_DURATION_S, 0.0, 1.0);
    let duration_reason = if t_seconds < FULL_SCORE_DURATION_S { "too_short" } else { "ok" };

    let mut subscores = BTreeMap::new();
    subscores.insert("continuity".to_string(), SubScore { value: finite_opt(continuity_value), reason: continuity_reason.to_string() });
    subscores.insert("dynamicRange".to_string(), SubScore { value: finite_opt(dynamic_value), reason: dynamic_reason.to_string() });
    subscores.insert("saturationFree".to_string(), SubScore { value: finite_opt(saturation_value), reason: "ok".to_string() });
    subscores.insert("baselineStability".to_string(), SubScore { value: finite_opt(baseline_value), reason: baseline_reason.to_string() });
    subscores.insert("signalStrength".to_string(), SubScore { value: signal_value, reason: signal_reason.to_string() });
    subscores.insert("recoveryCompleteness".to_string(), SubScore { value: recovery_value, reason: recovery_reason.to_string() });
    subscores.insert("durationAdequacy".to_string(), SubScore { value: finite_opt(duration_value), reason: duration_reason.to_string() });

    let mut weighted = 0.0;
    let mut sum_w = 0.0;
    for (k, sub) in subscores.iter() {
        if let Some(v) = sub.value {
            weighted += weight_of(k) * v;
            sum_w += weight_of(k);
        }
    }
    let mut total = if sum_w > 0.0 { Some(py_round(weighted / sum_w)) } else { None };

    // Dead channels are excluded from the live-channel means because a constant
    // channel carries no span, recovery or signal information. That exclusion
    // would otherwise *raise* the total, so each dead channel costs a fixed
    // penalty and losing hardware reads as the degradation it is.
    let mut dead_penalty = 0.0;
    if let Some(t) = total {
        if !flags.dead_sensors.is_empty() {
            dead_penalty = (DEAD_SENSOR_PENALTY * flags.dead_sensors.len() as f64).min(100.0);
            total = Some(py_round((t - dead_penalty).max(0.0)));
        }
    }
    let badge = match total {
        None => "Unknown",
        Some(t) if t >= 90.0 => "Excellent",
        Some(t) if t >= 75.0 => "Good",
        Some(t) if t >= 50.0 => "Fair",
        Some(_) => "Poor",
    };

    if !flags.dead_sensors.is_empty() {
        notes.push(format!("Dead sensors (cv < 0.001): {}", flags.dead_sensors.join(", ")));
        notes.push(format!(
            "{} dead channel(s) ({}) reduced the total by {}.",
            flags.dead_sensors.len(),
            flags.dead_sensors.join(", "),
            dead_penalty
        ));
    }
    if flags.non_finite_samples > 0 {
        notes.push(format!("{} non-finite values skipped.", flags.non_finite_samples));
    }
    if flags.unsorted_rows {
        notes.push("Rows were out of order and were sorted.".to_string());
    }
    if !rate_declared {
        notes.push("Sampling rate inferred from median gap; verify against hardware.".to_string());
    }
    if !adc_declared {
        notes.push("adcMax not declared; upper-rail clipping not checked (lower rail only).".to_string());
    }
    if flags.no_baseline {
        notes.push("No baseline; auto-R0 applied and baseline stability scores zero.".to_string());
    }

    QualityReport {
        format: "opensmell-quality".to_string(),
        version: "1".to_string(),
        computed_at: chrono::Utc::now().to_rfc3339(),
        total,
        badge: badge.to_string(),
        subscores,
        flags,
        reasons,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(id: &str, values: Vec<f64>) -> ChannelSeries {
        ChannelSeries::new(id, values)
    }

    const N: usize = 601;

    fn good_time() -> Vec<f64> {
        (0..N).map(|i| i as f64 * 100.0).collect()
    }

    #[test]
    fn pulse_exposure_explicit_baseline() {
        let values: Vec<f64> = (0..N)
            .map(|i| if (250..350).contains(&i) { 1150.0 } else { 1000.0 })
            .collect();
        let params = QualityParams {
            adc_max: Some(1500.0),
            sampling_rate_hz: Some(10.0),
            role: "exposure".into(),
            baseline_source: "explicit".into(),
            ..Default::default()
        };
        let r = compute_quality(&good_time(), &[series("A", values)], &params);

        assert_eq!(r.total, Some(65.0));
        assert_eq!(r.badge, "Fair");
        assert_subscore(&r, "continuity", Some(100.0), "ok");
        assert_subscore(&r, "dynamicRange", Some(100.0), "ok");
        assert_subscore(&r, "saturationFree", Some(100.0), "ok");
        assert_subscore(&r, "baselineStability", Some(0.0), "r0_window_cv_too_high");
        assert_subscore(&r, "signalStrength", Some(26.850699801687096), "ok");
        assert_subscore(&r, "recoveryCompleteness", Some(100.0), "ok");
        assert_subscore(&r, "durationAdequacy", Some(100.0), "ok");
        assert!(r.notes.is_empty());
        assert!(r.flags.dead_sensors.is_empty());
        assert!(!r.flags.used_default_adc_max);
        assert!(!r.flags.used_median_sampling_rate);
        assert!(!r.flags.no_baseline);
        assert!(!r.flags.empty_recording);
    }

    #[test]
    fn single_csv_irregular_gap() {
        let time = vec![0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 5000.0];
        let values: Vec<f64> = (0..10).map(|i| 1000.0 + 10.0 * i as f64).collect();
        let r = compute_quality(&time, &[series("A", values)], &QualityParams::default());

        assert_eq!(r.total, Some(38.0));
        assert_eq!(r.badge, "Poor");
        assert_subscore(&r, "continuity", Some(88.88888888888889), "irregular_gaps");
        // Robust 5th-95th span minus 3x the sample-to-sample noise floor.
        assert_subscore(&r, "dynamicRange", Some(11.913747634774993), "low_span");
        assert_subscore(&r, "saturationFree", Some(100.0), "ok");
        assert_subscore(&r, "baselineStability", Some(0.0), "no_baseline");
        assert_subscore(&r, "signalStrength", None, "no_exposure_signal");
        assert_subscore(&r, "recoveryCompleteness", None, "no_exposure_signal");
        assert_subscore(&r, "durationAdequacy", Some(1.5000000000000002), "too_short");

        assert_eq!(
            r.reasons.get("dynamicRange").map(|s| s.as_str()),
            Some("channel_span_below_10_percent_of_adc_range_after_noise_correction")
        );
        assert_eq!(
            r.notes,
            vec![
                "samplingRateHz not declared; nominal period taken as the median gap.",
                "Sampling rate inferred from median gap; verify against hardware.",
                "adcMax not declared; upper-rail clipping not checked (lower rail only).",
                "No baseline; auto-R0 applied and baseline stability scores zero.",
            ]
        );
        assert!(r.flags.used_default_adc_max);
        assert!(r.flags.used_median_sampling_rate);
        assert!(r.flags.no_baseline);
    }

    #[test]
    fn dead_channel_excluded() {
        let values = vec![1000.0; N];
        let params = QualityParams {
            sampling_rate_hz: Some(10.0),
            role: "exposure".into(),
            baseline_source: "auto".into(),
            ..Default::default()
        };
        let r = compute_quality(&good_time(), &[series("A", values)], &params);

        // A dead channel costs DEAD_SENSOR_PENALTY off the total, so losing
        // hardware reads as a degradation rather than an improvement.
        assert_eq!(r.total, Some(32.0));
        assert_eq!(r.badge, "Poor");
        assert_eq!(r.flags.dead_sensors, vec!["A"]);
        assert_subscore(&r, "baselineStability", Some(50.0), "auto_r0");
        assert_subscore(&r, "signalStrength", Some(0.0), "ok");
        assert_subscore(&r, "recoveryCompleteness", Some(0.0), "ok");
        assert_eq!(r.notes[0], "Dead sensors (cv < 0.001): A");
        assert!(r.notes.iter().any(|n| n.contains("reduced the total by 12.5")));
    }

    #[test]
    fn triangle_moderate_scores() {
        let values: Vec<f64> = (0..N)
            .map(|i| {
                if i <= 300 {
                    1000.0 + (i as f64 / 300.0) * 100.0
                } else {
                    1000.0 + ((600 - i) as f64 / 300.0) * 100.0
                }
            })
            .collect();
        let params = QualityParams {
            adc_max: Some(1500.0),
            sampling_rate_hz: Some(10.0),
            role: "exposure".into(),
            baseline_source: "explicit".into(),
            ..Default::default()
        };
        let r = compute_quality(&good_time(), &[series("A", values)], &params);

        assert_eq!(r.total, Some(54.0));
        assert_eq!(r.badge, "Fair");
        // Noise-corrected: a perfectly clean linear ramp now scores 59.5 rather
        // than 66.7, because the noise allowance is subtracted from span.
        assert_subscore(&r, "dynamicRange", Some(59.52859547920908), "ok");
        assert_subscore(&r, "baselineStability", Some(44.92246469412503), "ok");
        assert_subscore(&r, "signalStrength", Some(17.291640722335746), "ok");
        assert_subscore(&r, "recoveryCompleteness", Some(4.666666666666741), "ok");
    }

    #[test]
    fn auto_baseline_cap_badge_excellent() {
        let n = 1201usize;
        let time: Vec<f64> = (0..n).map(|i| i as f64 * 100.0).collect();
        let values: Vec<f64> = (0..n)
            .map(|i| if (400..600).contains(&i) { 1160.0 } else { 1000.0 })
            .collect();
        let params = QualityParams {
            adc_max: Some(1500.0),
            sampling_rate_hz: Some(10.0),
            role: "exposure".into(),
            baseline_source: "auto".into(),
            ..Default::default()
        };
        let r = compute_quality(&time, &[series("A", values)], &params);

        assert_eq!(r.total, Some(90.0));
        assert_eq!(r.badge, "Excellent");
        assert_subscore(&r, "baselineStability", Some(50.0), "auto_r0");
        assert_subscore(&r, "signalStrength", Some(100.0), "ok");
        assert_subscore(&r, "recoveryCompleteness", Some(100.0), "ok");
        assert_subscore(&r, "durationAdequacy", Some(100.0), "ok");
    }

    #[test]
    fn serializes_to_panel_shape() {
        let time = vec![0.0, 100.0, 200.0];
        let values = vec![1000.0, 1005.0, 1010.0];
        let r = compute_quality(&time, &[series("A", values)], &QualityParams::default());
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["format"], "opensmell-quality");
        assert_eq!(json["version"], "1");
        assert!(json["computedAt"].is_string());
        assert!(json["total"].is_number());
        assert!(json["badge"].is_string());
        assert!(json["subscores"]["continuity"]["value"].is_number());
        assert_eq!(json["subscores"]["continuity"]["reason"], "ok");
        assert!(json["flags"]["deadSensors"].is_array());
        assert!(json["flags"]["usedDefaultAdcMax"].is_boolean());
        assert!(json["flags"]["usedMedianSamplingRate"].is_boolean());
        assert!(json["flags"]["noBaseline"].is_boolean());
        assert!(json["flags"]["emptyRecording"].is_boolean());
        assert!(json["reasons"].is_object());
        assert!(json["notes"].is_array());
    }

    /// The pinned Python reference (`opensmell`.mox.quality) values are asserted
    /// within ±1 ulp: CPython computes `(v - m)**2` via libm `pow`, which can
    /// differ from Rust's exact `x * x` by one ULP. The integral total and the
    /// reason strings are asserted exactly.
    fn assert_subscore(r: &QualityReport, key: &str, value: Option<f64>, reason: &str) {
        let sub = &r.subscores[key];
        match (sub.value, value) {
            (Some(a), Some(b)) => assert!(
                (a - b).abs() <= (a.abs().max(b.abs()) * 1e-15 + 1e-12),
                "subscore {} value mismatch: {} vs {}",
                key,
                a,
                b
            ),
            (a, b) => assert_eq!(a, b, "subscore {} value mismatch", key),
        }
        assert_eq!(sub.reason, reason, "subscore {} reason mismatch", key);
    }
    // --- Regressions for the fixed defects found in the calibration study ---
    // Mirrors opensmell/tests/test_quality.py. See
    // opensmell/docs/quality-weight-calibration.md.

    #[test]
    fn time_column_in_seconds_is_flagged_not_scored_as_packet_loss() {
        let values: Vec<f64> = (0..N).map(|i| 1000.0 + i as f64).collect();
        let time: Vec<f64> = (0..N).map(|i| i as f64 * 0.1).collect(); // seconds
        let params = QualityParams {
            sampling_rate_hz: Some(10.0),
            ..Default::default()
        };
        let r = compute_quality(&time, &[series("A", values)], &params);

        assert!(r.flags.time_unit_mismatch);
        assert_subscore(&r, "continuity", None, "time_unit_mismatch");
        assert!(r.notes.iter().any(|n| n.contains("milliseconds")));
    }

    #[test]
    fn millisecond_time_column_is_not_flagged() {
        let values: Vec<f64> = (0..N).map(|i| 1000.0 + i as f64).collect();
        let params = QualityParams {
            sampling_rate_hz: Some(10.0),
            ..Default::default()
        };
        let r = compute_quality(&good_time(), &[series("A", values)], &params);

        assert!(!r.flags.time_unit_mismatch);
        assert_subscore(&r, "continuity", Some(100.0), "ok");
    }

    #[test]
    fn gap_jitter_within_tolerance_is_not_a_unit_mismatch() {
        let values: Vec<f64> = (0..N).map(|i| 1000.0 + i as f64).collect();
        // 10 Hz nominal is 100 ms; 1.5x jitter is well inside the plausibility
        // band, so this is jitter rather than a bad time unit.
        let time: Vec<f64> = (0..N)
            .map(|i| i as f64 * 100.0 + if i % 37 == 0 { 30.0 } else { 0.0 })
            .collect();
        let params = QualityParams {
            sampling_rate_hz: Some(10.0),
            ..Default::default()
        };
        let r = compute_quality(&time, &[series("A", values)], &params);

        assert!(!r.flags.time_unit_mismatch);
        assert!(r.subscores.get("continuity").and_then(|s| s.value).is_some());
    }

    #[test]
    fn noise_burst_does_not_raise_dynamic_range() {
        let clean: Vec<f64> = (0..N).map(|i| 1000.0 + (i as f64 / 60.0) * 400.0).collect();
        let noisy: Vec<f64> = (0..N)
            .map(|i| 1000.0 + (i as f64 / 60.0) * 400.0 + if (300..360).contains(&i) { 60.0 } else { 0.0 })
            .collect();
        let mk = |values: Vec<f64>| {
            compute_quality(
                &good_time(),
                &[series("A", values)],
                &QualityParams {
                    adc_max: Some(4095.0),
                    sampling_rate_hz: Some(10.0),
                    role: "exposure".into(),
                    baseline_source: "explicit".into(),
                    ..Default::default()
                },
            )
        };
        let q_clean = mk(clean);
        let q_noisy = mk(noisy);

        let d_clean = q_clean.subscores.get("dynamicRange").unwrap().value.unwrap();
        let d_noisy = q_noisy.subscores.get("dynamicRange").unwrap().value.unwrap();
        assert!(
            d_noisy <= d_clean,
            "interference must not raise dynamic range: {d_noisy} > {d_clean}"
        );
    }

    #[test]
    fn saturation_does_not_raise_dynamic_range() {
        // A pulse that rises and returns to baseline, so the recovery subscore
        // is healthy in both conditions and cannot confound the comparison.
        // Peak stays well below the 4095 rail in the control.
        let in_range = |i: usize| {
            let phase = if (250..350).contains(&i) { 1.0 } else { 0.0 };
            1000.0 + phase * 800.0
        };
        let clean: Vec<f64> = (0..N).map(in_range).collect();
        let clipped: Vec<f64> = (0..N)
            .map(|i| if (280..320).contains(&i) { 4095.0 } else { in_range(i) })
            .collect();
        assert!(clean.iter().all(|v| *v < 4095.0), "control must not clip");
        let mk = |values: Vec<f64>| {
            compute_quality(
                &good_time(),
                &[series("A", values)],
                &QualityParams {
                    adc_max: Some(4095.0),
                    sampling_rate_hz: Some(10.0),
                    role: "exposure".into(),
                    baseline_source: "explicit".into(),
                    ..Default::default()
                },
            )
        };
        let q_clean = mk(clean);
        let q_clipped = mk(clipped);

        let d_clean = q_clean.subscores.get("dynamicRange").unwrap().value.unwrap();
        let d_clipped = q_clipped.subscores.get("dynamicRange").unwrap().value.unwrap();
        assert!(
            d_clipped <= d_clean,
            "clipping to the rail must not widen the span: {d_clipped} > {d_clean}"
        );
        let s_clean = q_clean.subscores.get("saturationFree").unwrap().value.unwrap();
        let s_clipped = q_clipped.subscores.get("saturationFree").unwrap().value.unwrap();
        assert!(s_clipped < s_clean, "clipping must be visible in saturationFree");
        assert!(
            q_clipped.total.unwrap() < q_clean.total.unwrap(),
            "a saturated recording must not outscore a clean one: {} vs {}",
            q_clipped.total.unwrap(),
            q_clean.total.unwrap()
        );
    }

    #[test]
    fn real_exposure_still_earns_dynamic_range() {
        let values: Vec<f64> = (0..N).map(|i| 1000.0 + (i as f64 / 60.0) * 400.0).collect();
        let r = compute_quality(
            &good_time(),
            &[series("A", values)],
            &QualityParams {
                adc_max: Some(4095.0),
                sampling_rate_hz: Some(10.0),
                role: "exposure".into(),
                baseline_source: "explicit".into(),
                ..Default::default()
            },
        );

        let d = r.subscores.get("dynamicRange").unwrap().value.unwrap();
        assert!(d > 0.0, "the noise correction must not flatten a real ramp to zero");
        assert_eq!(r.subscores.get("dynamicRange").unwrap().reason, "ok");
    }

    #[test]
    fn dead_sensor_penalty_scales_with_channel_count() {
        let live: Vec<f64> = (0..N).map(|i| 1000.0 + (i as f64 / 60.0) * 400.0).collect();
        let params = QualityParams {
            adc_max: Some(4095.0),
            sampling_rate_hz: Some(10.0),
            role: "exposure".into(),
            baseline_source: "explicit".into(),
            ..Default::default()
        };
        let both_live = compute_quality(
            &good_time(),
            &[series("A", live.clone()), series("B", live.clone())],
            &params,
        );
        let one_dead = compute_quality(
            &good_time(),
            &[series("A", live.clone()), series("B", vec![1000.0; N])],
            &params,
        );
        let two_dead = compute_quality(&good_time(), &[series("A", vec![1000.0; N]), series("B", vec![1000.0; N])], &params);

        assert!(two_dead.total.unwrap() < one_dead.total.unwrap());
        assert!(one_dead.total.unwrap() < both_live.total.unwrap());
    }
}

#[cfg(test)]
mod parity {
    //! Numeric parity spot-checks against the Python implementation.
    //! Expected values were produced by opensmell/tests/test_quality.py
    //! fixtures on the same inputs.

    use super::*;

    /// Note: this ramp reaches 5000 against a declared `adc_max` of 4095, so
    /// roughly the top fifth of it sits on the converter rail. That is
    /// deliberate -- it exercises the clipping path on every parity check --
    /// but it means these totals are lower than a clean in-range recording's.
    fn ramp() -> Vec<f64> {
        (0..601).map(|i| 1000.0 + (i as f64 / 60.0) * 400.0).collect()
    }

    fn params() -> QualityParams {
        QualityParams {
            adc_max: Some(4095.0),
            sampling_rate_hz: Some(10.0),
            role: "exposure".into(),
            baseline_source: "explicit".into(),
            r0_samples: Some(30),
            ..Default::default()
        }
    }

    #[test]
    fn parity_dynamic_range_clean_and_noisy() {
        let noisy: Vec<f64> = (0..601)
            .map(|i| 1000.0 + (i as f64 / 60.0) * 400.0 + if (300..360).contains(&i) { 60.0 } else { 0.0 })
            .collect();
        let clean = compute_quality(&good_time_for_parity(), &[ChannelSeries::new("A", ramp())], &params());
        let dirty = compute_quality(&good_time_for_parity(), &[ChannelSeries::new("A", noisy)], &params());

        // Python: clean=100.0000 noisy=100.0000
        assert_eq!(clean.subscores.get("dynamicRange").unwrap().value, Some(100.0));
        assert_eq!(dirty.subscores.get("dynamicRange").unwrap().value, Some(100.0));
    }

    #[test]
    fn parity_dead_sensor_penalty() {
        let t = good_time_for_parity();
        let both_live = compute_quality(&t, &[ChannelSeries::new("A", ramp()), ChannelSeries::new("B", ramp())], &params());
        let one_dead = compute_quality(&t, &[ChannelSeries::new("A", ramp()), ChannelSeries::new("B", vec![1000.0; 601])], &params());
        let two_dead = compute_quality(&t, &[ChannelSeries::new("A", vec![1000.0; 601]), ChannelSeries::new("B", vec![1000.0; 601])], &params());

        // Python: live=53 one=42 two=30
        assert_eq!(both_live.total, Some(53.0));
        assert_eq!(one_dead.total, Some(42.0));
        assert_eq!(two_dead.total, Some(30.0));
    }

    #[test]
    fn parity_jitter_continuity() {
        let t: Vec<f64> = (0..601).map(|i| i as f64 * 100.0 + if i % 37 == 0 { 30.0 } else { 0.0 }).collect();
        let r = compute_quality(&t, &[ChannelSeries::new("A", ramp())], &params());
        // Python: 94.5
        assert_eq!(r.subscores.get("continuity").unwrap().value, Some(94.5));
    }

    fn good_time_for_parity() -> Vec<f64> {
        (0..601).map(|i| i as f64 * 100.0).collect()
    }
}
