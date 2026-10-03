//! Cadence-invariance proofs for the cadence-implicit time fix (Track 2c).
//!
//! The same physical event must yield the same *seconds-based* features whether
//! it was recorded at 10 Hz or 1 Hz. Crossing-time features (rise/decay/latency/
//! ttp) are sample-resolved, so the sharp-step cases below are exact; the
//! framework AUC (trapezoid over a piecewise-linear pulse) is exact because the
//! trapezoid rule integrates linear segments exactly; drift rate is a per-second
//! least-squares slope and is exact. The coarse sum-based classification AUC
//! carries an honest `1/fs` Riemann boundary term and is asserted within it.

use opensmell::framework::framework_window_features;
use opensmell::{
    Baseline,
    FailSafeSystem,
    FeatureGroup,
    extract_window_features_by_mode,
    extract_window_features_with_sr,
    paradigm_window_features,
};

fn baseline() -> Baseline {
    Baseline { r0: vec![1000.0], n_samples: 3, std: vec![1.0] }
}

/// Sample a time-domain signal `f(t)` on a uniform grid of `fs` Hz covering
/// `0..=duration`. Returns a single-channel window.
fn window_of(fs: f64, duration: f64, f: &dyn Fn(f64) -> f64) -> Vec<Vec<f64>> {
    let n = (duration * fs).round() as usize;
    (0..=n).map(|i| vec![f(i as f64 / fs)]).collect()
}

/// Piecewise-linear response pulse (ramp up, plateau, ramp down). Trapezoid
/// integration of this pulse is exact at any sampling grid that hits the knots.
fn pulse(t: f64) -> f64 {
    if t < 2.0 {
        1000.0
    } else if t < 4.0 {
        1000.0 + 250.0 * (t - 2.0)
    } else if t < 6.0 {
        1500.0
    } else if t < 8.0 {
        1500.0 - 250.0 * (t - 6.0)
    } else {
        1000.0
    }
}

#[test]
fn framework_auc_is_a_true_time_integral() {
    let w10 = window_of(10.0, 10.0, &pulse);
    let w1 = window_of(1.0, 10.0, &pulse);

    let f10 = framework_window_features(&w10, 3, 10.0).unwrap();
    let f1 = framework_window_features(&w1, 3, 1.0).unwrap();

    // f[5] = ch0 da auc, f[31] = global_total_auc. The exact pulse area is 2.0
    // (norm seconds) at any cadence. The old unit-spaced value grew ~10x at
    // 10 Hz; now both are ~2.0.
    assert!((f10[5] - 2.0).abs() < 1e-9, "aucc10 = {}", f10[5]);
    assert!((f1[5] - 2.0).abs() < 1e-9, "aucc1 = {}", f1[5]);
    assert!((f10[5] - f1[5]).abs() < 1e-9, "AUC must be cadence-invariant");
    assert!((f10[31] - f1[31]).abs() < 1e-9, "global AUC must be cadence-invariant");

    // Crossing-based rise/decay match within one 1 Hz sample per crossing.
    let (r10, d10) = (f10[10], f10[6]);
    let (r1, d1) = (f1[10], f1[6]);
    assert!(r10 > 0.0 && d10 > 0.0, "pulse must be rising/decaying");
    assert!((r10 - r1).abs() < 1.2, "rise_time 10Hz={r10}s 1Hz={r1}s");
    assert!((d10 - d1).abs() < 1.2, "decay_time 10Hz={d10}s 1Hz={d1}s");
}

#[test]
fn drift_rates_are_per_second_and_cadence_invariant() {
    // Linear ramp: v(t) = 1000 + 50*t -> raw slope 50/s, normalized 0.05/s.
    let ramp = |t: f64| 1000.0 + 50.0 * t;
    let w10 = window_of(10.0, 10.0, &ramp);
    let w1 = window_of(1.0, 10.0, &ramp);

    let b = baseline();
    let g = [FeatureGroup::Anomaly, FeatureGroup::Health];
    let f10 = extract_window_features_with_sr(&w10, &b, &g, 10.0).unwrap();
    let f1 = extract_window_features_with_sr(&w1, &b, &g, 1.0).unwrap();

    // ch0 anomaly drift_rate (position 0): normalized slope per second.
    assert!((f10[0] - 0.05).abs() < 1e-6, "anomaly drift10 = {}", f10[0]);
    assert!((f1[0] - 0.05).abs() < 1e-6, "anomaly drift1 = {}", f1[0]);
    assert!((f10[0] - f1[0]).abs() < 1e-6, "anomaly drift rate cadence-invariant");

    // ch0 health drift_rate (position 10): raw slope per second.
    assert!((f10[10] - 50.0).abs() < 1e-6, "health drift10 = {}", f10[10]);
    assert!((f1[10] - 50.0).abs() < 1e-6, "health drift1 = {}", f1[10]);
    assert!((f10[10] - f1[10]).abs() < 1e-6, "health drift rate cadence-invariant");
}

#[test]
fn kinetics_and_latency_are_seconds_not_sample_counts() {
    // Instantaneous step event at t=2s: rise/decay are 0s, ttp=2s, latency=2s.
    let step = |t: f64| if (2.0..8.0).contains(&t) { 1500.0 } else { 1000.0 };
    let w10 = window_of(10.0, 10.0, &step);
    let w1 = window_of(1.0, 10.0, &step);

    let b = baseline();
    let g = [FeatureGroup::Kinetics, FeatureGroup::Temporal];
    let f10 = extract_window_features_with_sr(&w10, &b, &g, 10.0).unwrap();
    let f1 = extract_window_features_with_sr(&w1, &b, &g, 1.0).unwrap();

    // A single 10 Hz sample is 0.1 s; the features must all be identical in
    // seconds. Old (sample-count) values were 10x offset at 10 Hz.
    // Kinetics block: 0=rise, 1=decay, 2=peak, 3=ttp; temporal block: 3=latency.
    assert_eq!(f10[0], 0.0); // rise_time at a sharp step is 0 s
    assert_eq!(f1[0], 0.0);
    assert_eq!(f10[1], 0.0); // decay_time at a sharp step is 0 s
    assert_eq!(f1[1], 0.0);
    for (name, idx) in [("time_to_peak", 3), ("response_latency", 9)] {
        assert!((f10[idx] - 2.0).abs() < 1e-9, "{name}: 10Hz={} expected 2.0s", f10[idx]);
        assert!((f1[idx] - 2.0).abs() < 1e-9, "{name}: 1Hz={} expected 2.0s", f1[idx]);
        assert!((f10[idx] - f1[idx]).abs() < 1e-9, "{name} must be cadence-invariant");
    }
}

#[test]
fn classification_auc_scales_with_1_over_fs() {
    // Plateau at 0.2 normalized over a 10 s window. The unit-spaced sum grows
    // ~10x at 10 Hz; /fs must cancel that. A soft Riemann boundary term of one
    // low-cadence sample (~0.2) remains between the two discrete grids.
    let plateau = |_t: f64| 1200.0;
    let b = baseline();
    let g = [FeatureGroup::Classification];
    let f10 = extract_window_features_with_sr(&window_of(10.0, 10.0, &plateau), &b, &g, 10.0).unwrap();
    let f1 = extract_window_features_with_sr(&window_of(1.0, 10.0, &plateau), &b, &g, 1.0).unwrap();

    let (a10, a1) = (f10[2], f1[2]);
    assert!(a10 > 0.0 && a1 > 0.0);
    assert!((a10 - a1).abs() <= 0.21, "auc10={a10} auc1={a1} differ by > 1 low-cadence sample");
    // Both approximate the exact integral 0.2 * 10s = 2.0.
    assert!((a10 - 2.0).abs() < 0.1, "auc10 = {a10}");
    assert!((a1 - 2.0).abs() < 0.3, "auc1 = {a1}");
}

#[test]
fn paradigm_auc_and_mean_slope_are_per_second_and_cadence_invariant() {
    // The paradigm features (model-training path) must match the Python
    // reference `compute_window_paradigms`: mean_slope is scaled by sr and auc
    // is the trapezoid time integral / sr. The same physical pulse sampled at
    // 10 Hz and 1 Hz therefore yields the same per-second slope and AUC.
    let w10 = window_of(10.0, 10.0, &pulse);
    let w1 = window_of(1.0, 10.0, &pulse);

    let f10 = paradigm_window_features(&w10, 3, 10.0);
    let f1 = paradigm_window_features(&w1, 3, 1.0);

    // Index 2 = ch0 mean_slope, index 3 = ch0 auc (5 features per channel).
    assert!((f10[2] - f1[2]).abs() < 1e-9, "mean_slope 10Hz={} 1Hz={} must be cadence-invariant", f10[2], f1[2]);
    assert!(f10[2] > 0.0, "pulse must have a nonzero per-second slope");
    // AUC of the piecewise-linear pulse (ramp 0.5 + plateau 1.0 + ramp 0.5)
    // is 2.0 at any cadence when divided by sr.
    assert!((f10[3] - f1[3]).abs() < 1e-9, "auc 10Hz={} 1Hz={} must be cadence-invariant", f10[3], f1[3]);
    assert!((f10[3] - 2.0).abs() < 1e-9, "auc10 = {}", f10[3]);

    // Dispatch path must scale identically (sr threaded through).
    let d10 = extract_window_features_by_mode(&w10, "paradigm", 3, 10.0);
    let d1 = extract_window_features_by_mode(&w1, "paradigm", 3, 1.0);
    assert!((d10[3] - d1[3]).abs() < 1e-9, "dispatch AUC must be cadence-invariant");
}

#[test]
fn failsafe_detect_with_dt_at_reference_cadence_matches_plain_detect() {
    // The new `FailSafeSystem::detect_with_dt(reading, dt_s)` path must be
    // byte-identical to `detect()` when dt_s equals the reference period
    // (0.1 s) — the same parity contract the anomaly engine enforces.
    let mut a = FailSafeSystem::new(1);
    let mut b = FailSafeSystem::new(1);

    let cal = vec![vec![1000.0], vec![1000.1], vec![999.9], vec![1000.0]];
    a.calibrate_baseline(&cal).unwrap();
    b.calibrate_baseline(&cal).unwrap();

    // Mild step: small deviation, well inside the adaptive threshold.
    let probe = vec![1010.0];
    let r1 = a.detect(&probe).unwrap();
    let r2 = b.detect_with_dt(&probe, 0.1).unwrap();
    assert_eq!(r1.is_anomaly, r2.is_anomaly);
    assert!((r1.raw_score - r2.raw_score).abs() < 1e-12);

    // Cadence-dependent path: a 6 s gap (TADI-like) must still score without
    // error and must not blow up the state filter.
    let mut c = FailSafeSystem::new(1);
    c.calibrate_baseline(&cal).unwrap();
    let r3 = c.detect_with_dt(&probe, 6.0).unwrap();
    assert!(r3.raw_score.is_finite());
}

#[test]
fn stuck_zero_flags_after_same_wall_clock_at_any_cadence() {
    use opensmell::STUCK_ZERO_SECONDS;
    let cal = vec![vec![1000.0], vec![1000.1], vec![999.9], vec![1000.0]];

    // Calibrated device streams a few healthy readings, then the channel pins
    // at zero. The failure must be reported ~STUCK_ZERO_SECONDS of wall-clock
    // after the first zero, at any cadence.
    fn trigger_time_s(fs: f64, cal: &[Vec<f64>]) -> f64 {
        let mut s = FailSafeSystem::new(1);
        s.calibrate_baseline(cal).unwrap();
        let dt = 1.0 / fs;
        let healthy = 4usize;
        let total = healthy + (STUCK_ZERO_SECONDS / dt).ceil() as usize + 3;
        for k in 0..total {
            let reading = if k >= healthy { vec![0.0] } else { vec![1000.0 + (k % 3) as f64 * 0.1] };
            let r = s.detect_with_dt(&reading, dt).unwrap();
            if r.sensor_failures.iter().any(|f| f.failure_type == "stuck_zero") {
                return k as f64 * dt;
            }
        }
        f64::MAX
    }

    for fs in [2.0, 10.0, 25.0] {
        let t = trigger_time_s(fs, &cal);
        let expected = 4.0 / fs + STUCK_ZERO_SECONDS; // first zero at k=4
        assert!(t != f64::MAX, "{fs} Hz never flagged stuck-zero");
        // Flag within ~2 inter-sample steps of the physical mark.
        assert!(
            (t - expected).abs() <= 2.0 / fs + 1e-6,
            "{fs} Hz flagged at {t:.3}s, expected ~{expected:.3}s (±{:.3}s)",
            2.0 / fs
        );
    }
}

#[test]
fn alert_escalation_tracks_wall_clock_not_sample_count() {
    use opensmell::EMERGENCY_SECONDS;
    let cal = vec![vec![1000.0], vec![1000.1], vec![999.9], vec![1000.0]];
    // A steep linear ramp keeps every reading anomalous (each is a fresh
    // deviation from a chasing baseline), so escalation accumulates wall-clock.
    // The alert must reach emergency after EMERGENCY_SECONDS *of real time*
    // at any cadence — not after a sample count.
    fn escalate_time_s(fs: f64, cal: &[Vec<f64>]) -> f64 {
        let mut s = FailSafeSystem::new(1);
        s.calibrate_baseline(cal).unwrap();
        let dt = 1.0 / fs;
        for k in 0..((EMERGENCY_SECONDS + 3.0) * fs) as usize {
            let x = vec![50000.0 + 200000.0 * (k as f64 * dt)];
            let r = s.detect_with_dt(&x, dt).unwrap();
            if r.alert_name == "emergency" {
                return k as f64 * dt;
            }
        }
        f64::MAX
    }
    for fs in [2.0, 10.0, 25.0] {
        let t = escalate_time_s(fs, &cal);
        assert!(t != f64::MAX, "{fs} Hz never reached emergency");
        // Reaches EMERGENCY_SECONDS within one inter-sample step of the mark.
        assert!(
            (t - EMERGENCY_SECONDS).abs() <= 1.5 / fs + 1e-6,
            "{fs} Hz emergency at {t:.3}s, expected ~{EMERGENCY_SECONDS:.3}s (±{:.3}s)",
            1.5 / fs
        );
    }
    // And the identical physical window yields warning (0.2 s) / critical
    // (0.5 s) on a reference 10 Hz run — the names map to duration, not count.
    let mut s = FailSafeSystem::new(1);
    s.calibrate_baseline(&cal).unwrap();
    let mut level = 0u8;
    for k in 0..12 {
        let x = vec![50000.0 + 200000.0 * (k as f64 * 0.1)];
        level = s.detect_with_dt(&x, 0.1).unwrap().alert_level;
    }
    assert!(s.anomalous_seconds >= 1.0, "accumulated {} s", s.anomalous_seconds);
    assert_eq!(level, 3);
}