//! R0 baseline window contract (electronic-nose/SAMPLING_CONTRACT.md, "The R0
//! window contract").
//!
//! `R0_WINDOW_DEFAULT` (and therefore `DEFAULT_R0_SAMPLES`) means "nothing
//! declared" and resolves to `clamp(floor(0.15 * n), 5, 30)`. These mirror the
//! Python and JS suites case-for-case so the three SDKs cannot drift apart again.

use opensmell::features::health;
use opensmell::{
    r0_window_samples, Baseline, R0_WINDOW_DEFAULT, R0_WINDOW_FRACTION, R0_WINDOW_MAX_SAMPLES,
    R0_WINDOW_MIN_SAMPLES,
};

const CADENCES: [f64; 4] = [1.0, 2.0, 10.0, 100.0];
// 0.15 * n is inside [5, 30] exactly for 34 <= n <= 200.
const FRACTION_REGION: [usize; 3] = [40, 80, 160];

/// Flat clean-air plateau then a monotone exposure, sampled at `fs` Hz.
/// 60 s long, so the sample count differs by 100x across CADENCES.
fn exposure(fs: f64, plateau_s: f64) -> Vec<Vec<f64>> {
    let duration_s = 60.0;
    let n = (duration_s * fs).round() as usize + 1;
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let ramp = ((t - plateau_s) / 10.0).clamp(0.0, 1.0);
            vec![1000.0 + 50.0 * ramp]
        })
        .collect()
}

fn median_of(series: &[f64]) -> f64 {
    let mut v: Vec<f64> = series.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

fn column(window: &[Vec<f64>]) -> Vec<f64> {
    window.iter().map(|row| row[0]).collect()
}

#[test]
fn window_is_a_floored_capped_fraction() {
    // Floor: a 20-sample window would be 3 samples at 15% — too few for a stable
    // median — so the floor of 5 binds.
    assert_eq!(
        r0_window_samples(1, R0_WINDOW_DEFAULT),
        R0_WINDOW_MIN_SAMPLES
    );
    assert_eq!(
        r0_window_samples(20, R0_WINDOW_DEFAULT),
        R0_WINDOW_MIN_SAMPLES
    );
    // Fraction region, 34 <= n <= 200.
    assert_eq!(r0_window_samples(60, R0_WINDOW_DEFAULT), 9);
    assert_eq!(r0_window_samples(100, R0_WINDOW_DEFAULT), 15); // DEFAULT_WINDOW_SIZE
    assert_eq!(r0_window_samples(200, R0_WINDOW_DEFAULT), 30);
    // Ceiling: an unbounded 15% of 600 would be 90 samples and would swallow the
    // onset on a long recording.
    assert_eq!(
        r0_window_samples(600, R0_WINDOW_DEFAULT),
        R0_WINDOW_MAX_SAMPLES
    );
    assert_eq!(
        r0_window_samples(18000, R0_WINDOW_DEFAULT),
        R0_WINDOW_MAX_SAMPLES
    );
}

#[test]
fn declared_window_wins_verbatim() {
    assert_eq!(r0_window_samples(600, 15), 15);
    assert_eq!(r0_window_samples(60, 180), 180);
    // 0 is the "not declared" sentinel, matching Python's None and JS's undefined.
    assert_eq!(r0_window_samples(100, 0), 15);
}

#[test]
fn window_spans_15_percent_of_seconds_at_every_cadence() {
    // A fixed sample count is cadence-*dependent*: 15 samples is 1.5 s at 10 Hz
    // and 15 s at 1 Hz. In the fraction region the window covers 0.15 * T seconds
    // at every rate, because the sample count grows with the rate.
    for fs in CADENCES {
        for n in FRACTION_REGION {
            let window = r0_window_samples(n, R0_WINDOW_DEFAULT) as f64;
            let duration_s = n as f64 / fs;
            let covered_s = window / fs;
            assert!(
                (covered_s - R0_WINDOW_FRACTION * duration_s).abs() < 1e-9,
                "{n} samples at {fs} Hz: window covers {covered_s} s of a {duration_s} s recording"
            );
        }
    }
}

#[test]
fn clamps_are_documented_cadence_dependence() {
    // 60 s at 1 Hz is 61 samples: the fraction binds and the window covers 9 s.
    // The same 60 s at 100 Hz is 6001 samples: the ceiling binds, 0.3 s. The
    // clamps are sample counts, so they are cadence-dependent by construction.
    let slow_s = r0_window_samples(61, R0_WINDOW_DEFAULT) as f64 / 1.0;
    let fast_s = r0_window_samples(6001, R0_WINDOW_DEFAULT) as f64 / 100.0;
    assert!((slow_s - 9.0).abs() < 1e-9, "slow_s = {slow_s}");
    assert!((fast_s - 0.3).abs() < 1e-9, "fast_s = {fast_s}");
    assert!((slow_s / fast_s - 30.0).abs() < 1e-9);
}

#[test]
fn declared_window_restores_cadence_invariance() {
    let duration_s = 60.0;
    for fs in CADENCES {
        let n = (duration_s * fs).round() as usize + 1;
        let declared = (R0_WINDOW_FRACTION * duration_s * fs).round() as usize;
        let covered_s = r0_window_samples(n, declared) as f64 / fs;
        assert!(
            (covered_s - R0_WINDOW_FRACTION * duration_s).abs() < 1e-9,
            "{fs} Hz: declared window covers {covered_s} s"
        );
    }
}

#[test]
fn baseline_reads_the_same_physical_level_at_every_cadence() {
    // 12 s plateau, so every resolved window lies wholly inside the baseline and
    // all four cadences must read the same R0. Under the old fixed 15-sample
    // window the 1 Hz and 100 Hz cases would have disagreed by the response
    // amplitude.
    for fs in CADENCES {
        let bl = Baseline::from_samples(&exposure(fs, 12.0));
        assert!(
            (bl.r0[0] - 1000.0).abs() < 1e-9,
            "{fs} Hz: R0 = {}",
            bl.r0[0]
        );
    }
}

#[test]
fn short_plateau_needs_a_declared_window() {
    // The limit the fraction does not fix. With a 1 s plateau in a 60 s recording
    // the plateau is shorter than 15% of the recording, so no fraction-based
    // window can find it: the 1 Hz window reaches 9 s and lands on the ramp.
    // Rule 5's declared window is the only thing that recovers it, at every rate.
    let at_1hz = column(&exposure(1.0, 1.0));
    let at_100hz = column(&exposure(100.0, 1.0));
    assert!(
        (Baseline::from_samples_with_window(&exposure(1.0, 1.0), 15).r0[0] - 1030.0).abs() < 1e-9
    );
    assert!(
        (Baseline::from_samples_with_window(&exposure(100.0, 1.0), 15).r0[0] - 1000.0).abs() < 1e-9
    );
    assert!((Baseline::from_samples(&exposure(1.0, 1.0)).r0[0] - 1015.0).abs() < 1e-9);
    assert!((Baseline::from_samples(&exposure(100.0, 1.0)).r0[0] - 1000.0).abs() < 1e-9);
    for fs in CADENCES {
        let declared = fs.round() as usize; // round(1 s * fs)
        let r0 = Baseline::from_samples_with_window(&exposure(fs, 1.0), declared).r0[0];
        assert!(
            (r0 - 1000.0).abs() < 1e-9,
            "{fs} Hz declared {declared}: R0 = {r0}"
        );
    }
    // Keep the two unused bindings honest about what they are.
    assert_eq!(at_1hz.len(), 61);
    assert_eq!(at_100hz.len(), 6001);
}

#[test]
fn baseline_n_samples_reports_the_resolved_window() {
    for fs in CADENCES {
        let window = exposure(fs, 12.0);
        let expected = r0_window_samples(window.len(), R0_WINDOW_DEFAULT).min(window.len());
        assert_eq!(
            Baseline::from_samples(&window).n_samples,
            expected,
            "{fs} Hz"
        );
        assert_eq!(Baseline::from_samples_with_window(&window, 7).n_samples, 7);
    }
}

#[test]
fn declared_window_is_used_by_the_health_block_end_to_end() {
    // Samples 0-6 sit at 1000 and samples 7-14 at 2000, so the median and the
    // spread both move when the window crosses index 7. `sensitivity_decay` is
    // divided by baseline.r0, so it is the feature the old 15-vs-15% split used
    // to disagree on.
    let series: Vec<Vec<f64>> = (0..101)
        .map(|i| {
            vec![match i {
                0..=6 => 1000.0,
                7..=14 => 2000.0,
                _ => 1500.0,
            }]
        })
        .collect();

    let narrow = Baseline::from_samples_with_window(&series, 7);
    let wide = Baseline::from_samples_with_window(&series, 15);
    let default = Baseline::from_samples(&series);
    assert!((narrow.r0[0] - 1000.0).abs() < 1e-9);
    assert!((wide.r0[0] - 2000.0).abs() < 1e-9);
    assert!(
        (default.r0[0] - wide.r0[0]).abs() < 1e-12,
        "n=101 resolves to 15"
    );
    // A 7-sample window is perfectly flat, so its baseline std is zero.
    assert!(narrow.std[0].abs() < 1e-12);
    assert!(wide.std[0] > 0.0);

    let narrow_health = health::extract_window(&series, &narrow, 10.0).unwrap();
    let wide_health = health::extract_window(&series, &wide, 10.0).unwrap();
    // names() order is [noise_floor, sensitivity_decay, drift_rate, hysteresis].
    // sensitivity_decay = (last_third_mean - first_third_mean) / r0, so the
    // numerator is fixed by the series and only the divisor moves. A 1000-vs-2000
    // window therefore halves the feature — the exact mechanism behind the
    // reported cross-SDK sensitivity_decay gap (R0 ratio 0.9535).
    let decay_narrow = narrow_health[1];
    let decay_wide = wide_health[1];
    assert!(
        decay_narrow < 0.0,
        "first third sits above the last third here"
    );
    assert!(
        (decay_narrow - 2.0 * decay_wide).abs() < 1e-12,
        "narrow {decay_narrow} vs wide {decay_wide}"
    );
}

#[test]
fn a_floor_of_five_never_panics_on_a_short_recording() {
    for n in 1..=8usize {
        let window: Vec<Vec<f64>> = (0..n).map(|i| vec![1000.0 + i as f64]).collect();
        let bl = Baseline::from_samples(&window);
        assert_eq!(bl.n_samples, n.min(R0_WINDOW_MIN_SAMPLES));
        assert!(bl.r0[0] > 0.0);
    }
}

#[test]
fn median_helper_matches_the_baseline_median() {
    // The test-side median must agree with the one under test, or the assertions
    // above would be measuring this file rather than the SDK.
    let odd = [3.0, 1.0, 2.0];
    let even = [4.0, 1.0, 3.0, 2.0];
    assert!(
        (median_of(&odd)
            - Baseline::from_samples_with_window(&vec![vec![3.0], vec![1.0], vec![2.0]], 3).r0[0])
            .abs()
            < 1e-12
    );
    assert!(
        (median_of(&even)
            - Baseline::from_samples_with_window(
                &vec![vec![4.0], vec![1.0], vec![3.0], vec![2.0]],
                4
            )
            .r0[0])
            .abs()
            < 1e-12
    );
}
