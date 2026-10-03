//! Wave-3 replay harness integration tests (synthetic sweep, no hardware).
//! Verifies the engine end-to-end reports confusion metrics + latency from a
//! labeled synthetic stream, per `anomaly-engine-design.md` §11.3.
use opensmell::anomaly::dual::DualKalmanEngine;
use opensmell::anomaly::replay::{ConfusionBin, Sample, SampleTruth};

fn baseline(c: usize) -> Vec<Vec<f64>> {
    (0..80)
        .map(|i| {
            let n = i as f64;
            (0..c).map(|ch| 1.0 + ch as f64 + 0.05 * (n * 1.7).sin()).collect()
        })
        .collect()
}

/// Synthetic stream: 300 normal samples, then a +2.5 sustained step for 200,
/// then 300 normal again. One labeled anomaly event at the step onset.
fn synthetic_sweep(c: usize) -> Vec<Sample> {
    let mut out = Vec::new();
    // Quiet phase.
    for i in 0..300 {
        let phase = i as f64;
        let reading: Vec<f64> = (0..c)
            .map(|ch| 1.0 + ch as f64 + 0.05 * (phase * 1.7).sin())
            .collect();
        out.push(Sample::new(reading));
    }
    // Sustained step: a genuine anomaly event (onset at sample 300).
    let onset = 300u64;
    for k in 0..200 {
        let phase = (k as f64 + 300.0) * 1.7;
        let reading: Vec<f64> = (0..c)
            .map(|ch| 1.0 + ch as f64 + 2.5 + 0.05 * phase.sin())
            .collect();
        let truth = if k == 0 {
            SampleTruth { is_anomaly: true, onset: Some(onset) }
        } else if k < 200 {
            SampleTruth { is_anomaly: true, onset: None }
        } else {
            SampleTruth::default()
        };
        out.push(Sample::new(reading).labeled(truth));
    }
    // Quiet phase after the event.
    for i in 0..300 {
        let phase = (i as f64 + 500.0) * 1.7;
        let reading: Vec<f64> = (0..c)
            .map(|ch| 1.0 + ch as f64 + 0.05 * phase.sin())
            .collect();
        out.push(Sample::new(reading));
    }
    out
}

#[test]
fn replay_detects_labeled_step_and_reports_metrics() {
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    // Warm-up samples so the filter settles before the event.
    let warmup: Vec<Sample> = (0..30)
        .map(|_| Sample::new(vec![1.0]))
        .collect();
    let _ = eng.replay_slice(&warmup).unwrap();

    let sweep = synthetic_sweep(1);
    let report = eng.replay_slice(&sweep).unwrap();
    let m = report.metrics;

    // The labeled event must be seen.
    assert_eq!(m.n_events, 1, "one labeled event in the sweep");
    assert!(
        m.true_positives > 0,
        "step must produce true positives (got tp={}, fn={})",
        m.true_positives,
        m.false_negatives
    );
    assert!(
        m.detected_events >= 1,
        "the event must be detected (detected {})",
        m.detected_events
    );
    // Latency: detection 0..~100 samples after onset is the design window.
    let lat = m.latency_samples.expect("latency computed").max(0.0);
    assert!(
        (0.0..300.0).contains(&lat),
        "detection latency {lat} must be within the step's footprint"
    );
    // False-positive rate in the quiet tail should be small.
    assert!(
        m.fpr() < 0.1,
        "quiet baseline must not alarm much (fpr {:.3})",
        m.fpr()
    );
    // PPV: with a single clean event the flagged positives are mostly true.
    assert!(
        m.ppv() > 0.5,
        "positive predictive value must stay meaningful (ppv {:.3})",
        m.ppv()
    );
}

#[test]
fn replay_marks_bins_and_keeps_verdicts() {
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    // A quiet normal sweep, fully labeled normal ⇒ all bins TN.
    let quiet: Vec<Sample> = (0..50)
        .map(|i| {
            let phase = i as f64;
            Sample::new(vec![1.0 + 0.02 * phase.sin()])
                .labeled(SampleTruth { is_anomaly: false, onset: None })
        })
        .collect();
    let _ = eng.replay_slice(&quiet).unwrap();
    let report = eng.replay_slice(&quiet).unwrap();
    assert!(
        report.verdicts.iter().all(|rv| rv.bin == Some(ConfusionBin::TrueNegative)),
        "quiet stream must label everything true-negative"
    );
    assert_eq!(report.metrics.true_negatives, 50);
    assert_eq!(report.metrics.false_positives, 0);
}

#[test]
fn replay_empty_stream_is_not_an_error() {
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let report = eng.replay_slice(&[]).unwrap();
    assert!(report.verdicts.is_empty());
    assert_eq!(report.metrics.n_samples, 0);
}

/// TADI-2019-style cadence-bug guard: a 6 s MOS logger used to be replayed as
/// the 10 Hz reference, so a physical recovery tail lasting well over a minute
/// looked "3× longer" in sample count and kept false-alarming. With
/// cadence-aware dt (`detect_with_dt`) the same plume + τ=60 s recovery must
/// be decided the same way at 10 Hz and 6 s, and the deep tail must not
/// false-alarm at either cadence.
#[test]
fn six_second_logger_matches_ten_hertz_decisions() {
    let tau = 60.0_f64;
    let t_end = 306.0_f64;
    let plume_start = 6.0_f64;
    let plume_end = 66.0_f64;

    let run = |dt: f64| {
        let mut eng = DualKalmanEngine::new(1);
        let mut cfg = eng.config.clone();
        cfg.adsorption.enabled = true;
        cfg.adsorption.tau_s = vec![tau];
        cfg.adsorption.q_adsorption = 1e-4;
        eng.set_config(cfg);
        eng.calibrate_baseline(&baseline(1)).unwrap();

        let mut t = 0.0;
        let mut flagged_plume = false;
        let mut first_alarm_t: Option<f64> = None;
        let mut deep_tail_alarms = 0usize;
        let mut grid: Vec<(f64, bool)> = Vec::new(); // (wall-clock s, verdict)
        while t < t_end {
            let level = if t < plume_start {
                1.0
            } else if t < plume_end {
                4.0
            } else {
                1.0 + 3.0 * (-(t - plume_end) / tau).exp()
            };
            let reading = [level + 0.02 * (t * 1.7).sin()];
            let v = eng.detect_with_dt(&reading, None, dt).unwrap();
            if (plume_start..plume_end).contains(&t) {
                flagged_plume |= v.is_anomaly;
                if v.is_anomaly && first_alarm_t.is_none() {
                    first_alarm_t = Some(t);
                }
            } else if t >= plume_end + 2.0 * tau && v.is_anomaly {
                deep_tail_alarms += 1;
            }
            grid.push((t, v.is_anomaly));
            t += dt;
        }
        (flagged_plume, first_alarm_t, deep_tail_alarms, grid)
    };

    let (f10, a10, r10, g10) = run(0.1);
    let (f6, a6, r6, g6) = run(6.0);

    assert!(f10 && f6, "plume must be flagged at 10 Hz ({f10}) and 6 s ({f6})");
    let l6 = a6.expect("6 s logger must detect the plume");
    let l10 = a10.expect("10 Hz reference must detect the plume");
    assert!(l6 < plume_end, "6 s logger flags within ~12 s of release (lat={l6:.1} s)");
    assert!(l10 < plume_start + 3.0, "10 Hz reference flags near the release (lat={l10:.1} s)");
    assert_eq!(r10, 0, "no recovery false alarms at 10 Hz");
    assert_eq!(r6, 0, "no recovery false alarms at 6 s — the memory-tail cadence bug is gone");
    // From 1 τ after the release end the two cadences must reach the same
    // decision at every wall-clock time they both sample.
    let agree_from = plume_end + tau;
    let disagree = g6
        .iter()
        .zip(g10.iter())
        .filter(|((t6, a6v), (t10, a10v))| {
            *t6 >= agree_from && (t6 - t10).abs() < 1e-9 && a6v != a10v
        })
        .count();
    assert_eq!(
        disagree, 0,
        "6 s-logger verdicts must match the 10 Hz reference on the shared wall-clock grid"
    );
}