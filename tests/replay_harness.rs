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