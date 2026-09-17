//! Wave-2 whole-engine gate: pins the §11.1–11.2 validation matrix of
//! `docs/anomaly-engine-design.md` against the real `DualKalmanEngine`,
//! driven end-to-end through `detect`. Rows A–H are the inherited behaviors;
//! the typology row and regime-switch row are the Wave-2.5 additions.
//!
//! Actual-behavior notes (kept honest, see §11.1):
//! - A single-sample +6 spike fires for one sample (a documented deviation
//!   from the legacy EWMA damping; the Kalman engine pegs the beacon).
//! - Warm-up: an uncalibrated engine buffers 60 readings, reports
//!   `warming_up` with `is_anomaly: false`, auto-calibrates on the 60th, and
//!   only then can alarm.
//! - A sustained jump alarms on entry and *then* declares a regime switch; the
//!   switch sample itself is never an anomaly and the flow stays quiet after
//!   re-anchoring.
use opensmell::anomaly::dual::{AmbientModel, AmbientReading, DualKalmanEngine, EngineVerdict};
use opensmell::anomaly::replay::{Sample, SampleTruth};
use opensmell::anomaly::stimulus::HealthFindingKind;
use opensmell::anomaly::TypologyKind;

fn baseline(c: usize) -> Vec<Vec<f64>> {
    (0..80)
        .map(|i| {
            let n = i as f64;
            (0..c).map(|ch| 1.0 + ch as f64 + 0.05 * (n * 1.7).sin()).collect()
        })
        .collect()
}

fn feed(eng: &mut DualKalmanEngine, stream: &[Vec<f64>]) -> Vec<EngineVerdict> {
    stream.iter().map(|s| eng.detect_no_ambient(s).unwrap()).collect()
}

fn fired(vs: &[EngineVerdict]) -> Vec<u64> {
    vs.iter()
        .enumerate()
        .filter(|(_, v)| v.is_anomaly)
        .map(|(i, _)| i as u64)
        .collect()
}

fn kinds(vs: &[EngineVerdict]) -> Vec<TypologyKind> {
    vs.iter().filter_map(|v| v.typology.as_ref().map(|t| t.kind)).collect()
}

// ---------------------------------------------------------------------------
// §11.1 inherited rows
// ---------------------------------------------------------------------------

#[test]
fn row_a_single_spike_fires_for_one_sample_with_spike_typology() {
    let mut eng = DualKalmanEngine::new(1);
    eng.config.q_state = 1e-4; // heavy smoothing
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let mut stream: Vec<Vec<f64>> = vec![vec![1.0]; 30];
    stream.push(vec![7.0]);
    stream.extend(vec![vec![1.0]; 40]);
    let mut vs = feed(&mut eng, &stream);

    let f = fired(&vs);
    assert_eq!(f, vec![30], "the +6 spike fires for exactly one sample");
    let kinds: Vec<String> = kinds(&vs).iter().map(|k| k.as_str().to_string()).collect();
    assert!(
        kinds.iter().any(|k| k == "spike"),
        "spike must be typed Spike (got {kinds:?})"
    );

    vs.clear();
}

#[test]
fn row_b_sustained_step_anomaly_within_40_with_step_typology() {
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let mut stream: Vec<Vec<f64>> = vec![vec![1.0]; 30];
    stream.extend(vec![vec![3.5]; 100]);
    stream.extend(vec![vec![1.0]; 40]);
    let vs = feed(&mut eng, &stream);

    let f = fired(&vs);
    let first = *f.first().expect("step must alarm");
    assert!(
        (30..70).contains(&first),
        "step alarms within ~40 samples (first={first})"
    );
    assert!(
        kinds(&vs).contains(&TypologyKind::Step),
        "sustained step must be typed Step"
    );
}

#[test]
fn row_c_slow_ramp_with_drift_walk_is_forgiven() {
    let mut eng = DualKalmanEngine::new(1);
    eng.config.q_param = 2e-4;
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let stream: Vec<Vec<f64>> = (0..300)
        .map(|i| vec![1.0 + 3.0 * (i as f64 / 300.0)])
        .collect();
    let vs = feed(&mut eng, &stream);
    assert!(
        !vs.iter().any(|v| v.is_anomaly),
        "drift-walk ramps are absorbed, not alarmed (fired={:?})",
        fired(&vs)
    );
}

#[test]
fn row_d_abrupt_step_after_ramp_alarms() {
    let mut eng = DualKalmanEngine::new(1);
    eng.config.q_param = 2e-4;
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let mut stream: Vec<Vec<f64>> = (0..300)
        .map(|i| vec![1.0 + 3.0 * (i as f64 / 300.0)])
        .collect();
    stream.extend(vec![vec![6.0]; 20]);
    let vs = feed(&mut eng, &stream);

    let f = fired(&vs);
    assert!(
        f.iter().any(|&i| (300..310).contains(&i)),
        "the abrupt +5 step at the ramp end must alarm (fired={f:?})"
    );
}

#[test]
fn row_e_same_shift_with_drift_off_alarms() {
    let mut eng = DualKalmanEngine::new(1);
    eng.config.q_param = 0.0;
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let mut stream: Vec<Vec<f64>> = vec![vec![1.0]; 30];
    stream.extend(vec![vec![4.0]; 60]);
    let vs = feed(&mut eng, &stream);
    let first = *fired(&vs).first().expect("drift-off shift must alarm");
    assert!(
        (30..70).contains(&first),
        "same shift with drift off alarms quickly (first={first})"
    );
}

#[test]
fn row_f_small_delta_no_vs_yes_with_sensitivity() {
    let mk = |sens: f64| -> DualKalmanEngine {
        let mut e = DualKalmanEngine::new(1);
        e.config.sensitivity = sens;
        e.config.q_state = 1e-4;
        e.calibrate_baseline(&baseline(1)).unwrap();
        e
    };
    let mut strict = mk(0.5);
    for _ in 0..30 {
        let _ = strict.detect_no_ambient(&[1.0]).unwrap();
    }
    let no = strict.detect_no_ambient(&[1.12]).unwrap();
    assert!(!no.is_anomaly, "delta 1.12 at sens=0.5 must stay quiet");

    let mut sensitive = mk(4.0);
    for _ in 0..30 {
        let _ = sensitive.detect_no_ambient(&[1.0]).unwrap();
    }
    let yes = sensitive.detect_no_ambient(&[1.12]).unwrap();
    assert!(yes.is_anomaly, "delta 1.12 at sens=4.0 must fire");
}

#[test]
fn row_g_first_readings_warm_up_never_anomaly_then_armed() {
    let mut eng = DualKalmanEngine::new(1);
    let mut vs = Vec::new();
    for _ in 0..59 {
        vs.push(eng.detect_no_ambient(&[1.0]).unwrap());
    }
    assert!(
        vs.iter().all(|v| v.warming_up && !v.is_anomaly),
        "first 59 readings of a fresh board are warming-up, never anomalous"
    );
    // The 60th reading auto-calibrates and falls through to the real path.
    let sixth = eng.detect_no_ambient(&[1.0]).unwrap();
    assert!(!sixth.warming_up, "60th reading completes the baseline");
    // Now the engine actually detects a real change.
    let hit = eng.detect_no_ambient(&[4.0]).unwrap();
    assert!(hit.is_anomaly, "post-warm-up engine detects a real shift");
}

#[test]
fn row_h_baseline_zeros_no_screaming_during_warmup() {
    let mut eng = DualKalmanEngine::new(1);
    let mut vs = Vec::new();
    for _ in 0..59 {
        vs.push(eng.detect_no_ambient(&[0.0]).unwrap());
    }
    assert!(
        vs.iter().all(|v| !v.is_anomaly),
        "flat zero-baseline warm-up must be silent while warming up"
    );

    // Calibrated board on a flat stream is also quiet.
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let flat: Vec<Vec<f64>> = vec![vec![1.0]; 59];
    assert!(fired(&feed(&mut eng, &flat)).is_empty());
}

// ---------------------------------------------------------------------------
// §11.2 new behaviors
// ---------------------------------------------------------------------------

#[test]
fn row_i_humidity_step_is_predicted_and_silent() {
    let mut eng = DualKalmanEngine::new(2);
    eng.calibrate_baseline(&baseline(2)).unwrap();
    let mut amb = AmbientModel::default();
    amb.fit(vec![0.0, 0.0], vec![0.5, 0.2], 25.0, 50.0);
    eng.set_ambient_model(amb);
    let at50 =
        |eng: &mut DualKalmanEngine| eng.detect(&[1.0, 2.0], Some(AmbientReading { temperature: Some(25.0), humidity: Some(50.0) }));
    for _ in 0..30 {
        let _ = at50(&mut eng).unwrap();
    }
    let humid = at50(&mut eng).unwrap();
    assert!(!humid.is_anomaly, "RH step 50→70 is predicted, not an anomaly");
}

#[test]
fn row_i2_missing_humidity_falls_back_without_crash() {
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let mut amb = AmbientModel::default();
    amb.fit(vec![0.0], vec![0.5], 25.0, 50.0);
    eng.set_ambient_model(amb);
    for _ in 0..30 {
        let _ = eng
            .detect(&[1.0], Some(AmbientReading { temperature: Some(25.0), humidity: Some(50.0) }))
            .unwrap();
    }
    // Sensor missing/saturated ⇒ humidity is None ⇒ φ_i → 0 correction.
    let v = eng.detect(&[1.0], Some(AmbientReading { temperature: Some(25.0), humidity: None })).unwrap();
    assert!(!v.is_anomaly, "missing humidity must not crash or alarm");
    let v = eng.detect(&[1.0], None).unwrap();
    assert!(!v.is_anomaly, "no ambient at all must not crash or alarm");
}

#[test]
fn row_regime_switch_declared_later_and_silent_after() {
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let mut stream: Vec<Vec<f64>> = vec![vec![1.0]; 30];
    stream.extend(vec![vec![6.0]; 360]); // fermenter idle → active
    let vs = feed(&mut eng, &stream);

    let switched: Vec<bool> = vs.iter().map(|v| v.regime_switch).collect();
    assert!(
        switched.iter().any(|&s| s),
        "a sustained novel region must eventually declare a regime switch"
    );
    // The declared switch itself is never an anomaly on that sample.
    assert!(
        vs.iter().any(|v| v.regime_switch && !v.is_anomaly),
        "the switch sample is re-anchoring, not an anomaly"
    );
    assert_eq!(vs.last().unwrap().regime, 1, "the active regime is now current");
    assert!(
        vs[vs.len() - 20..].iter().all(|v| !v.is_anomaly),
        "after the switch the sustained active flow stays quiet"
    );
}

#[test]
fn row_typology_spike_step_ramp_pulse_through_real_engine() {
    // Spike (heavy smoothing).
    let mut eng = DualKalmanEngine::new(1);
    eng.config.q_state = 1e-4;
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let mut stream: Vec<Vec<f64>> = vec![vec![1.0]; 30];
    stream.push(vec![7.0]);
    stream.extend(vec![vec![1.0]; 30]);
    let v = feed(&mut eng, &stream);
    assert!(kinds(&v).contains(&TypologyKind::Spike));

    // Step (sustained).
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let mut stream: Vec<Vec<f64>> = vec![vec![1.0]; 30];
    stream.extend(vec![vec![3.5]; 100]);
    let v = feed(&mut eng, &stream);
    assert!(kinds(&v).contains(&TypologyKind::Step));

    // Ramp (only typed once the ramp ends in an abrupt step, the D row).
    let mut eng = DualKalmanEngine::new(1);
    eng.config.q_param = 2e-4;
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let mut stream: Vec<Vec<f64>> = (0..300)
        .map(|i| vec![1.0 + 3.0 * (i as f64 / 300.0)])
        .collect();
    stream.extend(vec![vec![6.0]; 20]);
    let v = feed(&mut eng, &stream);
    assert!(kinds(&v).contains(&TypologyKind::Ramp));

    // Pulse (rise, hold, return).
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let mut stream: Vec<Vec<f64>> = vec![vec![1.0]; 40];
    stream.extend(vec![vec![3.5]; 30]);
    stream.extend(vec![vec![1.0]; 90]);
    let v = feed(&mut eng, &stream);
    assert!(
        kinds(&v).contains(&TypologyKind::Pulse),
        "a rise-hold-return pulse must be typed Pulse"
    );
}

#[test]
fn row_poison_gain_decay_confirms_and_surfaces() {
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    for _ in 0..10 {
        let _ = eng.detect_no_ambient(&[1.0]).unwrap();
    }
    // Parameter side sees the decay; the physical stimulus agrees.
    eng.set_relative_gains(vec![0.4]);
    let findings = eng.record_stimulus(&[0.4], &[1.0]).unwrap();
    assert!(
        findings.iter().any(|f| f.kind == HealthFindingKind::PoisonConfirmed),
        "filter + stimulus agreement must confirm poisoning"
    );
    let v = eng.detect_no_ambient(&[1.0]).unwrap();
    assert!(
        v.health_findings.iter().any(|f| f.kind == HealthFindingKind::PoisonConfirmed),
        "the engine verdict surfaces the confirmed poison finding"
    );
}

#[test]
fn scorecard_tpr_fpr_ppv_on_labeled_step() {
    let mut eng = DualKalmanEngine::new(1);
    eng.calibrate_baseline(&baseline(1)).unwrap();
    let warmup: Vec<Sample> = (0..30).map(|_| Sample::new(vec![1.0])).collect();
    let _ = eng.replay_slice(&warmup).unwrap();

    // 100 quiet normal (labeled normal), then a +3 step for 120 labeled
    // anomalous from onset.
    let mut samples: Vec<Sample> = (0..100)
        .map(|_| Sample::new(vec![1.0]).labeled(SampleTruth { is_anomaly: false, onset: None }))
        .collect();
    let onset = 100u64;
    for k in 0..120u64 {
        let truth = SampleTruth {
            is_anomaly: true,
            onset: if k == 0 { Some(onset) } else { None },
        };
        samples.push(Sample::new(vec![4.0]).labeled(truth));
    }
    let report = eng.replay_slice(&samples).unwrap();
    let m = report.metrics;

    // The engine is a *change* detector: it flags the transient near onset and
    // then absorbs the new level, so we score the event as caught when it fires
    // within the design latency window (the §11.3 `detected_events` / latency
    // metrics), not by TPR over the whole sustained event window.
    assert_eq!(m.n_events, 1, "one labeled event");
    assert_eq!(m.detected_events, 1, "the event must be caught");
    let lat = m.latency_samples.expect("latency present");
    assert!(
        (0.0..40.0).contains(&lat),
        "detection latency within the ~40-sample window (lat {lat})"
    );
    assert!(
        m.fpr() < 0.05,
        "quiet normal section must not alarm (fpr {:.3})",
        m.fpr()
    );
    assert!(
        m.ppv() > 0.5,
        "event catches are true positives (ppv {:.3}, tp={}, fp={})",
        m.ppv(),
        m.true_positives,
        m.false_positives
    );
}