use opensmell::framework::*;

/// Well-conditioned recovery: two clearly distinct time constants. The LM must
/// recover tau1=2.0 and tau2=10.0 closely (this is the regime where decay
/// parity with scipy curve_fit is achievable).
#[test]
fn well_conditioned_bi_exp_recovers_true_taus() {
    let sr = 10.0;
    let n = 120; // 12 s of data
    // Simulated flash recovery: series rises to a peak then decays as
    // r0 + 100*exp(-t/2) + 50*exp(-t/10) after the peak.
    let peak_idx = 30usize;
    let mut series: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let ti = i as f64 / sr;
        let base = 500.0 + 100.0 * (-ti / 2.0).exp() + 50.0 * (-ti / 10.0).exp();
        let val = if i < peak_idx {
            // rising transient
            500.0 + 150.0
        } else {
            base
        };
        series.push(val);
    }
    let dec = compute_multi_exp_decay(&series, Some(peak_idx), sr, None);
    // (tau1, tau2, tau3, a1, a2, a3, cost)
    assert!(dec.0 > 0.0, "tau1 must be positive, got {}", dec.0);
    assert!(dec.1 > 0.0, "tau2 must be positive, got {}", dec.1);
    // Both recovered; the fast component maps to the smaller tau.
    let taus = [dec.0, dec.1];
    let mut s = taus;
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        (s[0] - 2.0).abs() < 0.5,
        "fast tau should be ~2, got {} (taus={}, now sorted)",
        s[0], s[0]
    );
    assert!(
        (s[1] - 10.0).abs() < 1.5,
        "slow tau should be ~10, got {}",
        s[1]
    );
    // residual cost must be low (good fit)
    assert!(dec.6 < 1e-3, "fit cost too high: {}", dec.6);
}

/// Degenerate (near-linear / collinear) recovery: the exponential fit is
/// non-unique (amplitudes and time constants trade off). The port must return
/// a deterministic FINITE fit (never -1), matching the reference's behavior of
/// returning a finite collinear fit rather than a failure.
#[test]
fn degenerate_recovery_returns_finite_deterministic_fit() {
    let sr = 10.0;
    let n = 100;
    // R0 near-constant, recovery essentially flat (near-linear): the worst case.
    let series: Vec<f64> = (0..n)
        .map(|i| {
            let ti = i as f64 / sr;
            511.75 + 2.8 * (-ti / 3.0).exp() + 0.02 * ti
        })
        .collect();
    let r0 = 511.75;
    let peak_idx = series
        .iter()
        .enumerate()
        .max_by(|a, b| {
            (a.1 - r0)
                .abs()
                .partial_cmp(&(b.1 - r0).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap()
        .clamp(5, n - 10);
    let dec = compute_multi_exp_decay(&series, Some(peak_idx), sr, Some(r0));
    let values = [dec.0, dec.1, dec.2, dec.3, dec.4, dec.5, dec.6];
    for (i, v) in values.iter().enumerate() {
        assert!(v.is_finite(), "decay param {i} not finite: {v}");
    }
    // The reference returns finite values here too (collinear tau ~ thousands).
    // We assert our fit is finite (i.e. we never return the all -1 failure) and
    // has a low residual cost. Exact numeric values are non-unique.
    assert!(dec.0 > 0.0, "degenerate fit should yield finite tau1, got {}", dec.0);
    assert!(dec.6 < 1.0, "degenerate fit residual too high: {}", dec.6);
}
