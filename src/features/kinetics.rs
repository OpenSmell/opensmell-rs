use crate::{Baseline, Result};

pub fn extract(normalized: &[f64], _raw: &[f64], _baseline: &Baseline) -> Result<Vec<f64>> {
    let mut features = Vec::new();
    for ch in 0..normalized.len() {
        // Rise time (single reading approximation: time since baseline)
        features.push(normalized[ch].abs());
        // Decay indicator
        features.push(if normalized[ch] > 0.0 { 1.0 } else { -1.0 });
    }
    Ok(features)
}

pub fn extract_window(window: &[Vec<f64>], _baseline: &Baseline, sr: f64) -> Result<Vec<f64>> {
    let fs = if sr.is_finite() { sr.abs() } else { 1e-9 }.max(1e-9);
    let n_channels = window[0].len();
    let mut features = Vec::new();

    for ch in 0..n_channels {
        let raw_vals: Vec<f64> = window.iter().map(|s| s[ch]).collect();
        let n = raw_vals.len();

        // Rise time: 10% to 90% of peak
        let peak = raw_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let trough = raw_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = peak - trough;
        let low = trough + range * 0.1;
        let high = trough + range * 0.9;

        let rise_start = raw_vals.iter().position(|&v| v >= low).unwrap_or(0);
        let rise_end = raw_vals.iter().position(|&v| v >= high).unwrap_or(n);
        let rise_time = (rise_end - rise_start) as f64 / fs;
        features.push(rise_time);

        // Decay time: 90% to 10% of peak
        let decay_start = raw_vals.iter().rposition(|&v| v >= high).unwrap_or(n);
        let decay_end = raw_vals.iter().rposition(|&v| v >= low).unwrap_or(n);
        let decay_time = (decay_end - decay_start) as f64 / fs;
        features.push(decay_time);

        // Peak value
        features.push(peak);

        // Time to peak
        let ttp = raw_vals.iter().position(|&v| v == peak).unwrap_or(0) as f64 / fs;
        features.push(ttp);

        // Bi-exponential decay fit parameters (simplified): tau of each decay
        // segment is fit log-linearly against the time axis t = i/fs, so a
        // first-order time constant is cadence-independent (not a per-sample
        // mean |dy|).
        // tau1 = fast component (first 30% of decay)
        // tau2 = slow component (last 70% of decay)
        if decay_end > decay_start + 2 {
            let decay_vals: Vec<f64> = raw_vals[decay_start..=decay_end].to_vec();
            let n_decay = decay_vals.len();
            let fast_end = n_decay / 3;
            if fast_end > 1 {
                features.push(log_linear_tau(&decay_vals[..fast_end], fs));
                features.push(log_linear_tau(&decay_vals[fast_end..], fs));
            } else {
                features.push(0.0);
                features.push(0.0);
            }
        } else {
            features.push(0.0);
            features.push(0.0);
        }
    }
    Ok(features)
}

/// First-order exponential time constant via log-linear regression of
/// `|y - y_end|` against elapsed time `t = i/fs`: `ln(|y - y_end|) = ln(A) - t/tau`.
fn log_linear_tau(seg: &[f64], fs: f64) -> f64 {
    if seg.len() < 3 {
        return 0.0;
    }
    let end = seg[seg.len() - 1];
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for (i, &v) in seg.iter().enumerate() {
        let rel = (v - end).abs();
        if rel > 1e-9 {
            xs.push(i as f64 / fs);
            ys.push(rel.ln());
        }
    }
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean_x: f64 = xs.iter().sum::<f64>() / n;
    let mean_y: f64 = ys.iter().sum::<f64>() / n;
    let denom: f64 = xs.iter().map(|&x| (x - mean_x).powi(2)).sum();
    if denom <= 0.0 {
        return 0.0;
    }
    let slope: f64 = xs.iter().zip(ys.iter())
        .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>() / denom;
    if slope >= 0.0 {
        return 0.0; // not decaying: no meaningful time constant
    }
    (1.0 / -slope).min(1.0e6)
}

pub fn names(n_channels: usize) -> Vec<String> {
    let mut names = Vec::new();
    for ch in 0..n_channels {
        names.push(format!("ch{ch}_rise_time"));
        names.push(format!("ch{ch}_decay_time"));
        names.push(format!("ch{ch}_peak_value"));
        names.push(format!("ch{ch}_time_to_peak"));
        names.push(format!("ch{ch}_tau_fast"));
        names.push(format!("ch{ch}_tau_slow"));
    }
    names
}
