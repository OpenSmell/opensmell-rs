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

pub fn extract_window(window: &[Vec<f64>], _baseline: &Baseline) -> Result<Vec<f64>> {
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
        let rise_time = (rise_end - rise_start) as f64;
        features.push(rise_time);

        // Decay time: 90% to 10% of peak
        let decay_start = raw_vals.iter().rposition(|&v| v >= high).unwrap_or(n);
        let decay_end = raw_vals.iter().rposition(|&v| v >= low).unwrap_or(n);
        let decay_time = (decay_end - decay_start) as f64;
        features.push(decay_time);

        // Peak value
        features.push(peak);

        // Time to peak
        let ttp = raw_vals.iter().position(|&v| v == peak).unwrap_or(0) as f64;
        features.push(ttp);

        // Bi-exponential decay fit parameters (simplified)
        // tau1 = fast component (first 30% of decay)
        // tau2 = slow component (last 70% of decay)
        if decay_end > decay_start + 2 {
            let decay_vals: Vec<f64> = raw_vals[decay_start..=decay_end].to_vec();
            let n_decay = decay_vals.len();
            let fast_end = n_decay / 3;
            if fast_end > 1 {
                let fast_decay: f64 = decay_vals[..fast_end].windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .sum::<f64>() / fast_end as f64;
                let slow_decay: f64 = decay_vals[fast_end..].windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .sum::<f64>() / (n_decay - fast_end).max(1) as f64;
                features.push(fast_decay);
                features.push(slow_decay);
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
