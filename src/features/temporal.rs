use crate::{Baseline, Result};

pub fn extract(normalized: &[f64], raw: &[f64], baseline: &Baseline) -> Result<Vec<f64>> {
    let mut features = Vec::new();
    for ch in 0..normalized.len() {
        // Response latency: first index where signal exceeds 3x baseline noise
        let threshold = 3.0 * baseline.std[ch];
        let latency = raw.iter().position(|&v| (v - baseline.r0[ch]).abs() > threshold)
            .unwrap_or(raw.len()) as f64;
        features.push(latency);
    }
    Ok(features)
}

pub fn extract_window(window: &[Vec<f64>], baseline: &Baseline) -> Result<Vec<f64>> {
    let n_channels = window[0].len();
    let mut features = Vec::new();

    for ch in 0..n_channels {
        let raw_vals: Vec<f64> = window.iter().map(|s| s[ch]).collect();
        let norm_vals: Vec<f64> = raw_vals.iter()
            .map(|&r| if baseline.r0[ch] > 0.0 { (r - baseline.r0[ch]) / baseline.r0[ch] } else { 0.0 })
            .collect();

        // High-frequency transient: max absolute first difference
        let hf_transient = norm_vals.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f64, f64::max);
        features.push(hf_transient);

        // Oscillation frequency: zero crossings per sample
        let mean = norm_vals.iter().sum::<f64>() / norm_vals.len() as f64;
        let centered: Vec<f64> = norm_vals.iter().map(|v| v - mean).collect();
        let zero_crossings = centered.windows(2)
            .filter(|w| w[0] * w[1] < 0.0)
            .count() as f64 / (norm_vals.len() - 1).max(1) as f64;
        features.push(zero_crossings);

        // Oscillation amplitude: std of centered signal
        let variance = centered.iter().map(|v| v.powi(2)).sum::<f64>() / centered.len() as f64;
        features.push(variance.sqrt());

        // Response latency: time to first exceed 3x noise
        let threshold = 3.0 * baseline.std[ch];
        let latency = raw_vals.iter().position(|&v| (v - baseline.r0[ch]).abs() > threshold)
            .unwrap_or(raw_vals.len()) as f64;
        features.push(latency);
    }
    Ok(features)
}

pub fn names(n_channels: usize) -> Vec<String> {
    let mut names = Vec::new();
    for ch in 0..n_channels {
        names.push(format!("ch{ch}_hf_transient"));
        names.push(format!("ch{ch}_oscillation_freq"));
        names.push(format!("ch{ch}_oscillation_amp"));
        names.push(format!("ch{ch}_response_latency"));
    }
    names
}
