use crate::{Baseline, Result};

pub fn extract(normalized: &[f64], raw: &[f64], baseline: &Baseline) -> Result<Vec<f64>> {
    let mut features = Vec::new();
    for ch in 0..normalized.len() {
        // Absolute resistance (raw, unnormalized)
        features.push(raw[ch]);
        // Baseline resistance
        features.push(baseline.r0[ch]);
        // Normalized deviation (already computed)
        features.push(normalized[ch]);
        // Calibrated concentration estimate (power law approximation)
        let rr = if baseline.r0[ch] > 0.0 { raw[ch] / baseline.r0[ch] } else { 1.0 };
        features.push(rr);
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

        // Mean normalized response
        let mean_norm: f64 = norm_vals.iter().sum::<f64>() / norm_vals.len() as f64;
        features.push(mean_norm);

        // Peak normalized response
        let peak = norm_vals.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        features.push(peak);

        // Area under curve (AUC) of normalized response
        let auc: f64 = norm_vals.iter().map(|v| v.abs()).sum::<f64>();
        features.push(auc);

        // Endpoint delta (last - first)
        let endpoint = norm_vals.last().unwrap_or(&0.0) - norm_vals.first().unwrap_or(&0.0);
        features.push(endpoint);

        // Mean raw resistance
        let mean_raw: f64 = raw_vals.iter().sum::<f64>() / raw_vals.len() as f64;
        features.push(mean_raw);
    }
    Ok(features)
}

pub fn names(n_channels: usize) -> Vec<String> {
    let mut names = Vec::new();
    for ch in 0..n_channels {
        names.push(format!("ch{ch}_mean_normalized"));
        names.push(format!("ch{ch}_peak_response"));
        names.push(format!("ch{ch}_auc"));
        names.push(format!("ch{ch}_endpoint_delta"));
        names.push(format!("ch{ch}_mean_raw"));
    }
    names
}
