use crate::{Baseline, Result};

pub fn extract(normalized: &[f64], _raw: &[f64], _baseline: &Baseline) -> Result<Vec<f64>> {
    let mut features = Vec::new();
    // Cross-channel selectivity ratios for all pairs
    for i in 0..normalized.len() {
        for j in (i + 1)..normalized.len() {
            let ratio = if normalized[j].abs() > 1e-10 {
                normalized[i] / normalized[j]
            } else { 0.0 };
            features.push(ratio);
        }
    }
    Ok(features)
}

pub fn extract_window(window: &[Vec<f64>], baseline: &Baseline) -> Result<Vec<f64>> {
    let n_channels = window[0].len();
    let mut features = Vec::new();

    let normalized_window: Vec<Vec<f64>> = window.iter()
        .map(|s| baseline.normalize(s))
        .collect();

    // Mean selectivity ratios across the window
    for i in 0..n_channels {
        for j in (i + 1)..n_channels {
            let mean_i: f64 = normalized_window.iter().map(|s| s[i]).sum::<f64>() / window.len() as f64;
            let mean_j: f64 = normalized_window.iter().map(|s| s[j]).sum::<f64>() / window.len() as f64;
            let ratio = if mean_j.abs() > 1e-10 { mean_i / mean_j } else { 0.0 };
            features.push(ratio);
        }
    }

    // Correlation between channel pairs
    for i in 0..n_channels {
        for j in (i + 1)..n_channels {
            let vals_i: Vec<f64> = normalized_window.iter().map(|s| s[i]).collect();
            let vals_j: Vec<f64> = normalized_window.iter().map(|s| s[j]).collect();
            let corr = pearson_correlation(&vals_i, &vals_j);
            features.push(corr);
        }
    }

    Ok(features)
}

pub fn names(n_channels: usize) -> Vec<String> {
    let mut names = Vec::new();
    for i in 0..n_channels {
        for j in (i + 1)..n_channels {
            names.push(format!("sel_ratio_ch{i}_ch{j}"));
        }
    }
    for i in 0..n_channels {
        for j in (i + 1)..n_channels {
            names.push(format!("corr_ch{i}_ch{j}"));
        }
    }
    names
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }
    let denom = (ss_xx * ss_yy).sqrt();
    if denom > 0.0 { ss_xy / denom } else { 0.0 }
}
