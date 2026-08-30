use crate::{Baseline, Result, OpenSmellError};

/// Extract anomaly features from a single normalized reading.
pub fn extract(normalized: &[f64], raw: &[f64], baseline: &Baseline) -> Result<Vec<f64>> {
    let mut features = Vec::new();

    // 1. Drift rate: how far from baseline (Euclidean distance in normalized space)
    let drift: f64 = normalized.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    features.push(drift);

    // 2. Stability index: inverse of coefficient of variation of recent signal
    // (for single reading, use baseline CV as proxy)
    let cv: f64 = baseline.std.iter().zip(baseline.r0.iter())
        .map(|(&s, &r)| if r > 0.0 { s / r } else { 0.0 })
        .sum::<f64>() / normalized.len() as f64;
    let stability = if cv > 0.0 { 1.0 / (1.0 + cv) } else { 1.0 };
    features.push(stability);

    // 3. Noise floor: RMS of high-frequency component
    let hf: f64 = raw.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f64>()
        / (raw.len().max(1) - 1) as f64;
    features.push(hf.sqrt());

    // 4. Signal-to-noise ratio
    let signal_power: f64 = normalized.iter().map(|v| v.powi(2)).sum::<f64>() / normalized.len() as f64;
    let snr = if hf > 0.0 { signal_power / hf } else { 100.0 };
    features.push(snr);

    // 5. Normalized drift per channel
    for &v in normalized.iter() {
        features.push(v.abs());
    }

    // 6. Max absolute deviation across channels
    let max_dev = normalized.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    features.push(max_dev);

    // 7. Sum of absolute deviations
    let sum_dev: f64 = normalized.iter().map(|v| v.abs()).sum();
    features.push(sum_dev);

    // 8. Direction pattern: count of positive vs negative deviations
    let positive_count = normalized.iter().filter(|&&v| v > 0.0).count() as f64;
    let total = normalized.len() as f64;
    features.push(positive_count / total);  // oxidizing fraction

    Ok(features)
}

/// Extract anomaly features from a time series window (for continuous monitoring).
pub fn extract_window(window: &[Vec<f64>], baseline: &Baseline) -> Result<Vec<f64>> {
    if window.len() < 2 {
        return Err(OpenSmellError::InsufficientData { expected: 2, actual: window.len() });
    }
    let n_channels = window[0].len();
    let mut features = Vec::new();

    // Normalize entire window
    let normalized_window: Vec<Vec<f64>> = window.iter()
        .map(|s| baseline.normalize(s))
        .collect();

    for ch in 0..n_channels {
        let vals: Vec<f64> = normalized_window.iter().map(|s| s[ch]).collect();
        let raw_vals: Vec<f64> = window.iter().map(|s| s[ch]).collect();
        let n = vals.len() as f64;

        // 1. Drift rate: slope of linear fit to normalized values
        let mean_x = (n - 1.0) / 2.0;
        let mean_y: f64 = vals.iter().sum::<f64>() / n;
        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        for (i, &v) in vals.iter().enumerate() {
            let dx = i as f64 - mean_x;
            ss_xy += dx * (v - mean_y);
            ss_xx += dx * dx;
        }
        let drift_rate = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
        features.push(drift_rate);

        // 2. Stability: inverse of coefficient of variation
        let variance: f64 = vals.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let cv = if mean_y.abs() > 1e-10 { std / mean_y.abs() } else { 0.0 };
        let stability = if cv > 0.0 { 1.0 / (1.0 + cv) } else { 1.0 };
        features.push(stability);

        // 3. Noise floor: RMS of first differences
        let hf: f64 = vals.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f64>()
            / (vals.len().max(1) - 1) as f64;
        features.push(hf.sqrt());

        // 4. SNR
        let signal_power = vals.iter().map(|v| v.powi(2)).sum::<f64>() / n;
        features.push(if hf > 0.0 { signal_power / hf } else { 100.0 });

        // 5. Sensitivity decay: compare first-half mean to second-half mean
        let half = n as usize / 2;
        if half > 0 {
            let first_mean: f64 = vals[..half].iter().sum::<f64>() / half as f64;
            let second_mean: f64 = vals[half..].iter().sum::<f64>() / (n as usize - half) as f64;
            let decay = if first_mean.abs() > 1e-10 {
                (second_mean - first_mean) / first_mean.abs()
            } else { 0.0 };
            features.push(decay);
        } else {
            features.push(0.0);
        }

        // 6. Max deviation from baseline
        let max_dev = vals.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        features.push(max_dev);

        // 7. Hysteresis: difference between rising and falling edges
        let mut rising_mean = 0.0;
        let mut rising_count = 0;
        let mut falling_mean = 0.0;
        let mut falling_count = 0;
        for w in vals.windows(2) {
            if w[1] > w[0] {
                rising_mean += w[1] - w[0];
                rising_count += 1;
            } else if w[1] < w[0] {
                falling_mean += w[0] - w[1];
                falling_count += 1;
            }
        }
        let hysteresis = if rising_count > 0 && falling_count > 0 {
            (rising_mean / rising_count as f64) - (falling_mean / falling_count as f64)
        } else { 0.0 };
        features.push(hysteresis);

        // 8. Saturation index: fraction of samples near sensor limits
        let max_raw = raw_vals.iter().fold(0.0f64, |a, &b| a.max(b));
        let saturation_threshold = max_raw * 0.95;
        let saturated_fraction = raw_vals.iter()
            .filter(|&&v| v >= saturation_threshold)
            .count() as f64 / n;
        features.push(saturated_fraction);
    }

    // Cross-channel anomaly features
    if n_channels >= 2 {
        // 9. Correlation between channels (anomaly = decorrelation)
        for i in 0..n_channels {
            for j in (i + 1)..n_channels {
                let vals_i: Vec<f64> = normalized_window.iter().map(|s| s[i]).collect();
                let vals_j: Vec<f64> = normalized_window.iter().map(|s| s[j]).collect();
                let corr = pearson_correlation(&vals_i, &vals_j);
                features.push(corr);
            }
        }
    }

    Ok(features)
}

/// Feature names for anomaly detection.
pub fn names(n_channels: usize) -> Vec<String> {
    let mut names = Vec::new();
    for ch in 0..n_channels {
        names.push(format!("ch{ch}_drift_rate"));
        names.push(format!("ch{ch}_stability"));
        names.push(format!("ch{ch}_noise_floor"));
        names.push(format!("ch{ch}_snr"));
        names.push(format!("ch{ch}_sensitivity_decay"));
        names.push(format!("ch{ch}_max_deviation"));
        names.push(format!("ch{ch}_hysteresis"));
        names.push(format!("ch{ch}_saturation_index"));
    }
    // Cross-channel correlation
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
