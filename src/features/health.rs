use crate::{Baseline, Result};

pub fn extract(normalized: &[f64], _raw: &[f64], baseline: &Baseline) -> Result<Vec<f64>> {
    let mut features = Vec::new();
    for ch in 0..normalized.len() {
        // Noise floor: baseline standard deviation
        features.push(baseline.std[ch]);

        // Sensitivity: signal change per unit concentration (approximated by normalized response magnitude)
        let sensitivity = normalized[ch].abs();
        features.push(sensitivity);

        // Hysteresis: for single reading, use direction of deviation
        features.push(if normalized[ch] > 0.0 { 1.0 } else { -1.0 });
    }
    Ok(features)
}

pub fn extract_window(window: &[Vec<f64>], baseline: &Baseline) -> Result<Vec<f64>> {
    let n_channels = window[0].len();
    let mut features = Vec::new();

    for ch in 0..n_channels {
        let raw_vals: Vec<f64> = window.iter().map(|s| s[ch]).collect();
        let n = raw_vals.len() as f64;

        // Noise floor: RMS of high-frequency component
        let hf: f64 = raw_vals.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f64>()
            / (raw_vals.len().max(1) - 1) as f64;
        features.push(hf.sqrt());

        // Sensitivity decay: compare first-third to last-third response
        let third = raw_vals.len() / 3;
        if third > 0 && baseline.r0[ch] > 0.0 {
            let first_mean: f64 = raw_vals[..third].iter().sum::<f64>() / third as f64;
            let last_mean: f64 = raw_vals[raw_vals.len() - third..].iter().sum::<f64>() / third as f64;
            let decay = (last_mean - first_mean) / baseline.r0[ch];
            features.push(decay);
        } else {
            features.push(0.0);
        }

        // Drift rate
        let mean_x = (n - 1.0) / 2.0;
        let mean_y: f64 = raw_vals.iter().sum::<f64>() / n;
        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        for (i, &v) in raw_vals.iter().enumerate() {
            let dx = i as f64 - mean_x;
            ss_xy += dx * (v - mean_y);
            ss_xx += dx * dx;
        }
        let drift_rate = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
        features.push(drift_rate);

        // Hysteresis: rising vs falling edge magnitude difference
        let mut rising_total = 0.0;
        let mut rising_count = 0;
        let mut falling_total = 0.0;
        let mut falling_count = 0;
        for w in raw_vals.windows(2) {
            if w[1] > w[0] {
                rising_total += w[1] - w[0];
                rising_count += 1;
            } else if w[1] < w[0] {
                falling_total += w[0] - w[1];
                falling_count += 1;
            }
        }
        let hysteresis = if rising_count > 0 && falling_count > 0 {
            (rising_total / rising_count as f64) - (falling_total / falling_count as f64)
        } else { 0.0 };
        features.push(hysteresis);
    }
    Ok(features)
}

pub fn names(n_channels: usize) -> Vec<String> {
    let mut names = Vec::new();
    for ch in 0..n_channels {
        names.push(format!("ch{ch}_noise_floor"));
        names.push(format!("ch{ch}_sensitivity_decay"));
        names.push(format!("ch{ch}_drift_rate"));
        names.push(format!("ch{ch}_hysteresis"));
    }
    names
}
