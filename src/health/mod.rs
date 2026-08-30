use serde::{Deserialize, Serialize};
use crate::{Result, OpenSmellError};

/// Health status levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum HealthStatus {
    /// Sensor is operating normally.
    Healthy = 0,
    /// Minor degradation detected.
    Warning = 1,
    /// Significant degradation — replacement recommended.
    Critical = 2,
    /// Sensor has failed or is unresponsive.
    Failed = 3,
}

/// Per-channel health assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorHealth {
    pub channel: usize,
    pub status: HealthStatus,
    pub drift_rate: f64,
    pub noise_floor: f64,
    pub sensitivity_decay: f64,
    pub hysteresis: f64,
    pub estimated_lifetime_hours: f64,
    pub recommendation: String,
}

/// Fleet health summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetHealth {
    pub device_id: String,
    pub overall_status: HealthStatus,
    pub sensors: Vec<SensorHealth>,
    pub timestamp: f64,
}

/// Sensor health monitor that tracks degradation over time.
pub struct HealthMonitor {
    /// Rolling window of recent readings per channel.
    windows: Vec<Vec<f64>>,
    /// Window size for health assessment.
    window_size: usize,
    /// Number of channels.
    n_channels: usize,
    /// Initial baseline for comparison.
    initial_baseline: Option<Vec<f64>>,
    /// Thresholds for status transitions.
    drift_warning: f64,
    drift_critical: f64,
    noise_warning: f64,
    noise_critical: f64,
}

impl HealthMonitor {
    pub fn new(n_channels: usize, window_size: usize) -> Self {
        Self {
            windows: vec![Vec::new(); n_channels],
            window_size,
            n_channels,
            initial_baseline: None,
            drift_warning: 0.05,   // 5% drift triggers warning
            drift_critical: 0.15,  // 15% drift triggers critical
            noise_warning: 0.1,    // 10% noise increase triggers warning
            noise_critical: 0.3,   // 30% noise increase triggers critical
        }
    }

    /// Set initial baseline for comparison.
    pub fn set_baseline(&mut self, baseline: Vec<f64>) {
        self.initial_baseline = Some(baseline);
    }

    /// Add a reading and update health assessment.
    pub fn add_reading(&mut self, reading: &[f64]) -> Result<FleetHealth> {
        if reading.len() != self.n_channels {
            return Err(OpenSmellError::InvalidChannelCount {
                got: reading.len(),
                expected: self.n_channels,
            });
        }

        for (ch, &val) in reading.iter().enumerate() {
            self.windows[ch].push(val);
            if self.windows[ch].len() > self.window_size {
                self.windows[ch].remove(0);
            }
        }

        self.assess_health()
    }

    fn assess_health(&self) -> Result<FleetHealth> {
        let mut sensors = Vec::with_capacity(self.n_channels);
        let mut worst_status = HealthStatus::Healthy;

        for ch in 0..self.n_channels {
            let window = &self.windows[ch];
            if window.len() < 10 {
                sensors.push(SensorHealth {
                    channel: ch,
                    status: HealthStatus::Healthy,
                    drift_rate: 0.0,
                    noise_floor: 0.0,
                    sensitivity_decay: 0.0,
                    hysteresis: 0.0,
                    estimated_lifetime_hours: f64::INFINITY,
                    recommendation: "Insufficient data for assessment".to_string(),
                });
                continue;
            }

            let n = window.len() as f64;
            let mean = window.iter().sum::<f64>() / n;
            let variance = window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let _std = variance.sqrt();

            // Drift rate: compare first half to second half
            let half = window.len() / 2;
            let first_mean: f64 = window[..half].iter().sum::<f64>() / half as f64;
            let second_mean: f64 = window[half..].iter().sum::<f64>() / (window.len() - half) as f64;
            let drift_rate = if first_mean.abs() > 1e-10 {
                (second_mean - first_mean).abs() / first_mean.abs()
            } else { 0.0 };

            // Noise floor: RMS of first differences
            let noise_floor: f64 = window.windows(2)
                .map(|w| (w[1] - w[0]).powi(2))
                .sum::<f64>() / (window.len() - 1) as f64;
            let noise_floor = noise_floor.sqrt();

            // Sensitivity decay: compare to initial baseline
            let sensitivity_decay = if let Some(ref baseline) = self.initial_baseline {
                if baseline[ch].abs() > 1e-10 {
                    (mean - baseline[ch]).abs() / baseline[ch].abs()
                } else { 0.0 }
            } else { 0.0 };

            // Hysteresis: rising vs falling edge difference
            let mut rising = 0.0;
            let mut rising_n = 0;
            let mut falling = 0.0;
            let mut falling_n = 0;
            for w in window.windows(2) {
                if w[1] > w[0] {
                    rising += w[1] - w[0];
                    rising_n += 1;
                } else if w[1] < w[0] {
                    falling += w[0] - w[1];
                    falling_n += 1;
                }
            }
            let hysteresis = if rising_n > 0 && falling_n > 0 {
                (rising / rising_n as f64) - (falling / falling_n as f64)
            } else { 0.0 };

            // Determine status
            let status = if drift_rate > self.drift_critical || noise_floor > self.noise_critical {
                HealthStatus::Critical
            } else if drift_rate > self.drift_warning || noise_floor > self.noise_warning {
                HealthStatus::Warning
            } else {
                HealthStatus::Healthy
            };

            // Estimate lifetime based on drift rate
            let lifetime_hours = if drift_rate > 0.0 {
                // Linear extrapolation: if drifting at X%/hour, sensor fails at 100%
                1.0 / drift_rate * 100.0
            } else {
                f64::INFINITY
            };

            let recommendation = match status {
                HealthStatus::Healthy => "Operating normally".to_string(),
                HealthStatus::Warning => format!(
                    "Monitor closely — drift rate {:.1}%/hr, consider replacement within {:.0} hours",
                    drift_rate * 100.0, lifetime_hours
                ),
                HealthStatus::Critical => format!(
                    "Replace soon — drift rate {:.1}%/hr, noise floor elevated",
                    drift_rate * 100.0
                ),
                HealthStatus::Failed => "Sensor has failed — replace immediately".to_string(),
            };

            if status > worst_status {
                worst_status = status;
            }

            sensors.push(SensorHealth {
                channel: ch,
                status,
                drift_rate,
                noise_floor,
                sensitivity_decay,
                hysteresis,
                estimated_lifetime_hours: lifetime_hours,
                recommendation,
            });
        }

        Ok(FleetHealth {
            device_id: String::new(),
            overall_status: worst_status,
            sensors,
            timestamp: 0.0,
        })
    }
}

/// Fisher's discriminant ratio for measuring class separability.
/// Used to warn users when substances are too similar to classify.
pub fn fisher_discriminant_ratio(
    class_a: &[Vec<f64>],
    class_b: &[Vec<f64>],
) -> Result<Vec<f64>> {
    if class_a.is_empty() || class_b.is_empty() {
        return Err(OpenSmellError::InsufficientData {
            expected: 1,
            actual: 0,
        });
    }
    let n_features = class_a[0].len();
    let mut fdr = Vec::with_capacity(n_features);

    for f in 0..n_features {
        let vals_a: Vec<f64> = class_a.iter().map(|s| s[f]).collect();
        let vals_b: Vec<f64> = class_b.iter().map(|s| s[f]).collect();

        let mean_a = vals_a.iter().sum::<f64>() / vals_a.len() as f64;
        let mean_b = vals_b.iter().sum::<f64>() / vals_b.len() as f64;

        let var_a = vals_a.iter().map(|v| (v - mean_a).powi(2)).sum::<f64>() / vals_a.len() as f64;
        let var_b = vals_b.iter().map(|v| (v - mean_b).powi(2)).sum::<f64>() / vals_b.len() as f64;

        let pooled_var = (var_a + var_b) / 2.0;
        let fdr_val = if pooled_var > 0.0 {
            (mean_a - mean_b).powi(2) / pooled_var
        } else { 0.0 };

        fdr.push(fdr_val);
    }
    Ok(fdr)
}

/// Compute pairwise FDR between all classes in a dataset.
pub fn pairwise_fdr(
    classes: &[(&str, Vec<Vec<f64>>)],
) -> Result<Vec<(String, String, Vec<f64>, f64)>> {
    let mut results = Vec::new();
    for i in 0..classes.len() {
        for j in (i + 1)..classes.len() {
            let fdr = fisher_discriminant_ratio(&classes[i].1, &classes[j].1)?;
            let mean_fdr = fdr.iter().sum::<f64>() / fdr.len() as f64;
            results.push((
                classes[i].0.to_string(),
                classes[j].0.to_string(),
                fdr,
                mean_fdr,
            ));
        }
    }
    Ok(results)
}

/// Euclidean distance between two feature vectors.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Cosine similarity between two feature vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
    let norm_a: f64 = a.iter().map(|ai| ai.powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|bi| bi.powi(2)).sum::<f64>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
}

/// Warning when two substances are too similar to classify reliably.
pub fn similarity_warning(
    class_a_mean: &[f64],
    class_b_mean: &[f64],
    class_a_std: &[f64],
    class_b_std: &[f64],
    threshold: f64,
) -> (bool, String) {
    let dist = euclidean_distance(class_a_mean, class_b_mean);
    let pooled_std: f64 = class_a_std.iter().zip(class_b_std.iter())
        .map(|(a, b)| (a + b) / 2.0)
        .sum::<f64>() / class_a_std.len() as f64;
    let normalized_dist = if pooled_std > 0.0 { dist / pooled_std } else { dist };

    if normalized_dist < threshold {
        (true, format!(
            "Warning: These substances are very similar (distance {:.2}, threshold {:.2}). Classification may be unreliable. Consider adding more distinct substances or using different sensors.",
            normalized_dist, threshold
        ))
    } else {
        (false, format!(
            "Substances are distinguishable (distance {:.2}, threshold {:.2})",
            normalized_dist, threshold
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_discriminant_ratio() {
        let class_a = vec![
            vec![1.0, 2.0],
            vec![1.1, 2.1],
            vec![0.9, 1.9],
        ];
        let class_b = vec![
            vec![5.0, 6.0],
            vec![5.1, 6.1],
            vec![4.9, 5.9],
        ];
        let fdr = fisher_discriminant_ratio(&class_a, &class_b).unwrap();
        assert!(fdr[0] > 10.0, "FDR should be large for well-separated classes");
    }

    #[test]
    fn test_similarity_warning() {
        let a_mean = vec![1.0, 2.0];
        let b_mean = vec![1.1, 2.1];
        let a_std = vec![0.1, 0.1];
        let b_std = vec![0.1, 0.1];
        let (warn, _msg) = similarity_warning(&a_mean, &b_mean, &a_std, &b_std, 2.0);
        assert!(warn, "Should warn for similar substances");
    }
}
