use serde::{Deserialize, Serialize};
use crate::{Result, OpenSmellError};

/// Types of sensor degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegradationType {
    /// Catalyst poisoning (permanent sensitivity loss).
    SensitivityDecay,
    /// Electrical degradation (increased noise).
    NoiseIncrease,
    /// Surface contamination (slower recovery).
    RecoverySlowdown,
    /// Environmental drift.
    BaselineDrift,
}

/// Configuration for sensor health monitoring thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorHealthConfig {
    /// Max acceptable sensitivity decay per 24h window.
    pub sensitivity_decay_threshold: f64,
    /// Max acceptable noise floor increase per 24h window.
    pub noise_increase_threshold: f64,
    /// Max acceptable recovery time increase per 24h window.
    pub recovery_time_threshold: f64,
    /// Max acceptable baseline drift per 24h window.
    pub baseline_drift_threshold: f64,
    /// Minimum number of windows needed for trend detection.
    pub min_windows: usize,
    /// Hours per measurement window.
    pub window_size_hours: f64,
}

impl Default for SensorHealthConfig {
    fn default() -> Self {
        Self {
            sensitivity_decay_threshold: 0.05,  // 5% per day
            noise_increase_threshold: 0.10,     // 10% per day
            recovery_time_threshold: 0.15,      // 15% per day
            baseline_drift_threshold: 0.02,     // 2% per day
            min_windows: 3,
            window_size_hours: 24.0,
        }
    }
}

/// Health metrics computed from a single sensor data window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorMetrics {
    /// Peak response amplitude.
    pub sensitivity: f64,
    /// RMS of high-frequency components.
    pub noise_floor: f64,
    /// Time to return to 10% of baseline after peak.
    pub recovery_time: f64,
    /// Mean of first 10% of data.
    pub baseline_level: f64,
    /// Linear slope of signal over time.
    pub drift_rate: f64,
}

/// Current health status of a sensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorHealthStatus {
    pub channel: usize,
    pub is_healthy: bool,
    /// 0.0 = failed, 1.0 = perfect.
    pub health_score: f64,
    pub degradation_type: Option<DegradationType>,
    /// Rate of primary degradation (per window).
    pub degradation_rate: f64,
    /// Estimated hours until sensor needs replacement.
    pub estimated_remaining_life_hours: f64,
    /// "normal", "warning", or "critical".
    pub warning_level: String,
    pub metrics: SensorMetrics,
}

/// Sensor poisoning detector that tracks degradation over time.
///
/// Detection pipeline:
/// 1. Track sensitivity decay rate (slope of Rs/R0 over 24h windows)
/// 2. Track noise floor increase (RMS of high-frequency components)
/// 3. Track recovery time increase (time to return to baseline)
/// 4. If ANY metric exceeds threshold → flag sensor for replacement
///
/// Physical interpretation:
/// - Sensitivity decay: Catalyst poisoning (permanent)
/// - Noise floor increase: Electrical degradation (semi-permanent)
/// - Recovery time increase: Surface contamination (reversible)
pub struct PoisoningDetector {
    pub config: SensorHealthConfig,
    /// Per-channel history of metrics over time.
    history: Vec<Vec<SensorMetrics>>,
    /// Per-channel baseline metrics (initial calibration).
    baselines: Vec<Option<SensorMetrics>>,
}

impl PoisoningDetector {
    pub fn new(n_channels: usize, config: SensorHealthConfig) -> Self {
        Self {
            config,
            history: vec![Vec::new(); n_channels],
            baselines: vec![None; n_channels],
        }
    }

    /// Initialize sensor tracking with baseline measurements.
    pub fn initialize_channel(&mut self, channel: usize, data: &[f64]) -> Result<SensorMetrics> {
        if channel >= self.history.len() {
            return Err(OpenSmellError::InvalidChannelCount {
                got: channel + 1,
                expected: self.history.len(),
            });
        }

        let metrics = compute_metrics(data, 10.0);
        self.baselines[channel] = Some(metrics.clone());
        self.history[channel].push(metrics.clone());
        Ok(metrics)
    }

    /// Update sensor health with new data window.
    pub fn update_channel(&mut self, channel: usize, data: &[f64]) -> Result<SensorHealthStatus> {
        if channel >= self.history.len() {
            return Err(OpenSmellError::InvalidChannelCount {
                got: channel + 1,
                expected: self.history.len(),
            });
        }

        // Initialize if not yet done
        if self.baselines[channel].is_none() {
            let metrics = self.initialize_channel(channel, data)?;
            return Ok(SensorHealthStatus {
                channel,
                is_healthy: true,
                health_score: 1.0,
                degradation_type: None,
                degradation_rate: 0.0,
                estimated_remaining_life_hours: f64::INFINITY,
                warning_level: "normal".to_string(),
                metrics,
            });
        }

        let metrics = compute_metrics(data, 10.0);
        self.history[channel].push(metrics.clone());

        // Keep only recent history (last 7 days)
        let max_history = (7.0 * 24.0 / self.config.window_size_hours) as usize;
        if self.history[channel].len() > max_history {
            let drain_to = self.history[channel].len() - max_history;
            self.history[channel].drain(..drain_to);
        }

        self.analyze_degradation(channel, &metrics)
    }

    /// Analyze degradation trends over time using linear regression.
    fn analyze_degradation(&self, channel: usize, current: &SensorMetrics) -> Result<SensorHealthStatus> {
        let history = &self.history[channel];
        let baseline = self.baselines[channel].as_ref().unwrap();

        if history.len() < self.config.min_windows {
            return Ok(SensorHealthStatus {
                channel,
                is_healthy: true,
                health_score: 1.0,
                degradation_type: None,
                degradation_rate: 0.0,
                estimated_remaining_life_hours: f64::INFINITY,
                warning_level: "normal".to_string(),
                metrics: current.clone(),
            });
        }

        // Compute degradation rates via linear regression
        let time_points: Vec<f64> = (0..history.len())
            .map(|i| i as f64 * self.config.window_size_hours)
            .collect();

        let sensitivities: Vec<f64> = history.iter().map(|m| m.sensitivity).collect();
        let noise_floors: Vec<f64> = history.iter().map(|m| m.noise_floor).collect();
        let recovery_times: Vec<f64> = history.iter().map(|m| m.recovery_time).collect();
        let baseline_levels: Vec<f64> = history.iter().map(|m| m.baseline_level).collect();

        let sensitivity_slope = linear_slope(&time_points, &sensitivities) / (baseline.sensitivity + 1e-10);
        let noise_slope = linear_slope(&time_points, &noise_floors) / (baseline.noise_floor + 1e-10);
        let recovery_slope = linear_slope(&time_points, &recovery_times) / (baseline.recovery_time + 1e-10);
        let drift_slope = linear_slope(&time_points, &baseline_levels) / (baseline.baseline_level + 1e-10);

        // Find primary degradation type
        let rates = [
            (DegradationType::SensitivityDecay, sensitivity_slope.abs()),
            (DegradationType::NoiseIncrease, noise_slope.abs()),
            (DegradationType::RecoverySlowdown, recovery_slope.abs()),
            (DegradationType::BaselineDrift, drift_slope.abs()),
        ];

        let (primary_type, max_rate) = rates.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(t, r)| (Some(*t), *r))
            .unwrap_or((None, 0.0));

        // Compute health score
        let health_score = compute_health_score(
            sensitivity_slope, noise_slope, recovery_slope, drift_slope,
            &self.config,
        );

        // Determine warning level
        let warning_level = if health_score < 0.5 {
            "critical"
        } else if health_score < 0.7 {
            "warning"
        } else {
            "normal"
        }.to_string();

        // Estimate remaining life
        let remaining_life = estimate_remaining_life(max_rate, self.config.window_size_hours);

        Ok(SensorHealthStatus {
            channel,
            is_healthy: health_score > 0.7,
            health_score,
            degradation_type: primary_type,
            degradation_rate: max_rate,
            estimated_remaining_life_hours: remaining_life,
            warning_level,
            metrics: current.clone(),
        })
    }

    /// Get history for a channel.
    pub fn get_history(&self, channel: usize) -> &[SensorMetrics] {
        &self.history[channel]
    }
}

/// Compute health metrics from a single sensor data window.
fn compute_metrics(data: &[f64], sampling_rate: f64) -> SensorMetrics {
    if data.is_empty() {
        return SensorMetrics {
            sensitivity: 0.0,
            noise_floor: 0.0,
            recovery_time: 0.0,
            baseline_level: 0.0,
            drift_rate: 0.0,
        };
    }

    // Sensitivity: peak response amplitude
    let sensitivity = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

    // Noise floor: simplified RMS of high-frequency components
    // Use successive differences as a proxy for high-frequency content
    let noise_floor = if data.len() > 2 {
        let diffs: Vec<f64> = data.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        mean_diff
    } else {
        0.0
    };

    // Baseline level: mean of first 10% of data
    let baseline_end = (data.len() as f64 * 0.1).max(1.0) as usize;
    let baseline_level: f64 = data[..baseline_end].iter().sum::<f64>() / baseline_end as f64;

    // Recovery time: time to return to 10% of baseline after peak
    let peak_idx = data.iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let peak_val = data[peak_idx];
    let threshold_10 = if peak_val > baseline_level {
        baseline_level + 0.1 * (peak_val - baseline_level)
    } else {
        baseline_level - 0.1 * (baseline_level - peak_val)
    };

    let recovery_idx = if peak_val > baseline_level {
        data[peak_idx..].iter().position(|&v| v <= threshold_10)
    } else {
        data[peak_idx..].iter().position(|&v| v >= threshold_10)
    };

    let recovery_time = recovery_idx
        .map(|i| (peak_idx + i) as f64 / sampling_rate)
        .unwrap_or(data.len() as f64 / sampling_rate);

    // Drift rate: linear slope of signal over time
    let time_points: Vec<f64> = (0..data.len())
        .map(|i| i as f64 / sampling_rate)
        .collect();
    let drift_rate = linear_slope(&time_points, data);

    SensorMetrics {
        sensitivity,
        noise_floor,
        recovery_time,
        baseline_level,
        drift_rate,
    }
}

/// Compute health score from degradation rates.
///
/// Health score = 1 - Σ(w_i * |rate_i| / threshold_i)
fn compute_health_score(
    sensitivity_slope: f64,
    noise_slope: f64,
    recovery_slope: f64,
    drift_slope: f64,
    config: &SensorHealthConfig,
) -> f64 {
    let weights = [(0.4, sensitivity_slope), (0.3, noise_slope), (0.2, recovery_slope), (0.1, drift_slope)];
    let thresholds = [
        config.sensitivity_decay_threshold,
        config.noise_increase_threshold,
        config.recovery_time_threshold,
        config.baseline_drift_threshold,
    ];

    let total_degradation: f64 = weights.iter().zip(thresholds.iter())
        .map(|((w, rate), threshold)| w * rate.abs() / threshold)
        .sum();

    (1.0 - total_degradation).max(0.0)
}

/// Estimate remaining sensor life in hours via linear extrapolation.
fn estimate_remaining_life(max_rate: f64, window_size_hours: f64) -> f64 {
    if max_rate <= 0.0 {
        return f64::INFINITY;
    }
    // Time to reach 50% health from current 100%
    (0.5 / max_rate) * window_size_hours
}

/// Simple linear regression slope (β = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)).
fn linear_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return 0.0;
    }

    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    for i in 0..x.len().min(y.len()) {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
    }

    if ss_xx.abs() < 1e-10 {
        return 0.0;
    }
    ss_xy / ss_xx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_metrics() {
        // Simulated sensor data: baseline + peak + recovery
        let mut data: Vec<f64> = vec![100.0; 100]; // Baseline
        for i in 100..150 {
            data.push(100.0 + 500.0 * (-((i - 120) as f64).powi(2) / 200.0).exp()); // Gaussian peak
        }
        for _ in 150..200 {
            data.push(100.0); // Recovery
        }

        let metrics = compute_metrics(&data, 10.0);
        assert!(metrics.sensitivity > 500.0, "Peak should be captured");
        assert!(metrics.baseline_level > 90.0, "Baseline should be ~100");
        assert!(metrics.recovery_time > 0.0, "Recovery time should be positive");
    }

    #[test]
    fn test_poisoning_detector_healthy() {
        let config = SensorHealthConfig::default();
        let mut detector = PoisoningDetector::new(3, config);

        // Feed stable data (no degradation)
        for _ in 0..5 {
            let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.01).sin()).collect();
            detector.update_channel(0, &data).unwrap();
        }

        let status = detector.analyze_degradation(0, &compute_metrics(&vec![100.0; 100], 10.0)).unwrap();
        assert!(status.is_healthy);
        assert_eq!(status.warning_level, "normal");
    }

    #[test]
    fn test_poisoning_detector_degrading() {
        let config = SensorHealthConfig::default();
        let mut detector = PoisoningDetector::new(1, config);

        // Feed data with decreasing sensitivity
        for i in 0..10 {
            let decay = 1.0 - (i as f64 * 0.02); // 2% decay per window
            let data: Vec<f64> = (0..100).map(|j| decay * (100.0 + (j as f64 * 0.01).sin())).collect();
            detector.update_channel(0, &data).unwrap();
        }

        let status = detector.analyze_degradation(0, &compute_metrics(&vec![80.0; 100], 10.0)).unwrap();
        assert!(status.health_score < 1.0, "Should detect degradation");
        assert!(status.degradation_type.is_some(), "Should identify degradation type");
    }

    #[test]
    fn test_linear_slope() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let slope = linear_slope(&x, &y);
        assert!((slope - 2.0).abs() < 0.01, "Slope should be 2.0, got {}", slope);
    }

    #[test]
    fn test_health_score() {
        let config = SensorHealthConfig::default();
        // No degradation
        let score = compute_health_score(0.0, 0.0, 0.0, 0.0, &config);
        assert!((score - 1.0).abs() < 0.01);

        // At threshold
        let score = compute_health_score(
            config.sensitivity_decay_threshold,
            0.0, 0.0, 0.0, &config,
        );
        assert!(score < 0.7, "At threshold should be warning level, got {}", score);
    }
}
