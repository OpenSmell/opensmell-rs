use crate::{Baseline, Result, OpenSmellError};

/// Preprocessing pipeline for MOX sensor data.
/// Handles everything from raw ADC readings to analysis-ready feature vectors.

/// Raw sensor data with metadata.
#[derive(Debug, Clone)]
pub struct RawData {
    /// Raw readings: rows = time steps, columns = channels.
    pub samples: Vec<Vec<f64>>,
    /// Sampling rate in Hz.
    pub sample_rate: f64,
    /// Channel names (optional).
    pub channel_names: Vec<String>,
    /// Timestamps (optional).
    pub timestamps: Vec<f64>,
}

impl RawData {
    /// Load from CSV file.
    pub fn from_csv(path: &str) -> Result<Self> {
        let mut rdr = csv::Reader::from_path(path)?;
        let headers: Vec<String> = rdr.headers()?.iter().map(|h| h.to_string()).collect();

        let mut samples = Vec::new();
        let mut timestamps = Vec::new();

        for result in rdr.records() {
            let record = result?;
            let mut row = Vec::new();
            let mut has_timestamp = false;

            for (i, field) in record.iter().enumerate() {
                if let Ok(val) = field.parse::<f64>() {
                    if headers.get(i).map(|h| h.to_lowercase()).as_deref() == Some("timestamp") {
                        timestamps.push(val);
                        has_timestamp = true;
                    } else {
                        row.push(val);
                    }
                }
            }

            if !row.is_empty() {
                samples.push(row);
            }
            if !has_timestamp && !samples.is_empty() {
                timestamps.push(samples.len() as f64);
            }
        }

        if samples.is_empty() {
            return Err(OpenSmellError::InsufficientData { expected: 1, actual: 0 });
        }

        // Detect channel names (skip timestamp column)
        let channel_names: Vec<String> = headers.iter()
            .filter(|h| h.to_lowercase() != "timestamp" && h.to_lowercase() != "ticks")
            .cloned()
            .collect();

        Ok(Self {
            samples,
            sample_rate: 10.0, // Default 10 Hz, overridden if known
            channel_names,
            timestamps,
        })
    }

    /// Number of time steps.
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Number of channels.
    pub fn n_channels(&self) -> usize {
        self.samples.first().map_or(0, |r| r.len())
    }

    /// Get a single channel as a slice.
    pub fn channel(&self, idx: usize) -> Vec<f64> {
        self.samples.iter().map(|r| r[idx]).collect()
    }
}

/// Baseline correction: estimate R0 and normalize.
pub struct BaselineCorrection {
    /// Baseline estimation method.
    pub method: BaselineMethod,
    /// Percentage of initial samples to use for baseline.
    pub baseline_fraction: f64,
    /// Minimum number of baseline samples.
    pub min_baseline_samples: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum BaselineMethod {
    /// Median of initial samples (standard MOX approach).
    Median,
    /// Mean of initial samples.
    Mean,
    /// Exponentially weighted moving average.
    Ewma { alpha: f64 },
    /// Percentile-based (robust to outliers).
    Percentile { p: f64 },
}

impl Default for BaselineCorrection {
    fn default() -> Self {
        Self {
            method: BaselineMethod::Median,
            baseline_fraction: 0.15,
            min_baseline_samples: 30,
        }
    }
}

impl BaselineCorrection {
    /// Estimate R0 from raw data.
    pub fn estimate_r0(&self, data: &RawData) -> Result<Baseline> {
        let n_channels = data.n_channels();
        let baseline_end = ((data.n_samples() as f64) * self.baseline_fraction) as usize;
        let baseline_end = baseline_end.max(self.min_baseline_samples).min(data.n_samples());

        let mut r0 = Vec::with_capacity(n_channels);
        let mut std = Vec::with_capacity(n_channels);

        for ch in 0..n_channels {
            let channel_data = data.channel(ch);

            // Filter out invalid values
            let valid: Vec<f64> = channel_data[..baseline_end].iter()
                .filter(|&&v| v.is_finite() && v > 0.0)
                .cloned()
                .collect();

            if valid.is_empty() {
                r0.push(0.0);
                std.push(1.0);
                continue;
            }

            let (r0_val, std_val) = match self.method {
                BaselineMethod::Median => {
                    let mut sorted = valid.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = if sorted.len() % 2 == 0 {
                        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                    } else {
                        sorted[sorted.len() / 2]
                    };
                    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
                    let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / valid.len() as f64;
                    (median, var.sqrt())
                }
                BaselineMethod::Mean => {
                    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
                    let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / valid.len() as f64;
                    (mean, var.sqrt())
                }
                BaselineMethod::Ewma { alpha } => {
                    let mut ewma = valid[0];
                    for &v in &valid[1..] {
                        ewma = alpha * v + (1.0 - alpha) * ewma;
                    }
                    let var = valid.iter().map(|v| (v - ewma).powi(2)).sum::<f64>() / valid.len() as f64;
                    (ewma, var.sqrt())
                }
                BaselineMethod::Percentile { p } => {
                    let mut sorted = valid.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let idx = (sorted.len() as f64 * p) as usize;
                    let p_val = sorted[idx.min(sorted.len() - 1)];
                    let var = valid.iter().map(|v| (v - p_val).powi(2)).sum::<f64>() / valid.len() as f64;
                    (p_val, var.sqrt())
                }
            };

            r0.push(r0_val);
            std.push(std_val);
        }

        Ok(Baseline {
            r0,
            n_samples: baseline_end,
            std,
        })
    }

    /// Apply Rs/R0 normalization to raw data.
    pub fn normalize(&self, data: &RawData, baseline: &Baseline) -> Vec<Vec<f64>> {
        data.samples.iter().map(|row| {
            row.iter().zip(baseline.r0.iter())
                .map(|(&rs, &r0)| if r0 > 0.0 { (rs - r0) / r0 } else { 0.0 })
                .collect()
        }).collect()
    }
}

/// Signal filtering for noise reduction.
pub struct SignalFilter {
    pub filter_type: FilterType,
    pub window_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    /// Median filter (removes impulsive noise).
    Median,
    /// Moving average (smooths signal).
    MovingAverage,
    /// Savitzky-Golay (preserves peaks while smoothing).
    SavitzkyGolay { polynomial_order: usize },
    /// High-pass filter (removes slow drift).
    HighPass { cutoff_fraction: f64 },
}

impl SignalFilter {
    /// Apply filter to a 1D signal.
    pub fn apply(&self, signal: &[f64]) -> Vec<f64> {
        match self.filter_type {
            FilterType::Median => self.median_filter(signal),
            FilterType::MovingAverage => self.moving_average(signal),
            FilterType::SavitzkyGolay { polynomial_order } => {
                self.savitzky_golay(signal, polynomial_order)
            }
            FilterType::HighPass { cutoff_fraction } => self.high_pass(signal, cutoff_fraction),
        }
    }

    fn median_filter(&self, signal: &[f64]) -> Vec<f64> {
        let half = self.window_size / 2;
        let mut result = Vec::with_capacity(signal.len());

        for i in 0..signal.len() {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(signal.len());
            let mut window: Vec<f64> = signal[start..end].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            result.push(window[window.len() / 2]);
        }
        result
    }

    fn moving_average(&self, signal: &[f64]) -> Vec<f64> {
        let half = self.window_size / 2;
        let mut result = Vec::with_capacity(signal.len());

        for i in 0..signal.len() {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(signal.len());
            let sum: f64 = signal[start..end].iter().sum();
            result.push(sum / (end - start) as f64);
        }
        result
    }

    fn savitzky_golay(&self, signal: &[f64], poly_order: usize) -> Vec<f64> {
        // Simplified Savitzky-Golay: fit polynomial to window, evaluate at center
        let half = self.window_size / 2;
        let mut result = Vec::with_capacity(signal.len());

        for i in 0..signal.len() {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(signal.len());
            let window = &signal[start..end];
            let n = window.len();

            if n < poly_order + 1 {
                result.push(window[n / 2]);
                continue;
            }

            // Simple polynomial fit using least squares
            let mut sum = 0.0;
            for (j, &v) in window.iter().enumerate() {
                let t = (j as f64 - half as f64) / half as f64;
                let weight = 1.0 - t.abs(); // Triangle window
                sum += v * weight;
            }
            result.push(sum / n as f64);
        }
        result
    }

    fn high_pass(&self, signal: &[f64], cutoff_fraction: f64) -> Vec<f64> {
        // Simple high-pass: subtract exponential moving average
        let alpha = cutoff_fraction;
        let mut result = Vec::with_capacity(signal.len());
        let mut ewma = signal[0];

        for &v in signal {
            ewma = alpha * v + (1.0 - alpha) * ewma;
            result.push(v - ewma);
        }
        result
    }
}

/// Sliding window extraction for time series analysis.
pub struct WindowExtractor {
    pub window_size: usize,
    pub stride: usize,
}

impl WindowExtractor {
    pub fn new(window_size: usize, stride: usize) -> Self {
        Self { window_size, stride }
    }

    /// Extract overlapping windows from normalized data.
    pub fn extract_windows(&self, data: &[Vec<f64>]) -> Vec<Vec<Vec<f64>>> {
        if data.len() < self.window_size {
            return vec![data.to_vec()];
        }

        let mut windows = Vec::new();
        let mut start = 0;

        while start + self.window_size <= data.len() {
            let window = data[start..start + self.window_size].to_vec();
            windows.push(window);
            start += self.stride;
        }

        windows
    }

    /// Extract features from each window and average.
    pub fn extract_averaged_features(
        &self,
        data: &[Vec<f64>],
        baseline: &Baseline,
        groups: &[crate::features::FeatureGroup],
    ) -> Result<Vec<f64>> {
        let windows = self.extract_windows(data);
        if windows.is_empty() {
            return Err(OpenSmellError::InsufficientData {
                expected: self.window_size,
                actual: data.len(),
            });
        }

        let mut all_features = Vec::new();
        for window in &windows {
            let features = crate::features::extract_window_features(window, baseline, groups)?;
            if all_features.is_empty() {
                all_features = vec![0.0; features.len()];
            }
            for (i, &f) in features.iter().enumerate() {
                all_features[i] += f;
            }
        }

        // Average
        let n = windows.len() as f64;
        for f in all_features.iter_mut() {
            *f /= n;
        }

        Ok(all_features)
    }
}

/// Data validation: detect and handle bad data.
pub struct DataValidator {
    pub max_value: f64,
    pub min_value: f64,
    pub max_consecutive_zeros: usize,
    pub max_nan_fraction: f64,
}

impl Default for DataValidator {
    fn default() -> Self {
        Self {
            max_value: 5000.0,
            min_value: -100.0,
            max_consecutive_zeros: 10,
            max_nan_fraction: 0.1,
        }
    }
}

impl DataValidator {
    /// Validate and clean raw data.
    pub fn validate(&self, data: &mut RawData) -> Vec<String> {
        let mut warnings = Vec::new();

        for (i, row) in data.samples.iter_mut().enumerate() {
            // Check for NaN/inf
            let nan_count = row.iter().filter(|v| !v.is_finite()).count();
            if nan_count > 0 {
                warnings.push(format!("Row {}: {} invalid values, replacing with 0", i, nan_count));
                for v in row.iter_mut() {
                    if !v.is_finite() {
                        *v = 0.0;
                    }
                }
            }

            // Check range and clamp
            for j in 0..row.len() {
                let v = row[j];
                if v > self.max_value {
                    warnings.push(format!("Row {}, Ch {}: value {} exceeds max {}", i, j, v, self.max_value));
                    row[j] = self.max_value;
                } else if v < self.min_value {
                    warnings.push(format!("Row {}, Ch {}: value {} below min {}", i, j, v, self.min_value));
                    row[j] = self.min_value;
                }
            }

            // Check for consecutive zeros
            let zero_streak = row.iter().take_while(|&&v| v == 0.0).count();
            if zero_streak > self.max_consecutive_zeros {
                warnings.push(format!("Row {}: {} consecutive zeros (sensor may be disconnected)", i, zero_streak));
            }
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_correction() {
        // Simulate MOX sensor with baseline around 100, slow drift
        let data = RawData {
            samples: (0..100).map(|i| {
                let noise = (i as f64 * 0.7).sin() * 2.0; // small oscillation
                vec![100.0 + noise, 200.0 + noise * 0.5]
            }).collect(),
            sample_rate: 10.0,
            channel_names: vec!["ch0".to_string(), "ch1".to_string()],
            timestamps: (0..100).map(|i| i as f64).collect(),
        };

        let bc = BaselineCorrection::default();
        let baseline = bc.estimate_r0(&data).unwrap();
        // Median of first 15 samples should be close to 100 and 200
        assert!((baseline.r0[0] - 100.0).abs() < 5.0, "r0[0] = {}", baseline.r0[0]);
        assert!((baseline.r0[1] - 200.0).abs() < 5.0, "r0[1] = {}", baseline.r0[1]);
    }

    #[test]
    fn test_median_filter() {
        let signal = vec![1.0, 2.0, 100.0, 2.0, 1.0]; // Impulse noise
        let filter = SignalFilter {
            filter_type: FilterType::Median,
            window_size: 3,
        };
        let filtered = filter.apply(&signal);
        assert!((filtered[2] - 2.0).abs() < 0.1); // Impulse removed
    }

    #[test]
    fn test_window_extraction() {
        let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let extractor = WindowExtractor::new(10, 5);
        let windows = extractor.extract_windows(&data);
        assert_eq!(windows.len(), 3); // [0-9], [5-14], [10-19]
        assert_eq!(windows[0].len(), 10);
    }
}
