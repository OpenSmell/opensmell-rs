use ndarray::Array2;
use serde::{Deserialize, Serialize};
use crate::{Result, OpenSmellError};

/// Anomaly score with explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyScore {
    /// Overall anomaly score (0.0 = normal, 1.0 = highly anomalous).
    pub score: f64,
    /// Per-channel anomaly scores.
    pub channel_scores: Vec<f64>,
    /// Which detection method triggered.
    pub method: String,
    /// Human-readable explanation.
    pub explanation: String,
}

/// Anomaly detection methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyMethod {
    /// Mahalanobis distance from baseline.
    Mahalanobis,
    /// Isolation Forest (unsupervised).
    IsolationForest,
    /// Local Outlier Factor.
    LOF,
    /// Ensemble of all methods.
    Ensemble,
}

/// Anomaly detector that learns from baseline data.
pub struct AnomalyDetector {
    /// Baseline mean vector.
    mean: Vec<f64>,
    /// Inverse covariance matrix for Mahalanobis distance.
    inv_cov: Option<Vec<Vec<f64>>>,
    /// Threshold for anomaly detection (learned from baseline).
    threshold: f64,
    /// Number of channels.
    n_channels: usize,
}

impl AnomalyDetector {
    /// Create a new detector from baseline samples.
    pub fn fit(baseline_samples: &[Vec<f64>], sensitivity: f64) -> Result<Self> {
        if baseline_samples.is_empty() {
            return Err(OpenSmellError::InsufficientData { expected: 1, actual: 0 });
        }
        let n_channels = baseline_samples[0].len();
        let n_samples = baseline_samples.len();

        // Compute mean
        let mut mean = vec![0.0; n_channels];
        for sample in baseline_samples {
            for (i, &v) in sample.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n_samples as f64;
        }

        // Compute covariance and its inverse
        let inv_cov = if n_samples > n_channels {
            let mut cov = Array2::zeros((n_channels, n_channels));
            for sample in baseline_samples {
                for i in 0..n_channels {
                    for j in 0..n_channels {
                        cov[[i, j]] += (sample[i] - mean[i]) * (sample[j] - mean[j]);
                    }
                }
            }
            for val in cov.iter_mut() {
                *val /= (n_samples - 1) as f64;
            }

            // Add small regularization diagonal
            for i in 0..n_channels {
                cov[[i, i]] += 1e-6;
            }

            // Invert using Gauss-Jordan elimination
            let inv = invert_matrix(&cov)?;
            Some(inv)
        } else {
            None
        };

        // Compute threshold from baseline Mahalanobis distances
        let distances: Vec<f64> = baseline_samples.iter()
            .map(|s| {
                let d = Self::mahalanobis_distance_static(&mean, inv_cov.as_ref(), s);
                d
            })
            .collect();
        let mut sorted = distances.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Use 95th percentile as threshold
        let idx = (sorted.len() as f64 * 0.95) as usize;
        let threshold = sorted[idx.min(sorted.len() - 1)] * sensitivity;

        Ok(Self { mean, inv_cov, threshold, n_channels })
    }

    /// Detect anomaly in a new reading.
    pub fn detect(&self, reading: &[f64], method: AnomalyMethod) -> Result<AnomalyScore> {
        if reading.len() != self.n_channels {
            return Err(OpenSmellError::InvalidChannelCount {
                got: reading.len(),
                expected: self.n_channels,
            });
        }

        match method {
            AnomalyMethod::Mahalanobis => self.mahalanobis_detect(reading),
            AnomalyMethod::Ensemble => self.ensemble_detect(reading),
            _ => Err(OpenSmellError::AnomalyDetection(
                format!("Method {:?} not yet implemented", method)
            )),
        }
    }

    fn mahalanobis_detect(&self, reading: &[f64]) -> Result<AnomalyScore> {
        let dist = Self::mahalanobis_distance_static(&self.mean, self.inv_cov.as_ref(), reading);
        let score = (dist / self.threshold).min(1.0);

        let channel_scores: Vec<f64> = reading.iter().zip(self.mean.iter())
            .map(|(&r, &m)| {
                let diff = (r - m).abs();
                let std = if let Some(ref inv_cov) = self.inv_cov {
                    // Use diagonal of inverse covariance as per-channel importance
                    let idx = reading.iter().position(|x| (x - r).abs() < 1e-10).unwrap_or(0);
                    (inv_cov[idx][idx]).sqrt().recip().max(1e-6)
                } else { 1.0 };
                (diff * std / self.threshold).min(1.0)
            })
            .collect();

        let explanation = if score > 0.8 {
            "Strong anomaly detected — signal deviates significantly from baseline".to_string()
        } else if score > 0.5 {
            "Moderate anomaly — approaching threshold, monitor closely".to_string()
        } else if score > 0.3 {
            "Mild deviation — within normal range but worth tracking".to_string()
        } else {
            "Normal — signal within expected baseline range".to_string()
        };

        Ok(AnomalyScore {
            score,
            channel_scores,
            method: "mahalanobis".to_string(),
            explanation,
        })
    }

    fn ensemble_detect(&self, reading: &[f64]) -> Result<AnomalyScore> {
        // For now, ensemble is just Mahalanobis
        // Future: add Isolation Forest, LOF, and combine via voting/averaging
        self.mahalanobis_detect(reading)
    }

    fn mahalanobis_distance_static(mean: &[f64], inv_cov: Option<&Vec<Vec<f64>>>, x: &[f64]) -> f64 {
        let n = mean.len();
        let diff: Vec<f64> = x.iter().zip(mean.iter()).map(|(&xi, &mi)| xi - mi).collect();

        if let Some(inv) = inv_cov {
            // Mahalanobis distance: sqrt(diff^T * inv_cov * diff)
            let mut result = 0.0;
            for i in 0..n {
                for j in 0..n {
                    result += diff[i] * inv[i][j] * diff[j];
                }
            }
            result.sqrt().abs()
        } else {
            // Fallback: Euclidean distance
            diff.iter().map(|d| d.powi(2)).sum::<f64>().sqrt()
        }
    }
}

fn invert_matrix(m: &Array2<f64>) -> Result<Vec<Vec<f64>>> {
    let n = m.nrows();
    let mut aug = Array2::zeros((n, 2 * n));

    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = m[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Gauss-Jordan elimination
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < 1e-10 {
            return Err(OpenSmellError::AnomalyDetection("Matrix is singular".to_string()));
        }

        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let temp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Scale pivot row
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mahalanobis_basic() {
        let baseline = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
            vec![1.05, 2.05, 3.05],
        ];
        let detector = AnomalyDetector::fit(&baseline, 1.0).unwrap();
        let normal = vec![1.0, 2.0, 3.0];
        let anomalous = vec![5.0, 10.0, 15.0];

        let score_normal = detector.detect(&normal, AnomalyMethod::Mahalanobis).unwrap();
        let score_anomalous = detector.detect(&anomalous, AnomalyMethod::Mahalanobis).unwrap();

        assert!(score_normal.score < score_anomalous.score,
            "Normal {} should be less than anomalous {}", score_normal.score, score_anomalous.score);
    }
}
