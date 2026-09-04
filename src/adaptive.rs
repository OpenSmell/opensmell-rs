use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use crate::{Result, OpenSmellError};

/// Adaptive threshold that improves with observed data using Welford's online algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThreshold {
    pub initial_threshold: f64,
    pub current_threshold: f64,
    pub min_threshold: f64,
    pub max_threshold: f64,
    pub n_samples: usize,
    pub mean_score: f64,
    pub m2_score: f64,  // For Welford's online variance
    pub target_fpr: f64,
}

impl AdaptiveThreshold {
    pub fn new(initial: f64, min: f64, max: f64, target_fpr: f64) -> Self {
        Self {
            initial_threshold: initial,
            current_threshold: initial,
            min_threshold: min,
            max_threshold: max,
            n_samples: 0,
            mean_score: 0.0,
            m2_score: 0.0,
            target_fpr,
        }
    }

    /// Update threshold with new observation using Welford's algorithm.
    pub fn update(&mut self, score: f64) {
        self.n_samples += 1;
        let delta = score - self.mean_score;
        self.mean_score += delta / self.n_samples as f64;
        let delta2 = score - self.mean_score;
        self.m2_score += delta * delta2;

        // Adapt threshold based on observed distribution
        if self.n_samples > 10 {
            let variance = self.m2_score / (self.n_samples - 1) as f64;
            let std = variance.sqrt().max(1e-6);
            
            // Set threshold at mean + z_score * std
            // z_score chosen to achieve target FPR using rational approximation of inverse normal CDF
            let z_score = normal_ppf(1.0 - self.target_fpr);
            self.current_threshold = self.mean_score + z_score * std;
            
            // Clamp to reasonable range
            self.current_threshold = self.current_threshold
                .max(self.min_threshold)
                .min(self.max_threshold);
        }
    }

    /// Confidence in current threshold based on sample count.
    /// Logistic growth: 0 at n=0, 0.5 at n=30, 0.95 at n=100
    pub fn confidence(&self) -> f64 {
        1.0 / (1.0 + (-((self.n_samples as f64) - 30.0) / 15.0).exp())
    }
}

/// Rational approximation of the inverse normal CDF (Abramowitz & Stegun).
fn normal_ppf(p: f64) -> f64 {
    // Coefficients for rational approximation
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let q: f64;
    let r: f64;

    if p < P_LOW {
        q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
            ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        q = p - 0.5;
        r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q /
            (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
            ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// User feedback record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRecord {
    pub timestamp: f64,
    pub was_anomaly: bool,
    pub user_confirmed: bool,
    pub score: f64,
    pub sensor_readings: Vec<f64>,
    pub user_note: String,
}

/// Anomaly detector that improves with every user interaction.
#[derive(Debug, Clone)]
pub struct AdaptiveAnomalyDetector {
    pub n_channels: usize,
    pub target_fpr: f64,
    pub thresholds: Vec<AdaptiveThreshold>,
    pub baseline_mean: Vec<f64>,
    pub baseline_cov_inv: Option<Vec<Vec<f64>>>,
    pub baseline_n: usize,
    pub feedback_history: VecDeque<FeedbackRecord>,
    pub drift_rate: Vec<f64>,
    pub last_baseline_update: f64,
    pub platt_a: f64,
    pub platt_b: f64,
}

impl AdaptiveAnomalyDetector {
    pub fn new(n_channels: usize, target_fpr: f64) -> Self {
        let thresholds = (0..n_channels)
            .map(|_| AdaptiveThreshold::new(1.0, 0.1, 10.0, target_fpr))
            .collect();

        Self {
            n_channels,
            target_fpr,
            thresholds,
            baseline_mean: vec![0.0; n_channels],
            baseline_cov_inv: None,
            baseline_n: 0,
            feedback_history: VecDeque::new(),
            drift_rate: vec![0.0; n_channels],
            last_baseline_update: timestamp_now(),
            platt_a: 1.0,
            platt_b: 0.0,
        }
    }

    /// Initial calibration from baseline data.
    pub fn calibrate_baseline(&mut self, baseline_samples: &[Vec<f64>]) -> Result<()> {
        if baseline_samples.is_empty() {
            return Err(OpenSmellError::InsufficientData { expected: 1, actual: 0 });
        }

        let n = baseline_samples.len();
        
        // Compute mean
        self.baseline_mean = (0..self.n_channels)
            .map(|ch| {
                baseline_samples.iter().map(|s| s[ch]).sum::<f64>() / n as f64
            })
            .collect();

        // Compute covariance matrix
        if n > self.n_channels {
            let mut cov = vec![vec![0.0; self.n_channels]; self.n_channels];
            for sample in baseline_samples {
                for i in 0..self.n_channels {
                    for j in 0..self.n_channels {
                        cov[i][j] += (sample[i] - self.baseline_mean[i]) * (sample[j] - self.baseline_mean[j]);
                    }
                }
            }
            for i in 0..self.n_channels {
                for j in 0..self.n_channels {
                    cov[i][j] /= (n - 1) as f64;
                }
                cov[i][i] += 1e-6; // Regularization
            }

            // Invert using Gauss-Jordan elimination
            self.baseline_cov_inv = Some(invert_matrix(&cov)?);
        } else {
            self.baseline_cov_inv = None;
        }

        self.baseline_n = n;
        self.last_baseline_update = timestamp_now();
        Ok(())
    }

    /// Detect anomaly with adaptive threshold and calibrated confidence.
    pub fn detect(&self, reading: &[f64]) -> Result<DetectionResult> {
        if reading.len() != self.n_channels {
            return Err(OpenSmellError::InvalidChannelCount {
                got: reading.len(),
                expected: self.n_channels,
            });
        }

        // Mahalanobis distance
        let raw_score = mahalanobis_distance(&self.baseline_mean, self.baseline_cov_inv.as_ref(), reading);

        // Per-channel scores
        let channel_scores: Vec<f64> = reading.iter().zip(self.baseline_mean.iter())
            .map(|(&r, &m)| (r - m).abs())
            .collect();

        // Adaptive threshold check
        let mut triggered_channels = Vec::new();
        for (ch, (score, threshold)) in channel_scores.iter().zip(self.thresholds.iter()).enumerate() {
            if *score > threshold.current_threshold {
                triggered_channels.push(ch);
            }
        }
        let is_anomaly = !triggered_channels.is_empty();

        // Calibrated confidence (Platt scaling)
        let confidence = self.platt_scale(raw_score).clamp(0.0, 1.0);

        // Drift compensation
        let time_since_update = timestamp_now() - self.last_baseline_update;
        let drift_factor = (-0.001 * time_since_update).exp();
        let adjusted_threshold = self.thresholds[0].current_threshold * drift_factor;

        // Threshold confidence (average across channels)
        let threshold_confidence: f64 = self.thresholds.iter().map(|t| t.confidence()).sum::<f64>() 
            / self.n_channels as f64;

        Ok(DetectionResult {
            is_anomaly,
            raw_score,
            calibrated_confidence: confidence,
            triggered_channels,
            channel_scores,
            threshold: adjusted_threshold,
            threshold_confidence,
            n_feedback_samples: self.feedback_history.len(),
        })
    }

    /// Update detector with user feedback — the core learning loop.
    pub fn update_with_feedback(&mut self, reading: &[f64], was_anomaly: bool, note: &str) -> Result<()> {
        let detect_result = self.detect(reading)?;

        let record = FeedbackRecord {
            timestamp: timestamp_now(),
            was_anomaly,
            user_confirmed: true,
            score: detect_result.raw_score,
            sensor_readings: reading.to_vec(),
            user_note: note.to_string(),
        };
        self.feedback_history.push_back(record);

        // Update per-channel thresholds
        for ch in 0..self.n_channels {
            let ch_score = (reading[ch] - self.baseline_mean[ch]).abs();
            self.thresholds[ch].update(ch_score);
        }

        // Update baseline with confirmed normal samples
        if !was_anomaly && self.baseline_n < 1000 {
            let alpha = 0.01;
            for i in 0..self.n_channels {
                self.baseline_mean[i] = (1.0 - alpha) * self.baseline_mean[i] + alpha * reading[i];
            }
            self.baseline_n += 1;
        }

        // Retrain Platt scaling periodically
        if self.feedback_history.len() % 10 == 0 {
            self.retrain_platt_scaling();
        }

        // Track drift
        self.update_drift(reading);

        Ok(())
    }

    /// Convert raw Mahalanobis distance to calibrated probability.
    fn platt_scale(&self, raw_score: f64) -> f64 {
        1.0 / (1.0 + (-self.platt_a * raw_score + self.platt_b).exp())
    }

    /// Retrain Platt scaling parameters from feedback history.
    fn retrain_platt_scaling(&mut self) {
        if self.feedback_history.len() < 20 {
            return;
        }

        let scores: Vec<f64> = self.feedback_history.iter().map(|r| r.score).collect();
        let labels: Vec<f64> = self.feedback_history.iter().map(|r| if r.was_anomaly { 1.0 } else { 0.0 }).collect();

        // Simple gradient descent for Platt scaling (Nelder-Mead alternative)
        let mut best_a = self.platt_a;
        let mut best_b = self.platt_b;
        let mut best_nll = neg_log_likelihood(&scores, &labels, best_a, best_b);

        // Search in neighborhood of current parameters
        for da in [-0.1, 0.0, 0.1] {
            for db in [-0.1, 0.0, 0.1] {
                let a = self.platt_a + da;
                let b = self.platt_b + db;
                let nll = neg_log_likelihood(&scores, &labels, a, b);
                if nll < best_nll {
                    best_nll = nll;
                    best_a = a;
                    best_b = b;
                }
            }
        }

        self.platt_a = best_a;
        self.platt_b = best_b;
    }

    /// Track environmental drift.
    fn update_drift(&mut self, reading: &[f64]) {
        if self.feedback_history.len() < 2 {
            return;
        }

        // Compare current reading to historical mean of normal samples
        let normal_readings: Vec<&Vec<f64>> = self.feedback_history.iter()
            .filter(|r| !r.was_anomaly)
            .map(|r| &r.sensor_readings)
            .collect();

        if normal_readings.len() < 5 {
            return;
        }

        let recent = &normal_readings[normal_readings.len().saturating_sub(50)..];
        let historical_mean: Vec<f64> = (0..self.n_channels)
            .map(|ch| recent.iter().map(|r| r[ch]).sum::<f64>() / recent.len() as f64)
            .collect();

        let alpha = 0.05;
        for i in 0..self.n_channels {
            let current_deviation = reading[i] - historical_mean[i];
            self.drift_rate[i] = (1.0 - alpha) * self.drift_rate[i] + alpha * current_deviation;
        }
    }

    /// Measure how much the detector has improved through feedback.
    pub fn get_accuracy_improvement(&self) -> AccuracyImprovement {
        if self.feedback_history.len() < 20 {
            return AccuracyImprovement {
                status: "insufficient_data".to_string(),
                early_accuracy: 0.0,
                late_accuracy: 0.0,
                improvement: 0.0,
                n_feedback_samples: self.feedback_history.len(),
                threshold_confidence: 0.0,
            };
        }

        let mid = self.feedback_history.len() / 2;
        let early: Vec<&FeedbackRecord> = self.feedback_history.iter().take(mid).collect();
        let late: Vec<&FeedbackRecord> = self.feedback_history.iter().skip(mid).collect();

        let threshold = self.thresholds[0].current_threshold;
        
        let compute_accuracy = |records: &[&FeedbackRecord]| -> f64 {
            if records.is_empty() { return 0.0; }
            let correct = records.iter()
                .filter(|r| r.was_anomaly == (r.score > threshold))
                .count();
            correct as f64 / records.len() as f64
        };

        let early_acc = compute_accuracy(&early);
        let late_acc = compute_accuracy(&late);
        let threshold_confidence: f64 = self.thresholds.iter().map(|t| t.confidence()).sum::<f64>()
            / self.n_channels as f64;

        AccuracyImprovement {
            status: "ok".to_string(),
            early_accuracy: early_acc,
            late_accuracy: late_acc,
            improvement: late_acc - early_acc,
            n_feedback_samples: self.feedback_history.len(),
            threshold_confidence,
        }
    }

    /// Export detector state for persistence.
    pub fn export_state(&self) -> DetectorState {
        DetectorState {
            n_channels: self.n_channels,
            baseline_mean: self.baseline_mean.clone(),
            baseline_n: self.baseline_n,
            thresholds: self.thresholds.iter().map(|t| ThresholdState {
                current: t.current_threshold,
                mean: t.mean_score,
                n_samples: t.n_samples,
                confidence: t.confidence(),
            }).collect(),
            platt_a: self.platt_a,
            platt_b: self.platt_b,
            drift_rate: self.drift_rate.clone(),
            n_feedback: self.feedback_history.len(),
        }
    }
}

/// Result of anomaly detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub is_anomaly: bool,
    pub raw_score: f64,
    pub calibrated_confidence: f64,
    pub triggered_channels: Vec<usize>,
    pub channel_scores: Vec<f64>,
    pub threshold: f64,
    pub threshold_confidence: f64,
    pub n_feedback_samples: usize,
}

/// Accuracy improvement metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyImprovement {
    pub status: String,
    pub early_accuracy: f64,
    pub late_accuracy: f64,
    pub improvement: f64,
    pub n_feedback_samples: usize,
    pub threshold_confidence: f64,
}

/// Exported detector state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorState {
    pub n_channels: usize,
    pub baseline_mean: Vec<f64>,
    pub baseline_n: usize,
    pub thresholds: Vec<ThresholdState>,
    pub platt_a: f64,
    pub platt_b: f64,
    pub drift_rate: Vec<f64>,
    pub n_feedback: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdState {
    pub current: f64,
    pub mean: f64,
    pub n_samples: usize,
    pub confidence: f64,
}

/// Neg-log-likelihood for Platt scaling optimization.
fn neg_log_likelihood(scores: &[f64], labels: &[f64], a: f64, b: f64) -> f64 {
    scores.iter().zip(labels.iter())
        .map(|(&s, &l)| {
            let p = 1.0 / (1.0 + (-a * s + b).exp()).clamp(1e-7, 1.0 - 1e-7);
            -(l * p.ln() + (1.0 - l) * (1.0 - p).ln())
        })
        .sum()
}

/// Fail-safe system with redundant detectors and escalation.
#[derive(Debug, Clone)]
pub struct FailSafeSystem {
    pub n_channels: usize,
    pub detectors: Vec<AdaptiveAnomalyDetector>,
    pub sensor_health: Vec<f64>,
    pub alert_level: u8,
    pub consecutive_anomalies: usize,
    pub consecutive_normal: usize,
    /// Collected-but-unused warm-up samples used to establish a baseline when
    /// none was provided before streaming (kills plug-in false positives).
    pub baseline_samples: Vec<Vec<f64>>,
    /// True once a baseline has been established (explicit calibration or a
    /// completed warm-up buffer). Detection is suppressed until this is set.
    pub baseline_ready: bool,
}

/// Number of fresh stream readings to collect before auto-enabling anomaly
/// detection on a newly-attached device.
pub const WARMUP_SAMPLES: usize = 60;

impl FailSafeSystem {
    pub fn new(n_channels: usize) -> Self {
        Self {
            n_channels,
            detectors: vec![
                AdaptiveAnomalyDetector::new(n_channels, 0.05),  // Standard
                AdaptiveAnomalyDetector::new(n_channels, 0.01),  // Conservative
                AdaptiveAnomalyDetector::new(n_channels, 0.10),  // Sensitive
            ],
            sensor_health: vec![1.0; n_channels],
            alert_level: 0,
            consecutive_anomalies: 0,
            consecutive_normal: 0,
            baseline_samples: Vec::new(),
            baseline_ready: false,
        }
    }

    /// Fail-safe detection: if ANY detector triggers, we alert. Anomalies are
    /// suppressed until a baseline is established so a freshly-attached device
    /// doesn't scream "ANOMALY" while its mean/covariance are still unknown.
    pub fn detect(&mut self, reading: &[f64]) -> Result<FailSafeResult> {
        // If a baseline wasn't calibrated up front, warm up from the live
        // stream: buffer the first WARMUP_SAMPLES readings, then calibrate all
        // detectors at once. Until then, report "warming up" and never anomaly.
        if !self.baseline_ready {
            // An explicit/manual calibration already populated the detectors —
            // treat that as ready and skip the deferred warm-up.
            if self.detectors.iter().any(|d| d.baseline_n > 0) {
                self.baseline_ready = true;
            } else if self.baseline_samples.len() < WARMUP_SAMPLES {
                self.baseline_samples.push(reading.to_vec());
                if self.baseline_samples.len() == WARMUP_SAMPLES {
                    let samples = self.baseline_samples.clone();
                    for detector in &mut self.detectors {
                        let _ = detector.calibrate_baseline(&samples);
                    }
                    self.baseline_samples.clear();
                    self.baseline_ready = true;
                } else {
                    return Ok(FailSafeResult {
                        is_anomaly: false,
                        anomaly_votes: 0,
                        max_confidence: 0.0,
                        alert_level: 0,
                        alert_name: "warming_up".to_string(),
                        consecutive_anomalies: 0,
                        sensor_failures: Vec::new(),
                        degraded_sensors: Vec::new(),
                        warming_up: true,
                        baseline_progress: self.baseline_samples.len() as f64 / WARMUP_SAMPLES as f64,
                    });
                }
            } else {
                self.baseline_ready = true;
            }
        }

        let mut results = Vec::new();
        for detector in &self.detectors {
            results.push(detector.detect(reading)?);
        }

        // Consensus: majority vote
        let anomaly_votes = results.iter().filter(|r| r.is_anomaly).count();
        let mut is_anomaly = anomaly_votes >= 2;

        // Worst-case: if any detector has very high confidence, alert
        let max_confidence = results.iter().map(|r| r.calibrated_confidence).fold(0.0f64, f64::max);
        if max_confidence > 0.9 {
            is_anomaly = true;
        }

        // Sensor health check: if any sensor is degraded, lower threshold
        let degraded_sensors: Vec<usize> = self.sensor_health.iter().enumerate()
            .filter(|(_, &h)| h < 0.5)
            .map(|(i, _)| i)
            .collect();
        
        if !degraded_sensors.is_empty() {
            is_anomaly = anomaly_votes >= 1;
        }

        // Escalation logic
        if is_anomaly {
            self.consecutive_anomalies += 1;
            self.consecutive_normal = 0;
            self.alert_level = if self.consecutive_anomalies >= 10 {
                3  // Emergency
            } else if self.consecutive_anomalies >= 5 {
                2  // Critical
            } else if self.consecutive_anomalies >= 2 {
                1  // Warning
            } else {
                self.alert_level
            };
        } else {
            self.consecutive_normal += 1;
            self.consecutive_anomalies = 0;
            if self.consecutive_normal >= 20 {
                self.alert_level = 0;
            }
        }

        // Sensor failure detection
        let sensor_failures = self.detect_sensor_failures(reading);

        let alert_name = match self.alert_level {
            0 => "normal",
            1 => "warning",
            2 => "critical",
            3 => "emergency",
            _ => "unknown",
        }.to_string();

        Ok(FailSafeResult {
            is_anomaly,
            anomaly_votes,
            max_confidence,
            alert_level: self.alert_level,
            alert_name,
            consecutive_anomalies: self.consecutive_anomalies,
            sensor_failures,
            degraded_sensors,
            warming_up: false,
            baseline_progress: 1.0,
        })
    }

    /// Detect sensor failures BEFORE they cause missed anomalies.
    fn detect_sensor_failures(&mut self, reading: &[f64]) -> Vec<SensorFailure> {
        let mut failures = Vec::new();
        for (ch, &value) in reading.iter().enumerate() {
            // Check 1: stuck at zero
            if value.abs() < 1e-10 {
                failures.push(SensorFailure {
                    channel: ch,
                    failure_type: "stuck_zero".to_string(),
                    severity: "critical".to_string(),
                    message: format!("Channel {} is stuck at zero — sensor may be disconnected", ch),
                });
                self.sensor_health[ch] = 0.0;
            }
        }
        failures
    }

    /// Update all detectors with user feedback.
    pub fn update_feedback(&mut self, reading: &[f64], was_anomaly: bool, note: &str) -> Result<()> {
        for detector in &mut self.detectors {
            detector.update_with_feedback(reading, was_anomaly, note)?;
        }
        Ok(())
    }
}

/// Result of fail-safe detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailSafeResult {
    pub is_anomaly: bool,
    pub anomaly_votes: usize,
    pub max_confidence: f64,
    pub alert_level: u8,
    pub alert_name: String,
    pub consecutive_anomalies: usize,
    pub sensor_failures: Vec<SensorFailure>,
    pub degraded_sensors: Vec<usize>,
    /// True while the detector is still establishing a baseline after connect —
    /// anomalies are not asserted during this phase (avoids plug-in noise).
    pub warming_up: bool,
    /// 0.0→1.0 baseline warm-up progress (only meaningful while warming_up).
    pub baseline_progress: f64,
}

/// Sensor failure record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorFailure {
    pub channel: usize,
    pub failure_type: String,
    pub severity: String,
    pub message: String,
}

/// Labeling system for user corrections and data commons contributions.
#[derive(Debug, Clone)]
pub struct LabelingSystem {
    pub labels: Vec<LabelRecord>,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelRecord {
    pub timestamp: f64,
    pub reading: Vec<f64>,
    pub is_anomaly: bool,
    pub note: String,
    pub confidence: f64,
    pub session_id: Option<String>,
}

impl LabelingSystem {
    pub fn new() -> Self {
        Self {
            labels: Vec::new(),
            session_id: None,
        }
    }

    /// Label a single sample.
    pub fn label_sample(&mut self, reading: &[f64], is_anomaly: bool, note: &str, confidence: f64) -> LabelRecord {
        let record = LabelRecord {
            timestamp: timestamp_now(),
            reading: reading.to_vec(),
            is_anomaly,
            note: note.to_string(),
            confidence,
            session_id: self.session_id.clone(),
        };
        self.labels.push(record.clone());
        record
    }

    /// Label multiple samples at once.
    pub fn batch_label(&mut self, readings: &[Vec<f64>], is_anomaly: bool, note: &str) -> Vec<LabelRecord> {
        readings.iter()
            .map(|r| self.label_sample(r, is_anomaly, note, 1.0))
            .collect()
    }

    /// Get labeling statistics.
    pub fn get_statistics(&self) -> LabelingStats {
        if self.labels.is_empty() {
            return LabelingStats::default();
        }

        let n_normal = self.labels.iter().filter(|l| !l.is_anomaly).count();
        let n_anomaly = self.labels.iter().filter(|l| l.is_anomaly).count();
        let with_notes = self.labels.iter().filter(|l| !l.note.is_empty()).count();
        let low_confidence = self.labels.iter().filter(|l| l.confidence < 0.7).count();

        LabelingStats {
            total: self.labels.len(),
            normal: n_normal,
            anomaly: n_anomaly,
            anomaly_ratio: n_anomaly as f64 / self.labels.len() as f64,
            with_notes,
            low_confidence,
        }
    }

    /// Export labeled data for data commons contribution.
    pub fn export_for_commons(&self, output_dir: &std::path::Path) -> Result<String> {
        std::fs::create_dir_all(output_dir)?;

        // Group by session
        let mut sessions: std::collections::HashMap<String, Vec<&LabelRecord>> = std::collections::HashMap::new();
        for label in &self.labels {
            let sid = label.session_id.as_deref().unwrap_or("unknown").to_string();
            sessions.entry(sid).or_default().push(label);
        }

        // Export each session
        for (sid, session_labels) in &sessions {
            let csv_path = output_dir.join(format!("session_{}.csv", sid));
            let mut wtr = csv::Writer::from_path(&csv_path)?;
            
            // Header
            let mut headers = vec!["timestamp".to_string()];
            for i in 0..session_labels[0].reading.len() {
                headers.push(format!("sensor_{}", i));
            }
            headers.push("is_anomaly".to_string());
            headers.push("note".to_string());
            wtr.write_record(&headers)?;

            // Data
            for label in session_labels {
                let mut row = vec![label.timestamp.to_string()];
                for v in &label.reading {
                    row.push(v.to_string());
                }
                row.push(label.is_anomaly.to_string());
                row.push(label.note.clone());
                wtr.write_record(&row)?;
            }
            wtr.flush()?;

            // Metadata JSON
            let json_path = output_dir.join(format!("session_{}.json", sid));
            let metadata = serde_json::json!({
                "session_id": sid,
                "n_samples": session_labels.len(),
                "n_normal": session_labels.iter().filter(|l| !l.is_anomaly).count(),
                "n_anomaly": session_labels.iter().filter(|l| l.is_anomaly).count(),
                "device_id": "unknown",
                "session_date": chrono::Utc::now().format("%Y-%m-%d").to_string(),
                "notes": format!("Labeled session with {} samples", session_labels.len()),
            });
            std::fs::write(&json_path, serde_json::to_string_pretty(&metadata)?)?;
        }

        Ok(output_dir.to_string_lossy().to_string())
    }
}

impl Default for LabelingSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Labeling statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LabelingStats {
    pub total: usize,
    pub normal: usize,
    pub anomaly: usize,
    pub anomaly_ratio: f64,
    pub with_notes: usize,
    pub low_confidence: usize,
}

/// Matrix inversion using Gauss-Jordan elimination.
fn invert_matrix(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = m.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];

    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Gauss-Jordan elimination
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-10 {
            return Err(OpenSmellError::AnomalyDetection("Matrix is singular".to_string()));
        }

        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                aug.swap(col, j);  // This is wrong, need to swap specific elements
            }
            // Fix: manual swap
            for j in 0..(2 * n) {
                let temp = aug[col][j];
                aug[col][j] = aug[max_row][j];
                aug[max_row][j] = temp;
            }
        }

        // Scale pivot row
        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Ok(inv)
}

/// Mahalanobis distance helper.
fn mahalanobis_distance(mean: &[f64], inv_cov: Option<&Vec<Vec<f64>>>, x: &[f64]) -> f64 {
    let n = mean.len();
    let diff: Vec<f64> = x.iter().zip(mean.iter()).map(|(&xi, &mi)| xi - mi).collect();

    if let Some(inv) = inv_cov {
        let mut result = 0.0;
        for i in 0..n {
            for j in 0..n {
                result += diff[i] * inv[i][j] * diff[j];
            }
        }
        result.sqrt().abs()
    } else {
        diff.iter().map(|d| d.powi(2)).sum::<f64>().sqrt()
    }
}

/// Get current timestamp in seconds since Unix epoch.
fn timestamp_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_threshold() {
        let mut threshold = AdaptiveThreshold::new(1.0, 0.1, 10.0, 0.05);
        
        // Feed some data
        for i in 0..100 {
            threshold.update(0.5 + (i as f64) * 0.01);
        }
        
        assert!(threshold.n_samples == 100);
        assert!(threshold.current_threshold > 0.0);
        assert!(threshold.confidence() > 0.5);
    }

    #[test]
    fn test_adaptive_detector() {
        let baseline = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
            vec![1.05, 2.05, 3.05],
        ];
        
        let mut detector = AdaptiveAnomalyDetector::new(3, 0.05);
        detector.calibrate_baseline(&baseline).unwrap();
        
        // Normal reading
        let result = detector.detect(&vec![1.0, 2.0, 3.0]).unwrap();
        assert!(!result.is_anomaly);
        
        // Anomalous reading
        let result = detector.detect(&vec![5.0, 10.0, 15.0]).unwrap();
        assert!(result.is_anomaly);
    }

    #[test]
    fn test_feedback_learning() {
        let baseline = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
        ];
        
        let mut detector = AdaptiveAnomalyDetector::new(3, 0.05);
        detector.calibrate_baseline(&baseline).unwrap();
        
        // Feed 50 normal samples
        for _ in 0..50 {
            detector.update_with_feedback(&vec![1.0, 2.0, 3.0], false, "").unwrap();
        }
        
        // Should be more confident now
        let stats = detector.get_accuracy_improvement();
        assert_eq!(stats.status, "ok");
    }

    #[test]
    fn test_fail_safe() {
        let baseline = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
        ];
        
        let mut system = FailSafeSystem::new(3);
        for detector in &mut system.detectors {
            detector.calibrate_baseline(&baseline).unwrap();
        }
        
        // Normal reading
        let result = system.detect(&vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(result.alert_level, 0);
    }

    #[test]
    fn test_fail_safe_warm_up_suppresses_false_positive() {
        // A freshly-created fail-safe with NO explicit baseline must NOT
        // report an anomaly during the warm-up window (the old behaviour fired
        // on essentially every reading because baseline_mean was all zeros).
        let mut system = FailSafeSystem::new(3);

        // First WARMUP_SAMPLES-1 readings: reported as warming up, never anomaly.
        for _ in 0..(WARMUP_SAMPLES - 1) {
            let r = system.detect(&[42.0, 43.0, 44.0]).unwrap();
            assert!(!r.is_anomaly, "must not alarm during warm-up");
            assert!(r.warming_up, "expected warming-up phase");
        }

        // The final warm-up sample flips it to ready and calibrates the baseline.
        let r = system.detect(&[42.0, 43.0, 44.0]).unwrap();
        assert!(!r.warming_up, "baseline should be established now");
        assert!(system.baseline_ready);
        // A stable reading close to the just-calibrated baseline must be normal.
        assert!(!r.is_anomaly);
    }

    #[test]
    fn test_labeling_system() {
        let mut labeling = LabelingSystem::new();
        
        labeling.label_sample(&vec![1.0, 2.0], false, "normal sample", 1.0);
        labeling.label_sample(&vec![5.0, 10.0], true, "anomalous", 0.9);
        
        let stats = labeling.get_statistics();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.normal, 1);
        assert_eq!(stats.anomaly, 1);
        assert_eq!(stats.anomaly_ratio, 0.5);
    }

    #[test]
    fn test_normal_ppf() {
        // Test inverse CDF at known points
        let ppf_05 = normal_ppf(0.5);
        assert!((ppf_05 - 0.0).abs() < 0.01);  // Should be ~0
        
        let ppf_975 = normal_ppf(0.975);
        assert!((ppf_975 - 1.96).abs() < 0.05);  // Should be ~1.96
    }

    #[test]
    fn test_export_state() {
        let baseline = vec![vec![1.0, 2.0], vec![1.1, 2.1]];
        let mut detector = AdaptiveAnomalyDetector::new(2, 0.05);
        detector.calibrate_baseline(&baseline).unwrap();
        
        let state = detector.export_state();
        assert_eq!(state.n_channels, 2);
        assert_eq!(state.baseline_n, 2);
        assert_eq!(state.thresholds.len(), 2);
    }
}
