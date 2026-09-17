//! EWMA control-chart baseline detector.
//!
//! Diagnostic / research detector used to cross-check the shipping
//! `DualKalmanEngine` on the second (home-activity) corpus. Per channel it
//! keeps an exponential moving average of the reading and of the squared
//! residual (an adaptive variance); each new reading is scored as
//! `z = (x - mu) / sd`, and a sample is anomalous when `min_votes` channels
//! simultaneously exceed `threshold_sigma` z. The baseline chases always, so
//! slow ramps are integrated rather than tracked-away — the property that
//! exposes the weak, slow stimuli the Kalman follows before it fires.

use serde::{Deserialize, Serialize};

use crate::{OpenSmellError, Result};

/// Tuning surface for the control chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwmaConfig {
    /// EWMA weight on the reading and on the squared residual (per update).
    pub alpha: f64,
    /// Sample is anomalous when a channel's |z| exceeds this many sigma.
    pub threshold_sigma: f64,
    /// Minimum number of simultaneously-exceeding channels for a vote.
    pub min_votes: usize,
}

impl Default for EwmaConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            threshold_sigma: 5.0,
            min_votes: 2,
        }
    }
}

/// One per-reading verdict.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EwmaVerdict {
    pub is_anomaly: bool,
    pub max_z: f64,
    pub z_scores: Vec<f64>,
    pub anomaly_votes: usize,
}

/// Streaming EWMA control chart over `n_channels` parallel channels.
#[derive(Debug, Clone)]
pub struct EwmaControlChart {
    n_channels: usize,
    config: EwmaConfig,
    mu: Vec<f64>,
    var: Vec<f64>,
    calibrated: bool,
}

impl EwmaControlChart {
    /// Create with default tuning.
    pub fn new(n_channels: usize) -> Self {
        Self {
            n_channels,
            config: EwmaConfig::default(),
            mu: vec![0.0; n_channels],
            var: vec![1e-12; n_channels],
            calibrated: false,
        }
    }

    /// Create with explicit tuning.
    pub fn with_config(n_channels: usize, config: EwmaConfig) -> Self {
        let mut s = Self::new(n_channels);
        s.config = config;
        s
    }

    pub fn config(&self) -> &EwmaConfig {
        &self.config
    }

    /// Initialise `mu`/`var` from a clean-air calibration window
    /// (sample mean / biased variance, like the engine's window stats).
    pub fn calibrate(&mut self, samples: &[Vec<f64>]) -> Result<()> {
        if samples.is_empty() {
            return Err(OpenSmellError::InsufficientData { expected: 1, actual: 0 });
        }
        let c = self.n_channels;
        for s in samples {
            if s.len() != c {
                return Err(OpenSmellError::InvalidChannelCount {
                    got: s.len(),
                    expected: c,
                });
            }
        }
        let n = samples.len() as f64;
        for i in 0..c {
            let mean = samples.iter().map(|s| s[i]).sum::<f64>() / n;
            let var = samples
                .iter()
                .map(|s| {
                    let d = s[i] - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            self.mu[i] = mean;
            self.var[i] = var.max(1e-12);
        }
        self.calibrated = true;
        Ok(())
    }

    /// Score one reading (causal: state before this sample drives the verdict,
    /// then the baseline updates from it).
    pub fn detect(&mut self, reading: &[f64]) -> Result<EwmaVerdict> {
        if !self.calibrated {
            return Err(OpenSmellError::AnomalyDetection(
                "EwmaControlChart used before calibrate".to_string(),
            ));
        }
        if reading.len() != self.n_channels {
            return Err(OpenSmellError::InvalidChannelCount {
                got: reading.len(),
                expected: self.n_channels,
            });
        }
        let mut z_scores = Vec::with_capacity(self.n_channels);
        let mut votes = 0usize;
        let mut max_z = 0.0f64;
        for (i, &x) in reading.iter().enumerate().take(self.n_channels) {
            let sd = self.var[i].sqrt();
            let zi = (x - self.mu[i]) / sd;
            z_scores.push(zi);
            let a = zi.abs();
            if a > self.config.threshold_sigma {
                votes += 1;
            }
            if a > max_z {
                max_z = a;
            }
        }
        Ok(EwmaVerdict {
            is_anomaly: votes >= self.config.min_votes,
            max_z,
            z_scores,
            anomaly_votes: votes,
        })
    }

    /// Advance the baseline by one reading (called after `detect`).
    pub fn update(&mut self, reading: &[f64]) {
        let a = self.config.alpha;
        let inv = 1.0 - a;
        for (i, &x) in reading.iter().enumerate().take(self.n_channels) {
            let res = x - self.mu[i];
            self.mu[i] = inv * self.mu[i] + a * x;
            self.var[i] = (inv * self.var[i] + a * res * res).max(1e-12);
        }
    }

    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calm_baseline_stays_quiet() {
        let mut c = EwmaControlChart::new(2);
        c.calibrate(&vec![vec![10.0; 2]; 100]).unwrap();
        let mut votes = 0usize;
        for _ in 0..200 {
            let v = c.detect(&[10.0, 10.0]).unwrap();
            c.update(&[10.0, 10.0]);
            votes += usize::from(v.is_anomaly);
        }
        assert_eq!(votes, 0);
    }

    #[test]
    fn step_fires_after_integration() {
        let mut c = EwmaControlChart::with_config(
            1,
            EwmaConfig {
                alpha: 0.05,
                threshold_sigma: 4.0,
                min_votes: 1,
            },
        );
        c.calibrate(&vec![vec![1.0]; 200]).unwrap();
        let mut fired = false;
        for _ in 0..60 {
            let v = c.detect(&[2.5]).unwrap();
            fired |= v.is_anomaly;
            c.update(&[2.5]);
        }
        assert!(fired, "sustained +1.5 step must trip the control chart");
    }

    #[test]
    fn uncalibrated_is_an_error() {
        let mut c = EwmaControlChart::new(1);
        assert!(c.detect(&[1.0]).is_err());
    }
}