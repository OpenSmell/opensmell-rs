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
    /// EWMA weight on the reading and on the squared residual, per reference-
    /// cadence sample. Scaled to the actual inter-reading gap at update time so
    /// a slow-logged stream smooths at the same physical time constant.
    pub alpha: f64,
    /// Sample is anomalous when a channel's |z| exceeds this many sigma.
    pub threshold_sigma: f64,
    /// Minimum number of simultaneously-exceeding channels for a vote.
    pub min_votes: usize,
    /// Reference cadence `alpha` is quoted at (default 10 Hz). `update_with_dt`
    /// folds the true gap to this cadence.
    #[serde(default = "default_sample_period_s")]
    pub sample_period_s: f64,
    /// Wall-clock confirmation window: a raw per-sample vote (≥ `min_votes`
    /// channels) only hardens into an anomaly after it has *persisted*
    /// `confirm_window_s` seconds of real time (accumulated via `dt_s`). 0.0
    /// disables confirmation (a single-sample vote fires immediately — the
    /// legacy behaviour). At any cadence, an episode must endure the same
    /// physical window before the decision layer asserts it, so a ~6 s TADI
    /// logger and a 10 Hz stream share one decision time constant.
    #[serde(default)]
    pub confirm_window_s: f64,
}

fn default_sample_period_s() -> f64 {
    0.1
}

fn default_confirm_window_s() -> f64 {
    0.0
}

impl Default for EwmaConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            threshold_sigma: 5.0,
            min_votes: 2,
            sample_period_s: default_sample_period_s(),
            confirm_window_s: default_confirm_window_s(),
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
    /// Wall-clock seconds the raw vote has persisted (only meaningful when
    /// `confirm_window_s` is configured).
    #[serde(default)]
    pub confirm_hold_s: f64,
}

/// Streaming EWMA control chart over `n_channels` parallel channels.
#[derive(Debug, Clone)]
pub struct EwmaControlChart {
    n_channels: usize,
    config: EwmaConfig,
    mu: Vec<f64>,
    var: Vec<f64>,
    calibrated: bool,
    /// Wall-clock seconds the current raw vote has persisted without an
    /// intervening normal sample (drives `confirm_window_s` gating).
    confirm_hold_s: f64,
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
            confirm_hold_s: 0.0,
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

    /// Score one reading with the reference-cadence assumption (dt = the
    /// configured `sample_period_s`). Applies the confirmation window over the
    /// reference time constant.
    pub fn detect(&mut self, reading: &[f64]) -> Result<EwmaVerdict> {
        self.detect_with_dt(reading, self.config.sample_period_s.max(1e-6))
    }

    /// Score one reading `dt_s` seconds after the previous one. The verdict is
    /// gated by a wall-clock confirmation window: a raw vote (≥ `min_votes`
    /// channels) only asserts `is_anomaly` after it has persisted
    /// `confirm_window_s` seconds of accumulated inter-sample time. A slow-
    /// logged (e.g. 6 s TADI) stream and a 10 Hz stream then demand the same
    /// physical episode length before the decision layer fires.
    pub fn detect_with_dt(&mut self, reading: &[f64], dt_s: f64) -> Result<EwmaVerdict> {
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
        // Wall-clock persistence gate (pure decision layer, no baseline effect).
        let raw_anomaly = votes >= self.config.min_votes;
        if self.config.confirm_window_s > 0.0 {
            if raw_anomaly {
                self.confirm_hold_s += dt_s.max(1e-6);
            } else {
                self.confirm_hold_s = 0.0;
            }
        }
        let confirmed = if self.config.confirm_window_s > 0.0 {
            self.confirm_hold_s + 1e-9 >= self.config.confirm_window_s
        } else {
            raw_anomaly
        };
        Ok(EwmaVerdict {
            is_anomaly: confirmed,
            max_z,
            z_scores,
            anomaly_votes: votes,
            confirm_hold_s: self.confirm_hold_s,
        })
    }

    /// Advance the baseline by one reading (called after `detect`).
    pub fn update(&mut self, reading: &[f64]) {
        self.update_with_dt(reading, self.config.sample_period_s.max(1e-6))
    }

    /// Advance the baseline by one reading taken `dt_s` seconds after the
    /// previous one. The EWMA weight is per physical time: over a longer gap
    /// the baseline chases harder (`∑α = α·dt/period`), so a slow-logged
    /// stream smooths over the same wall-clock window it would at 10 Hz.
    pub fn update_with_dt(&mut self, reading: &[f64], dt_s: f64) {
        let period = self.config.sample_period_s.max(1e-6);
        let dr = (dt_s.max(1e-6) / period).max(1e-6);
        // At the reference cadence use the configured α exactly (byte-identical
        // to the legacy per-sample update).
        let a = if dr == 1.0 {
            self.config.alpha
        } else {
            1.0 - (1.0 - self.config.alpha).powf(dr)
        };
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
                    sample_period_s: 0.1,
                    confirm_window_s: 0.0,
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

    #[test]
    fn update_with_dt_at_reference_cadence_is_identical_to_update() {
        let mut a = EwmaControlChart::new(2);
        a.calibrate(&vec![vec![10.0; 2]; 100]).unwrap();
        let mut b = EwmaControlChart::new(2);
        b.calibrate(&vec![vec![10.0; 2]; 100]).unwrap();
        for &r in &[10.6_f64, 10.2, 9.7, 11.1, 10.3] {
            a.update(&[r, r]);
            b.update_with_dt(&[r, r], 0.1);
            assert_eq!(a.mu, b.mu, "mu byte-identical at reference cadence");
            assert_eq!(a.var, b.var, "var byte-identical at reference cadence");
        }
    }

    #[test]
    fn update_with_dt_is_time_homogeneous_for_mean() {
        // Geometric-EWMA invariant: one 0.5 s step moves the level exactly like
        // five 0.1 s steps, since (1−α)^(5·dr=5) applied once equals α stepped
        // five times. (The variance is *not* identical — residuals are taken
        // against the moving mean — so only the level must match.)
        let mut one = EwmaControlChart::new(1);
        one.calibrate(&vec![vec![1.0]; 200]).unwrap();
        let mut five = EwmaControlChart::new(1);
        five.calibrate(&vec![vec![1.0]; 200]).unwrap();
        one.detect(&[2.5]).unwrap();
        one.update_with_dt(&[2.5], 0.5);
        five.detect(&[2.5]).unwrap();
        for _ in 0..5 {
            five.update_with_dt(&[2.5], 0.1);
        }
        assert!((one.mu[0] - five.mu[0]).abs() < 1e-9, "level: 1×0.5 s ≡ 5×0.1 s");
        assert!(one.var[0].is_finite() && one.var[0] > 0.0);
        assert!(five.var[0].is_finite() && five.var[0] > 0.0);
    }

    #[test]
    fn confirm_window_gates_on_wall_clock_not_sample_count() {
        // A strong step produces a transient multi-sample vote (the EWMA's
        // variance inflates and the baseline absorbs it after a few samples —
        // exactly why single-sample blips are the FP mechanism). With
        // confirm_window_s = 0.2 s the decision must fire on the reading that
        // crosses 0.2 s of *accumulated* anomaly: sample 2 at 10 Hz (2 × 0.1 s)
        // but sample 1 at 6 s cadence (1 × 6 s). Same wall-clock recipe,
        // different sample counts — that IS the cadence invariance.
        let cfg = |window: f64| EwmaConfig {
            alpha: 0.05,
            threshold_sigma: 4.0,
            min_votes: 1,
            sample_period_s: 0.1,
            confirm_window_s: window,
        };
        let step = vec![1.0e6];
        let fire_on_sample = |dt: f64| {
            let mut c = EwmaControlChart::with_config(1, cfg(0.2));
            c.calibrate(&vec![vec![1.0]; 200]).unwrap();
            for k in 0..8 {
                let v = c.detect_with_dt(&step, dt).unwrap();
                if v.is_anomaly {
                    return (k + 1, v.confirm_hold_s);
                }
                c.update_with_dt(&step, dt);
            }
            panic!("window never fired at dt={dt}");
        };
        let (n10, hold10) = fire_on_sample(0.1);
        assert_eq!(n10, 2, "10 Hz crosses 0.2 s on the 2nd sample");
        assert!(hold10 >= 0.2, "10 Hz hold {hold10}");
        let (n6, hold6) = fire_on_sample(6.0);
        assert_eq!(n6, 1, "6 s cadence crosses 0.2 s on its 1st reading (6 s > 0.2 s)");
        assert!(hold6 >= 0.2, "6 s hold {hold6}");
        // A normal sample in between resets the persistence: an episode shorter
        // than the window must NOT fire even though a raw vote occurred.
        let mut c = EwmaControlChart::with_config(1, cfg(6.0));
        c.calibrate(&vec![vec![1.0]; 200]).unwrap();
        assert!(!c.detect_with_dt(&step, 0.1).unwrap().is_anomaly, "0.1 s < 6 s window");
        c.update_with_dt(&step, 0.1);
        c.update_with_dt(&[1.0], 0.1); // normal sample resets the window
        assert!(!c.detect_with_dt(&step, 0.1).unwrap().is_anomaly, "reset must not fire");
        // Disabled (legacy): a lone single-sample vote fires immediately.
        let mut c0 = EwmaControlChart::with_config(1, cfg(0.0));
        c0.calibrate(&vec![vec![1.0]; 200]).unwrap();
        assert!(c0.detect_with_dt(&step, 6.0).unwrap().is_anomaly,
            "legacy single-sample vote must fire instantly");
    }
}