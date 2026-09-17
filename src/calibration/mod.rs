use serde::{Deserialize, Serialize};
use crate::{Result, OpenSmellError};

use crate::anomaly::EngineConfig;

/// Stream cadence the engine assumes (10 Hz) — matches `dual.rs::SAMPLE_PERIOD_S`.
const SAMPLE_PERIOD_S: f64 = 0.1;

/// Calibration profile for a sensor rig.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationProfile {
    /// Per-channel R0 baseline values.
    pub r0: Vec<f64>,
    /// Per-channel baseline standard deviation.
    pub baseline_std: Vec<f64>,
    /// Number of baseline samples used.
    pub baseline_samples: usize,
    /// Timestamp of calibration.
    pub timestamp: f64,
    /// Device identifier.
    pub device_id: String,
    /// Sensor cartridge IDs (for razor-blade tracking).
    pub cartridge_ids: Vec<String>,
    /// Calibration quality score (0.0-1.0).
    pub quality: f64,
}

/// Sensor swap event record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapEvent {
    /// Timestamp of swap.
    pub timestamp: f64,
    /// Device ID.
    pub device_id: String,
    /// Channel index that was swapped.
    pub channel: usize,
    /// Old cartridge ID.
    pub old_cartridge: String,
    /// New cartridge ID.
    pub new_cartridge: String,
    /// Whether calibration was transferred successfully.
    pub transfer_success: bool,
    /// Post-swap verification score.
    pub verification_score: f64,
}

/// Calibration engine for sensor cartridge swaps.
pub struct Calibrator {
    /// Current calibration profile.
    pub profile: CalibrationProfile,
    /// History of swap events.
    swap_history: Vec<SwapEvent>,
}

impl Calibrator {
    /// Create a new calibrator with zero-calibration baseline.
    /// This is the v0 approach: 30 min in normal air, no reference chemicals.
    pub fn zero_calibration(device_id: String, n_channels: usize) -> Self {
        Self {
            profile: CalibrationProfile {
                r0: vec![0.0; n_channels],
                baseline_std: vec![1.0; n_channels],
                baseline_samples: 0,
                timestamp: 0.0,
                device_id,
                cartridge_ids: vec![String::new(); n_channels],
                quality: 0.0,
            },
            swap_history: Vec::new(),
        }
    }

    /// Update baseline from calibration samples (zero-calibration or reference-based).
    pub fn calibrate(&mut self, samples: &[Vec<f64>], timestamp: f64) -> Result<()> {
        if samples.is_empty() {
            return Err(OpenSmellError::InsufficientData { expected: 1, actual: 0 });
        }
        let n_channels = samples[0].len();
        let baseline_end = (samples.len() as f64 * 0.15) as usize;
        let baseline_end = baseline_end.max(1);

        let mut r0 = Vec::with_capacity(n_channels);
        let mut std = Vec::with_capacity(n_channels);

        for ch in 0..n_channels {
            let mut vals: Vec<f64> = samples[..baseline_end]
                .iter()
                .map(|s| s[ch])
                .filter(|v| v.is_finite() && *v > 0.0)
                .collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if vals.len().is_multiple_of(2) {
                (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.0
            } else {
                vals[vals.len() / 2]
            };
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            r0.push(median);
            std.push(variance.sqrt());
        }

        // Quality score: based on baseline stability
        let mean_cv: f64 = std.iter().zip(r0.iter())
            .map(|(&s, &r)| if r > 0.0 { s / r } else { 1.0 })
            .sum::<f64>() / n_channels as f64;
        let quality = (1.0 - mean_cv).max(0.0).min(1.0);

        self.profile = CalibrationProfile {
            r0,
            baseline_std: std,
            baseline_samples: baseline_end,
            timestamp,
            device_id: self.profile.device_id.clone(),
            cartridge_ids: self.profile.cartridge_ids.clone(),
            quality,
        };
        Ok(())
    }

    /// Execute a sensor cartridge swap.
    /// Returns the new calibration profile after transfer.
    pub fn swap_cartridge(
        &mut self,
        channel: usize,
        new_cartridge_id: String,
        timestamp: f64,
    ) -> Result<SwapEvent> {
        if channel >= self.profile.r0.len() {
            return Err(OpenSmellError::InvalidChannelCount {
                got: channel + 1,
                expected: self.profile.r0.len(),
            });
        }

        let old_cartridge = self.profile.cartridge_ids[channel].clone();

        // Calibration transfer strategy:
        // 1. Keep R0 from old cartridge (warm sensor has stable baseline)
        // 2. Reset std to default (new sensor needs new baseline)
        // 3. Mark as needing re-baseline
        let transfer_success = true; // Always succeeds for MOX (same model)
        let verification_score = 0.5; // Needs re-baseline to reach full quality

        self.profile.cartridge_ids[channel] = new_cartridge_id.clone();
        self.profile.baseline_std[channel] = 1.0; // Reset std
        self.profile.quality *= 0.8; // Reduce quality until re-baseline

        let event = SwapEvent {
            timestamp,
            device_id: self.profile.device_id.clone(),
            channel,
            old_cartridge,
            new_cartridge: new_cartridge_id,
            transfer_success,
            verification_score,
        };
        self.swap_history.push(event.clone());
        Ok(event)
    }

    /// Get fleet status: which cartridges need replacement.
    pub fn cartridge_status(&self) -> Vec<CartridgeStatus> {
        self.profile.r0.iter().enumerate().map(|(i, &r0)| {
            CartridgeStatus {
                channel: i,
                cartridge_id: self.profile.cartridge_ids[i].clone(),
                r0,
                baseline_std: self.profile.baseline_std[i],
                age_hours: 0.0, // Would need swap history to compute
                needs_replacement: false,
            }
        }).collect()
    }

    /// Normalize a reading using current calibration.
    pub fn normalize(&self, raw: &[f64]) -> Vec<f64> {
        raw.iter().zip(self.profile.r0.iter())
            .map(|(&rs, &r0)| if r0 > 0.0 { (rs - r0) / r0 } else { 0.0 })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartridgeStatus {
    pub channel: usize,
    pub cartridge_id: String,
    pub r0: f64,
    pub baseline_std: f64,
    pub age_hours: f64,
    pub needs_replacement: bool,
}

/// Cross-device calibration: align features between different rigs.
pub struct CrossDeviceCalibrator {
    /// Reference device calibration.
    reference: CalibrationProfile,
    /// Target device calibration.
    target: CalibrationProfile,
    /// Per-channel gain alignment.
    gain_map: Vec<f64>,
}

impl CrossDeviceCalibrator {
    /// Create a new cross-device calibrator from reference and target profiles.
    pub fn new(reference: CalibrationProfile, target: CalibrationProfile) -> Self {
        let n = reference.r0.len().min(target.r0.len());
        let mut gain_map = Vec::with_capacity(n);
        for i in 0..n {
            let gain = if target.r0[i] > 0.0 {
                reference.r0[i] / target.r0[i]
            } else { 1.0 };
            gain_map.push(gain);
        }
        Self { reference, target, gain_map }
    }

    /// Align a feature vector from target device to reference device space.
    pub fn align(&self, features: &[f64]) -> Vec<f64> {
        features.iter().zip(self.gain_map.iter())
            .map(|(&f, &g)| f * g)
            .collect()
    }

    /// Reference device calibration profile.
    pub fn reference(&self) -> &CalibrationProfile {
        &self.reference
    }

    /// Target device calibration profile.
    pub fn target(&self) -> &CalibrationProfile {
        &self.target
    }

    /// Alignment quality score (0.0-1.0).
    pub fn alignment_quality(&self) -> f64 {
        let mean_gain = self.gain_map.iter().sum::<f64>() / self.gain_map.len() as f64;
        let variance = self.gain_map.iter()
            .map(|g| (g - mean_gain).powi(2))
            .sum::<f64>() / self.gain_map.len() as f64;
        // Good alignment: gains are close to 1.0 and consistent
        let closeness = 1.0 / (1.0 + (mean_gain - 1.0).abs());
        let consistency = 1.0 / (1.0 + variance.sqrt());
        (closeness + consistency) / 2.0
    }
}

/// Physics-derived engine tuning from calibration data.
///
/// Replaces the engineer-guessed defaults in `EngineConfig` with values
/// estimated from the sensor's own window:
/// - `q_state`  — state process noise, from the first-differenced baseline
///   variance (`Var(Δy)/2` for a drifted random-walk level; an over-bound when
///   measurement noise dominates, which keeps the state filter able to follow
///   real events instead of pinning on noise).
/// - `q_param`  — parameter-walk noise, from the drift path's first-difference
///   variance distributed over the underlying 10 Hz cadence (so a 24 h drift
///   history normalizes into a per-sample variance ≪ `q_state`).
/// - `tau_s`    — per-channel desorption time constants from an exponential fit
///   of the post-exposure recovery tail (meters the adsorption-memory
///   transient's `m` decay).
/// - `alpha`    — per-channel power-law exponents from the log-log slope of
///   known-concentration response pairs (1.0 = linear).
/// - `baseline_std` — per-channel calibrated σ of the baseline window.
///
/// `apply` writes the fitted values into an `EngineConfig`, so a deployment
/// gets physics-based process noise, adsorption memory, and response shape
/// instead of constants.
#[derive(Debug, Clone, Default)]
pub struct AutoTune {
    pub q_state: f64,
    pub q_param: f64,
    pub baseline_std: Vec<f64>,
    pub tau_s: Vec<f64>,
    pub alpha: Vec<f64>,
}

fn channel_mean(samples: &[Vec<f64>], i: usize) -> f64 {
    samples.iter().map(|s| s[i]).sum::<f64>() / samples.len() as f64
}

fn channel_variance(samples: &[Vec<f64>], i: usize) -> f64 {
    let mean = channel_mean(samples, i);
    samples.iter().map(|s| (s[i] - mean).powi(2)).sum::<f64>() / samples.len() as f64
}

impl AutoTune {
    /// Fit the baseline-derived knobs (`q_state`, `baseline_std`) from a
    /// steady calibration window (same input shape as
    /// `DualKalmanEngine::calibrate_baseline`).
    pub fn from_baseline(samples: &[Vec<f64>]) -> Result<Self> {
        if samples.is_empty() {
            return Err(OpenSmellError::InsufficientData { expected: 1, actual: 0 });
        }
        if samples.len() < 2 {
            // Need at least two samples for a first difference.
            return Err(OpenSmellError::InsufficientData { expected: 2, actual: samples.len() });
        }
        let c = samples[0].len();
        for s in samples {
            if s.len() != c {
                return Err(OpenSmellError::InvalidChannelCount { got: s.len(), expected: c });
            }
        }
        // First-difference variance per channel: for a level x that random-walks
        // with process noise Q and iid measurement noise σ², Var(Δy) = 2Q + 2σ²,
        // so Q = Var(Δy)/2 − σ². We report the roadmap's upper-bound form
        // Var(Δ)/2 and clamp at a floor derived from the measurement noise, so a
        // nearly-flat baseline never pins the state filter rigidly.
        let n = samples.len() - 1;
        let sigma2: Vec<f64> = (0..c).map(|i| channel_variance(samples, i)).collect();
        let mut q_state = 0.0;
        for i in 0..c {
            let mut d2 = 0.0;
            for w in samples.windows(2) {
                let d = w[1][i] - w[0][i];
                d2 += d * d;
            }
            d2 /= n as f64;
            let floor = (sigma2[i] * 1e-3).max(1e-6);
            let qi = (d2 / 2.0).max(floor);
            q_state += qi;
        }
        q_state /= c as f64;
        let q_state = q_state.max(1e-6);

        Ok(Self {
            q_state,
            q_param: 1e-6,
            baseline_std: (0..c).map(|i| sigma2[i].sqrt()).collect(),
            tau_s: Vec::new(),
            alpha: Vec::new(),
        })
    }

    /// Fit `q_param` from a drift path: `(time_seconds, per-channel level)`
    /// samples of how the baseline wandered (e.g. hourly medians over 24 h).
    /// The per-sample walk variance is the drift path's first-difference
    /// variance divided by the number of 10 Hz samples each gap spans.
    pub fn with_drift(&mut self, drift: &[(f64, Vec<f64>)]) -> Result<&mut Self> {
        if drift.len() < 2 {
            return Err(OpenSmellError::InsufficientData { expected: 2, actual: drift.len() });
        }
        let c = self.baseline_std.len();
        if c == 0 {
            return Err(OpenSmellError::InsufficientData { expected: 1, actual: 0 });
        }
        let mut q = 0.0;
        let mut gap_samples = 0.0;
        for w in drift.windows(2) {
            let (t0, l0) = (&w[0].0, &w[0].1);
            let (t1, l1) = (&w[1].0, &w[1].1);
            if t1 <= t0 {
                continue;
            }
            gap_samples += (t1 - t0) / SAMPLE_PERIOD_S;
            for i in 0..c {
                let d = l1[i] - l0[i];
                q += d * d;
            }
        }
        gap_samples = gap_samples.max(1.0);
        let q = (q / (gap_samples * c.max(1) as f64)).max(1e-8);
        self.q_param = q.min(self.q_state.max(q * 10.0));
        Ok(self)
    }

    /// Fit per-channel desorption time constants (seconds) from the
    /// post-exposure recovery tail (the sensor "still smells" decay). For each
    /// channel the exponential's baseline cancels in the first difference
    /// (`y(t)−y(t+Δt) = A·e^(−t/τ)·(1−e^(−Δt/τ))`), so `ln(Δy)` is regressed
    /// against t and τ = −Δt/slope without estimating the asymptote.
    pub fn with_desorption(&mut self, tail: &[Vec<f64>]) -> Result<&mut Self> {
        if tail.len() < 3 {
            return Err(OpenSmellError::InsufficientData { expected: 3, actual: tail.len() });
        }
        let c = tail[0].len();
        self.tau_s = (0..c)
            .map(|i| {
                let mut pts: Vec<(f64, f64)> = tail
                    .windows(2)
                    .enumerate()
                    .filter_map(|(k, w)| {
                        let d = w[0][i] - w[1][i];
                        if d > 1e-9 {
                            Some((k as f64 * SAMPLE_PERIOD_S, d.ln()))
                        } else {
                            None
                        }
                    })
                    .collect();
                if pts.len() < 3 {
                    return default_tau_s();
                }
                pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let n = pts.len();
                let t_mean = pts.iter().map(|p| p.0).sum::<f64>() / n as f64;
                let u_mean = pts.iter().map(|p| p.1).sum::<f64>() / n as f64;
                let mut num = 0.0;
                let mut den = 0.0;
                for &(t, u) in &pts {
                    num += (t - t_mean) * (u - u_mean);
                    den += (t - t_mean).powi(2);
                }
                if den.abs() < 1e-12 {
                    return default_tau_s();
                }
                let slope = num / den;
                if slope >= 0.0 {
                    // No decay (or growth) → no useful memory; keep the default.
                    return default_tau_s();
                }
                let tau = -1.0 / slope;
                tau.clamp(1.0, 3600.0)
            })
            .collect();
        Ok(self)
    }

    /// Fit per-channel power-law exponents from known-concentration response
    /// pairs `(x, y)`: `y = g·x^α`, so a log-log slope of the pairs is α.
    /// At least two pairs per channel are needed; a monotone fit is enforced
    /// by clamping to the physically-sensible exponent range.
    pub fn with_response(&mut self, pairs: &[(Vec<f64>, Vec<f64>)]) -> Result<&mut Self> {
        if pairs.len() < 2 {
            return Err(OpenSmellError::InsufficientData { expected: 2, actual: pairs.len() });
        }
        let c = pairs[0].0.len();
        self.alpha = (0..c)
            .map(|i| {
                let mut logx: Vec<f64> = Vec::new();
                let mut logy: Vec<f64> = Vec::new();
                for (x, y) in pairs {
                    let (xi, yi) = (x[i], y[i]);
                    if xi > 0.0 && yi > 0.0 {
                        logx.push(xi.ln());
                        logy.push(yi.ln());
                    }
                }
                if logx.len() < 2 {
                    return default_alpha_s();
                }
                let n = logx.len();
                let sx = logx.iter().sum::<f64>() / n as f64;
                let sy = logy.iter().sum::<f64>() / n as f64;
                let mut num = 0.0;
                let mut den = 0.0;
                for k in 0..n {
                    num += (logx[k] - sx) * (logy[k] - sy);
                    den += (logx[k] - sx).powi(2);
                }
                if den.abs() < 1e-12 {
                    return default_alpha_s();
                }
                let alpha = num / den;
                alpha.clamp(0.1, 3.0)
            })
            .collect();
        Ok(self)
    }

    /// Write the fitted values into an `EngineConfig`. The adsorption and
    /// response feature switches themselves stay as configured — only their
    /// physics parameters are overridden from the calibration data.
    pub fn apply(&self, config: &mut EngineConfig) {
        config.q_state = self.q_state;
        config.q_param = self.q_param;
        if !self.tau_s.is_empty() {
            config.adsorption.tau_s = self.tau_s.clone();
            config.adsorption.tau_default_s = self.tau_s[0];
        }
        if !self.alpha.is_empty() {
            config.response.alpha = self.alpha.clone();
            config.response.alpha_default = self.alpha[0];
        }
    }
}

fn default_tau_s() -> f64 {
    300.0
}

fn default_alpha_s() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_walk(n: usize, q: f64, sigma: f64) -> Vec<Vec<f64>> {
        // Deterministic-ish pseudo-random walk wrapped as a 1-channel series:
        // x_{t+1} = x_t + w, y = x + v; w~N(0,q), v~N(0,sigma²).
        let mut x = 0.0;
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            // LCG noise in [-1,1].
            let w = ((k as u64 * 2654435761 % 1000) as f64 / 1000.0 - 0.5) * 2.0;
            let v = (((k as u64 * 40503 % 1000) as f64 / 1000.0) - 0.5) * 2.0;
            x += w * q.sqrt();
            let y = x + v * sigma;
            out.push(vec![y]);
        }
        out
    }

    #[test]
    fn baseline_infers_q_state_scale() {
        // A strong random walk (Q = 1e-3) plus modest measurement noise must
        // infer q_state ~ 1e-3 (first-difference variance / 2 dominates).
        let series = rand_walk(2000, 1e-3, 0.03);
        let t = AutoTune::from_baseline(&series).unwrap();
        assert!(
            (t.q_state - 1e-3).abs() / 1e-3 < 0.5,
            "q_state {:.4} should sit near 1e-3",
            t.q_state
        );
        assert_eq!(t.baseline_std.len(), 1);
    }

    #[test]
    fn baseline_empty_is_error() {
        let r = AutoTune::from_baseline(&[]);
        assert!(r.is_err());
    }

    #[test]
    fn drift_path_infers_q_param() {
        let mut t = AutoTune {
            q_state: 1e-3,
            baseline_std: vec![1.0],
            ..Default::default()
        };
        // 24 h drift: baseline wanders by ±0.2 once an hour, so the per-second
        // drift rate is tiny and the per-10Hz-sample variance smaller still.
        let mut drift = Vec::with_capacity(25);
        for k in 0..25 {
            let lev = 1.0 + 0.2 * ((k as f64 * 0.7).sin());
            drift.push((k as f64 * 3600.0, vec![lev]));
        }
        t.with_drift(&drift).unwrap();
        assert!(
            t.q_param > 0.0 && t.q_param < t.q_state,
            "drift-derived q_param {:.2e} must be positive and ≪ q_state",
            t.q_param
        );
    }

    #[test]
    fn tau_least_squares_recovers_desorption_constant() {
        // True τ = 50 s: y = 1 + exp(-t/50) over 120 s at 10 Hz.
        let tau_true = 50.0;
        let tail: Vec<Vec<f64>> = (0..1200)
            .map(|k| {
                let t = k as f64 * SAMPLE_PERIOD_S;
                vec![1.0 + (-t / tau_true).exp()]
            })
            .collect();
        let mut t = AutoTune {
            baseline_std: vec![1.0],
            ..Default::default()
        };
        t.with_desorption(&tail).unwrap();
        assert_eq!(t.tau_s.len(), 1);
        assert!(
            (t.tau_s[0] - tau_true).abs() < 10.0,
            "recovered τ {} vs true {tau_true}",
            t.tau_s[0]
        );
    }

    #[test]
    fn alpha_loglog_recovers_exponent() {
        // True α = 0.5: y = 2·x^0.5 at known levels x = 1, 4, 16.
        let mut t = AutoTune::default();
        let pairs = vec![
            (vec![1.0], vec![2.0]),
            (vec![4.0], vec![4.0]),
            (vec![16.0], vec![8.0]),
        ];
        t.with_response(&pairs).unwrap();
        assert_eq!(t.alpha.len(), 1);
        assert!(
            (t.alpha[0] - 0.5).abs() < 1e-9,
            "log-log slope {} should be 0.5",
            t.alpha[0]
        );
    }

    #[test]
    fn apply_writes_physic_based_config() {
        let mut config = EngineConfig::default();
        config.adsorption.enabled = true;
        config.response.power_law_enabled = true;
        let tuned = AutoTune {
            q_state: 2.5e-3,
            q_param: 1.2e-8,
            baseline_std: vec![0.03],
            tau_s: vec![42.0],
            alpha: vec![0.6],
        };
        tuned.apply(&mut config);
        assert_eq!(config.q_state, 2.5e-3);
        assert_eq!(config.q_param, 1.2e-8);
        assert_eq!(config.adsorption.tau_s, vec![42.0]);
        assert_eq!(config.response.alpha, vec![0.6]);
        // Feature switches stay caller-controlled (auto-tune only fills params).
        assert!(config.adsorption.enabled);
        assert!(config.response.power_law_enabled);
    }
}
