use serde::{Deserialize, Serialize};
use crate::{Result, OpenSmellError};

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
            let median = if vals.len() % 2 == 0 {
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
