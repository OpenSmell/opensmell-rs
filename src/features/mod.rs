use serde::{Deserialize, Serialize};

/// Feature groups organized by use case.
/// Developers select only what they need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureGroup {
    /// Anomaly detection: drift rate, stability, noise floor, fractional derivatives.
    /// Use for: monitoring, spoilage detection, leak detection, cold-chain.
    Anomaly,
    /// Classification: absolute resistance, calibrated concentration.
    /// Use for: substance identification, fingerprinting.
    Classification,
    /// Sensor health: hysteresis, sensitivity decay, thermal profile, ADC noise.
    /// Use for: predicting sensor failure, scheduling maintenance.
    Health,
    /// Kinetics: rise time, decay time, multi-exponential decay parameters.
    /// Use for: understanding adsorption/desorption dynamics.
    Kinetics,
    /// Selectivity: cross-channel ratios for gas discrimination.
    /// Use for: improving classifier accuracy, understanding sensor overlap.
    Selectivity,
    /// Temporal: high-frequency transients, oscillation, response latency.
    /// Use for: detecting rapid events (gas leaks, spoilage onset).
    Temporal,
    /// Hardware: circuit response, thermal profile, ADC noise.
    /// Use for: diagnosing hardware issues, quality control.
    Hardware,
}

impl FeatureGroup {
    /// All available feature groups.
    pub fn all() -> &'static [FeatureGroup] {
        &[
            FeatureGroup::Anomaly,
            FeatureGroup::Classification,
            FeatureGroup::Health,
            FeatureGroup::Kinetics,
            FeatureGroup::Selectivity,
            FeatureGroup::Temporal,
            FeatureGroup::Hardware,
        ]
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            FeatureGroup::Anomaly => "Anomaly Detection",
            FeatureGroup::Classification => "Substance Classification",
            FeatureGroup::Health => "Sensor Health",
            FeatureGroup::Kinetics => "Adsorption Kinetics",
            FeatureGroup::Selectivity => "Cross-Channel Selectivity",
            FeatureGroup::Temporal => "Temporal Patterns",
            FeatureGroup::Hardware => "Hardware Diagnostics",
        }
    }

    /// Description of what this group is used for.
    pub fn description(&self) -> &'static str {
        match self {
            FeatureGroup::Anomaly => "Drift rate, stability indices, noise floor, fractional derivatives. For monitoring applications where you detect 'something is different' without knowing what.",
            FeatureGroup::Classification => "Absolute resistance values, calibrated concentrations, baseline-corrected signals. For identifying specific substances.",
            FeatureGroup::Health => "Hysteresis, sensitivity decay, noise floor, thermal profile. For predicting when sensors need replacement.",
            FeatureGroup::Kinetics => "Rise time, decay time, multi-exponential decay parameters (tau1-3, a1-3). For understanding molecular adsorption/desorption dynamics.",
            FeatureGroup::Selectivity => "Cross-channel selectivity ratios. For quantifying how well sensors distinguish between target gases.",
            FeatureGroup::Temporal => "High-frequency transients, oscillation frequency/amplitude, response latency. For detecting rapid events.",
            FeatureGroup::Hardware => "Circuit response characteristics, thermal profile, ADC noise. For diagnosing hardware issues.",
        }
    }
}

mod anomaly;
mod classification;
mod health;
mod kinetics;
mod selectivity;
mod temporal;

pub use anomaly::*;

use crate::{Baseline, SensorReading, Result};

/// Extract features from a single reading using specified feature groups.
pub fn extract_features(
    reading: &SensorReading,
    baseline: &Baseline,
    groups: &[FeatureGroup],
) -> Result<Vec<f64>> {
    let normalized = baseline.normalize(&reading.channels);
    let mut features = Vec::new();

    for group in groups {
        let mut group_features = match group {
            FeatureGroup::Anomaly => anomaly::extract(&normalized, &reading.channels, baseline)?,
            FeatureGroup::Classification => classification::extract(&normalized, &reading.channels, baseline)?,
            FeatureGroup::Health => health::extract(&normalized, &reading.channels, baseline)?,
            FeatureGroup::Kinetics => kinetics::extract(&normalized, &reading.channels, baseline)?,
            FeatureGroup::Selectivity => selectivity::extract(&normalized, &reading.channels, baseline)?,
            FeatureGroup::Temporal => temporal::extract(&normalized, &reading.channels, baseline)?,
            FeatureGroup::Hardware => hardware::extract(&normalized, &reading.channels, baseline)?,
        };
        features.append(&mut group_features);
    }
    Ok(features)
}

/// Get feature names for specified groups.
pub fn feature_names(groups: &[FeatureGroup], n_channels: usize) -> Vec<String> {
    let mut names = Vec::new();
    for group in groups {
        let group_names = match group {
            FeatureGroup::Anomaly => anomaly::names(n_channels),
            FeatureGroup::Classification => classification::names(n_channels),
            FeatureGroup::Health => health::names(n_channels),
            FeatureGroup::Kinetics => kinetics::names(n_channels),
            FeatureGroup::Selectivity => selectivity::names(n_channels),
            FeatureGroup::Temporal => temporal::names(n_channels),
            FeatureGroup::Hardware => hardware::names(n_channels),
        };
        names.extend(group_names);
    }
    names
}

mod hardware {
    use super::*;

    pub fn extract(_normalized: &[f64], raw: &[f64], _baseline: &crate::Baseline) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        // Circuit response: ratio of signal variance to baseline variance
        let variance: f64 = raw.iter().map(|v| v.powi(2)).sum::<f64>() / raw.len() as f64
            - raw.iter().sum::<f64>().powi(2) / (raw.len() as f64).powi(2);
        features.push(variance.sqrt());
        // ADC noise: high-frequency component (adjacent sample differences)
        if raw.len() > 1 {
            let hf: f64 = raw.windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .sum::<f64>() / (raw.len() - 1) as f64;
            features.push(hf);
        } else {
            features.push(0.0);
        }
        Ok(features)
    }

    pub fn names(_n_channels: usize) -> Vec<String> {
        vec![
            "circuit_response".to_string(),
            "adc_noise".to_string(),
        ]
    }
}

/// Extract features from a time series window (for monitoring/anomaly detection).
pub fn extract_window_features(
    window: &[Vec<f64>],
    baseline: &Baseline,
    groups: &[FeatureGroup],
) -> Result<Vec<f64>> {
    if window.is_empty() {
        return Err(crate::OpenSmellError::InsufficientData { expected: 1, actual: 0 });
    }
    let _n_channels = window[0].len();
    let mut features = Vec::new();

    for group in groups {
        let mut group_features = match group {
            FeatureGroup::Anomaly => anomaly::extract_window(window, baseline)?,
            FeatureGroup::Classification => classification::extract_window(window, baseline)?,
            FeatureGroup::Health => health::extract_window(window, baseline)?,
            FeatureGroup::Kinetics => kinetics::extract_window(window, baseline)?,
            FeatureGroup::Selectivity => selectivity::extract_window(window, baseline)?,
            FeatureGroup::Temporal => temporal::extract_window(window, baseline)?,
            FeatureGroup::Hardware => hardware_extract_window(window, baseline)?,
        };
        features.append(&mut group_features);
    }
    Ok(features)
}

fn hardware_extract_window(window: &[Vec<f64>], _baseline: &crate::Baseline) -> Result<Vec<f64>> {
    let n_channels = window[0].len();
    let mut features = Vec::new();

    for ch in 0..n_channels {
        let vals: Vec<f64> = window.iter().map(|s| s[ch]).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        let hf: f64 = vals.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>()
            / (vals.len().max(1) - 1) as f64;
        features.push(variance.sqrt());
        features.push(hf);
    }
    Ok(features)
}
