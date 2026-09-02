
/// Errors that can occur during feature extraction or detection.
#[derive(Debug, thiserror::Error)]
pub enum OpenSmellError {
    #[error("Insufficient data: need at least {expected} samples, got {actual}")]
    InsufficientData { expected: usize, actual: usize },

    #[error("Invalid channel count: got {got}, expected {expected}")]
    InvalidChannelCount { got: usize, expected: usize },

    #[error("All channels are dead (zero variance)")]
    AllChannelsDead,

    #[error("Feature extraction failed: {0}")]
    FeatureExtraction(String),

    #[error("Anomaly detection failed: {0}")]
    AnomalyDetection(String),

    #[error("Calibration failed: {0}")]
    Calibration(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, OpenSmellError>;

/// A single sensor reading: raw resistance values across N channels.
#[derive(Debug, Clone)]
pub struct SensorReading {
    pub channels: Vec<f64>,
    pub timestamp: f64,
    pub active_channels: Vec<usize>,
}

impl SensorReading {
    pub fn new(channels: Vec<f64>, timestamp: f64) -> Self {
        let active_channels = channels.iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0 && v.is_finite())
            .map(|(i, _)| i)
            .collect();
        Self { channels, timestamp, active_channels }
    }

    pub fn n_active(&self) -> usize {
        self.active_channels.len()
    }

    pub fn active_values(&self) -> Vec<f64> {
        self.active_channels.iter().map(|&i| self.channels[i]).collect()
    }
}

/// Baseline calibration data (R0 values per channel).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Baseline {
    pub r0: Vec<f64>,
    pub n_samples: usize,
    pub std: Vec<f64>,
}

impl Baseline {
    pub fn from_samples(samples: &[Vec<f64>]) -> Self {
        if samples.is_empty() {
            return Self { r0: vec![], n_samples: 0, std: vec![] };
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
        Self { r0, n_samples: baseline_end, std }
    }

    pub fn normalize(&self, raw: &[f64]) -> Vec<f64> {
        raw.iter().zip(self.r0.iter())
            .map(|(&rs, &r0)| if r0 > 0.0 { (rs - r0) / r0 } else { 0.0 })
            .collect()
    }
}

pub mod features;
pub mod anomaly;
pub mod calibration;
pub mod health;
pub mod protocol;
pub mod preprocessing;
pub mod framework;
pub mod adaptive;
pub mod poisoning;
pub mod quality;

pub mod training;

pub mod live;

pub mod smellability;
pub use features::{FeatureGroup, extract_features, extract_window_features, feature_names};
pub use anomaly::{AnomalyDetector, AnomalyScore, AnomalyMethod};
pub use calibration::{Calibrator, CalibrationProfile, CrossDeviceCalibrator};
pub use health::{HealthMonitor, SensorHealth, HealthStatus, FleetHealth, fisher_discriminant_ratio, pairwise_fdr, euclidean_distance, cosine_similarity, similarity_warning};
pub use protocol::{OsmProtocol, OsmMessage};
pub use preprocessing::{RawData, BaselineCorrection, BaselineMethod, SignalFilter, FilterType, WindowExtractor, DataValidator};
pub use adaptive::{AdaptiveAnomalyDetector, AdaptiveThreshold, FailSafeSystem, LabelingSystem, DetectionResult, AccuracyImprovement, DetectorState, LabelingStats, FailSafeResult, LabelRecord};
pub use poisoning::{PoisoningDetector, SensorHealthConfig, SensorHealthStatus, SensorMetrics, DegradationType};
pub use quality::{compute_quality, ChannelSeries, QualityParams, QualityReport};
pub use training::{train_classifier, TrainOptions, TrainingReport, ClassifierModel, ModelCard,
                   ConfusionCell, PairSimilarity, LabeledRecording, paradigm_window_features,
                   extract_window_features_by_mode, feature_length_for, framework_feature_len,
                   extract_training_windows, compute_warning, DEFAULT_WINDOW_SIZE, TRAIN_STRIDE,
                   PythonModelExport, PythonScalerExport, PythonLrExport, PythonMetadataExport};
pub use framework::{framework_window_features, compute_multi_exp_decay};
pub use live::{LiveClassifier, LiveSnapshot, Prediction, ROLLING_WINDOW, LOCK_THRESHOLD,
               LOCK_CONSECUTIVE, UNKNOWN_THRESHOLD, UNKNOWN_CONSECUTIVE};
pub use smellability::{
    Chemical, ChemicalProperties, ChainOptions, ChainStep, ChainValue, ConstituentVerdict,
    CrossCheck, DataSource, FeasibilityVerdict, IncidentFluxInput, Property, ResolvedEntityKind,
    ResponseSpeed, SignalBand, SignalStrength, Verdict, VerdictConfidence,
    delta_h_vap_trouton, diffusion_coefficient_fuller, incident_flux, incident_flux_proportional,
    resolve_and_run, run_chemical_verdict, signal_band_label, signal_ratio_vs_ref, signal_score,
    vapor_pressure_antoine, vapor_pressure_clausius_clapeyron, worst_verdict, AMBIENT_TEMP_C,
    AMBIENT_TEMP_K, DEFAULT_DISTANCE_M, DEFAULT_SENSOR_COUNT, MOX_FLOOR_PPM, P_ATM,
    REFERENCE_CHEMICAL_ID, R, N_A, max_substances, reference_by_id,
};
