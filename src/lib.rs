
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

// --- R0 baseline window (SAMPLING_CONTRACT.md, "The R0 window contract") ---
//
// A declared window (`Baseline::from_samples_with_window`, or an explicit
// `r0_samples` threaded from a manifest / preset / model) always wins and is used
// verbatim: whoever declares a window owns the duration-to-count conversion the
// contract requires, `round(duration_s * sr)`. `R0_WINDOW_DEFAULT` means "no
// window declared", *not* "zero samples" -- reduce the recording with
// `r0_window_samples` below.
pub const R0_WINDOW_DEFAULT: usize = 0;
/// Fraction of the recording the baseline window spans when nothing is declared.
/// The same fraction `HARDWARE.md` (`cutoff = sample_count * 0.15`) and
/// `data-commons/docs/wire-protocol.md` ("median of first 15%") specify.
pub const R0_WINDOW_FRACTION: f64 = 0.15;
/// Floor: below ~5 samples the median is one or two readings and a single ADC LSB
/// moves R0 by 10-20%.
pub const R0_WINDOW_MIN_SAMPLES: usize = 5;
/// Ceiling: on a long recording an unbounded 15% would swallow the onset, so the
/// baseline must stay inside the leading plateau.
pub const R0_WINDOW_MAX_SAMPLES: usize = 30;

/// Number of leading samples forming the R0 baseline window.
///
/// `declared` is a caller/manifest-declared window and is returned verbatim
/// (`R0_WINDOW_DEFAULT` means "not declared"). Otherwise the contract default
/// applies: `clamp(floor(0.15 * n_samples), 5, 30)`.
///
/// **The default window is cadence-independent**; a fixed sample count is not.
/// `n_samples` grows with the rate, so `0.15 * n_samples` spans `0.15 * T`
/// *seconds* of recording whether it was sampled at 1, 2, 10 or 100 Hz -- the same
/// invariance `SAMPLING_CONTRACT.md` hard rule 5 demands, obtained without needing
/// a declared rate (the auto-R0 path exists precisely for recordings that have no
/// separate baseline session and so no trusted `sr`). The superseded fixed
/// 15-sample default spanned 1.5 s at 10 Hz against 15 s at 1 Hz, silently
/// rescaling R0 and every feature divided by it by up to 10x.
///
/// The clamps are the documented cost and are themselves sample counts, so they
/// are cadence-*dependent*: below 34 samples the floor binds and above 200 the
/// ceiling does. Invariance is exact only for `34 <= n_samples <= 200`.
pub fn r0_window_samples(n_samples: usize, declared: usize) -> usize {
    if declared != R0_WINDOW_DEFAULT {
        return declared.max(1);
    }
    let fraction = (n_samples as f64 * R0_WINDOW_FRACTION).floor() as usize;
    fraction.clamp(R0_WINDOW_MIN_SAMPLES, R0_WINDOW_MAX_SAMPLES)
}

impl Baseline {
    /// R0 = per-channel median of the leading contract window
    /// (`r0_window_samples(n, R0_WINDOW_DEFAULT)`).
    pub fn from_samples(samples: &[Vec<f64>]) -> Self {
        Self::from_samples_with_window(samples, R0_WINDOW_DEFAULT)
    }

    /// R0 = per-channel median of the leading `r0_samples` rows. Pass
    /// `R0_WINDOW_DEFAULT` (or use [`Baseline::from_samples`]) for the
    /// cadence-independent contract default; pass any other value to honour a
    /// window declared by a manifest, preset or trained model.
    pub fn from_samples_with_window(samples: &[Vec<f64>], r0_samples: usize) -> Self {
        if samples.is_empty() {
            return Self { r0: vec![], n_samples: 0, std: vec![] };
        }
        let n_channels = samples[0].len();
        // The floor of 5 can exceed a very short recording, so bound the slice
        // by the row count: `n_samples` reports what was actually used.
        let baseline_end = r0_window_samples(samples.len(), r0_samples).min(samples.len());

        let mut r0 = Vec::with_capacity(n_channels);
        let mut std = Vec::with_capacity(n_channels);

        for ch in 0..n_channels {
            let mut vals: Vec<f64> = samples[..baseline_end]
                .iter()
                .map(|s| s[ch])
                .filter(|v| v.is_finite() && *v > 0.0)
                .collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // No finite positive sample in the baseline window: the channel is
            // dead or disconnected. 1.0 is the neutral placeholder used
            // throughout the feature contract so normalization stays finite and
            // downstream guards (r0 > 0.0) behave predictably. A zero would
            // divide by zero in normalize() and poison every window feature.
            if vals.is_empty() {
                r0.push(1.0);
                std.push(0.0);
                continue;
            }
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
pub mod timing;

pub mod training;

pub mod live;

pub mod smellability;
pub use features::{FeatureGroup, extract_features, extract_features_with_sr,
                   extract_window_features, extract_window_features_with_sr, feature_names};
pub use anomaly::{AnomalyDetector, AnomalyScore, AnomalyMethod};
pub use calibration::{AutoTune, Calibrator, CalibrationProfile, CrossDeviceCalibrator};
pub use health::{HealthMonitor, SensorHealth, HealthStatus, FleetHealth, fisher_discriminant_ratio, pairwise_fdr, euclidean_distance, cosine_similarity, similarity_warning};
pub use protocol::{OsmProtocol, OsmMessage, format_event, KNOWN_PHASE_LABELS,
                   SAMPLE_INDEX_DEVICE_ASSIGNED};
pub use preprocessing::{RawData, BaselineCorrection, BaselineMethod, SignalFilter, FilterType, WindowExtractor, DataValidator};
pub use adaptive::{AdaptiveAnomalyDetector, AdaptiveThreshold, DetectionConfig, FailSafeSystem, LabelingSystem, DetectionResult, AccuracyImprovement, DetectorState, LabelingStats, FailSafeResult, LabelRecord, WARMUP_SECONDS, STUCK_ZERO_SECONDS, WARNING_SECONDS, CRITICAL_SECONDS, EMERGENCY_SECONDS, RESET_NORMAL_SECONDS};
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
