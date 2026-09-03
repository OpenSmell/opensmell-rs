//! Classifier training for n-of-1 MOX recordings.
//!
//! Ports the training semantics of the legacy `train_tab.py` / `realtime_classifier.py`
//! (logistic regression, L2, C=1.0, balanced class weights, sliding windows with
//! stride 5, lock 0.7x10 / unknown 0.5x20 thresholds) and adds reliability guards:
//!
//! 1. LORO (leave-one-recording-out) evaluation — an honest generalization estimate.
//! 2. Quality-gated training data (optional per-recording quality floor).
//! 3. Class-similarity / degeneracy gates (FDR + prototype distance).
//! 4. Data floors (>= 2 classes, >= 2 recordings, minimum windows per class).
//! 5. Honest model card (OOS metrics, per-class precision/recall, confusion matrix).
//!
//! Feature extraction uses the reference *framework* window features by default
//! (`compute_framework_features`: 28 per channel + 4 global + selection ratios =
//! 187 dims @ 6 channels, in sorted-name order — matching the reference Python
//! app). The legacy *paradigm* path (`compute_window_paradigms`, 5 per channel)
//! remains available as a fallback via `TrainOptions::feature_mode = "paradigm"`.
//! Feature-mode dispatch is centralized in [`extract_window_features_by_mode`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{OpenSmellError, Result};

// --------------------------------------------------------------- constants

/// Window stride during training (matches `TRAIN_STRIDE = 5`).
pub const TRAIN_STRIDE: usize = 5;

/// Reference window-size clamp (matches realtime classifier 20..=500).
pub const WINDOW_SIZE_MIN: usize = 20;
pub const WINDOW_SIZE_MAX: usize = 500;

/// Default training window size (matches reference `window_size=100`).
pub const DEFAULT_WINDOW_SIZE: usize = 100;

/// Leading samples used to estimate R0 per window (matches the reference
/// `r0_samples=15` default used by `extract_all_framework_features`).
pub const DEFAULT_R0_SAMPLES: usize = 15;

/// Regularization strength (matches `C=1.0`).
pub const DEFAULT_C: f64 = 1.0;

/// Default gradient-inf-norm tolerance for the logistic optimizer (matches
/// sklearn's `LogisticRegression(tol=1e-4)` default).
pub const LR_GRAD_TOL: f64 = 1e-4;

/// Cap on optimizer (L-BFGS) iterations.
pub const MAX_LR_ITERATIONS: usize = 1000;

/// Minimum windows per class below which training is refused.
pub const MIN_WINDOWS_PER_CLASS: usize = 8;

/// Minimum recordings required to train (matches reference UI rule of >= 2).
pub const MIN_RECORDINGS: usize = 2;

/// Max classes per sensor count (matches `MAX_SUBSTANCES` in train_tab.py).
pub const MAX_SUBSTANCES: &[(usize, usize)] = &[(3, 7), (4, 12), (5, 20), (6, 40)];

/// Fallback class cap for sensor counts not in `MAX_SUBSTANCES`.
pub const MAX_SUBSTANCES_DEFAULT: usize = 12;

/// Cosine threshold for the hard similarity refusal (near-identical prototypes).
pub const SIMILAR_REFUSE_COSINE: f64 = 0.99;

/// Normalized prototype distance threshold for the hard similarity refusal.
pub const SIMILAR_REFUSE_DISTANCE: f64 = 0.5;

/// Mean FDR below which the model card warns about overlap.
pub const SIMILAR_WARN_FDR: f64 = 0.25;

// ------------------------------------------------------------ simple math

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

fn population_std(x: &[f64]) -> f64 {
    let mu = mean(x);
    (x.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Composite trapezoid integral with unit spacing (matches `np.trapezoid(y)`).
fn trapezoid(y: &[f64]) -> f64 {
    match y.len() {
        0 | 1 => 0.0,
        2 => (y[0] + y[1]) / 2.0,
        n => (y[0] + y[n - 1]) / 2.0 + y[1..n - 1].iter().sum::<f64>(),
    }
}

/// Solve `a x = b` with partial-pivoting Gaussian elimination.
/// Returns `None` when the system is numerically singular.
/// Only used by unit tests (the production LR optimizes with L-BFGS).
#[cfg(test)]
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Some(vec![]);
    }
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        let mut pivot = col;
        let mut best = m[col][col].abs();
        for r in (col + 1)..n {
            let v = m[r][col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        m.swap(col, pivot);

        for r in (col + 1)..n {
            let factor = m[r][col] / m[col][col];
            if factor == 0.0 {
                continue;
            }
            for c in col..(n + 1) {
                m[r][c] -= factor * m[col][c];
            }
        }
    }

    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut sum = m[r][n];
        for c in (r + 1)..n {
            sum -= m[r][c] * x[c];
        }
        x[r] = sum / m[r][r];
    }
    Some(x)
}

// -------------------------------------------------- paradigm window features

/// Paradigm window features — the reference `compute_window_paradigms` (5 per
/// channel): `delta_ratio`, `direction`, `mean_slope`, `auc`, `endpoint_delta`.
///
/// Dead/constant channels produce five zeros, matching the reference.
pub fn paradigm_window_features(window: &[Vec<f64>], r0_samples: usize) -> Vec<f64> {
    if window.is_empty() {
        return vec![];
    }
    let n_channels = window[0].len();
    if n_channels == 0 {
        return vec![];
    }
    let r0 = r0_samples.max(1);
    let mut feats = Vec::with_capacity(n_channels * 5);

    for c in 0..n_channels {
        let mut ch: Vec<f64> = window.iter().map(|s| s[c]).collect();
        for v in ch.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        let std = population_std(&ch);
        let all_zero = ch.iter().all(|&v| v == 0.0);
        if std < 1e-8 || all_zero {
            feats.extend([0.0; 5]);
            continue;
        }

        let n_base = r0.min(ch.len());
        let mut r0_val = mean(&ch[..n_base]);
        if r0_val <= 0.0 {
            let positives: Vec<f64> = ch.iter().copied().filter(|&v| v > 0.0).collect();
            r0_val = if positives.is_empty() { 1.0 } else { mean(&positives) };
        }

        let delta_ratio = ch.iter().map(|&v| (v - r0_val).abs()).fold(0.0f64, f64::max) / r0_val;

        let last_mean = if ch.len() >= 3 {
            mean(&ch[ch.len() - 3..])
        } else {
            ch[ch.len() - 1]
        };
        let direction = if last_mean > r0_val * 1.02 {
            1.0
        } else if last_mean < r0_val * 0.98 {
            -1.0
        } else {
            0.0
        };

        let mean_slope = if ch.len() > 1 {
            ch.windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .sum::<f64>()
                / (ch.len() - 1) as f64
                / r0_val
        } else {
            0.0
        };

        let normalized: Vec<f64> = ch.iter().map(|&v| (v - r0_val).abs() / r0_val).collect();
        let auc = if normalized.len() > 1 {
            trapezoid(&normalized)
        } else {
            normalized[0]
        };

        let n_first = 3.min(ch.len());
        let n_last = 3.min(ch.len());
        let first_mean = mean(&ch[..n_first]);
        let endpoint_delta = (mean(&ch[ch.len() - n_last..]) - first_mean) / r0_val;

        feats.extend([delta_ratio, direction, mean_slope, auc, endpoint_delta]);
    }

    for f in feats.iter_mut() {
        if !f.is_finite() {
            *f = 0.0;
        }
    }
    feats
}

// ------------------------------------------------------ feature mode dispatch

/// Framework (187-dim @6ch) feature-count for `n` channels.
pub fn framework_feature_len(n_channels: usize) -> usize {
    crate::framework::framework_feature_len(n_channels)
}

/// Length of the feature vector for a given channel count and feature mode.
pub fn feature_length_for(n_channels: usize, feature_mode: &str) -> usize {
    if feature_mode == "framework" {
        framework_feature_len(n_channels)
    } else {
        n_channels * 5 // paradigm
    }
}

/// Extract a feature vector from one window according to `feature_mode`.
/// `sr` (samples/second) is only used by the framework path (time constants,
/// oscillation frequency). The paradigm path ignores it.
///
/// The framework path needs at least ~15 samples to produce sensible values;
/// very short windows degrade through the internal guards, not here.
pub fn extract_window_features_by_mode(
    window: &[Vec<f64>],
    feature_mode: &str,
    r0_samples: usize,
    sr: f64,
) -> Vec<f64> {
    if feature_mode == "framework" {
        // The framework path returns None only for empty/zero-channel windows.
        crate::framework::framework_window_features(window, r0_samples, sr)
            .unwrap_or_default()
    } else {
        paradigm_window_features(window, r0_samples)
    }
}

/// Extract features for many independent windows in parallel. Each window's
/// feature vector is computed identically to the serial path (feature
/// extraction is pure/deterministic), so results and ordering are unchanged —
/// only the CPU work is spread across cores. Thread count is bounded to the
/// available parallelism (capped at 4) to respect memory-fragile hosts.
fn parallel_extract(
    windows: &[Vec<Vec<f64>>],
    feature_mode: &str,
    r0_samples: usize,
    sr: f64,
) -> Vec<Vec<f64>> {
    let n = windows.len();
    if n <= 1 {
        return windows
            .iter()
            .map(|w| extract_window_features_by_mode(w, feature_mode, r0_samples, sr))
            .collect();
    }
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(4);
    if threads <= 1 {
        return windows
            .iter()
            .map(|w| extract_window_features_by_mode(w, feature_mode, r0_samples, sr))
            .collect();
    }

    let chunk = n.div_ceil(threads);
    let chunked: Vec<Vec<Vec<f64>>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for start in (0..n).step_by(chunk) {
            let end = (start + chunk).min(n);
            let slice = &windows[start..end];
            handles.push(scope.spawn(move || {
                slice
                    .iter()
                    .map(|w| extract_window_features_by_mode(w, feature_mode, r0_samples, sr))
                    .collect::<Vec<Vec<f64>>>()
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_default())
            .collect()
    });
    chunked.into_iter().flatten().collect()
}

// ------------------------------------------------------------- window bounds

/// Sliding training windows exactly like the reference:
/// `range(0, n - window_size + 1, stride)` slices, or a single edge-padded window
/// for recordings shorter than `window_size`.
pub fn extract_training_windows(
    samples: &[Vec<f64>],
    window_size: usize,
    stride: usize,
) -> Vec<Vec<Vec<f64>>> {
    let n = samples.len();
    if n < window_size {
        let mut padded: Vec<Vec<f64>> = samples.to_vec();
        let last = samples.last().cloned().unwrap_or_default();
        while padded.len() < window_size {
            padded.push(last.clone());
        }
        return vec![padded];
    }
    let mut windows = Vec::new();
    let mut start = 0;
    while start + window_size <= n {
        windows.push(samples[start..start + window_size].to_vec());
        start += stride;
    }
    windows
}

// ------------------------------------------------------------- training types

/// One labeled recording that may participate in training.
#[derive(Debug, Clone)]
pub struct LabeledRecording {
    pub label: String,
    pub samples: Vec<Vec<f64>>,
    /// Optional quality total (0..=1). Recordings below the floor are skipped.
    pub quality: Option<f64>,
}

impl LabeledRecording {
    pub fn new(label: impl Into<String>, samples: Vec<Vec<f64>>) -> Self {
        Self {
            label: label.into(),
            samples,
            quality: None,
        }
    }
}

/// Training configuration. Defaults match the reference application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainOptions {
    /// Samples per classification window (clamped 20..=500).
    pub window_size: usize,
    /// Sensor count used for the class-cap warning and model metadata.
    pub n_sensors: usize,
    /// Only recordings with `quality >= min_quality` are used (0..=1, 0 disables).
    pub min_quality: f64,
    /// Window stride during training.
    pub stride: usize,
    /// Feature extraction mode: `"framework"` (default; 187-dim @6ch) or
    /// `"paradigm"` (legacy; 5 per channel).
    pub feature_mode: String,
    /// Sampling rate in samples/second, used only by the framework path
    /// (time constants, oscillation frequency). Defaults to 10.
    pub sr: f64,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            window_size: DEFAULT_WINDOW_SIZE,
            n_sensors: 3,
            min_quality: 0.0,
            stride: TRAIN_STRIDE,
            feature_mode: "framework".to_string(),
            sr: 10.0,
        }
    }
}

/// One cell of the (aggregated, out-of-sample) confusion matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionCell {
    pub actual: String,
    pub predicted: String,
    pub count: usize,
}

/// Similarity summary for one class pair, recorded in the model card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairSimilarity {
    pub class_a: String,
    pub class_b: String,
    pub cosine: f64,
    pub fdr_mean: f64,
    pub scaled_distance: f64,
}

/// Honest model card: out-of-sample metrics and provenance, stored in the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    /// Training algorithm used, e.g. `"logistic_regression"`. Recorded so a
    /// model card never overstates what was actually fit.
    pub algorithm: String,
    /// Out-of-sample LORO accuracy (windows aggregated over all leave-out folds).
    pub accuracy: f64,
    /// Mean of per-recording (leave-one-out) fold accuracies.
    pub loro_mean_accuracy: f64,
    /// In-sample accuracy on the full training set (reference-style metric).
    pub in_sample_accuracy: f64,
    /// Out-of-sample per-class precision and recall.
    pub per_class_precision: BTreeMap<String, f64>,
    pub per_class_recall: BTreeMap<String, f64>,
    /// Out-of-sample confusion matrix (events from leave-one-out folds).
    pub confusion: Vec<ConfusionCell>,
    /// Similarity analysis between every class pair.
    pub similarity: Vec<PairSimilarity>,
    /// Human-readable warnings surfaced by the gates (class cap, FDR, quality…).
    pub warnings: Vec<String>,
}

/// The portable, serializable model consumed by the desktop realtime runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierModel {
    pub name: String,
    /// Sorted class labels (label-encoder order).
    pub classes: Vec<String>,
    pub n_sensors: usize,
    pub window_size: usize,
    pub n_features: usize,
    /// Feature extraction mode: `"framework"` (default; 187-dim @6ch, matching
    /// the reference Python app) or `"paradigm"` (legacy; 5 per channel).
    pub feature_mode: String,
    /// Leading samples used to estimate R0 within a window.
    pub r0_samples: usize,
    /// StandardScaler fit on the full training set.
    pub scaler_mean: Vec<f64>,
    pub scaler_scale: Vec<f64>,
    /// Logistic regression coefficients: `[class][feature]`.
    pub coef: Vec<Vec<f64>>,
    /// Intercept per class.
    pub intercept: Vec<f64>,
    pub n_windows: usize,
    pub windows_per_class: BTreeMap<String, usize>,
    pub recordings_per_class: BTreeMap<String, usize>,
    pub model_card: ModelCard,
}

/// Lightweight, self-describing export of a trained model intended for
/// consumption by Python tooling (research / experimentation / interop).
///
/// The classifier parameters map 1:1 onto a scikit-learn *multinomial*
/// `LogisticRegression`: apply the StandardScaler, then softmax over
/// `coef @ x_scaled + intercept` (matching the Rust runtime's
/// [`ClassifierModel::predict_proba`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonModelExport {
    /// Format tag; bump on any incompatible schema change.
    pub format: String,
    /// The estimator family the parameters describe.
    pub engine: String,
    /// Class labels in encoder order (index == row of `classifier.coef`).
    pub classes: Vec<String>,
    /// Feature normalization applied *before* the classifier.
    pub preprocessing: PythonScalerExport,
    /// Multinomial logistic-regression parameters.
    pub classifier: PythonLrExport,
    /// Provenance / sizing metadata.
    pub metadata: PythonMetadataExport,
}

/// StandardScaler parameters (fit on the full training set).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonScalerExport {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

/// Multinomial LogisticRegression parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonLrExport {
    /// `[class][feature]` coefficients in scaled-feature space.
    pub coef: Vec<Vec<f64>>,
    /// Per-class intercept.
    pub intercept: Vec<f64>,
}

/// Sizing / provenance metadata carried alongside the parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonMetadataExport {
    pub name: String,
    pub feature_mode: String,
    pub n_sensors: usize,
    pub n_channels: usize,
    pub n_features: usize,
    pub window_size: usize,
    pub r0_samples: usize,
    pub sr: f64,
    pub n_windows: usize,
    pub windows_per_class: BTreeMap<String, usize>,
    pub recordings_per_class: BTreeMap<String, usize>,
    pub accuracy: f64,
    pub loro_mean_accuracy: f64,
    pub in_sample_accuracy: f64,
}

impl ClassifierModel {
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(OpenSmellError::from)
    }

    /// Build a self-describing Python-friendly export of this model.
    ///
    /// The output is a plain JSON object that Python can load with `json`.
    /// No pickle / arbitrary-code path is involved — this is a safe, portable
    /// interchange format.
    pub fn to_python_export(&self) -> PythonModelExport {
        let card = &self.model_card;
        PythonModelExport {
            format: "opensmell-model/v1".to_string(),
            engine: "multinomial-logistic-regression".to_string(),
            classes: self.classes.clone(),
            preprocessing: PythonScalerExport {
                mean: self.scaler_mean.clone(),
                scale: self.scaler_scale.clone(),
            },
            classifier: PythonLrExport {
                coef: self.coef.clone(),
                intercept: self.intercept.clone(),
            },
            metadata: PythonMetadataExport {
                name: self.name.clone(),
                feature_mode: self.feature_mode.clone(),
                n_sensors: self.n_sensors,
                n_channels: self.n_channels(),
                n_features: self.n_features,
                window_size: self.window_size,
                r0_samples: self.r0_samples,
                // `sr` is not persisted on the model; the runtime uses the
                // framework default (10 Hz) when predicting, so report that.
                sr: 10.0,
                n_windows: self.n_windows,
                windows_per_class: self.windows_per_class.clone(),
                recordings_per_class: self.recordings_per_class.clone(),
                accuracy: card.accuracy,
                loro_mean_accuracy: card.loro_mean_accuracy,
                in_sample_accuracy: card.in_sample_accuracy,
            },
        }
    }

    /// Serialize the Python-friendly export to pretty JSON.
    pub fn to_python_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.to_python_export()).map_err(OpenSmellError::from)
    }

    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(OpenSmellError::from)
    }

    /// Number of channels this model expects.
    ///
    /// For the framework path this is the stored sensor count (the feature
    /// dimension grows non-linearly with channels). For paradigm it is
    /// `n_features / 5` (5 features per channel), used as a cross-check.
    pub fn n_channels(&self) -> usize {
        if self.feature_mode == "framework" {
            self.n_sensors
        } else {
            self.n_features / 5
        }
    }

    fn scale(&self, features: &[f64]) -> Vec<f64> {
        features
            .iter()
            .zip(self.scaler_mean.iter().zip(self.scaler_scale.iter()))
            .map(|(&x, (&mu, &scale))| (x - mu) / scale)
            .collect()
    }

    /// Predict class probabilities (softmax) for one raw window.
    /// Returns `None` when the window's channel count does not match the model.
    pub fn predict_proba(&self, window: &[Vec<f64>]) -> Option<Vec<f64>> {
        if window.is_empty() {
            return None;
        }
        let n_channels = window[0].len();
        if n_channels != self.n_channels() {
            return None;
        }
        let raw = extract_window_features_by_mode(window, &self.feature_mode, self.r0_samples, 10.0);
        if raw.len() != self.n_features {
            return None;
        }
        let scaled = self.scale(&raw);
        let z: Vec<f64> = self
            .coef
            .iter()
            .zip(self.intercept.iter())
            .map(|(w, b)| dot(w, &scaled) + b)
            .collect();
        Some(softmax(&z))
    }

    /// Predict the most likely class with its probability for one raw window.
    pub fn predict(&self, window: &[Vec<f64>]) -> Option<(String, f64)> {
        let probs = self.predict_proba(window)?;
        let mut best = 0usize;
        for (i, &p) in probs.iter().enumerate() {
            if p > probs[best] {
                best = i;
            }
        }
        Some((self.classes[best].clone(), probs[best]))
    }
}

/// Full training result including the serialized model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub success: bool,
    pub error: Option<String>,
    pub name: String,
    pub classes: Vec<String>,
    pub n_windows: usize,
    pub windows_per_class: BTreeMap<String, usize>,
    pub recordings_per_class: BTreeMap<String, usize>,
    /// Out-of-sample aggregate accuracy (the honest number).
    pub accuracy: f64,
    /// Mean leave-one-recording-out fold accuracy.
    pub loro_mean_accuracy: f64,
    /// Reference-style in-sample accuracy, kept for comparison.
    pub in_sample_accuracy: f64,
    pub warnings: Vec<String>,
    pub model_json: String,
}

// --------------------------------------------------------------- warnings

/// Class-vs-sensor warning mirroring `compute_warning` in train_tab.py.
pub fn compute_warning(n_classes: usize, n_sensors: usize) -> String {
    let max_ok = MAX_SUBSTANCES
        .iter()
        .find(|(s, _)| *s == n_sensors)
        .map(|(_, m)| *m)
        .unwrap_or(MAX_SUBSTANCES_DEFAULT);
    if n_classes <= 3 {
        return String::new();
    }
    if n_classes > max_ok {
        format!(
            "{} classes with {} sensors (~{}-{} max). Predictions will overlap. Add more sensors or reduce classes.",
            n_classes, n_sensors, max_ok / 2, max_ok
        )
    } else if n_classes as f64 > max_ok as f64 * 0.7 {
        format!(
            "{} classes approaching the ~{} limit for {} sensors. Consider reducing classes.",
            n_classes, max_ok, n_sensors
        )
    } else {
        String::new()
    }
}

fn softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// ------------------------------------------------------------- scaler & LR

#[derive(Debug, Clone)]
struct StandardScaler {
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl StandardScaler {
    fn fit(xs: &[Vec<f64>]) -> Self {
        let n = xs.len();
        let d = xs.first().map_or(0, |x| x.len());
        let mut mean = vec![0.0; d];
        let mut scale = vec![1.0; d];
        if n == 0 || d == 0 {
            return Self { mean, scale };
        }
        for j in 0..d {
            mean[j] = xs.iter().map(|x| x[j]).sum::<f64>() / n as f64;
        }
        for j in 0..d {
            let var = xs.iter().map(|x| (x[j] - mean[j]).powi(2)).sum::<f64>() / n as f64;
            scale[j] = if var > 0.0 { var.sqrt() } else { 1.0 };
        }
        Self { mean, scale }
    }

    fn transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.mean.iter().zip(self.scale.iter()))
            .map(|(&v, (&mu, &s))| (v - mu) / s)
            .collect()
    }
}

/// Multinomial (softmax) logistic regression with L2 regularization, fit by
/// damped Newton-IRLS. Matches the reference `LogisticRegression(max_iter=2000,
/// class_weight="balanced", C=1.0)` family: the objective is
/// `f = (1/(2C)) Σ_k ‖w_k‖² + (1/N) Σ_i s_i · ce(p_i, y_i)` with balanced sample
/// weights. sklearn includes C in the loss via `-C·ll + ½‖w‖²`, which shares the
/// same minimizer up to a positive constant — predictions are equivalent.
struct MultiLogistic {
    coef: Vec<Vec<f64>>,
    intercept: Vec<f64>,
    n_classes: usize,
    n_features: usize,
}

impl MultiLogistic {
    fn new(n_classes: usize, n_features: usize) -> Self {
        Self {
            coef: vec![vec![0.0; n_features]; n_classes],
            intercept: vec![0.0; n_classes],
            n_classes,
            n_features,
        }
    }

    fn predict_one(&self, xs: &[f64]) -> usize {
        let mut z = Vec::with_capacity(self.n_classes);
        for k in 0..self.n_classes {
            z.push(self.intercept[k] + dot(&self.coef[k], xs));
        }
        let p = softmax(&z);
        p.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Fit on already-scaled features. `ys` are 0..K target indices, `weights`
    /// are the balanced per-sample weights. Uses an L-BFGS optimizer (first
    /// order, O(d) per step) matching sklearn's logistic regression solver,
    /// with the same `C`-regularized cross-entropy objective.
    fn fit(&mut self, xs: &[Vec<f64>], ys: &[usize], weights: &[f64], c: f64) {
        let (n, d) = (xs.len(), self.n_features);
        if n == 0 {
            return;
        }
        let lambda = 1.0 / c.max(1e-6);
        let inv_n = 1.0 / n as f64;
        let kk = self.n_classes;
        let d1 = d + 1;
        let p = kk * d1;

        // Flat parameter layout: per class k, block at k*d1 where slot 0 is the
        // intercept and slots 1..=d are the feature coefficients.
        let mut x = vec![0.0; p];
        for k in 0..kk {
            x[k * d1] = self.intercept[k];
            for j in 0..d {
                x[k * d1 + 1 + j] = self.coef[k][j];
            }
        }

        // Evaluate objective + gradient at a flat parameter vector.
        let obj_grad = |w: &[f64]| -> (f64, Vec<f64>) {
            let mut f = 0.0;
            let mut g = vec![0.0; p];
            for t in 0..p {
                f += 0.5 * lambda * w[t] * w[t];
                g[t] = lambda * w[t];
            }
            for i in 0..n {
                let mut z = vec![0.0; kk];
                for k in 0..kk {
                    z[k] = w[k * d1];
                    for j in 0..d {
                        z[k] += w[k * d1 + 1 + j] * xs[i][j];
                    }
                }
                let lse: f64 = {
                    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    max + z.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
                };
                let sw = weights[i] * inv_n;
                let py = ys[i];
                f += sw * (lse - z[py]);
                let mut sum = 0.0;
                for k in 0..kk {
                    sum += (z[k] - z[py]).exp();
                }
                for k in 0..kk {
                    let pr = (z[k] - z[py]).exp() / sum;
                    let err = pr - (ys[i] == k) as usize as f64;
                    g[k * d1] += sw * err;
                    for j in 0..d {
                        g[k * d1 + 1 + j] += sw * err * xs[i][j];
                    }
                }
            }
            (f, g)
        };

        let (mut f, mut g) = obj_grad(&x);

        let m = 10;
        let mut s_buf: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut y_buf: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut rho_buf: Vec<f64> = Vec::with_capacity(m);

        for _iter in 0..MAX_LR_ITERATIONS {
            let inf_norm = g.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            if inf_norm < LR_GRAD_TOL {
                break;
            }

            // Two-loop recursion for the search direction (q = H⁻¹·g).
            let mcnt = s_buf.len();
            let mut q = g.clone();
            let mut alphas = vec![0.0; mcnt];
            for i in (0..mcnt).rev() {
                let alpha = rho_buf[i] * dot(&s_buf[i], &q);
                alphas[i] = alpha;
                for t in 0..p {
                    q[t] -= alpha * y_buf[i][t];
                }
            }
            let mut gamma = 1.0;
            if mcnt > 0 {
                let yy = dot(&y_buf[mcnt - 1], &y_buf[mcnt - 1]);
                if yy > 0.0 {
                    gamma = dot(&y_buf[mcnt - 1], &s_buf[mcnt - 1]) / yy;
                }
            }
            let mut r = q.clone();
            for t in 0..p {
                r[t] *= gamma;
            }
            for i in 0..mcnt {
                let beta = rho_buf[i] * dot(&y_buf[i], &r);
                let a2 = alphas[i];
                for t in 0..p {
                    r[t] += (a2 - beta) * s_buf[i][t];
                }
            }
            let gd = dot(&g, &r); // descent requires -r, so require gd > 0
            if gd <= 0.0 {
                break;
            }

            // Backtracking Armijo line search along -r.
            let mut found = false;
            let mut step = 1.0;
            for _ls in 0..40 {
                let mut xnew = vec![0.0; p];
                for t in 0..p {
                    xnew[t] = x[t] - step * r[t];
                }
                let (fnew, gnew) = obj_grad(&xnew);
                if fnew <= f + 1e-4 * step * gd {
                    let mut s_new = vec![0.0; p];
                    let mut y_new = vec![0.0; p];
                    for t in 0..p {
                        s_new[t] = xnew[t] - x[t];
                        y_new[t] = gnew[t] - g[t];
                    }
                    let sy = dot(&s_new, &y_new);
                    if sy > 1e-12 {
                        s_buf.push(s_new);
                        y_buf.push(y_new);
                        rho_buf.push(1.0 / sy);
                        if s_buf.len() > m {
                            s_buf.remove(0);
                            y_buf.remove(0);
                            rho_buf.remove(0);
                        }
                    }
                    x = xnew;
                    f = fnew;
                    g = gnew;
                    found = true;
                    break;
                }
                step *= 0.5;
            }
            if !found {
                break;
            }
        }

        // Write back results.
        for k in 0..kk {
            self.intercept[k] = x[k * d1];
            for j in 0..d {
                self.coef[k][j] = x[k * d1 + 1 + j];
            }
        }
    }

    fn accuracy(&self, xs: &[Vec<f64>], ys: &[usize]) -> f64 {
        let mut correct = 0usize;
        for (x, &y) in xs.iter().zip(ys.iter()) {
            if self.predict_one(x) == y {
                correct += 1;
            }
        }
        if ys.is_empty() {
            0.0
        } else {
            correct as f64 / ys.len() as f64
        }
    }
}

/// sklearn-style balanced sample weights: `n / (K * count_k)` per window.
fn balanced_weights(ys: &[usize], n_classes: usize) -> Vec<f64> {
    let n = ys.len();
    let mut counts = vec![0usize; n_classes];
    for &y in ys {
        counts[y] += 1;
    }
    ys.iter()
        .map(|&y| n as f64 / (n_classes as f64 * counts[y].max(1) as f64))
        .collect()
}

// ----------------------------------------------------------- gates & metrics

struct NamedWindows {
    label: String,
    windows: Vec<Vec<Vec<f64>>>,
}

impl NamedWindows {
    fn count(&self) -> usize {
        self.windows.len()
    }
}

fn validate_and_group(
    recordings: &[LabeledRecording],
    options: &TrainOptions,
) -> Result<(Vec<NamedWindows>, Vec<String>, Vec<String>, usize)> {
    let mut warnings = Vec::new();
    let mut quality_notes = Vec::new();
    let mut usable = 0usize;
    let mut skipped_invalid = 0usize;
    let mut skipped_quality = 0usize;
    let mut n_channels: Option<usize> = None;

    let mut groups: Vec<NamedWindows> = Vec::new();
    for rec in recordings {
        let label = rec.label.trim().to_lowercase().replace(' ', "_");
        if label.is_empty() {
            continue;
        }
        if rec.samples.is_empty() {
            skipped_invalid += 1;
            continue;
        }
        let ch = rec.samples[0].len();
        if ch == 0 {
            skipped_invalid += 1;
            continue;
        }
        match n_channels {
            Some(c) if c != ch => {
                return Err(OpenSmellError::FeatureExtraction(format!(
                    "Mixed channel counts: {} vs {}. Recordings must share the same sensor count for one classifier.",
                    c, ch
                )));
            }
            Some(_) => {}
            None => n_channels = Some(ch),
        }
        if let Some(q) = rec.quality {
            if q < options.min_quality {
                skipped_quality += 1;
                quality_notes.push(format!(
                    "Rec('{}', quality {:.2}) below floor {:.2} — skipped.",
                    label, q, options.min_quality
                ));
                continue;
            }
        }
        let windows = extract_training_windows(&rec.samples, options.window_size, options.stride);
        if windows.is_empty() {
            skipped_invalid += 1;
            continue;
        }
        groups.push(NamedWindows { label, windows });
        usable += 1;
    }

    let effective_channels = n_channels.unwrap_or(options.n_sensors.clamp(3, 6));
    if usable == 0 && skipped_quality > 0 {
        return Err(OpenSmellError::FeatureExtraction(format!(
            "No recordings passed the quality floor ({:.0}% or better).",
            options.min_quality * 100.0
        )));
    }
    if usable < MIN_RECORDINGS {
        return Err(OpenSmellError::FeatureExtraction(format!(
            "Select at least {} recordings to train ({} usable).",
            MIN_RECORDINGS, usable
        )));
    }
    if skipped_invalid > 0 {
        warnings.push(format!("{} recording(s) could not be read and were skipped.", skipped_invalid));
    }
    Ok((groups, warnings, quality_notes, effective_channels))
}

fn class_lists(groups: &[NamedWindows]) -> Result<(Vec<String>, Vec<&NamedWindows>)> {
    let mut by_class: BTreeMap<String, Vec<&NamedWindows>> = BTreeMap::new();
    for g in groups {
        by_class.entry(g.label.clone()).or_default().push(g);
    }
    let classes: Vec<String> = by_class.keys().cloned().collect();
    if classes.len() < 2 {
        return Err(OpenSmellError::FeatureExtraction(
            "Assign at least 2 different substance labels.".to_string(),
        ));
    }
    // Window floors per class.
    for (label, recs) in &by_class {
        let total: usize = recs.iter().map(|g| g.count()).sum();
        if total < MIN_WINDOWS_PER_CLASS {
            return Err(OpenSmellError::FeatureExtraction(format!(
                "Class '{label}' has too few windows ({}; need at least {}). Record longer or add more samples.",
                total, MIN_WINDOWS_PER_CLASS
            )));
        }
    }
    let ordered: Vec<&NamedWindows> = classes
        .iter()
        .flat_map(|c| by_class[c].iter().copied())
        .collect();
    Ok((classes, ordered))
}

/// Class-count / similarity analysis on scaled feature space.
fn analyze_similarity(
    classes: &[String],
    scaled: &[Vec<f64>],
    ys: &[usize],
) -> Vec<PairSimilarity> {
    let mut out = Vec::new();
    for (i, a) in classes.iter().enumerate() {
        for (j, b) in classes.iter().enumerate() {
            if j <= i {
                continue;
            }
            let va: Vec<Vec<f64>> = scaled
                .iter()
                .zip(ys.iter())
                .filter(|(_, &y)| y == i)
                .map(|(x, _)| x.clone())
                .collect();
            let vb: Vec<Vec<f64>> = scaled
                .iter()
                .zip(ys.iter())
                .filter(|(_, &y)| y == j)
                .map(|(x, _)| x.clone())
                .collect();
            if va.is_empty() || vb.is_empty() {
                continue;
            }
            let d = va[0].len();
            let mean_a: Vec<f64> = (0..d).map(|f| mean_over(&va, f)).collect();
            let mean_b: Vec<f64> = (0..d).map(|f| mean_over(&vb, f)).collect();
            let cosine = crate::health::cosine_similarity(&mean_a, &mean_b);
            let scaled_distance = crate::health::euclidean_distance(&mean_a, &mean_b);
            // Mean per-feature Fisher discriminant ratio between the two classes.
            let fdr_mean = crate::health::fisher_discriminant_ratio(&va, &vb)
                .map(|f| if f.is_empty() { 0.0 } else { f.iter().sum::<f64>() / f.len() as f64 })
                .unwrap_or(0.0);
            out.push(PairSimilarity {
                class_a: a.clone(),
                class_b: b.clone(),
                cosine,
                fdr_mean,
                scaled_distance,
            });
        }
    }
    out
}

fn mean_over(rows: &[Vec<f64>], col: usize) -> f64 {
    rows.iter().map(|r| r[col]).sum::<f64>() / rows.len() as f64
}

/// Render the aggregated confusion counts as explicit cells.
fn confusion_from_counts(
    classes: &[String],
    counts: &BTreeMap<(usize, usize), usize>,
) -> Vec<ConfusionCell> {
    let mut out = Vec::new();
    for (a, actual) in classes.iter().enumerate() {
        for (b, predicted) in classes.iter().enumerate() {
            let cnt = counts.get(&(a, b)).copied().unwrap_or(0);
            if cnt > 0 {
                out.push(ConfusionCell {
                    actual: actual.clone(),
                    predicted: predicted.clone(),
                    count: cnt,
                });
            }
        }
    }
    out
}

/// Out-of-sample per-class precision and recall from the aggregated confusion.
fn precision_recall(
    classes: &[String],
    counts: &BTreeMap<(usize, usize), usize>,
) -> (BTreeMap<String, f64>, BTreeMap<String, f64>) {
    let mut precision = BTreeMap::new();
    let mut recall = BTreeMap::new();
    for (i, class) in classes.iter().enumerate() {
        let tp = counts.get(&(i, i)).copied().unwrap_or(0) as f64;
        let pred_total: usize = counts.iter().filter(|((_, b), _)| *b == i).map(|(_, &c)| c).sum();
        let act_total: usize = counts.iter().filter(|((a, _), _)| *a == i).map(|(_, &c)| c).sum();
        precision.insert(
            class.clone(),
            if pred_total > 0 {
                tp / pred_total as f64
            } else {
                0.0
            },
        );
        recall.insert(
            class.clone(),
            if act_total > 0 {
                tp / act_total as f64
            } else {
                0.0
            },
        );
    }
    (precision, recall)
}

/// Full LORO evaluation. Splits per recording (all windows of one recording are
/// held out), refits scaler + LR, and accumulates an out-of-sample confusion.
/// Full LORO evaluation. Splits per recording (all windows of one recording are
/// held out), refits scaler + LR, and accumulates an out-of-sample confusion.
fn loro_evaluate(
    ordered: &[&NamedWindows],
    classes: &[String],
    feature_mode: &str,
    sr: f64,
) -> (f64, f64, BTreeMap<(usize, usize), usize>) {
    let mut global_confusion: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut correct_total = 0usize;
    let mut total = 0usize;
    let mut fold_accums = Vec::new();

    // Extract every window's raw features exactly once, per group, outside folds.
    let n_groups = ordered.len();
    let group_lens: Vec<usize> = ordered.iter().map(|g| g.windows.len()).collect();
    let all_windows: Vec<Vec<Vec<f64>>> = ordered
        .iter()
        .flat_map(|g| g.windows.iter().cloned())
        .collect();
    let extracted = parallel_extract(&all_windows, feature_mode, DEFAULT_R0_SAMPLES, sr);
    let mut raw_by_group: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_groups);
    let mut cursor = 0usize;
    for len in group_lens {
        raw_by_group.push(extracted[cursor..cursor + len].to_vec());
        cursor += len;
    }

    for (holdout_idx, _holdout) in ordered.iter().enumerate() {
        // Training groups = all except the holdout group.
        let train_idx: Vec<usize> = (0..n_groups).filter(|&i| i != holdout_idx).collect();

        // Gather training raw features (no extraction; just index).
        let raw_x: Vec<Vec<f64>> = train_idx
            .iter()
            .flat_map(|&gi| raw_by_group[gi].iter().cloned())
            .collect();
        let scaler = StandardScaler::fit(&raw_x);

        // Build scaled training rows.
        let mut tx = Vec::new();
        let mut ty = Vec::new();
        for &gi in &train_idx {
            let cls = classes.iter().position(|c| c == &ordered[gi].label).unwrap_or(0);
            for raw in &raw_by_group[gi] {
                tx.push(scaler.transform(raw));
                ty.push(cls);
            }
        }

        let w = balanced_weights(&ty, classes.len());
        let mut lr = MultiLogistic::new(classes.len(), scaler.mean.len());
        lr.fit(&tx, &ty, &w, DEFAULT_C);

        // Predict held-out group's windows (precomputed raw).
        let actual = classes.iter().position(|c| c == &ordered[holdout_idx].label).unwrap_or(0);
        let mut correct_fold = 0usize;
        let mut fold_total = 0usize;
        for raw in &raw_by_group[holdout_idx] {
            let scaled = scaler.transform(raw);
            let pred = lr.predict_one(&scaled);
            *global_confusion.entry((actual, pred)).or_insert(0) += 1;
            total += 1;
            fold_total += 1;
            if pred == actual {
                correct_total += 1;
                correct_fold += 1;
            }
        }
        let fold_acc = if fold_total > 0 {
            correct_fold as f64 / fold_total as f64
        } else {
            0.0
        };
        fold_accums.push(fold_acc);
    }

    let aggregate = if total > 0 {
        correct_total as f64 / total as f64
    } else {
        0.0
    };
    let loro_mean = if fold_accums.is_empty() {
        0.0
    } else {
        fold_accums.iter().sum::<f64>() / fold_accums.len() as f64
    };
    (aggregate, loro_mean, global_confusion)
}

// ------------------------------------------------------------- entry point

/// Train a classifier from labeled recordings.
///
/// Returns a `TrainingReport` with the serialized `ClassifierModel`. When the
/// reliability gates reject the training set, `Err` carries the reason (the
/// desktop layer maps this to a user-visible message).
pub fn train_classifier(
    recordings: &[LabeledRecording],
    name: &str,
    options: &TrainOptions,
) -> Result<TrainingReport> {
    let opts = TrainOptions {
        window_size: options.window_size.clamp(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX),
        n_sensors: options.n_sensors.clamp(3, 6),
        min_quality: options.min_quality.clamp(0.0, 1.0),
        stride: options.stride.max(1),
        feature_mode: options.feature_mode.clone(),
        sr: options.sr,
    };

    let (groups, mut warnings, quality_notes, n_channels) = validate_and_group(recordings, &opts)?;
    warnings.extend(quality_notes);

    let (classes, ordered) = class_lists(&groups)?;

    // Class-cap warning (reference `compute_warning`).
    let cap_warning = compute_warning(classes.len(), n_channels);
    if !cap_warning.is_empty() {
        warnings.push(cap_warning);
    }

    // Full-scaled feature set (extract features once and scale in place).
    let all_windows: Vec<Vec<Vec<f64>>> = ordered
        .iter()
        .flat_map(|g| g.windows.iter().cloned())
        .collect();
    let raw_x: Vec<Vec<f64>> =
        parallel_extract(&all_windows, &opts.feature_mode, DEFAULT_R0_SAMPLES, opts.sr);
    let n_features = raw_x.first().map_or(0, |x| x.len());
    let scaler = StandardScaler::fit(&raw_x);
    let mut xs = Vec::with_capacity(raw_x.len());
    for r in &raw_x {
        xs.push(scaler.transform(r));
    }
    let mut ys = Vec::with_capacity(raw_x.len());
    for g in &ordered {
        let idx = classes.iter().position(|c| c == &g.label).unwrap_or(0);
        ys.extend(std::iter::repeat(idx).take(g.windows.len()));
    }
    let n_windows = ys.len();

    // LORO evaluation (honest out-of-sample numbers).
    let (accuracy, loro_mean, confusion_counts) =
        loro_evaluate(&ordered, &classes, &opts.feature_mode, opts.sr);

    // Full fit + in-sample accuracy (reference-style metric).
    let weights = balanced_weights(&ys, classes.len());
    let mut lr = MultiLogistic::new(classes.len(), n_features);
    lr.fit(&xs, &ys, &weights, DEFAULT_C);
    let in_sample = lr.accuracy(&xs, &ys);

    // Similarity gates.
    let similarity = analyze_similarity(&classes, &xs, &ys);
    let mut warnings_sim = Vec::new();
    for pair in &similarity {
        if pair.cosine >= SIMILAR_REFUSE_COSINE || pair.scaled_distance < SIMILAR_REFUSE_DISTANCE {
            return Err(OpenSmellError::FeatureExtraction(format!(
                "Classes '{a}' and '{b}' are near-identical (cosine {c:.3}, distance {d:.3}). \
                 Merge them or add samples that separate them before training.",
                a = pair.class_a,
                b = pair.class_b,
                c = pair.cosine,
                d = pair.scaled_distance
            )));
        }
        if pair.fdr_mean < SIMILAR_WARN_FDR {
            warnings_sim.push(format!(
                "'{a}' and '{b}' overlap heavily (FDR {f:.3}) — predictions between them will be unreliable.",
                a = pair.class_a,
                b = pair.class_b,
                f = pair.fdr_mean
            ));
        }
    }
    warnings.extend(warnings_sim);

    let (per_class_precision, per_class_recall) =
        precision_recall(&classes, &confusion_counts);

    let model_card = ModelCard {
        algorithm: "logistic_regression".to_string(),
        accuracy,
        loro_mean_accuracy: loro_mean,
        in_sample_accuracy: in_sample,
        per_class_precision,
        per_class_recall,
        confusion: confusion_from_counts(&classes, &confusion_counts),
        similarity,
        warnings: warnings.clone(),
    };

    let model = ClassifierModel {
        name: name.to_string(),
        classes: classes.clone(),
        n_sensors: n_channels,
        window_size: opts.window_size,
        n_features,
        feature_mode: opts.feature_mode.clone(),
        r0_samples: DEFAULT_R0_SAMPLES,
        scaler_mean: scaler.mean.clone(),
        scaler_scale: scaler.scale.clone(),
        coef: lr.coef,
        intercept: lr.intercept,
        n_windows,
        windows_per_class: class_window_counts(&ordered),
        recordings_per_class: class_recording_counts(&ordered),
        model_card,
    };

    let model_json = model.to_json()?;

    Ok(TrainingReport {
        success: true,
        error: None,
        name: name.to_string(),
        classes,
        n_windows,
        windows_per_class: class_window_counts(&ordered),
        recordings_per_class: class_recording_counts(&ordered),
        accuracy,
        loro_mean_accuracy: loro_mean,
        in_sample_accuracy: in_sample,
        warnings,
        model_json,
    })
}

fn class_window_counts(ordered: &[&NamedWindows]) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for g in ordered {
        *m.entry(g.label.clone()).or_insert(0) += g.count();
    }
    m
}

fn class_recording_counts(ordered: &[&NamedWindows]) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for g in ordered {
        *m.entry(g.label.clone()).or_insert(0) += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recording(label: &str, n: usize, base: f64, amp: f64, channels: usize) -> LabeledRecording {
        let samples: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 / 10.0;
                (0..channels)
                    .map(|c| {
                        base + amp * (t * 1.0 + c as f64 * 0.7).sin()
                    })
                    .collect()
            })
            .collect();
        LabeledRecording::new(label, samples)
    }

    fn flat_recording(label: &str, n: usize, base: f64) -> LabeledRecording {
        let samples: Vec<Vec<f64>> = (0..n)
            .map(|_| vec![base, base * 0.95, base * 1.05])
            .collect();
        LabeledRecording::new(label, samples)
    }

    /// Realistic MOX-like recording: a flash rise to a peak then exponential
    /// recovery per channel. The framework decay LM fits these well and fast
    /// (unlike the sinusoidal `recording`, whose oscillating recovery is an
    /// adversarial fit that runs the full LM budget in debug builds).
    fn mox_recording(label: &str, n: usize, base: f64, amp: f64) -> LabeledRecording {
        let peak = (n as f64 * 0.25) as usize;
        let channels = 6;
        let samples: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..channels)
                    .map(|c| {
                        let tau = 2.5 + c as f64 * 0.4;
                        let after = if i > peak {
                            -((i - peak) as f64 / 10.0 / tau)
                        } else {
                            0.0
                        };
                        base + c as f64 * 40.0 + amp * after.exp()
                    })
                    .collect()
            })
            .collect();
        LabeledRecording::new(label, samples)
    }

    /// Fast pipeline-test options: paradigm features (framework is validated
    /// separately by `framework_parity.rs` / `decay_parity.rs` and by the
    /// dedicated `test_framework_training_integration` below).
    fn paradigm_opts() -> TrainOptions {
        TrainOptions {
            feature_mode: "paradigm".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_paradigm_features_matches_hand_computation() {
        // 1 channel, 5 samples. R0 = mean(first 3) = (100+100+101)/3.
        let window = vec![vec![100.0], vec![100.0], vec![101.0], vec![105.0], vec![110.0]];
        let f = paradigm_window_features(&window, 3);
        let r0: f64 = (100.0 + 100.0 + 101.0) / 3.0;
        let exp_delta = (105.0 - r0).abs().max((110.0 - r0).abs()) / r0;
        assert!((f[0] - exp_delta).abs() < 1e-12, "delta_ratio {} vs {}", f[0], exp_delta);
        // last_mean = mean [105,110] = 107.5 > r0*1.02 -> direction up
        assert_eq!(f[1], 1.0);
        let diffs = [0.0f64, 1.0, 4.0, 5.0];
        let exp_slope = diffs.iter().map(|d| d.abs()).sum::<f64>() / 4.0 / r0;
        assert!((f[2] - exp_slope).abs() < 1e-12, "mean_slope {} vs {}", f[2], exp_slope);
        assert_eq!(f.len(), 5);
    }

    #[test]
    fn test_paradigm_dead_channel_zeroes() {
        let window = vec![vec![0.0, 5.0], vec![0.0, 6.0], vec![0.0, 5.5]];
        let f = paradigm_window_features(&window, 3);
        assert_eq!(f.len(), 10);
        assert_eq!(&f[..5], &[0.0; 5]);
        assert!(f[5..].iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_sliding_window_counts() {
        let data: Vec<Vec<f64>> = (0..200).map(|i| vec![i as f64]).collect();
        // N=200, ws=100, stride=5 -> j in 0,5,...,100 -> 21 windows.
        assert_eq!(extract_training_windows(&data, 100, 5).len(), 21);
        // N=100 -> exactly one window.
        let data100: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        assert_eq!(extract_training_windows(&data100, 100, 5).len(), 1);
        // Short recording -> one edge-padded window at full size.
        let short: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64]).collect();
        let ws = extract_training_windows(&short, 100, 5);
        assert_eq!(ws.len(), 1);
        assert_eq!(ws[0].len(), 100);
        assert_eq!(ws[0][99], vec![39.0]);
    }

    #[test]
    fn test_balanced_weights() {
        // Class 0 has 1 sample, class 1 has 9 -> weights 5x and 0.556x.
        let ys = vec![0usize, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let w = balanced_weights(&ys, 2);
        assert!((w[0] - 5.0).abs() < 1e-9);
        assert!((w[1] - 10.0 / 18.0).abs() < 1e-9);
    }

    #[test]
    fn test_multinomial_lr_separable() {
        // Two well-separated Gaussians in 2D.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let rng_base = 12345u64;
        let mut s = rng_base;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        for _ in 0..60 {
            xs.push(vec![next(), next()]);
            ys.push(0);
        }
        for _ in 0..60 {
            xs.push(vec![8.0 + next(), 8.0 + next()]);
            ys.push(1);
        }
        let scaler = StandardScaler::fit(&xs);
        let xs_s: Vec<Vec<f64>> = xs.iter().map(|x| scaler.transform(x)).collect();
        let w = balanced_weights(&ys, 2);
        let mut lr = MultiLogistic::new(2, 2);
        lr.fit(&xs_s, &ys, &w, 1.0);
        let acc = lr.accuracy(&xs_s, &ys);
        assert!(acc >= 0.99, "separable data accuracy {}", acc);
    }

    #[test]
    fn test_solver_simple() {
        // 2x2 system: x + y = 3, x - y = 1 -> x=2, y=1.
        let a = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
        let b = vec![3.0, 1.0];
        let x = solve_linear(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_no_quality_floor_ok() {
        let recs = vec![
            recording("garlic", 300, 100.0, 4.0, 3),
            recording("garlic", 300, 120.0, 4.0, 3),
            recording("ginger", 300, 900.0, 4.0, 3),
            recording("ginger", 300, 950.0, 4.0, 3),
        ];
        let opts = paradigm_opts();
        let rep = train_classifier(&recs, "spices", &opts).unwrap();
        assert!(rep.success);
        assert_eq!(rep.classes, vec!["garlic", "ginger"]);
        assert!(rep.accuracy >= 0.9, "accuracy {}", rep.accuracy);
        assert!(rep.in_sample_accuracy >= 0.9);
        assert!(rep.accuracy <= rep.in_sample_accuracy + 1e-9);
    }

    #[test]
    fn test_quality_floor_filters() {
        let mut g = recording("garlic", 300, 100.0, 4.0, 3);
        g.quality = Some(0.4);
        let mut g2 = recording("garlic", 300, 120.0, 4.0, 3);
        g2.quality = Some(0.9);
        let mut i1 = recording("ginger", 300, 900.0, 4.0, 3);
        i1.quality = Some(0.9);
        let i2 = recording("ginger", 300, 950.0, 4.0, 3);
        let recs = vec![g, g2, i1, i2];
        let opts = TrainOptions {
            min_quality: 0.8,
            feature_mode: "paradigm".to_string(),
            ..Default::default()
        };
        let rep = train_classifier(&recs, "spices", &opts).unwrap();
        assert_eq!(rep.recordings_per_class["garlic"], 1);
        assert!(rep.warnings.iter().any(|w| w.contains("quality")));
    }

    #[test]
    fn test_all_quality_below_floor_errors() {
        let mut recs: Vec<LabeledRecording> = vec![
            recording("garlic", 300, 100.0, 4.0, 3),
            recording("ginger", 300, 900.0, 4.0, 3),
        ];
        for r in recs.iter_mut() {
            r.quality = Some(0.1);
        }
        let opts = TrainOptions {
            min_quality: 0.8,
            feature_mode: "paradigm".to_string(),
            ..Default::default()
        };
        let err = train_classifier(&recs, "spices", &opts).unwrap_err();
        assert!(err.to_string().contains("quality floor"));
    }

    #[test]
    fn test_too_few_classes_rejected() {
        let recs = vec![
            recording("garlic", 300, 100.0, 4.0, 3),
            recording("garlic", 300, 120.0, 4.0, 3),
        ];
        let err = train_classifier(&recs, "spices", &paradigm_opts()).unwrap_err();
        assert!(err.to_string().contains("at least 2 different substance labels"));
    }

    #[test]
    fn test_similarity_gate_refuses_near_identical_classes() {
        let recs = vec![
            flat_recording("a", 300, 100.0),
            flat_recording("a", 300, 101.0),
            flat_recording("b", 300, 100.0),
            flat_recording("b", 300, 101.0),
        ];
        let err = train_classifier(&recs, "dup", &paradigm_opts()).unwrap_err();
        assert!(
            err.to_string().contains("near-identical"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_max_substances_warning() {
        let w = compute_warning(8, 3);
        assert!(w.contains("Predictions will overlap"));
        let w2 = compute_warning(5, 3); // above 70% of 7
        assert!(w2.contains("approaching"));
        assert!(compute_warning(3, 3).is_empty());
    }

    #[test]
    fn test_model_serialize_roundtrip() {
        let recs = vec![
            recording("garlic", 300, 100.0, 4.0, 3),
            recording("garlic", 300, 120.0, 4.0, 3),
            recording("ginger", 300, 900.0, 4.0, 3),
            recording("ginger", 300, 950.0, 4.0, 3),
        ];
        let rep = train_classifier(&recs, "spices", &paradigm_opts()).unwrap();
        let model = ClassifierModel::from_json(&rep.model_json).unwrap();
        assert_eq!(model.n_channels(), 3);
        assert_eq!(model.classes, vec!["garlic", "ginger"]);
        assert_eq!(model.n_features, 15);
        // Predict roundtrip on a novel window.
        let win = recording("ginger", 100, 950.0, 4.0, 3).samples;
        let (label, conf) = model.predict(&win).unwrap();
        assert_eq!(label, "ginger");
        assert!(conf > 0.5);
        // Channel-count mismatch -> None.
        let bad = recording("ginger", 100, 950.0, 4.0, 6).samples;
        assert!(model.predict(&bad).is_none());
    }

    #[test]
    fn test_python_export_matches_native_predictions() {
        let recs = vec![
            recording("garlic", 300, 100.0, 4.0, 3),
            recording("garlic", 300, 120.0, 4.0, 3),
            recording("ginger", 300, 900.0, 4.0, 3),
            recording("ginger", 300, 950.0, 4.0, 3),
        ];
        let rep = train_classifier(&recs, "spices", &paradigm_opts()).unwrap();
        let model = ClassifierModel::from_json(&rep.model_json).unwrap();

        let exp = model.to_python_export();
        assert_eq!(exp.format, "opensmell-model/v1");
        assert_eq!(exp.engine, "multinomial-logistic-regression");
        assert_eq!(exp.classes, vec!["garlic", "ginger"]);
        // Parameters mirror the model exactly.
        assert_eq!(exp.preprocessing.mean, model.scaler_mean);
        assert_eq!(exp.preprocessing.scale, model.scaler_scale);
        assert_eq!(exp.classifier.coef, model.coef);
        assert_eq!(exp.classifier.intercept, model.intercept);
        assert_eq!(exp.metadata.n_features, model.n_features);
        assert_eq!(exp.metadata.n_channels, model.n_channels());
        assert_eq!(exp.metadata.accuracy, model.model_card.accuracy);

        // The exported parameters must reproduce the model's softmax prediction
        // for a novel window (independent reimplementation of the math).
        let win = recording("ginger", 100, 950.0, 4.0, 3).samples;
        let native = model.predict_proba(&win).unwrap();
        let raw = extract_window_features_by_mode(&win, &exp.metadata.feature_mode, exp.metadata.r0_samples, 10.0);
        let scaled: Vec<f64> = raw
            .iter()
            .zip(exp.preprocessing.mean.iter().zip(exp.preprocessing.scale.iter()))
            .map(|(&x, (&mu, &s))| (x - mu) / s)
            .collect();
        let z: Vec<f64> = exp
            .classifier
            .coef
            .iter()
            .zip(exp.classifier.intercept.iter())
            .map(|(w, b)| dot(w, &scaled) + b)
            .collect();
        let probs = softmax(&z);
        assert!(probs.iter().zip(&native).all(|(a, b)| (a - b).abs() < 1e-12));
        assert_eq!(probs.len(), 2);

        // JSON serializes without error and round-trips.
        let json = model.to_python_json().unwrap();
        let back: PythonModelExport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.format, "opensmell-model/v1");
        assert_eq!(back.classes, exp.classes);
    }

    #[test]
    fn test_loro_metrics_reported() {
        let recs = vec![
            recording("garlic", 300, 100.0, 4.0, 3),
            recording("garlic", 300, 120.0, 4.0, 3),
            recording("ginger", 300, 900.0, 4.0, 3),
            recording("ginger", 300, 950.0, 4.0, 3),
        ];
        let rep = train_classifier(&recs, "spices", &paradigm_opts()).unwrap();
        let card = ClassifierModel::from_json(&rep.model_json).unwrap().model_card;
        assert_eq!(card.algorithm, "logistic_regression");
        assert!(card.accuracy > 0.0 && card.accuracy <= 1.0);
        assert!(card.loro_mean_accuracy > 0.0 && card.loro_mean_accuracy <= 1.0);
        assert!(card.per_class_precision.contains_key("garlic"));
        assert!(card.per_class_recall.contains_key("ginger"));
    }

    /// End-to-end framework integration: the model records `feature_mode =
    /// "framework"` and its feature count matches the framework formula for the
    /// sensor count. Uses a small window + short recordings to keep the decay-LM
    /// cost bounded.
    #[test]
    fn test_framework_training_integration() {
        // Minimal recordings to satisfy the >=8-windows-per-class floor while
        // exercising the 6-channel / 187-dim framework path end-to-end.
        let recs = vec![
            mox_recording("garlic", 90, 100.0, 40.0),
            mox_recording("garlic", 90, 110.0, 40.0),
            mox_recording("garlic", 90, 120.0, 40.0),
            mox_recording("ginger", 90, 900.0, 40.0),
            mox_recording("ginger", 90, 910.0, 40.0),
            mox_recording("ginger", 90, 920.0, 40.0),
        ];
        let opts = TrainOptions {
            window_size: 30,
            n_sensors: 6,
            min_quality: 0.0,
            stride: 15,
            ..Default::default() // feature_mode = "framework"
        };
        let rep = train_classifier(&recs, "spices", &opts).unwrap();
        let model = ClassifierModel::from_json(&rep.model_json).unwrap();
        assert_eq!(model.feature_mode, "framework");
        assert_eq!(model.n_sensors, 6);
        assert_eq!(model.n_channels(), 6);
        // framework_feature_len(6) == 187
        assert_eq!(model.n_features, 187);
        // Predict on a novel 6-channel window.
        let win = mox_recording("ginger", 40, 950.0, 40.0).samples;
        let (label, conf) = model.predict(&win).unwrap();
        assert_eq!(label, "ginger");
        assert!(conf > 0.5);
    }
}