//! Realtime classifier runtime — port of `Osmograph/viz/realtime_classifier.py`.
//!
//! A rolling sample buffer feeds a trained [`ClassifierModel`], producing
//! per-class probabilities and a lock/unknown state machine with the exact
//! reference thresholds:
//!
//! - **Lock**: max probability `>= 0.70` for 10 consecutive windows.
//! - **Unknown**: max probability `< 0.50` for 20 consecutive windows.
//!
//! Feature extraction follows the model's `feature_mode` (the *framework*
//! 187-dim path by default, matching the reference Python app; `"paradigm"`
//! remains available as a legacy fallback). See the training module.

use serde::{Deserialize, Serialize};

use crate::training::ClassifierModel;

pub const ROLLING_WINDOW: usize = 30;
pub const LOCK_THRESHOLD: f64 = 0.7;
pub const LOCK_CONSECUTIVE: usize = 10;
pub const UNKNOWN_THRESHOLD: f64 = 0.5;
pub const UNKNOWN_CONSECUTIVE: usize = 20;
pub const WINDOW_SIZE_MIN: usize = 20;
pub const WINDOW_SIZE_MAX: usize = 500;

/// One classifier verdict: the winning class (or `"unknown"`) and its
/// probability. Mirrors the reference `(label, confidence)` tuple.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub label: String,
    pub confidence: f64,
}

/// Serializable snapshot of the live engine (the desktop polls this).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSnapshot {
    pub loaded: bool,
    pub classifier_name: String,
    pub loaded_path: String,
    pub classes: Vec<String>,
    pub n_sensors: usize,
    pub window_size: usize,
    pub confidence_threshold: f64,
    pub training_accuracy: f64,
    pub current_probs: Vec<f64>,
    pub current_prediction: Prediction,
    pub lock_count: usize,
    pub unknown_count: usize,
    pub locked: bool,
    pub locked_class: String,
    pub is_unknown: bool,
    pub buffer_len: usize,
}

/// The realtime classifier engine.
pub struct LiveClassifier {
    model: Option<ClassifierModel>,
    loaded_path: Option<String>,
    classifier_name: String,
    training_accuracy: f64,
    n_sensors: usize,
    window_size: usize,
    confidence_threshold: f64,
    buffer: Vec<Vec<f64>>,
    current_probs: Vec<f64>,
    current_prediction: (String, f64),
    prev_prediction: (String, f64),
    lock_count: usize,
    unknown_count: usize,
    locked: bool,
    locked_class: String,
}

impl Default for LiveClassifier {
    fn default() -> Self {
        Self {
            model: None,
            loaded_path: None,
            classifier_name: String::new(),
            training_accuracy: 0.0,
            n_sensors: 0,
            window_size: ROLLING_WINDOW,
            confidence_threshold: UNKNOWN_THRESHOLD,
            buffer: Vec::new(),
            current_probs: Vec::new(),
            current_prediction: (String::new(), 0.0),
            prev_prediction: (String::new(), 0.0),
            lock_count: 0,
            unknown_count: 0,
            locked: false,
            locked_class: String::new(),
        }
    }
}

impl LiveClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    pub fn model(&self) -> Option<&ClassifierModel> {
        self.model.as_ref()
    }

    pub fn classes(&self) -> &[String] {
        self.model.as_ref().map(|m| m.classes.as_slice()).unwrap_or(&[])
    }

    pub fn classifier_name(&self) -> &str {
        &self.classifier_name
    }

    pub fn loaded_path(&self) -> &str {
        self.loaded_path.as_deref().unwrap_or("")
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn n_sensors(&self) -> usize {
        self.n_sensors
    }

    pub fn current_probabilities(&self) -> &[f64] {
        &self.current_probs
    }

    pub fn current_prediction(&self) -> (&str, f64) {
        (&self.current_prediction.0, self.current_prediction.1)
    }

    pub fn is_locked(&self) -> bool {
        self.locked
    }

    pub fn locked_class(&self) -> &str {
        &self.locked_class
    }

    pub fn lock_count(&self) -> usize {
        self.lock_count
    }

    pub fn is_unknown(&self) -> bool {
        self.unknown_count >= UNKNOWN_CONSECUTIVE
    }

    pub fn unknown_count(&self) -> usize {
        self.unknown_count
    }

    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Clamp and set the rolling window size, trimming the buffer like the
    /// reference setter (`max(20, min(500, size))`, keeps the trailing samples).
    pub fn set_window_size(&mut self, size: usize) {
        let size = size.clamp(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX);
        if size != self.window_size {
            self.window_size = size;
            if self.buffer.len() > size {
                let keep = self.buffer.len() - size;
                self.buffer.drain(..keep);
            }
        }
    }

    /// Load a trained model for live use. Resets the buffer and the state
    /// machine; the window size follows the model, mirroring the reference.
    pub fn load(&mut self, model: ClassifierModel, path: Option<String>) {
        let window_size = model.window_size.clamp(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX);
        self.classifier_name = model.name.clone();
        self.training_accuracy = model.model_card.in_sample_accuracy;
        self.n_sensors = model.n_sensors;
        self.window_size = window_size;
        self.model = Some(model);
        self.loaded_path = path;
        self.buffer.clear();
        self.current_probs.clear();
        self.current_prediction = (String::new(), 0.0);
        self.prev_prediction = (String::new(), 0.0);
        self.reset_locks();
    }

    /// Drop the model and reset everything (reference `RealtimeClassifier.unload`).
    pub fn unload(&mut self) {
        *self = Self::default();
    }

    /// Reset lock/unknown counters but keep the loaded model and buffer
    /// (reference `RealtimeClassifier.reset_locks`).
    pub fn reset_locks(&mut self) {
        self.locked = false;
        self.locked_class = String::new();
        self.lock_count = 0;
        self.unknown_count = 0;
    }

    pub fn snapshot(&self) -> LiveSnapshot {
        LiveSnapshot {
            loaded: self.is_loaded(),
            classifier_name: self.classifier_name.clone(),
            loaded_path: self.loaded_path().to_string(),
            classes: self.classes().to_vec(),
            n_sensors: self.n_sensors,
            window_size: self.window_size,
            confidence_threshold: self.confidence_threshold,
            training_accuracy: self.training_accuracy,
            current_probs: self.current_probs.clone(),
            current_prediction: Prediction {
                label: self.current_prediction.0.clone(),
                confidence: self.current_prediction.1,
            },
            lock_count: self.lock_count,
            unknown_count: self.unknown_count,
            locked: self.locked,
            locked_class: self.locked_class.clone(),
            is_unknown: self.is_unknown(),
            buffer_len: self.buffer.len(),
        }
    }

    /// Feed one raw sample (`[channel, ...]`). Returns a prediction once the
    /// buffer has `window_size` samples; `None` before then or on channel
    /// mismatch (reference `RealtimeClassifier.add_sample`).
    pub fn add_sample(&mut self, sample: &[f64]) -> Option<Prediction> {
        if !self.is_loaded() {
            return None;
        }
        self.buffer.push(sample.to_vec());
        if self.buffer.len() < self.window_size {
            return None;
        }
        if self.buffer.len() > self.window_size * 2 {
            let keep = self.buffer.len() - self.window_size;
            self.buffer.drain(..keep);
        }
        self.predict_on_current_window()
    }

    /// Compute features for the trailing `window_size` samples and run the LR.
    fn predict_on_current_window(&mut self) -> Option<Prediction> {
        let model = self.model.as_ref()?;
        let start = self.buffer.len() - self.window_size;
        let window: Vec<Vec<f64>> = self.buffer[start..].to_vec();
        let probs = model.predict_proba(&window)?;
        Some(self.apply_probs(probs))
    }

    /// Reference `_predict` post-probability logic (thresholding + state machine).
    fn apply_probs(&mut self, probs: Vec<f64>) -> Prediction {
        let confidence = probs.iter().copied().fold(0.0f64, f64::max);
        let mut best = 0usize;
        for (i, &p) in probs.iter().enumerate() {
            if p > probs[best] {
                best = i;
            }
        }
        let label = self
            .model
            .as_ref()
            .map(|m| m.classes[best].clone())
            .unwrap_or_default();

        self.current_probs = probs;
        self.prev_prediction = self.current_prediction.clone();

        if confidence < self.confidence_threshold {
            self.current_prediction = ("unknown".to_string(), confidence);
        } else {
            self.current_prediction = (label.clone(), confidence);
        }

        if confidence >= LOCK_THRESHOLD {
            self.lock_count += 1;
            self.unknown_count = 0;
            if self.lock_count >= LOCK_CONSECUTIVE && !self.locked {
                self.locked = true;
                self.locked_class = label;
            }
        } else {
            self.lock_count = 0;
            self.locked = false;
        }

        if confidence < UNKNOWN_THRESHOLD {
            self.unknown_count += 1;
        } else {
            self.unknown_count = 0;
        }

        Prediction {
            label: self.current_prediction.0.clone(),
            confidence: self.current_prediction.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{train_classifier, LabeledRecording, TrainOptions};

    fn recording(label: &str, base: f64, n: usize, channels: usize) -> LabeledRecording {
        let samples: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 / 10.0;
                (0..channels)
                    .map(|c| base + 4.0 * (t * 1.0 + c as f64 * 0.7).sin())
                    .collect()
            })
            .collect();
        LabeledRecording::new(label, samples)
    }

    /// Small garlic/ginger model with a 30-sample window (fast live fixture).
    fn fixture_model(window_size: usize) -> ClassifierModel {
        let recs = vec![
            recording("garlic", 100.0, 300, 3),
            recording("garlic", 120.0, 300, 3),
            recording("ginger", 900.0, 300, 3),
            recording("ginger", 950.0, 300, 3),
        ];
        let report = train_classifier(
            &recs,
            "fixture",
            &TrainOptions {
                window_size,
                n_sensors: 3,
                min_quality: 0.0,
                stride: 5,
                // The live state-machine tests exercise prediction thresholds, not
                // feature extraction; paradigm is far cheaper than the framework LM.
                feature_mode: "paradigm".to_string(),
                sr: 10.0,
            },
        )
        .expect("fixture model trains");
        ClassifierModel::from_json(&report.model_json).expect("model is serializable")
    }

    #[test]
    fn no_model_returns_none() {
        let mut live = LiveClassifier::new();
        assert!(!live.is_loaded());
        assert!(live.add_sample(&[100.0, 100.0, 100.0]).is_none());
        assert_eq!(live.window_size(), ROLLING_WINDOW);
        assert_eq!(live.classes(), &[] as &[String]);
        assert_eq!(live.buffer_len(), 0);
    }

    #[test]
    fn load_sets_window_and_resets() {
        let mut live = LiveClassifier::new();
        live.set_window_size(WINDOW_SIZE_MIN);
        assert_eq!(live.window_size(), WINDOW_SIZE_MIN);
        // Not loaded -> samples are not buffered at all.
        for i in 0..20 {
            let _ = live.add_sample(&[i as f64, 0.0, 0.0]);
        }
        assert_eq!(live.buffer_len(), 0);

        let model = fixture_model(30);
        live.load(model, Some("kitchen.json".to_string()));
        assert!(live.is_loaded());
        assert_eq!(live.window_size(), 30);
        assert_eq!(live.classifier_name(), "fixture");
        assert_eq!(live.loaded_path(), "kitchen.json");
        assert_eq!(live.classes(), &["garlic", "ginger"]);
        assert_eq!(live.buffer_len(), 0);
        assert_eq!(live.lock_count(), 0);
    }

    #[test]
    fn window_size_clamp_matches_reference() {
        let mut live = LiveClassifier::new();
        live.set_window_size(5);
        assert_eq!(live.window_size(), WINDOW_SIZE_MIN);
        live.set_window_size(600);
        assert_eq!(live.window_size(), WINDOW_SIZE_MAX);
    }

    #[test]
    fn add_sample_predicts_after_window_fills() {
        let model = fixture_model(30);
        let mut live = LiveClassifier::new();
        live.load(model, None);
        let garlic: Vec<Vec<f64>> = recording("garlic", 100.0, 40, 3).samples;
        // Not enough samples yet.
        for s in &garlic[..29] {
            let _ = live.add_sample(s);
        }
        assert_eq!(live.buffer_len(), 29);
        // The 30th sample completes the window.
        let p = live.add_sample(&garlic[29]).expect("predicts");
        assert_eq!(p.label, "garlic");
        assert!(p.confidence > 0.5);
        assert_eq!(live.current_probs.len(), 2);
    }

    #[test]
    fn buffer_trims_at_twice_window_size() {
        let model = fixture_model(20);
        let mut live = LiveClassifier::new();
        live.load(model, None);
        // window*2 = 40 samples: no trim needed at 40, trim only > 40.
        for _i in 0..41 {
            let _ = live.add_sample(&[100.0, 0.0, 0.0]);
        }
        assert_eq!(live.buffer_len(), live.window_size());
    }

    #[test]
    fn channel_mismatch_returns_none() {
        let model = fixture_model(30);
        let mut live = LiveClassifier::new();
        live.load(model, None);
        for _ in 0..100 {
            // 4 channels instead of the required 3.
            assert!(live.add_sample(&[100.0, 100.0, 100.0, 100.0]).is_none());
        }
        assert!(!live.is_loaded() || live.buffer_len() > 0);
        // Model untouched: counters stay idle.
        assert_eq!(live.lock_count(), 0);
    }

    #[test]
    fn lock_after_consecutive_high_confidence() {
        let model = fixture_model(20);
        let mut live = LiveClassifier::new();
        live.load(model, None);
        let mut got_high = false;
        for _ in 0..LOCK_CONSECUTIVE {
            let p = live.apply_probs(vec![0.1, 1.0]).confidence;
            got_high = p >= LOCK_THRESHOLD;
        }
        assert!(got_high, "fixture confidence should reach the lock threshold");
        assert_eq!(live.lock_count(), LOCK_CONSECUTIVE);
        assert!(live.is_locked());
        assert_eq!(live.locked_class(), "ginger");
    }

    #[test]
    fn lock_releases_below_threshold() {
        let mut live = LiveClassifier::new();
        // Drive the lock up with high-confidence class 0.
        for _ in 0..LOCK_CONSECUTIVE {
            live.apply_probs(vec![0.9, 0.1]);
        }
        assert!(live.is_locked());
        // A single sub-lock window resets the counter and unlocks.
        live.apply_probs(vec![0.51, 0.49]);
        assert!(!live.is_locked());
        assert_eq!(live.lock_count(), 0);
    }

    #[test]
    fn unknown_accumulates_and_triggers() {
        let mut live = LiveClassifier::new();
        assert!(!live.is_unknown());
        for _ in 0..(UNKNOWN_CONSECUTIVE - 1) {
            live.apply_probs(vec![0.4, 0.3]);
        }
        assert!(!live.is_unknown());
        assert_eq!(live.unknown_count(), UNKNOWN_CONSECUTIVE - 1);
        live.apply_probs(vec![0.4, 0.3]);
        assert!(live.is_unknown());
        // A confident window resets the unknown streak.
        live.apply_probs(vec![0.9, 0.1]);
        assert_eq!(live.unknown_count(), 0);
        assert!(!live.is_unknown());
    }

    #[test]
    fn below_confidence_threshold_aliases_unknown() {
        let model = fixture_model(20);
        let mut live = LiveClassifier::new();
        live.load(model, None);
        let p = live.apply_probs(vec![0.4, 0.6]);
        // 0.6 >= 0.5 -> winning class is reported.
        assert_eq!(p.label, "ginger");
        let p = live.apply_probs(vec![0.45, 0.30]);
        // max 0.45 < 0.5 -> aliased to "unknown".
        assert_eq!(p.label, "unknown");
        assert!(p.confidence < 0.5);
        assert_eq!(live.current_prediction().0, "unknown");
    }

    #[test]
    fn unknown_and_lock_do_not_coincide() {
        let mut live = LiveClassifier::new();
        // Confidence in [0.5, 0.7): unknown resets, lock does not build.
        for _ in 0..10 {
            live.apply_probs(vec![0.51, 0.49]);
        }
        assert_eq!(live.lock_count(), 0);
        assert_eq!(live.unknown_count(), 0);
        assert!(!live.is_locked());
    }

    #[test]
    fn snapshot_reflects_state() {
        let model = fixture_model(20);
        let mut live = LiveClassifier::new();
        live.load(model, None);
        live.apply_probs(vec![0.95, 0.05]);
        let s = live.snapshot();
        assert!(s.loaded);
        assert_eq!(s.classes, vec!["garlic".to_string(), "ginger".to_string()]);
        assert_eq!(s.current_prediction.label, "garlic");
        assert_eq!(s.current_probs, vec![0.95, 0.05]);
        assert_eq!(s.window_size, 20);
        assert_eq!(s.buffer_len, 0);
    }
}