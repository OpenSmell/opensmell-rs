//! Wave-3 replay harness.
//!
//! Drives a `DualKalmanEngine` end-to-end over a recorded (or Monte-Carlo
//! synthetic) sample stream and emits per-sample verdicts plus the
//! TPR/FPR/PPV/latency report of `anomaly-engine-design.md` §11.3. Pure
//! software — no hardware required.
use crate::{Result, OpenSmellError};
use super::dual::{AmbientReading, DualKalmanEngine, EngineVerdict};

/// Ground truth for one replayed sample.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SampleTruth {
    /// Whether this sample really was an anomaly.
    pub is_anomaly: bool,
    /// Index where a labeled event starts (for detection-latency reporting).
    pub onset: Option<u64>,
}

/// One sample of a replay stream: the reading, optional ambient, the ground-
/// truth label used only for scoring (not fed to the engine), and the optional
/// real inter-sample gap in seconds. `dt_s = None` lets the engine's configured
/// reference cadence apply (the legacy 10 Hz behaviour).
#[derive(Debug, Clone, Default)]
pub struct Sample {
    pub reading: Vec<f64>,
    pub temperature: Option<f64>,
    pub humidity: Option<f64>,
    pub dt_s: Option<f64>,
    pub truth: Option<SampleTruth>,
}

impl Sample {
    /// Build a plain (unlabeled) sample — used for calibration or blind sweeps.
    pub fn new(reading: Vec<f64>) -> Self {
        Self { reading, temperature: None, humidity: None, dt_s: None, truth: None }
    }

    /// Attach the real seconds elapsed since the previous sample.
    pub fn with_dt(mut self, dt_s: f64) -> Self {
        self.dt_s = Some(dt_s);
        self
    }

    /// Attach a ground-truth label.
    pub fn labeled(mut self, truth: SampleTruth) -> Self {
        self.truth = Some(truth);
        self
    }
}

fn ambient_option(s: &Sample) -> Option<AmbientReading> {
    if s.temperature.is_some() || s.humidity.is_some() {
        Some(AmbientReading { temperature: s.temperature, humidity: s.humidity })
    } else {
        None
    }
}

/// Per-sample replay output: the engine verdict plus its scoring bin.
#[derive(Debug, Clone)]
pub struct ReplayVerdict {
    pub sample_index: u64,
    pub verdict: EngineVerdict,
    /// tp / tn / fp / fn relative to the labeled truth (None when unlabeled).
    pub bin: Option<ConfusionBin>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfusionBin {
    TruePositive,
    TrueNegative,
    FalsePositive,
    FalseNegative,
}

/// Aggregate confusion-matrix metrics from a replay run.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReplayMetrics {
    pub n_samples: u64,
    pub true_positives: u64,
    pub true_negatives: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub n_events: u64,
    pub detected_events: u64,
    /// Mean detection latency in samples (first flag ≥ onset − onset), over
    /// the events that were detected in time.
    pub latency_samples: Option<f64>,
    /// Mean detection latency in wall-clock seconds (sum of the per-sample
    /// `dt_s` over the latency window), over the same detected events. Only
    /// populated when the samples carry real cadence.
    pub latency_seconds: Option<f64>,
}

impl ReplayMetrics {
    pub fn tpr(&self) -> f64 {
        let p = self.true_positives + self.false_negatives;
        if p == 0 { 0.0 } else { self.true_positives as f64 / p as f64 }
    }
    pub fn fpr(&self) -> f64 {
        let n = self.false_positives + self.true_negatives;
        if n == 0 { 0.0 } else { self.false_positives as f64 / n as f64 }
    }
    pub fn ppv(&self) -> f64 {
        let p = self.true_positives + self.false_positives;
        if p == 0 { 0.0 } else { self.true_positives as f64 / p as f64 }
    }
}

/// Full result of a replay: the scorecard and every per-sample verdict.
#[derive(Debug, Clone, Default)]
pub struct ReplayReport {
    pub metrics: ReplayMetrics,
    pub verdicts: Vec<ReplayVerdict>,
}

impl DualKalmanEngine {
    /// Replay a recorded sample stream end-to-end. The engine must already be
    /// calibrated (`calibrate_baseline`) or the Warm-up path of the caller
    /// must be in place. Emits one verdict per sample and scorecard metrics.
    /// Channels must match `self.n_channels`.
    pub fn replay_dataset<'a, I>(&mut self, samples: I) -> Result<ReplayReport>
    where
        I: Iterator<Item = &'a Sample>,
    {
        let mut report = ReplayReport::default();
        let mut latencies: Vec<u64> = Vec::new();
        // Actual per-sample dt used (seconds), for wall-clock latency and so the
        // engine's process model matches the real cadence.
        let mut wall_seconds: Vec<f64> = Vec::new();
        // Pending labeled events: (onset, first-hit index, closed?).
        let mut pending_onsets: Vec<(u64, Option<u64>)> = Vec::new();

        for (index, sample) in (0u64..).zip(samples) {
            let reading = sample.reading.clone();
            let ambient = ambient_option(sample);
            let dt_s = sample.dt_s.unwrap_or(self.config.sample_period_s.max(1e-4));
            wall_seconds.push(dt_s);
            let verdict = self.detect_with_dt(&reading, ambient, dt_s).map_err(|e| {
                OpenSmellError::AnomalyDetection(format!(
                    "replay sample {index}: {e}"
                ))
            })?;

            let bin = match sample.truth {
                Some(t) => {
                    let flag = verdict.is_anomaly;
                    let bin = match (flag, t.is_anomaly) {
                        (true, true) => ConfusionBin::TruePositive,
                        (false, false) => ConfusionBin::TrueNegative,
                        (true, false) => ConfusionBin::FalsePositive,
                        (false, true) => ConfusionBin::FalseNegative,
                    };
                    let m = &mut report.metrics;
                    m.n_samples += 1;
                    match bin {
                        ConfusionBin::TruePositive => m.true_positives += 1,
                        ConfusionBin::TrueNegative => m.true_negatives += 1,
                        ConfusionBin::FalsePositive => m.false_positives += 1,
                        ConfusionBin::FalseNegative => m.false_negatives += 1,
                    }
                    if t.is_anomaly && t.onset == Some(index) {
                        m.n_events += 1;
                        pending_onsets.push((index, None));
                    }
                    if flag && !pending_onsets.is_empty() {
                        // The flag closes the earliest pending event (FIFO).
                        let entry = pending_onsets.first_mut().unwrap();
                        if entry.1.is_none() {
                            entry.1 = Some(index);
                        }
                    }
                    Some(bin)
                }
                None => None,
            };

            report.verdicts.push(ReplayVerdict {
                sample_index: index,
                verdict,
                bin,
            });
        }

        // Sweep the closed latencies.
        let mut latencies_sec: Vec<f64> = Vec::new();
        for (onset, first_hit) in pending_onsets {
            if let Some(hit) = first_hit {
                report.metrics.detected_events += 1;
                latencies.push(hit - onset);
                if (onset as usize) < wall_seconds.len() && (hit as usize) < wall_seconds.len() {
                    latencies_sec.push(
                        wall_seconds[onset as usize..=hit as usize].iter().sum::<f64>(),
                    );
                }
            }
        }
        if !latencies.is_empty() {
            report.metrics.latency_samples =
                Some(latencies.iter().sum::<u64>() as f64 / latencies.len() as f64);
        }
        if !latencies_sec.is_empty() {
            report.metrics.latency_seconds =
                Some(latencies_sec.iter().sum::<f64>() / latencies_sec.len() as f64);
        }
        if report.metrics.n_samples == 0 && report.verdicts.is_empty() {
            // Empty stream ⇒ not an error, but nothing was scored.
        }
        Ok(report)
    }

    /// Convenience wrapper: replay from an owned sample slice.
    pub fn replay_slice(&mut self, samples: &[Sample]) -> Result<ReplayReport> {
        self.replay_dataset(samples.iter())
    }
}