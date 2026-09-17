//! Wave-5 process adapters.
//!
//! The `DualKalmanEngine` is deliberately general-purpose: it reports *that*
//! something changed (`EngineVerdict`), at what scale (`raw_score`, `max_z`),
//! and what shape (`Typology`), without knowing what the device is watching.
//! A `ProcessAdapter` is the single place domain interpretation lives: it turns
//! those generic signals into process events ("fermentation entered the
//! exponential stage", "sensor fault confirmed on channel 2") that a status
//! line or dashboard can show. Per `master-architecture.md` §5 the engine
//! never hard-codes a process vocabulary; the adapter owns that mapping.

use serde::{Deserialize, Serialize};

use super::dual::EngineVerdict;
use super::stimulus::HealthFindingKind;
use super::typology::TypologyKind;

/// What class of process phenomenon an event represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessEventKind {
    /// The process moved between named stages (e.g. a regime switch).
    StageTransition,
    /// A sharp, sustained change (typology: step).
    RapidChange,
    /// A slow, monotone change (typology: ramp).
    SlowChange,
    /// A short-lived excursion (typology: spike/pulse).
    Transient,
    /// A hardware/health problem surfaced by the engine (e.g. poison confirmed).
    SensorFault,
    /// Status noise worth showing but not alarming.
    Info,
}

impl ProcessEventKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProcessEventKind::StageTransition => "stage-transition",
            ProcessEventKind::RapidChange => "rapid-change",
            ProcessEventKind::SlowChange => "slow-change",
            ProcessEventKind::Transient => "transient",
            ProcessEventKind::SensorFault => "sensor-fault",
            ProcessEventKind::Info => "info",
        }
    }

    /// 0 = info, 1 = minor, 2 = major, 3 = critical.
    pub fn severity(&self) -> u8 {
        match self {
            ProcessEventKind::StageTransition | ProcessEventKind::Info => 0,
            ProcessEventKind::Transient => 1,
            ProcessEventKind::RapidChange | ProcessEventKind::SlowChange => 2,
            ProcessEventKind::SensorFault => 3,
        }
    }
}

/// One domain-level event derived from an engine verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessEvent {
    pub kind: ProcessEventKind,
    /// Process-domain stage at the time of the event (domain vocabulary).
    pub stage: String,
    /// Channel the event concerns (None for stage/process-wide events).
    pub channel: Option<usize>,
    pub severity: u8,
    pub message: String,
}

impl ProcessEvent {
    pub fn new(kind: ProcessEventKind, stage: &str, message: impl Into<String>) -> Self {
        let severity = kind.severity();
        Self { kind, stage: stage.to_string(), channel: None, severity, message: message.into() }
    }

    pub fn on_channel(
        kind: ProcessEventKind,
        stage: &str,
        channel: usize,
        message: impl Into<String>,
    ) -> Self {
        Self { channel: Some(channel), ..Self::new(kind, stage, message) }
    }
}

/// A process adapter turns generic engine verdicts into domain events.
///
/// Adapters are stateful when they need to know the previous verdict (e.g. a
/// stage transition is only a transition relative to the prior stage).
pub trait ProcessAdapter {
    /// Consume one engine verdict, returning any domain events it implies.
    fn update(&mut self, verdict: &EngineVerdict) -> Vec<ProcessEvent>;
    /// One-line status suitable for a dashboard / status line.
    fn summarize(&self, verdict: &EngineVerdict, events: &[ProcessEvent]) -> String;
}

/// Fermentation-specific stage vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FermentationStage {
    /// Sensor watch, no active fermentation.
    Idle,
    /// Inoculated / waiting for onset.
    Lag,
    /// Growing rapidly (the most informative stage).
    Exponential,
    /// Yield plateau / slowing down.
    Stationary,
    /// Decline / end of run.
    Decline,
    /// The engine's regime could not be mapped.
    #[default]
    Unknown,
}

impl FermentationStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            FermentationStage::Idle => "idle",
            FermentationStage::Lag => "lag",
            FermentationStage::Exponential => "exponential",
            FermentationStage::Stationary => "stationary",
            FermentationStage::Decline => "decline",
            FermentationStage::Unknown => "unknown",
        }
    }
}

/// Maps the engine's regime index onto a fermentation stage and turns its
/// verdicts into fermentation process events.
///
/// The regime → stage mapping is the *only* domain knowledge here; it is a
/// configuration supplied by whoever knows this process (how many regimes the
/// rig tends to exhibit, what the lowest regime means, etc.). Everything else
/// follows mechanically from the `EngineVerdict`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FermentationAdapter {
    /// Regime index → fermentation stage (index aligned with the engine's
    /// regime ordering, lowest environmental level first). Missing regimes are
    /// `Unknown`.
    pub regime_stages: Vec<FermentationStage>,
    /// Only surface change events whose confidence is at least this high.
    pub confidence_threshold: f64,
    /// Current stage (adapter state, tracked across verdicts).
    #[serde(skip)]
    pub current_stage: FermentationStage,
}

impl Default for FermentationAdapter {
    fn default() -> Self {
        Self {
            regime_stages: vec![
                FermentationStage::Idle,
                FermentationStage::Lag,
                FermentationStage::Exponential,
                FermentationStage::Stationary,
            ],
            confidence_threshold: 0.6,
            current_stage: FermentationStage::Unknown,
        }
    }
}

impl FermentationAdapter {
    fn stage_of(&self, regime: usize) -> FermentationStage {
        self.regime_stages.get(regime).copied().unwrap_or(FermentationStage::Unknown)
    }
}

impl ProcessAdapter for FermentationAdapter {
    fn update(&mut self, verdict: &EngineVerdict) -> Vec<ProcessEvent> {
        let mut events = Vec::new();
        let stage = self.stage_of(verdict.regime);
        let confident = verdict.confidence >= self.confidence_threshold;

        // 1. Stage transition (regime switch or a crossed confidence threshold).
        if stage != self.current_stage || verdict.regime_switch {
            let mut label = format!(
                "Stage transition: {} → {}",
                self.current_stage.as_str(),
                stage.as_str()
            );
            if verdict.regime_switch {
                label.push_str(" (regime switch)");
            }
            let mut ev = ProcessEvent::new(
                ProcessEventKind::StageTransition,
                stage.as_str(),
                label,
            );
            ev.channel = None;
            events.push(ev);
            self.current_stage = stage;
        }

        // 2. Anomaly shape → process phenomenon.
        if verdict.is_anomaly && confident {
            match verdict.typology.as_ref().map(|t| t.kind) {
                Some(TypologyKind::Step) => events.push(ProcessEvent::new(
                    ProcessEventKind::RapidChange,
                    stage.as_str(),
                    format!("Rapid change (step, votes {})", verdict.anomaly_votes),
                )),
                Some(TypologyKind::Ramp) => events.push(ProcessEvent::new(
                    ProcessEventKind::SlowChange,
                    stage.as_str(),
                    "Slow drift (ramp) — stage may be transitioning gradually".to_string(),
                )),
                Some(TypologyKind::Spike | TypologyKind::Pulse) => events.push(
                    ProcessEvent::new(
                        ProcessEventKind::Transient,
                        stage.as_str(),
                        "Short transient (spike/pulse)".to_string(),
                    ),
                ),
                None => events.push(ProcessEvent::new(
                    ProcessEventKind::Info,
                    stage.as_str(),
                    format!("Anomaly (votes {})", verdict.anomaly_votes),
                )),
            }
        }

        // 3. Confirmed poison from the engine's own health findings.
        for f in &verdict.health_findings {
            if f.kind == HealthFindingKind::PoisonConfirmed {
                events.push(ProcessEvent::on_channel(
                    ProcessEventKind::SensorFault,
                    stage.as_str(),
                    f.channel,
                    format!("Sensor fault: confirmed poisoned on channel {}", f.channel),
                ));
            }
        }

        events
    }

    fn summarize(&self, verdict: &EngineVerdict, events: &[ProcessEvent]) -> String {
        let stage = self.stage_of(verdict.regime);
        let mut parts = vec![format!("stage={}", stage.as_str())];
        let mut anomaly = false;
        let mut faults = 0usize;
        for ev in events {
            match ev.kind {
                ProcessEventKind::StageTransition => parts.push(ev.message.clone()),
                ProcessEventKind::RapidChange | ProcessEventKind::SlowChange => {
                    anomaly = true;
                    parts.push(ev.message.clone());
                }
                ProcessEventKind::SensorFault => faults += 1,
                _ => {}
            }
        }
        if anomaly {
            parts.push("CHANGE".to_string());
        }
        if faults > 0 {
            parts.push(format!("{faults} sensor fault(s)"));
        }
        parts.join(" · ")
    }
}

/// Convenience: run an adapter over a whole replayed verdict stream and emit
/// the process events (used by dashboards and the Wave-3 metrics report).
pub fn summarize_replay<A: ProcessAdapter>(
    adapter: &mut A,
    verdicts: &[crate::anomaly::replay::ReplayVerdict],
) -> Vec<(u64, String)> {
    let mut out = Vec::new();
    for rv in verdicts {
        let events = adapter.update(&rv.verdict);
        let line = adapter.summarize(&rv.verdict, &events);
        if !events.is_empty() {
            out.push((rv.sample_index, line));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anomaly::dual::DualKalmanEngine;

    #[test]
    fn stage_transition_fires_on_regime_change() {
        let mut eng = DualKalmanEngine::new(1);
        eng.calibrate_baseline(&baseline()).unwrap();
        let mut adapter = FermentationAdapter::default();
        // Warm into regime 0.
        for _ in 0..30 {
            let v = eng.detect_no_ambient(&[1.0]).unwrap();
            let _ = adapter.update(&v);
        }
        assert_eq!(adapter.current_stage, FermentationStage::Idle);
        // A sustained level change moves the regime; the adapter must surface it.
        let mut transition = false;
        for _ in 0..300 {
            let v = eng.detect_no_ambient(&[6.0]).unwrap();
            if adapter
                .update(&v)
                .iter()
                .any(|e| e.kind == ProcessEventKind::StageTransition)
            {
                transition = true;
                break;
            }
        }
        assert!(transition, "regime change must emit a stage transition");
    }

    #[test]
    fn poison_fault_surfaces_finding() {
        let mut eng = DualKalmanEngine::new(1);
        eng.calibrate_baseline(&baseline()).unwrap();
        let mut adapter = FermentationAdapter::default();
        for _ in 0..10 {
            let _ = adapter.update(&eng.detect_no_ambient(&[1.0]).unwrap());
        }
        eng.set_relative_gains(vec![0.4]);
        let _ = eng.record_stimulus(&[0.4], &[1.0]).unwrap();
        let faulty = adapter
            .update(&eng.detect_no_ambient(&[1.0]).unwrap())
            .iter()
            .any(|e| e.kind == ProcessEventKind::SensorFault);
        assert!(faulty, "confirmed poison must become a sensor-fault event");
    }

    #[test]
    fn summarize_mentions_stage() {
        let mut eng = DualKalmanEngine::new(1);
        eng.calibrate_baseline(&baseline()).unwrap();
        let adapter = FermentationAdapter::default();
        for _ in 0..30 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        let v = eng.detect_no_ambient(&[1.0]).unwrap();
        let line = adapter.summarize(&v, &[]);
        assert!(line.contains("stage="), "summary must carry the stage: {line}");
    }

    fn baseline() -> Vec<Vec<f64>> {
        (0..80)
            .map(|i| {
                let n = i as f64;
                vec![1.0 + 0.05 * (n * 1.7).sin()]
            })
            .collect()
    }
}