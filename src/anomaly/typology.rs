/// Anomaly typology head — classify the recent innovation stream into
/// spike / step / ramp / pulse so the operator learns *what* happened, not just
/// that something happened.
///
/// The Kalman filter *internalizes* a sustained level change in a few samples —
/// the innovation spikes at onset, then decays even though the environment is
/// still elevated. Classification on the innovation alone therefore mistakes a
/// step for a spike and misses pulses/ramps entirely. So, per §7 of
/// `anomaly-engine-design.md`, this head blends **innovation** features
/// (short window `W` = 60 samples @ 10 Hz = 6 s) with **state-level** features
/// (the filter's `x̂`, which remembers where the level actually is):
///
/// - `peak_z`     — max standardized innovation in `W` (sharp onset evidence)
/// - `hold`       — samples with `z > z_hold` (does the *innovation* persist?)
/// - `span`       — max per-channel `(max x̂ − min x̂)` over `W_long` / σ_i
/// - `below`      — max per-channel `(x̂_now − min x̂)` over `W_long` / σ_i (how
///   far above the floor the level is *right now*)
/// - `now` / `prev` — level change across the latest / preceding `W` window
///   (still moving ⇒ ramp, finished moving ⇒ step)
/// - `lifted`     — samples in `W_long` at least `u_notable / 2` above the
///   floor (state-based persistence: pulse vs mere blip)
///
/// Emission gate: report while `peak_z` is hot, and — for `W_long` after the
/// last hot sample — keep typing a subject whose excursion `span ≥ u_notable`
/// is still inside the long window (a pulse's quiet return to floor lands
/// here). Decision table (calibrated-σ units; `u_notable` = 6σ by default):
///
/// | peak_z ≥ z_note OR (subject alive AND span ≥ u_notable) | emit         |
/// | span < u_notable                                        | **spike**    |
/// | span ≥ u_notable AND below at floor                     | **pulse** / **spike** (lifted ≥ 3) |
/// | span ≥ u_notable, elevated, now & prev moving           | **ramp**     |
/// | span ≥ u_notable, elevated, settled                     | **step**     |
use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// What kind of change the innovation stream represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypologyKind {
    Spike,
    Step,
    Ramp,
    Pulse,
}

impl TypologyKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            TypologyKind::Spike => "spike",
            TypologyKind::Step => "step",
            TypologyKind::Ramp => "ramp",
            TypologyKind::Pulse => "pulse",
        }
    }
}

/// A classified change, ready for the status line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Typology {
    pub kind: TypologyKind,
    pub channels: Vec<usize>,
    /// Sample index at detection (10 Hz units).
    pub onset_sample: u64,
    pub amplitude: f64,
    pub duration_samples: usize,
}

/// Typology-head configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypologyConfig {
    pub window: usize,
    pub window_long: usize,
    /// z above which a deviation "holds" (persistence threshold).
    pub z_hold: f64,
    /// z above which a deviation is a candidate event.
    pub z_note: f64,
    /// State-level change (in calibrated σ) above which a change is a lasting
    /// level change (step/ramp/pulse) rather than a transient (spike).
    #[serde(default = "default_u_notable")]
    pub u_notable: f64,
}

fn default_u_notable() -> f64 {
    6.0
}

impl Default for TypologyConfig {
    fn default() -> Self {
        Self {
            window: 60,
            window_long: 600,
            z_hold: 1.5,
            z_note: 2.0,
            u_notable: default_u_notable(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct StepStat {
    z: VecDeque<f64>,
}

impl StepStat {
    fn new() -> Self {
        Self::default()
    }
    fn push(&mut self, z: f64, cap: usize) {
        self.z.push_back(z);
        if self.z.len() > cap {
            self.z.pop_front();
        }
    }
    fn len(&self) -> usize {
        self.z.len()
    }
}

#[derive(Debug, Clone)]
pub struct TypologyHead {
    pub config: TypologyConfig,
    short: StepStat,
    /// Rolling level block `x̂` over the long window (for floor/span/ramp).
    levels_long: VecDeque<Vec<f64>>,
    /// Most recent sample with a hot innovation. The subject stays "alive" for
    /// `window_long` samples after it, so a pulse's quiet return is still typed.
    last_event: Option<u64>,
    sample: u64,
}

impl TypologyHead {
    pub fn new(config: TypologyConfig) -> Self {
        let window_long = config.window_long;
        Self {
            config,
            short: StepStat::new(),
            levels_long: VecDeque::with_capacity(window_long),
            last_event: None,
            sample: 0,
        }
    }

    /// Feed one filtered sample and ask whether a change has taken shape.
    ///
    /// `level` is the filter's current environmental state block `x̂` (same units
    /// as the readings) and `sigma` the per-channel calibrated measurement σ
    /// `sqrt(R_ii)` — the level features are expressed in those units.
    pub fn update(
        &mut self,
        z_scores: &[f64],
        big_channels: &[usize],
        level: &[f64],
        sigma: &[f64],
    ) -> Option<Typology> {
        self.sample += 1;
        let n_ch = z_scores.len();
        let z = z_scores.iter().cloned().fold(0.0f64, f64::max).max(0.0);
        self.short.push(z, self.config.window);
        self.levels_long.push_back(level.to_vec());
        if self.levels_long.len() > self.config.window_long {
            self.levels_long.pop_front();
        }

        let u_note = self.config.u_notable;
        let w = self.config.window;
        if self.short.len() < w {
            return None;
        }
        let peak_z = self.short.z.iter().cloned().fold(0.0f64, f64::max);
        let hold = self.short.z.iter().filter(|&&v| v > self.config.z_hold).count();

        let sigma_of = |i: usize| sigma.get(i).copied().filter(|&s| s > 0.0).unwrap_or(1.0);

        // Level geometry over the long window (per channel, in calibrated σ):
        // `span`  = total up-and-back excursion family,
        // `below` = how far the level is above its floor *right now*,
        // `lifted` = how many recent samples sat at least half a notable
        //            displacement above the floor (state-based persistence).
        let len = self.levels_long.len();
        let (span, below, lifted) = if len >= w {
            (0..n_ch.min(level.len()))
                .map(|i| {
                    let s = sigma_of(i);
                    let mut mx = f64::MIN;
                    let mut mn = f64::MAX;
                    for v in &self.levels_long {
                        mx = mx.max(v[i]);
                        mn = mn.min(v[i]);
                    }
                    let thr = u_note * 0.5 * s;
                    let lifted = self
                        .levels_long
                        .iter()
                        .filter(|v| v[i] - mn >= thr)
                        .count();
                    let cur = level.get(i).copied().unwrap_or(0.0);
                    ((mx - mn) / s, (cur - mn) / s, lifted)
                })
                .fold((0.0f64, 0.0f64, 0usize), |(as_, bs, li), (a, b, l)| {
                    (as_.max(a), bs.max(b), li.max(l))
                })
        } else {
            (0.0, 0.0, 0)
        };

        // Emission gate: report while the innovation is hot (peak_z); after a
        // notable excursion also keep typing the subject while its level geometry
        // is still inside the long window — a pulse's quiet "return to floor"
        // lands here, long after the innovation has settled.
        let subject_alive = self
            .last_event
            .is_some_and(|t| self.sample - t <= self.config.window_long as u64);
        let live = peak_z >= self.config.z_note || (subject_alive && span >= u_note);
        if !live {
            return None;
        }
        if peak_z >= self.config.z_note {
            self.last_event = Some(self.sample);
        }

        // Movement freshness: the level change across the latest `W` (`now`) vs
        // the `W` before that (`prev`). A ramp is moving in both; a settled
        // step moved once and stopped.
        let (now, prev) = if len > 2 * w {
            let cur = self.levels_long.back().unwrap();
            let mid = &self.levels_long[len - 1 - w];
            let ref_l = &self.levels_long[len - 1 - 2 * w];
            (
                (0..n_ch.min(level.len()))
                    .map(|i| {
                        let s = sigma_of(i);
                        (cur.get(i).copied().unwrap_or(0.0) - mid.get(i).copied().unwrap_or(0.0)).abs() / s
                    })
                    .fold(0.0f64, f64::max),
                (0..n_ch.min(level.len()))
                    .map(|i| {
                        let s = sigma_of(i);
                        (mid.get(i).copied().unwrap_or(0.0) - ref_l.get(i).copied().unwrap_or(0.0)).abs() / s
                    })
                    .fold(0.0f64, f64::max),
            )
        } else {
            (0.0, 0.0)
        };

        let channels = if big_channels.is_empty() {
            z_scores
                .iter()
                .enumerate()
                .filter(|&(_, v)| *v > self.config.z_note)
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        } else {
            big_channels.to_vec()
        };

        let onset = self.sample.saturating_sub(hold.max(1) as u64);
        let amplitude = peak_z;

        // §7 decision table, state-anchored: the floor/span geometry survives
        // the innovation settling, so a settled step still reads as a step and
        // a recovered pulse reads as a pulse.
        let typ = if span < u_note {
            // The level never really moved: a pure innovation blip.
            TypologyKind::Spike
        } else if below < u_note * 0.5 {
            // Big excursion in the long window, but the level is back at its
            // floor now: held above the floor for a while (pulse) vs a mere
            // blip that never persisted (spike).
            if lifted >= 3 {
                TypologyKind::Pulse
            } else {
                TypologyKind::Spike
            }
        } else if now >= u_note * 0.6 && prev >= u_note * 0.6 {
            // Still above the floor and still moving in both windows: climb.
            TypologyKind::Ramp
        } else {
            TypologyKind::Step
        };

        Some(Typology {
            kind: typ,
            channels,
            onset_sample: onset,
            amplitude,
            duration_samples: hold.max(1),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the head with synthetic (z, level, sigma) streams in the same
    /// style the engine does: `z` is the standardized innovation, `level` the
    /// filter's environmental state, `sigma` the calibrated σ.
    fn run(kind: TypologyKind) -> Option<TypologyKind> {
        let mut head = TypologyHead::new(TypologyConfig::default());
        let mut out = None;
        let sigma = [0.1];
        match kind {
            TypologyKind::Spike => {
                let mut z = vec![0.0f64; 80];
                z[10] = 8.0;
                for (i, &zz) in z.iter().enumerate() {
                    let lvl = vec![1.0]; // level never moves
                    if let Some(t) = head.update(&[0.0, zz], &[], &lvl, &sigma) {
                        out = Some(t.kind);
                    }
                    let _ = i;
                }
            }
            TypologyKind::Step => {
                for i in 0..70 {
                    let zz = if i < 10 { 0.0 } else { 5.0 };
                    let lvl = vec![if i < 10 { 1.0 } else { 1.8 }]; // +0.8 → 8σ
                    if let Some(t) = head.update(&[0.0, zz], &[], &lvl, &sigma) {
                        out = Some(t.kind);
                    }
                }
            }
            TypologyKind::Pulse => {
                for i in 0..160 {
                    let elevated = (20..50).contains(&i); // 30 samples
                    let zz = if elevated { 5.0 } else { 0.0 };
                    let lvl = vec![if elevated { 1.8 } else { 1.0 }];
                    if let Some(t) = head.update(&[0.0, zz], &[], &lvl, &sigma) {
                        out = Some(t.kind);
                    }
                }
            }
            TypologyKind::Ramp => {
                for i in 0..300 {
                    let zz = 5.0 + 3.0 * (i as f64 / 300.0); // climbing innovation
                    let lvl = vec![1.0 + 3.0 * (i as f64 / 300.0)]; // climbing level
                    if let Some(t) = head.update(&[0.0, zz], &[], &lvl, &sigma) {
                        out = Some(t.kind);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn spike_classified() {
        assert_eq!(run(TypologyKind::Spike), Some(TypologyKind::Spike));
    }

    #[test]
    fn step_classified() {
        assert_eq!(run(TypologyKind::Step), Some(TypologyKind::Step));
    }

    #[test]
    fn pulse_classified() {
        assert_eq!(run(TypologyKind::Pulse), Some(TypologyKind::Pulse));
    }

    #[test]
    fn ramp_classified() {
        assert_eq!(run(TypologyKind::Ramp), Some(TypologyKind::Ramp));
    }

    #[test]
    fn quiet_stream_emits_nothing() {
        let mut head = TypologyHead::new(TypologyConfig::default());
        for _ in 0..200 {
            assert!(head.update(&[0.05, 0.02], &[], &[1.0, 1.0], &[0.1, 0.1]).is_none());
        }
    }
}