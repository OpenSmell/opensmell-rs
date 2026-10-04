//! The policy layer that turns detections into an actionable answer.
//!
//! Everything below this file *detects*: the dual engine fires on multivariate
//! outliers, the EWMA chart watches for level shifts, `poisoning.rs` tracks
//! sensitivity decay and noise growth over 24 h windows, and the health
//! features report drift rate and noise floor per channel. None of them answer
//! the question a user actually has, which is:
//!
//! > Do I do something about this, and what?
//!
//! A binary `is_anomaly` cannot. The literature is explicit on why
//! ([`docs/research/literature-drift-poisoning-anomaly.md`]): a drift-correction
//! layer that *forgives* drift is exactly right for gradual drift and actively
//! dangerous for poisoning or end-of-life, because both present as a slow change
//! in level. A detector that learns to ignore slow change will eventually learn
//! to ignore a dying sensor.
//!
//! So the verdict is four-way, and the split is not cosmetic — it decides
//! whether correction is permitted:
//!
//! | Verdict | Correctable | Meaning |
//! |---|---|---|
//! | [`SensorVerdict::Normal`] | — | within expected behaviour |
//! | [`SensorVerdict::GradualDrift`] | **yes** | slow baseline movement, keep running |
//! | [`SensorVerdict::Poisoning`] | **no** | sensitivity lost, replace sensor |
//! | [`SensorVerdict::Dead`] | — | no response at all, channel is gone |
//!
//! Poisoning outranks everything except `Dead`. If a channel both drifts and
//! decays, it is poisoned; re-anchoring a poisoned sensor destroys the very
//! baseline you would need to prove the loss.
//!
//! Thresholds are named constants with their basis recorded. They are defaults
//! for MOX arrays on a typical duty cycle, not laws; every one is overridable
//! through [`VerdictThresholds`] so a user with a different sensor or a faster
//! sampling rate can retune without forking the policy.

use serde::{Deserialize, Serialize};

use crate::features::health;
use crate::poisoning::{DegradationType, SensorHealthStatus};

/// What the user should do about a channel right now.
///
/// Ordered by severity, not by frequency. `Ord` is derived so the worst
/// condition always wins when several channels are reduced at once — a
/// vectorised consumer that takes `.max()` gets the most urgent answer without
/// needing to know the ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SensorVerdict {
    /// Within expected behaviour. No action.
    Normal,
    /// Slow baseline movement, recoverable by re-anchoring.
    ///
    /// The only verdict a drift-correction layer may act on.
    GradualDrift,
    /// Sensitivity or noise degrading beyond correction.
    ///
    /// Permanent. Re-anchoring is *forbidden* here — see module docs.
    Poisoning,
    /// Channel has stopped responding.
    ///
    /// Distinct from `Poisoning`: dead means replace now, poisoning means
    /// schedule a replacement.
    Dead,
}

impl SensorVerdict {
    /// Whether a drift-correction layer is permitted to re-anchor this channel.
    ///
    /// The single most consequential method here. Everything that "just fixes
    /// drift" downstream must consult this rather than re-checking for slowness,
    /// because slowness is exactly what poisoning looks like.
    ///
    /// Equivalent to [`SensorVerdict::allows_reanchoring`]; kept as the
    /// name callers reach for at the decision site.
    pub fn is_correctable(self) -> bool {
        self.allows_reanchoring()
    }

    /// Whether the channel is still usable without intervention.
    pub fn is_operational(self) -> bool {
        matches!(self, SensorVerdict::Normal | SensorVerdict::GradualDrift)
    }

    /// Whether re-anchoring this channel is safe.
    ///
    /// True for `Normal` as well as `GradualDrift`: re-anchoring a healthy
    /// channel is a no-op that keeps its baseline current, and refusing it
    /// would leave a long-running unit permanently frozen on its
    /// original baseline. What matters is that `Poisoning` and `Dead` are
    /// excluded — those are the verdicts where re-anchoring would destroy
    /// evidence of the degradation.
    pub fn allows_reanchoring(self) -> bool {
        matches!(self, SensorVerdict::Normal | SensorVerdict::GradualDrift)
    }

    /// Whether a human needs to be involved.
    pub fn needs_action(self) -> bool {
        !matches!(self, SensorVerdict::Normal)
    }

    /// Short machine-readable label, for logs and serial output.
    pub fn as_str(self) -> &'static str {
        match self {
            SensorVerdict::Normal => "normal",
            SensorVerdict::GradualDrift => "gradual_drift",
            SensorVerdict::Poisoning => "poisoning",
            SensorVerdict::Dead => "dead",
        }
    }
}

impl std::fmt::Display for SensorVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per-channel input to the policy. Assembled by the caller from whatever
/// detectors are in use, so the policy has no dependency on any of them.
#[derive(Debug, Clone, Default)]
pub struct ChannelEvidence {
    /// Least-squares `dR/dt` from the health features, in sensor units/second.
    pub drift_rate: f64,
    /// `True` when a multivariate detector fired on this channel this step.
    pub outlier: bool,
    /// `poisoning.rs` status, when a 24 h window is available.
    pub health: Option<SensorHealthStatus>,
}

impl ChannelEvidence {
    /// Evidence for a channel with no detectors reporting — used on cold start
    /// and in tests. Never sufficient on its own to condemn a channel.
    pub fn quiet() -> Self {
        Self::default()
    }
}

/// Thresholds for [`classify`]. Every field is overridable; the defaults target
/// MOX arrays on a multi-minute sampling cadence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VerdictThresholds {
    /// Signal-to-noise below this is a dead channel, not a quiet one.
    ///
    /// Basis: a working MOX channel sits far above its own baseline
    /// scatter; 0.02 is ~2% relative variation, well under any healthy
    /// array's response to its target gas.
    pub dead_noise_ceiling: f64,

    /// Absolute `drift_rate` below which movement is ignored, sensor units/s.
    ///
    /// Basis: chosen so that thermal settling after a stimulus (order 1e-3 to
    /// 1e-2 units/s) does not register as drift, while genuine baseline
    /// creep does.
    pub drift_abs_floor: f64,

    /// Relative drift, as a fraction of baseline per window.
    ///
    /// Basis: literature drift benchmarks place meaningful baseline creep
    /// above ~1%/window; below that, correction is indistinguishable from noise.
    pub drift_rel_ceiling: f64,
}

impl Default for VerdictThresholds {
    fn default() -> Self {
        Self {
            dead_noise_ceiling: 0.02,
            drift_abs_floor: 1e-3,
            drift_rel_ceiling: 0.01,
        }
    }
}

/// Classify one channel.
///
/// Precedence is fixed and deliberate:
///
/// 1. **Dead** — no response. Nothing else can be trusted about the channel,
///    so it is decided first and short-circuits.
/// 2. **Poisoning** — `poisoning.rs` has already done multi-window work to
///    reach this, and it is irreversible. Checked before drift because a
///    poisoned channel also drifts, and drift would otherwise mask it.
/// 3. **GradualDrift** — slow, bounded, correctable movement.
/// 4. **Normal** — everything else, including isolated outliers, which are
///    expected in a noisy array and must not condemn a channel on one sample.
pub fn classify(ev: &ChannelEvidence, r0: f64, th: &VerdictThresholds) -> SensorVerdict {
    if is_dead(ev, th) {
        return SensorVerdict::Dead;
    }

    if let Some(h) = &ev.health {
        if !h.is_healthy && irreversible(h.degradation_type) {
            return SensorVerdict::Poisoning;
        }
    }

    if is_gradual_drift(ev, r0, th) {
        return SensorVerdict::GradualDrift;
    }

    SensorVerdict::Normal
}

/// A channel with no measurable response. `health.rs` already reports
/// `noise_floor` as baseline-relative, so this is directly comparable across
/// devices.
fn is_dead(ev: &ChannelEvidence, th: &VerdictThresholds) -> bool {
    if let Some(h) = &ev.health {
        // `noise_floor` is absolute, `baseline_level` is the signal the channel
        // was sitting at. A channel whose scatter has collapsed relative to its
        // own level has stopped responding. Both must be present: without a
        // health window there is not enough evidence to condemn a channel, and
        // a false "dead" would silently discard a working sensor.
        if h.metrics.baseline_level > 0.0
            && h.metrics.noise_floor / h.metrics.baseline_level < th.dead_noise_ceiling
        {
            return true;
        }
    }
    false
}

/// Only the degradation modes that do not reverse on their own.
///
/// `RecoverySlowdown` is surface contamination and `BaselineDrift` is
/// environmental — both are correctable, so routing them to `Poisoning` would
/// condemn sensors that a re-anchor would rescue.
fn irreversible(kind: Option<DegradationType>) -> bool {
    matches!(
        kind,
        Some(DegradationType::SensitivityDecay) | Some(DegradationType::NoiseIncrease)
    )
}

fn is_gradual_drift(ev: &ChannelEvidence, r0: f64, th: &VerdictThresholds) -> bool {
    let abs_ok = ev.drift_rate.abs() < th.drift_abs_floor;
    if abs_ok {
        return false;
    }
    // Relative check keeps the verdict device-agnostic: the same physical creep
    // is a bigger absolute number on a high-resistance sensor.
    if r0 > 0.0 {
        ev.drift_rate.abs() / r0 <= th.drift_rel_ceiling
    } else {
        // No baseline to normalise against; fall back to absolute only.
        true
    }
}

/// Verdict for a whole array, with the worst channel identified.
///
/// A single poisoned channel condemns the array for unattended operation even if
/// the rest are healthy: a gas classification over a poisoned sensor is not a
/// measurement, and the failure is silent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayVerdict {
    pub per_channel: Vec<SensorVerdict>,
    /// Index of the most severe channel, if any needs action.
    pub worst_channel: Option<usize>,
}

impl ArrayVerdict {
    pub fn worst(&self) -> SensorVerdict {
        self.per_channel.iter().copied().max().unwrap_or(SensorVerdict::Normal)
    }

    /// Whether the array as a whole is trustworthy for unattended use.
    pub fn is_operational(&self) -> bool {
        self.per_channel.iter().all(|v| v.is_operational())
    }

    /// Whether drift correction may run. Requires every channel to be
    /// re-anchorable: a single poisoned or dead channel blocks correction for
    /// the whole array, since re-anchoring one channel shifts the reference the
    /// others are compared against.
    pub fn correction_permitted(&self) -> bool {
        self.per_channel.iter().all(|v| v.allows_reanchoring())
    }

    /// Channels needing human attention.
    pub fn degraded_channels(&self) -> Vec<usize> {
        self.per_channel
            .iter()
            .enumerate()
            .filter(|(_, v)| v.needs_action())
            .map(|(i, _)| i)
            .collect()
    }
}

/// Classify every channel of an array.
pub fn classify_array(
    evidence: &[ChannelEvidence],
    r0: &[f64],
    th: &VerdictThresholds,
) -> ArrayVerdict {
    let per_channel: Vec<SensorVerdict> = (0..evidence.len())
        .map(|i| {
            let base = r0.get(i).copied().unwrap_or(0.0);
            classify(&evidence[i], base, th)
        })
        .collect();
    let worst_channel = per_channel
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != SensorVerdict::Normal)
        .max_by_key(|(_, v)| **v)
        .map(|(i, _)| i);
    ArrayVerdict { per_channel, worst_channel }
}

/// Build evidence straight from a recorded window plus its baseline.
///
/// Convenience for the common case where `poisoning.rs` has not yet accumulated
/// a 24 h window: health features still give drift and noise, which is enough
/// for `Dead` and `GradualDrift`.
pub fn evidence_from_window(
    window: &[Vec<f64>],
    baseline: &crate::Baseline,
    sr: f64,
    outliers: &[usize],
) -> crate::Result<Vec<ChannelEvidence>> {
    let features = health::extract_window(window, baseline, sr)?;
    let n_channels = window.first().map(|s| s.len()).unwrap_or(0);
    Ok((0..n_channels)
        .map(|ch| ChannelEvidence {
            drift_rate: features[ch * 4 + 2],
            outlier: outliers.contains(&ch),
            health: None,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Baseline;

    fn ev(drift: f64, outlier: bool) -> ChannelEvidence {
        ChannelEvidence { drift_rate: drift, outlier, health: None }
    }

    fn metrics(sensitivity: f64, noise: f64, recovery: f64) -> crate::poisoning::SensorMetrics {
        crate::poisoning::SensorMetrics {
            sensitivity,
            noise_floor: noise,
            recovery_time: recovery,
            baseline_level: 1.0,
            drift_rate: 0.001,
        }
    }

    fn status(kind: DegradationType, healthy: bool, m: crate::poisoning::SensorMetrics) -> SensorHealthStatus {
        SensorHealthStatus {
            channel: 0,
            is_healthy: healthy,
            health_score: if healthy { 1.0 } else { 0.3 },
            degradation_type: Some(kind),
            degradation_rate: 0.02,
            estimated_remaining_life_hours: 100.0,
            warning_level: if healthy { "normal" } else { "critical" }.to_string(),
            metrics: m,
        }
    }

    #[test]
    fn quiet_channel_is_normal() {
        assert_eq!(classify(&ev(0.0, false), 1.0, &VerdictThresholds::default()), SensorVerdict::Normal);
    }

    #[test]
    fn isolated_outlier_alone_never_condemns() {
        // One noisy sample must not condemn a channel.
        assert_eq!(classify(&ev(0.0, true), 1.0, &VerdictThresholds::default()), SensorVerdict::Normal);
    }

    #[test]
    fn small_relative_creep_is_drift_not_poisoning() {
        let th = VerdictThresholds::default();
        // 0.5% of baseline per window: above the absolute floor, below the
        // relative ceiling. Correctable.
        let v = classify(&ev(5e-3, false), 1.0, &th);
        assert_eq!(v, SensorVerdict::GradualDrift);
        assert!(v.is_correctable(), "gradual drift must permit correction");
    }

    #[test]
    fn large_relative_creep_outside_ceiling_is_normal_not_drift() {
        // Beyond the relative ceiling we decline to call it correctable drift:
        // too large to be ordinary creep, and `poisoning.rs` has not confirmed
        // it, so the honest answer is "no evidence of a correctable problem".
        let th = VerdictThresholds::default();
        assert_eq!(classify(&ev(0.5, false), 1.0, &th), SensorVerdict::Normal);
    }

    #[test]
    fn sensitivity_decay_is_poisoning_and_blocks_correction() {
        let h = SensorHealthStatus {
            channel: 0,
            is_healthy: false,
            health_score: 0.3,
            degradation_type: Some(DegradationType::SensitivityDecay),
            degradation_rate: 0.02,
            estimated_remaining_life_hours: 100.0,
            warning_level: "critical".to_string(),
            metrics: crate::poisoning::SensorMetrics {
                sensitivity: 0.5,
                noise_floor: 0.5,
                recovery_time: 10.0,
                baseline_level: 1.0,
                drift_rate: 0.001,
            },
        };
        let e = ChannelEvidence { drift_rate: 1e-3, outlier: false, health: Some(h) };
        let v = classify(&e, 1.0, &VerdictThresholds::default());
        assert_eq!(v, SensorVerdict::Poisoning);
        assert!(!v.is_correctable(), "poisoning must NEVER be correctable");
    }

    #[test]
    fn poisoning_outranks_drift() {
        // A poisoned channel also drifts. Drift must not mask it.
        let h = SensorHealthStatus {
            channel: 0,
            is_healthy: false,
            health_score: 0.2,
            degradation_type: Some(DegradationType::NoiseIncrease),
            degradation_rate: 0.05,
            estimated_remaining_life_hours: 20.0,
            warning_level: "critical".to_string(),
            metrics: crate::poisoning::SensorMetrics {
                sensitivity: 0.3,
                noise_floor: 0.5,
                recovery_time: 12.0,
                baseline_level: 1.0,
                drift_rate: 0.01,
            },
        };
        let e = ChannelEvidence { drift_rate: 5e-3, outlier: false, health: Some(h) };
        assert_eq!(classify(&e, 1.0, &VerdictThresholds::default()), SensorVerdict::Poisoning);
    }

    #[test]
    fn reversible_degradation_is_not_poisoning() {
        // Surface contamination and environmental drift correct themselves.
        for kind in [DegradationType::RecoverySlowdown, DegradationType::BaselineDrift] {
            let h = SensorHealthStatus {
                channel: 0,
                is_healthy: false,
                health_score: 0.7,
                degradation_type: Some(kind),
                degradation_rate: 0.01,
                estimated_remaining_life_hours: 500.0,
                warning_level: "warning".to_string(),
                metrics: crate::poisoning::SensorMetrics {
                    sensitivity: 0.8,
                    noise_floor: 0.3,
                    recovery_time: 30.0,
                    baseline_level: 1.0,
                    drift_rate: 0.005,
                },
            };
            let e = ChannelEvidence { drift_rate: 5e-3, outlier: false, health: Some(h) };
            assert_eq!(
                classify(&e, 1.0, &VerdictThresholds::default()),
                SensorVerdict::GradualDrift,
                "{kind:?} must stay correctable"
            );
        }
    }

    #[test]
    fn dead_channel_wins_over_everything() {
        let h = SensorHealthStatus {
            channel: 0,
            is_healthy: false,
            health_score: 0.0,
            degradation_type: Some(DegradationType::SensitivityDecay),
            degradation_rate: 0.5,
            estimated_remaining_life_hours: 0.0,
            warning_level: "critical".to_string(),
            metrics: crate::poisoning::SensorMetrics {
                sensitivity: 0.0,
                noise_floor: 0.0,
                recovery_time: 0.0,
                baseline_level: 1.0,
                drift_rate: 0.0,
            },
        };
        let e = ChannelEvidence { drift_rate: 0.0, outlier: false, health: Some(h) };
        assert_eq!(classify(&e, 1.0, &VerdictThresholds::default()), SensorVerdict::Dead);
    }

    #[test]
    fn verdict_ordering_is_severity_ordering() {
        assert!(SensorVerdict::Dead > SensorVerdict::Poisoning);
        assert!(SensorVerdict::Poisoning > SensorVerdict::GradualDrift);
        assert!(SensorVerdict::GradualDrift > SensorVerdict::Normal);
    }

    #[test]
    fn one_poisoned_channel_blocks_correction_array_wide() {
        // Re-anchoring one channel shifts the reference the others are judged
        // against, so a single bad channel must stop correction everywhere.
        let arr = ArrayVerdict {
            per_channel: vec![SensorVerdict::GradualDrift, SensorVerdict::Poisoning, SensorVerdict::Normal],
            worst_channel: Some(1),
        };
        assert!(!arr.correction_permitted());
        assert!(!arr.is_operational());
        assert_eq!(arr.worst(), SensorVerdict::Poisoning);
        assert_eq!(arr.degraded_channels(), vec![0, 1]);
    }

    #[test]
    fn all_healthy_array_is_operational() {
        let arr = classify_array(
            &[ev(0.0, false), ev(1e-4, false), ev(0.0, true)],
            &[1.0, 1.0, 1.0],
            &VerdictThresholds::default(),
        );
        assert!(arr.is_operational());
        assert!(arr.correction_permitted());
        assert_eq!(arr.worst_channel, None);
    }

    #[test]
    fn worst_channel_identifies_the_severe_one() {
        let arr = classify_array(
            &[ev(5e-3, false), ev(0.0, false)],
            &[1.0, 1.0],
            &VerdictThresholds::default(),
        );
        assert_eq!(arr.worst_channel, Some(0));
    }

    #[test]
    fn evidence_from_window_reads_the_right_feature_slot() {
        let series: Vec<f64> = (0..60).map(|i| 1.0 + 0.5 * (-0.06 * i as f64).exp()).collect();
        let window: Vec<Vec<f64>> = series.iter().map(|v| vec![*v]).collect();
        let b = Baseline::from_samples(&window);
        let ev = evidence_from_window(&window, &b, 10.0, &[]).expect("evidence");
        assert_eq!(ev.len(), 1);
        // Slot 2 of each channel's group is drift_rate; it must match the
        // extractor exactly rather than being re-derived.
        let direct = health::extract_window(&window, &b, 10.0).expect("features");
        assert_eq!(ev[0].drift_rate, direct[2]);
    }
}