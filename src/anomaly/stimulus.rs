/// Stimulus-based poisoning detection: gain measured *per reference stimulus*.
///
/// The legacy `poisoning.rs` infers degradation from ambient window statistics;
/// this module is the physical confirmation layer the design doc requires: a
/// controlled heater-pulse / reference exposure measures the transducer's gain
/// directly, and the retention ratio `ρ = ĝ / g₀` is reconciled against the
/// dual-Kalman parameter filter's relative gain. Poisoning that is real shows
/// up in *both* independently.
use serde::{Deserialize, Serialize};

/// A health finding produced by stimulus reconciliation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthFinding {
    pub channel: usize,
    /// `PoisonConfirmed` (both measurements agree on decay) or
    /// `GainDisagreement` (the two sources diverged — re-anchor needed).
    pub kind: HealthFindingKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthFindingKind {
    PoisonConfirmed,
    GainDisagreement,
}

/// A single recorded stimulus measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StimulusMeasurement {
    pub sample: u64,
    /// Measured retained gain per channel (fraction of burn-in reference).
    pub rho: Vec<f64>,
    /// Reference-stimulus sensor units this cycle.
    pub stimulus_units: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct StimulusGainTracker {
    /// Burn-in reference gain `g₀` per channel (from calibration / array burn-in).
    pub g0: Vec<f64>,
    /// Most recent retained-gain estimates `ρ`.
    pub rho: Vec<f64>,
    /// Per-channel smoothed relative gain (feed from the parameter filter).
    pub filter_relative_gain: Vec<f64>,
    n_channels: usize,
    history: Vec<StimulusMeasurement>,
    config: StimulusConfig,
}

/// Configuration for stimulus tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StimulusConfig {
    /// Schedule period for automatic reference stimuli, in seconds.
    pub period_s: u64,
    /// Relative gain (or ρ) below which a channel is confirmed poisoned.
    pub poison_relative_gain: f64,
    /// Relative disagreement between filter and stimulus that means re-anchoring.
    pub gain_disagreement: f64,
}

impl Default for StimulusConfig {
    fn default() -> Self {
        Self {
            period_s: 3600,
            poison_relative_gain: 0.5,
            gain_disagreement: 0.20,
        }
    }
}

impl StimulusGainTracker {
    pub fn new(n_channels: usize, g0: Vec<f64>, config: StimulusConfig) -> Self {
        let rho = if g0.is_empty() {
            vec![1.0; n_channels]
        } else {
            g0.iter()
                .map(|&g| if g > 0.0 { 1.0 } else { 1.0 })
                .collect()
        };
        Self {
            g0: if g0.is_empty() { vec![1.0; n_channels] } else { g0 },
            rho,
            filter_relative_gain: vec![1.0; n_channels],
            n_channels,
            history: Vec::new(),
            config,
        }
    }

    /// Set the parameter filter's current relative gain (per channel) for
    /// reconciliation. Feed this every filter step.
    pub fn set_filter_relative_gain(&mut self, relative: Vec<f64>) {
        if relative.len() != self.n_channels {
            return;
        }
        self.filter_relative_gain = relative;
    }

    /// Record the response of one reference stimulus.
    ///
    /// `channel_response` is the measured per-channel response to this
    /// stimulus; `expected_ref` is the burn-in response per stimulus unit.
    /// Returns the fresh retention ratio `ρ` and any health findings.
    pub fn record_stimulus(
        &mut self,
        sample: u64,
        channel_response: &[f64],
        expected_ref: &[f64],
    ) -> Result<Vec<HealthFinding>, crate::OpenSmellError> {
        if channel_response.len() != self.n_channels || expected_ref.len() != self.n_channels {
            return Err(crate::OpenSmellError::InvalidChannelCount {
                got: channel_response.len(),
                expected: self.n_channels,
            });
        }
        let stimulus_units: Vec<f64> = channel_response.to_vec();
        let mut rho = vec![1.0; self.n_channels];
        let mut findings = Vec::new();
        for ch in 0..self.n_channels {
            let ref_units = expected_ref[ch];
            let est_gain = if ref_units.abs() > 1e-12 {
                channel_response[ch] / ref_units
            } else {
                1.0
            };
            rho[ch] = est_gain / self.g0[ch].abs().max(1e-12);
            let r_filter = self.filter_relative_gain[ch];

            // Reconciliation: physical stimulus vs statistical filter.
            if (rho[ch] - r_filter).abs() > self.config.gain_disagreement {
                findings.push(HealthFinding {
                    channel: ch,
                    kind: HealthFindingKind::GainDisagreement,
                    message: format!(
                        "channel {}: stimulus gain {:.2} vs filter gain {:.2} diverge — re-anchor filters",
                        ch, rho[ch], r_filter
                    ),
                });
            } else if rho[ch] < self.config.poison_relative_gain
                && r_filter < self.config.poison_relative_gain
            {
                findings.push(HealthFinding {
                    channel: ch,
                    kind: HealthFindingKind::PoisonConfirmed,
                    message: format!(
                        "channel {}: poisoned (stimulus ρ={:.2}, filter gain {:.2}) — needs service",
                        ch, rho[ch], r_filter
                    ),
                });
            }
        }
        self.rho = rho.clone();
        self.history.push(StimulusMeasurement {
            sample,
            rho: rho.clone(),
            stimulus_units,
        });
        if self.history.len() > 128 {
            self.history.remove(0);
        }
        Ok(findings)
    }

    pub fn recent_history(&self) -> &[StimulusMeasurement] {
        &self.history
    }

    /// True only when the agreement path says a channel is genuinely dead.
    pub fn is_poisoned(&self, channel: usize) -> bool {
        if channel >= self.n_channels {
            return false;
        }
        self.rho[channel] < self.config.poison_relative_gain
            && self.filter_relative_gain[channel] < self.config.poison_relative_gain
    }

    pub fn config(&self) -> &StimulusConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_channel_no_findings() {
        let mut tracker = StimulusGainTracker::new(2, vec![1.0, 1.0], StimulusConfig::default());
        tracker.set_filter_relative_gain(vec![1.0, 1.0]);
        let findings = tracker
            .record_stimulus(10, &[1.0, 1.0], &[1.0, 1.0])
            .unwrap();
        assert!(findings.is_empty(), "healthy stimulus must not flag");
        assert!((tracker.rho[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn poison_confirmed_only_when_both_agree() {
        let mut tracker = StimulusGainTracker::new(1, vec![1.0], StimulusConfig::default());
        // Filter already saw decay; stimulus halves too.
        tracker.set_filter_relative_gain(vec![0.4]);
        let findings = tracker
            .record_stimulus(10, &[0.4], &[1.0])
            .unwrap();
        assert!(
            findings.iter().any(|f| f.kind == HealthFindingKind::PoisonConfirmed),
            "filter + stimulus agreement on decay must confirm poisoning"
        );
        assert!(tracker.is_poisoned(0));
    }

    #[test]
    fn disagreement_is_not_poison_confirmation() {
        let mut tracker = StimulusGainTracker::new(1, vec![1.0], StimulusConfig::default());
        tracker.set_filter_relative_gain(vec![1.0]);
        // Stimulus says 0.4 (decayed) but the filter still says 1.0.
        let findings = tracker.record_stimulus(10, &[0.4], &[1.0]).unwrap();
        assert!(
            findings.iter().any(|f| f.kind == HealthFindingKind::GainDisagreement),
            "single-source signal must be disagreement, not confirmation"
        );
        assert!(!tracker.is_poisoned(0));
    }
}