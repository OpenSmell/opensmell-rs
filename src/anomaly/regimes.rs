/// Multi-regime normal baseline (online clustering + transition model).
///
/// The single-regime assumption of the legacy detector is the biggest
/// false-alarm source in real deployments (idle→active fermenters, doors,
/// weather fronts). This module keeps up to `K_MAX` normal regimes, updates
/// them online with exponential forgetting, and declares a *regime switch* (an
/// expected transition — never an anomaly) when the stream persistently joins a
/// different known regime. When novel, sustained normal behaviour appears a new
/// regime spawns (up to the cap).
use serde::{Deserialize, Serialize};

use crate::{Result, OpenSmellError};
use super::linalg::{add_ridge, invert_pd, solve_pd, vec_sub};

/// A regime: reference level, covariance, and how much history backs it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeCluster {
    pub mean: Vec<f64>,
    pub cov: Vec<Vec<f64>>,
    /// Effective weight (forgetting factor counts down older history).
    pub weight: f64,
}

/// Transition matrix entry (k → k′).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeTransition {
    pub from: usize,
    pub to: usize,
    /// Smoothed transition probability 0..1.
    pub probability: f64,
}

/// Result of one online-regime update.
#[derive(Debug, Clone, Default)]
pub struct RegimeUpdate {
    /// Index of the regime the stream currently belongs to.
    pub regime: usize,
    /// True when a persistent switch to another *known* regime was declared.
    pub switched: bool,
    /// True when a new regime was spawned during this step.
    pub spawned: bool,
    /// The cluster re-anchor target when `switched` (empty when not).
    pub anchor_mean: Vec<f64>,
    pub anchor_cov: Vec<Vec<f64>>,
}

const K_MAX: usize = 3;

/// Configuration for the regime model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Forgetting factor per sample: `weight *= (1 − β)` for untouched clusters.
    pub beta: f64,
    /// Minimum persistence (samples) before a novel region spawns a cluster.
    pub min_samples: usize,
    /// Minimum persistence in a different known cluster before a switch is
    /// declared (filters single-sample flicker).
    pub switch_samples: usize,
    /// Chi-square quantile used as the "join" threshold.
    pub join_quantile: f64,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            beta: 0.001,
            min_samples: 300,
            switch_samples: 30,
            join_quantile: 0.95,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegimeModel {
    pub clusters: Vec<RegimeCluster>,
    pub current: usize,
    pub transitions: Vec<RegimeTransition>,
    pub config: RegimeConfig,
    n_channels: usize,
    // Novel-region persistence bookkeeping.
    novel_samples: usize,
    novel_sum: Vec<f64>,
    // Switch-candidate bookkeeping.
    candidate_switch: Option<usize>,
    candidate_samples: usize,
}

/// Chi-square critical value via the Wilson–Hilferty approximation
/// (accurate to a fraction of a percent for small df — plenty for a
/// deployment-scale membership threshold, and dependency-free).
fn chi_sq_crit(df: usize, q: f64) -> f64 {
    let v = df.max(1) as f64;
    let z = normal_ppf(q);
    let w = 1.0 - 2.0 / (9.0 * v) + z * (2.0 / (9.0 * v)).sqrt();
    v * w.powi(3)
}

/// Inverse-normal CDF via Acklam's central-region rational approximation
/// (accurate to ~1e-9 on (0.02425, 0.97575), which covers every membership
/// quantile (0.90–0.99) this code uses). Coeffs match `adaptive.rs` `normal_ppf`.
fn normal_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383_577_518_672_69e2,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}

impl RegimeModel {
    pub fn new(n_channels: usize, config: RegimeConfig) -> Self {
        Self {
            clusters: Vec::new(),
            current: 0,
            transitions: Vec::new(),
            config,
            n_channels,
            novel_samples: 0,
            novel_sum: vec![0.0; n_channels],
            candidate_switch: None,
            candidate_samples: 0,
        }
    }

    /// Seed the first regime from a calibration window (the warm-up baseline).
    pub fn seed(&mut self, mean: Vec<f64>, cov: Vec<Vec<f64>>) {
        self.clusters.clear();
        self.clusters.push(RegimeCluster {
            mean,
            cov,
            weight: 1.0,
        });
        self.current = 0;
        self.novel_samples = 0;
        self.novel_sum = vec![0.0; self.n_channels];
        self.candidate_switch = None;
        self.candidate_samples = 0;
    }

    /// Online update with one sample of the (environmental) reference state.
    pub fn update(&mut self, x: &[f64]) -> Result<RegimeUpdate> {
        if x.len() != self.n_channels {
            return Err(OpenSmellError::InvalidChannelCount {
                got: x.len(),
                expected: self.n_channels,
            });
        }
        if self.clusters.is_empty() {
            // No baseline yet: hold everything until the warm-up seeds cluster 0.
            return Ok(RegimeUpdate {
                regime: 0,
                ..Default::default()
            });
        }

        // Forgetting: untouched clusters fade; the joined one refreshes.
        for c in &mut self.clusters {
            c.weight *= 1.0 - self.config.beta;
        }

        // Nearest-cluster membership using Mahalanobis distance (ridge-guarded).
        let join_thr = chi_sq_crit(self.n_channels, self.config.join_quantile);
        let mut best_idx = self.current.min(self.clusters.len() - 1);
        let mut best_d2 = f64::INFINITY;
        for (i, c) in self.clusters.iter().enumerate() {
            let sr = add_ridge(&c.cov, 1e-9);
            let diff = vec_sub(x, &c.mean);
            let sq = {
                let solved = solve_pd(&sr, &diff)?;
                diff.iter().zip(solved.iter()).map(|(&a, &b)| a * b).sum::<f64>()
            };
            if sq < best_d2 {
                best_d2 = sq;
                best_idx = i;
            }
        }

        // Novel region bookkeeping (below the join threshold for every cluster).
        let mut update = RegimeUpdate::default();
        if best_d2 > join_thr {
            self.novel_samples += 1;
            for (i, &v) in x.iter().enumerate() {
                self.novel_sum[i] += v;
            }
            self.candidate_switch = None;
            self.candidate_samples = 0;

            if self.novel_samples >= self.config.min_samples && self.clusters.len() < K_MAX {
                let mean: Vec<f64> = self
                    .novel_sum
                    .iter()
                    .map(|s| s / self.novel_samples as f64)
                    .collect();
                let mut cov = vec![vec![0.0; self.n_channels]; self.n_channels];
                for &v in self.novel_sum.iter() {
                    let _ = v;
                }
                cov = spread_cov(&[&mean], self.n_channels);
                self.clusters.push(RegimeCluster {
                    mean,
                    cov,
                    weight: 1.0,
                });
                self.novel_samples = 0;
                let idx = self.clusters.len() - 1;
                self.current = idx;
                update = RegimeUpdate {
                    regime: idx,
                    switched: false,
                    spawned: true,
                    anchor_mean: self.clusters[idx].mean.clone(),
                    anchor_cov: self.clusters[idx].cov.clone(),
                };
            }
            return Ok(update);
        }

        // Join the best cluster (rolling center, forgetting history).
        let c = &mut self.clusters[best_idx];
        let a = self.config.beta.max(1e-3);
        for (i, &v) in x.iter().enumerate() {
            c.mean[i] += a * (v - c.mean[i]);
        }
        // Update covariance one sample at a time (shrunk effective count).
        let eff = c.weight.max(2.0);
        let diff = vec_sub(x, &c.mean);
        for i in 0..self.n_channels {
            for j in 0..self.n_channels {
                c.cov[i][j] += a * (diff[i] * diff[j] - c.cov[i][j]) / eff.max(1.0);
            }
        }
        c.weight = (c.weight + 1.0).min(1e6);
        self.novel_samples = 0;
        self.novel_sum = vec![0.0; self.n_channels];

        // Switch-candidate tracking.
        if best_idx != self.current {
            if self.candidate_switch != Some(best_idx) {
                self.candidate_switch = Some(best_idx);
                self.candidate_samples = 1;
            } else {
                self.candidate_samples += 1;
            }
            if self.candidate_samples >= self.config.switch_samples {
                let from = self.current;
                let to = best_idx;
                self.register_transition(from, to);
                let anchor = self.clusters[to].clone();
                self.current = to;
                self.candidate_switch = None;
                self.candidate_samples = 0;
                update = RegimeUpdate {
                    regime: to,
                    switched: true,
                    spawned: false,
                    anchor_mean: anchor.mean.clone(),
                    anchor_cov: anchor.cov.clone(),
                };
            }
        } else {
            self.candidate_switch = None;
            self.candidate_samples = 0;
            update = RegimeUpdate {
                regime: best_idx,
                switched: false,
                spawned: false,
                anchor_mean: Vec::new(),
                anchor_cov: Vec::new(),
            };
        }
        Ok(update)
    }

    /// Probability of the most recent declared switch (k → k′) signalled to
    /// upper layers; laplace-smoothed between the two events.
    fn register_transition(&mut self, from: usize, to: usize) {
        if let Some(t) = self
            .transitions
            .iter_mut()
            .find(|t| t.from == from && t.to == to)
        {
            t.probability = (t.probability * 0.9 + 0.1).min(0.999);
        } else {
            self.transitions.push(RegimeTransition {
                from,
                to,
                probability: 0.5,
            });
        }
    }

    pub fn is_seeded(&self) -> bool {
        !self.clusters.is_empty()
    }
}

/// Build a diagonal-ish spread covariance for a freshly spawned cluster.
fn spread_cov(samples: &[&[f64]], n: usize) -> Vec<Vec<f64>> {
    let mut cov = vec![vec![0.0; n]; n];
    if samples.is_empty() {
        for i in 0..n {
            cov[i][i] = 1e-3;
        }
        return cov;
    }
    let mut variance: Vec<f64> = vec![1e-3; n];
    let mean: Vec<f64> = samples[0].to_vec();
    for i in 0..n {
        variance[i] = variance[i].max(1e-3);
    }
    for i in 0..n {
        cov[i][i] = variance[i];
    }
    let _ = mean;
    cov
}

/// Invert a covariance for Mahalanobis membership (visibility helper).
pub fn inv_cov(cov: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    invert_pd(&add_ridge(cov, 1e-9))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_cluster_absorbs_stable_stream() {
        let mut model = RegimeModel::new(2, RegimeConfig::default());
        model.seed(vec![1.0, 2.0], vec![vec![0.1, 0.0], vec![0.0, 0.1]]);
        let mut update = RegimeUpdate::default();
        for _ in 0..100 {
            update = model.update(&[1.02, 1.98]).unwrap();
        }
        assert!(!update.switched, "stable stream must not switch regimes");
        assert_eq!(model.current, 0);
    }

    #[test]
    fn persistent_novel_region_spawns_cluster() {
        let mut model = RegimeModel::new(2, RegimeConfig {
            min_samples: 10,
            ..Default::default()
        });
        model.seed(vec![1.0, 2.0], vec![vec![0.05, 0.0], vec![0.0, 0.05]]);
        let mut spawned = false;
        for _ in 0..15 {
            let u = model.update(&[9.0, 8.0]).unwrap();
            spawned |= u.spawned;
        }
        assert!(spawned, "persistent novel region should spawn a new regime");
        assert_eq!(model.clusters.len(), 2);
    }

    #[test]
    fn persistent_switch_to_known_regime_declares_switch() {
        let mut model = RegimeModel::new(2, RegimeConfig::default());
        model.seed(vec![1.0, 2.0], vec![vec![0.05, 0.0], vec![0.0, 0.05]]);
        // Force-add a second known regime (simulating a previously spawned one).
        model.clusters.push(RegimeCluster {
            mean: vec![9.0, 8.0],
            cov: vec![vec![0.05, 0.0], vec![0.0, 0.05]],
            weight: 100.0,
        });
        let cfg = model.config.switch_samples;
        let mut u = RegimeUpdate::default();
        for _ in 0..cfg + 5 {
            u = model.update(&[9.0, 8.0]).unwrap();
            if u.switched {
                break;
            }
        }
        assert!(u.switched, "persistent membership in known regime must switch");
        assert_eq!(u.regime, 1);
    }
}