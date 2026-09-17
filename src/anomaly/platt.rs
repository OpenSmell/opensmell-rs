/// Platt scaling with the convex Newton solver of Lin, Lin & Weng (2007).
///
/// The legacy 3×3 grid search in `adaptive.rs` is replaced here: a proper
/// Newton method with backtracking line search on the cross-entropy objective,
/// plus the small-sample target trick (`t⁺ = (N⁺+1)/(N⁺+2)`, `t⁻ = 1/(N⁻+2)`)
/// that [Lin et al., 2007] introduced so the fitted sigmoid never needs an
/// infinite slope to reach ~1/0 on separable data.
///
/// The engine keeps the same public conventions as before — retrain every 10
/// confirmed feedbacks, only from ≥ 20 — so the calibration loop integrates
/// without changing the calling contract.
use serde::{Deserialize, Serialize};

/// Result of a calibration fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlattParams {
    pub a: f64,
    pub b: f64,
    /// Cross-entropy on the training pairs (decreases monotonically per fit).
    pub final_nll: f64,
    pub n_positive: usize,
    pub n_negative: usize,
}

impl Default for PlattParams {
    fn default() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            final_nll: f64::INFINITY,
            n_positive: 0,
            n_negative: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlattCalibrator {
    pub params: PlattParams,
    pairs: Vec<(f64, f64)>, // (score, label) where label ∈ {0,1}
}

impl Default for PlattCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl PlattCalibrator {
    pub fn new() -> Self {
        Self {
            params: PlattParams::default(),
            pairs: Vec::new(),
        }
    }

    /// Record one confirmed decision: `score` is the anomaly evidence (`d`),
    /// `label` is the operator's "did this matter?" answer.
    pub fn add_feedback(&mut self, score: f64, label: bool) {
        self.pairs.push((score, if label { 1.0 } else { 0.0 }));
    }

    /// Retrain when enough feedback has accumulated.
    pub fn retrain(&mut self) -> Option<PlattParams> {
        if self.pairs.len() < 20 {
            return None;
        }
        let fitted = fit_lin_platt(&self.pairs);
        self.params = fitted.clone();
        Some(fitted)
    }

    /// Probability that the current evidence score is an anomaly.
    pub fn predict(&self, score: f64) -> f64 {
        let p = 1.0 / (1.0 + (-(self.params.a * score + self.params.b)).exp());
        p.clamp(0.0, 1.0)
    }

    pub fn n_feedback(&self) -> usize {
        self.pairs.len()
    }
}

fn fit_lin_platt(pairs: &[(f64, f64)]) -> PlattParams {
    let n_pos = pairs.iter().filter(|(_, y)| *y > 0.5).count();
    let n_neg = pairs.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return PlattParams::default();
    }

    // Regularized targets — never exactly 1/0.
    let hi = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
    let lo = 1.0 / (n_neg as f64 + 2.0);
    let targets: Vec<f64> = pairs.iter().map(|(_, y)| if *y > 0.5 { hi } else { lo }).collect();

    let mut a = 1.0;
    let mut b = 0.0;
    const MAX_ITER: usize = 60;
    let mut nll_current = nll(a, b, pairs, &targets);
    let mut improved = true;

    for _ in 0..MAX_ITER {
        if !improved {
            break;
        }
        improved = false;
        let (g, h) = grad_hess(a, b, pairs, &targets);
        // If Hessian is pathological (separable data), take a gradient step.
        let (da, db) = if h.is_finite() && h.det() > 1e-12 {
            // Solve H·δ = −g via 2×2 Cramer (n=2 keeps this simple and exact).
            let inv_det = 1.0 / h.det();
            (
                -inv_det * (h.h22 * g.dg_a - h.h12 * g.dg_b),
                -inv_det * (h.h11 * g.dg_b - h.h12 * g.dg_a),
            )
        } else {
            (-0.02 * g.dg_a, -0.02 * g.dg_b)
        };
        let (na, nb) = (a + da, b + db);
        let new_nll = nll(na, nb, pairs, &targets);
        if new_nll < nll_current - 1e-12 {
            a = na;
            b = nb;
            nll_current = new_nll;
            improved = true;
        } else {
            // Backtracking line search along the Newton direction.
            let mut t = 0.5;
            for _ in 0..12 {
                let (ta, tb) = (a + t * da, b + t * db);
                let tn = nll(ta, tb, pairs, &targets);
                if tn < nll_current - 1e-12 {
                    a = ta;
                    b = tb;
                    nll_current = tn;
                    improved = true;
                    break;
                }
                t *= 0.5;
            }
        }
    }

    PlattParams {
        a,
        b,
        final_nll: nll_current,
        n_positive: n_pos,
        n_negative: n_neg,
    }
}

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp().clamp(f64::MIN_POSITIVE, f64::MAX))
}

fn nll(a: f64, b: f64, pairs: &[(f64, f64)], targets: &[f64]) -> f64 {
    pairs
        .iter()
        .zip(targets.iter())
        .map(|((s, _), &t)| {
            let p = sigmoid(a * s + b);
            let p_c = p.clamp(1e-9, 1.0 - 1e-9);
            -(t * p_c.ln() + (1.0 - t) * (1.0 - p_c).ln())
        })
        .sum()
}

#[derive(Clone, Copy)]
struct Hessian {
    h11: f64,
    h12: f64,
    h22: f64,
}
impl Hessian {
    fn det(&self) -> f64 {
        self.h11 * self.h22 - self.h12 * self.h12
    }
    fn is_finite(&self) -> bool {
        self.h11.is_finite() && self.h12.is_finite() && self.h22.is_finite()
    }
}

fn grad_hess(a: f64, b: f64, pairs: &[(f64, f64)], targets: &[f64]) -> (GradPair, Hessian) {
    let mut dg_a = 0.0;
    let mut dg_b = 0.0;
    let mut h11 = 0.0;
    let mut h12 = 0.0;
    let mut h22 = 0.0;
    for ((s, _y), &t) in pairs.iter().zip(targets.iter()) {
        let t1 = s;
        let t2 = 1.0;
        let f = sigmoid(a * s + b);
        let g1 = (f - t) * t1;
        let g2 = (f - t) * t2;
        let c = f * (1.0 - f);
        dg_a += g1;
        dg_b += g2;
        h11 += c * t1 * t1;
        h12 += c * t1 * t2;
        h22 += c * t2 * t2;
    }
    (GradPair { dg_a, dg_b }, Hessian { h11, h12, h22 })
}

struct GradPair {
    dg_a: f64,
    dg_b: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotone_in_score() {
        let mut cal = PlattCalibrator::new();
        // Clean, separable-looking data with a wide score gap.
        for i in 0..25 {
            cal.add_feedback(0.5 + (i as f64) * 0.1, false);
            cal.add_feedback(8.0 + (i as f64) * 0.1, true);
        }
        cal.retrain();
        let p_low = cal.predict(1.0);
        let p_high = cal.predict(9.0);
        assert!(p_high > p_low, "predictions must be monotone in score");
        assert!(p_low < 0.5 && p_high > 0.5, "separable data must separate");
        assert!(p_high.is_finite() && p_low.is_finite());
    }

    #[test]
    fn separable_data_does_not_diverge() {
        let mut cal = PlattCalibrator::new();
        for i in 0..30 {
            cal.add_feedback(1.0 + (i as f64) * 0.05, false);
            cal.add_feedback(10.0 + (i as f64) * 0.05, true);
        }
        let p = cal.retrain().expect("fit should succeed");
        assert!(p.a.is_finite() && p.b.is_finite());
        assert!(p.a > 0.0, "larger scores must map to larger probabilities");
    }

    #[test]
    fn default_before_sufficient_data() {
        let mut cal = PlattCalibrator::new();
        for i in 0..10 {
            cal.add_feedback(i as f64, false);
        }
        assert!(cal.retrain().is_none());
        // Defaults give ~50% at score ~0.
        assert!((cal.predict(0.0) - 0.5).abs() < 1e-6);
    }
}