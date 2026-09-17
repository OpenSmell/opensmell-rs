/// Kalman filter implementations behind one struct.
///
/// Provided in one place:
/// - `StateFilterKind::Kalman` — the linear (KF) update; also the correct EKF
///   result whenever the measurement model is linear in the state (gain/offset
///   enter linearly), since EKF ≡ KF for linear models.
/// - `StateFilterKind::Unscented` — the sigma-point (UKF) update used by the
///   state filter by default: no Jacobians, and the innovation covariance is
///   positive-definite by construction (sigma-point spread + measurement noise),
///   which is exactly the property the anomaly decision relies on.
///
/// See `anomaly-engine-design.md` §4 for the derivation.
use serde::{Deserialize, Serialize};

use crate::{Result, OpenSmellError};

use super::linalg::{
    add_ridge, cholesky, invert_pd, mat_add, mat_mul, mat_scale, mat_vec, transpose,
    vec_sub,
};

/// Output of a filter update: everything the anomaly engine needs from one step.
#[derive(Debug, Clone)]
pub struct UpdateOutcome {
    /// Innovation `z − ŷ` (the measurement residual driving the anomaly verdict).
    pub innovation: Vec<f64>,
    /// Innovation covariance `S = P_yy + R` (positive-definite by construction).
    pub innovation_cov: Vec<Vec<f64>>,
    /// Predicted measurement `ŷ`.
    pub y_hat: Vec<f64>,
    /// Kalman gain (kept for inspection/debugging).
    pub gain: Vec<Vec<f64>>,
    /// Per-element standardized residual `r_i / sqrt(S_ii)`.
    pub z_scores: Vec<f64>,
    /// Multivariate deviation `sqrt(rᵀ S⁻¹ r)` — the honest magnitude answer.
    pub mahalanobis: f64,
}

/// Standardized residuals and total Mahalanobis deviation from an innovation.
/// A tiny ridge keeps borderline innovation covariances well-conditioned; the
/// design no longer needs the legacy Gauss-Jordan blow-up guard.
pub fn innovation_stats(r: &[f64], s: &[Vec<f64>], ridge: f64) -> Result<(Vec<f64>, f64)> {
    let sr = add_ridge(s, ridge);
    let solved = solve_ok(&sr, r)?;
    let d2: f64 = r.iter().zip(solved.iter()).map(|(&a, &b)| a * b).sum();
    let zs: Vec<f64> = r
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let s_ii = sr[i][i].max(1e-12).sqrt();
            v / s_ii
        })
        .collect();
    Ok((zs, d2.max(0.0).sqrt()))
}

fn solve_ok(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    super::linalg::solve_pd(a, b)
}

/// UKF sigma-point tuning constants.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UkfParams {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
}

impl Default for UkfParams {
    fn default() -> Self {
        Self {
            // α = 0.5 keeps the sigma weights O(1): with α = 1e-3 and small n,
            // (n+λ) ≈ 2e-6 forces wm0 ≈ −1e6 and y_hat is rebuilt by
            // catastrophic cancellation (errors ~1.0, enough to false-alarm).
            // The measurement model is linear (g·x + o), so UKF ≡ KF for any α;
            // a moderate α is exact AND numerically stable.
            alpha: 0.5,
            beta: 2.0,
            kappa: 0.0,
        }
    }
}

/// Which update rule the state filter uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateFilterKind {
    #[serde(rename = "kalman")]
    Kalman,
    #[serde(rename = "unscented")]
    Unscented,
}

/// A single filter object; `kind` selects the update rule.
#[derive(Debug, Clone)]
pub struct KalmanFilterImpl {
    pub x: Vec<f64>,
    pub p: Vec<Vec<f64>>,
    pub kind: StateFilterKind,
    pub ukf: UkfParams,
    pub last: Option<UpdateOutcome>,
}

impl KalmanFilterImpl {
    pub fn new(x: Vec<f64>, p: Vec<Vec<f64>>, kind: StateFilterKind) -> Self {
        Self {
            x,
            p,
            kind,
            ukf: UkfParams::default(),
            last: None,
        }
    }

    /// Time update: `x ← F x`, `P ← F P Fᵀ + Q`.
    pub fn predict(&mut self, f: &[Vec<f64>], q: &[Vec<f64>]) -> Result<()> {
        if f.is_empty() {
            return Err(OpenSmellError::AnomalyDetection(
                "predict: empty transition matrix".into(),
            ));
        }
        let n = self.x.len();
        if q.len() != n || q.iter().any(|row| row.len() != n) {
            return Err(OpenSmellError::AnomalyDetection(format!(
                "predict: process noise not {n}×{n}"
            )));
        }
        self.x = mat_vec(f, &self.x);
        let fpf = mat_mul(&mat_mul(f, &self.p), &transpose(f));
        self.p = mat_add(&fpf, q);
        Ok(())
    }

    /// Measurement update for a **linear** observation `z ≈ H x`:
    /// `H` doubles as the sigma-point map when `kind == Unscented`.
    pub fn update_linear(
        &mut self,
        z: &[f64],
        h: &[Vec<f64>],
        r_mat: &[Vec<f64>],
        ridge: f64,
    ) -> Result<UpdateOutcome> {
        let n = self.x.len();
        if h.is_empty() || h[0].len() != n {
            return Err(OpenSmellError::AnomalyDetection(format!(
                "linear update: observation matrix {}×{} not shaped for state dim {n}",
                h.len(),
                h.first().map_or(0, |r| r.len())
            )));
        }
        match self.kind {
            StateFilterKind::Kalman => self.linear_update(z, h, r_mat, ridge),
            StateFilterKind::Unscented => {
                let map = |xp: &[f64]| mat_vec(h, xp);
                self.sigma_update(z, &map, r_mat, ridge)
            }
        }
    }

    /// Measurement update for a **nonlinear** observation `z ≈ h(x)` (UKF).
    pub fn update_sigma(
        &mut self,
        z: &[f64],
        h: &dyn Fn(&[f64]) -> Vec<f64>,
        r_mat: &[Vec<f64>],
        ridge: f64,
    ) -> Result<UpdateOutcome> {
        self.sigma_update(z, h, r_mat, ridge)
    }

    fn linear_update(
        &mut self,
        z: &[f64],
        h: &[Vec<f64>],
        r_mat: &[Vec<f64>],
        ridge: f64,
    ) -> Result<UpdateOutcome> {
        let n = self.x.len();
        let y_hat = mat_vec(h, &self.x);
        let innovation = vec_sub(z, &y_hat);
        let hp = mat_mul(h, &self.p);
        let s = add_ridge(&mat_add(&mat_mul(&hp, &transpose(h)), r_mat), ridge);
        let s_inv = invert_pd(&s)?;
        let pht = mat_mul(&self.p, &transpose(h));
        let gain = mat_mul(&pht, &s_inv);
        let correction = mat_vec(&gain, &innovation);
        for i in 0..n {
            self.x[i] += correction[i];
        }
        let p_new = mat_sub_psd(&self.p, &mat_mul(&mat_mul(&gain, &s), &transpose(&gain)));
        self.p = symmetrize(&p_new);
        let (z_scores, mahalanobis) = innovation_stats(&innovation, &s, ridge)?;
        let outcome = UpdateOutcome {
            innovation,
            innovation_cov: s,
            y_hat,
            gain,
            z_scores,
            mahalanobis,
        };
        self.last = Some(outcome.clone());
        Ok(outcome)
    }

    fn sigma_update(
        &mut self,
        z: &[f64],
        h: &dyn Fn(&[f64]) -> Vec<f64>,
        r_mat: &[Vec<f64>],
        ridge: f64,
    ) -> Result<UpdateOutcome> {
        let n = self.x.len();
        if n == 0 {
            return Err(OpenSmellError::AnomalyDetection("empty state".into()));
        }
        let lambda = self.ukf.alpha * self.ukf.alpha * (n as f64 + self.ukf.kappa) - n as f64;
        let n_l = n as f64 + lambda;
        let wm0 = lambda / n_l;
        let wc0 = lambda / n_l + (1.0 - self.ukf.alpha * self.ukf.alpha + self.ukf.beta);
        let wmi = 1.0 / (2.0 * n_l);

        // sqrt((n+λ) P): symmetrize (numerical drift keeps P symmetric-PD), Cholesky.
        let scaled = mat_scale(&self.p, n_l);
        let sym = symmetrize(&scaled);
        let root = cholesky(&add_ridge(&sym, ridge))?;

        // Sigma points.
        let mut x_pts: Vec<Vec<f64>> = Vec::with_capacity(2 * n + 1);
        x_pts.push(self.x.clone());
        for j in 0..n {
            let plus: Vec<f64> = (0..n).map(|i| self.x[i] + root[i][j]).collect();
            let minus: Vec<f64> = (0..n).map(|i| self.x[i] - root[i][j]).collect();
            x_pts.push(plus);
            x_pts.push(minus);
        }

        // Transform, then reconstruct the statistical moments.
        let z_pts: Vec<Vec<f64>> = x_pts.iter().map(|xp| h(xp)).collect();
        let m = z_pts[0].len();
        let mut y_hat = vec![0.0; m];
        for (i, zp) in z_pts.iter().enumerate() {
            let w = if i == 0 { wm0 } else { wmi };
            for (j, &v) in zp.iter().enumerate() {
                y_hat[j] += w * v;
            }
        }
        let mut p_yy = vec![vec![0.0; m]; m];
        for (i, zp) in z_pts.iter().enumerate() {
            let w = if i == 0 { wc0 } else { wmi };
            let d = vec_sub(zp, &y_hat);
            for (a, &da) in d.iter().enumerate() {
                for (b, &db) in d.iter().enumerate() {
                    p_yy[a][b] += w * da * db;
                }
            }
        }
        let s_full = mat_add(&p_yy, r_mat);
        let s = add_ridge(&s_full, ridge);

        let mut p_xy = vec![vec![0.0; m]; n];
        for i in 0..n {
            for k in 0..=2 * n {
                let w = if k == 0 { wc0 } else { wmi };
                for j in 0..m {
                    p_xy[i][j] += w * (x_pts[k][i] - self.x[i]) * (z_pts[k][j] - y_hat[j]);
                }
            }
        }

        // K = P_xy S⁻¹ ; x += K r ; P = P − K S Kᵀ
        let s_inv = invert_pd(&s)?;
        let gain = mat_mul(&p_xy, &s_inv);
        let innovation = vec_sub(z, &y_hat);
        let correction = mat_vec(&gain, &innovation);
        for i in 0..n {
            self.x[i] += correction[i];
        }
        let p_new = mat_sub_psd(&self.p, &mat_mul(&mat_mul(&gain, &s), &transpose(&gain)));
        self.p = symmetrize(&p_new);

        let (z_scores, mahalanobis) = innovation_stats(&innovation, &s, ridge)?;
        let outcome = UpdateOutcome {
            innovation,
            innovation_cov: s,
            y_hat,
            gain,
            z_scores,
            mahalanobis,
        };
        self.last = Some(outcome.clone());
        Ok(outcome)
    }
}

fn mat_sub_psd(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut out = a.to_vec();
    for i in 0..n {
        for j in 0..n {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    out
}

fn symmetrize(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut out = a.to_vec();
    for i in 0..n {
        for j in 0..n {
            out[i][j] = (a[i][j] + a[j][i]) / 2.0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye(n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            m[i][i] = 1.0;
        }
        m
    }

    #[test]
    fn kalman_tracks_constant() {
        // 1-D constant system F=1, H=1, no noise: filter converges to the mean.
        let mut kf = KalmanFilterImpl::new(vec![2.0], vec![vec![1.0]], StateFilterKind::Kalman);
        let r = vec![vec![0.5]];
        let mut last = 0.0;
        for z in [5.0, 5.0, 5.0, 5.0, 5.0] {
            kf.predict(&eye(1), &vec![vec![1e-3]]).unwrap();
            let out = kf.update_linear(&[z], &eye(1), &r, 0.0).unwrap();
            last = out.innovation[0].abs();
        }
        assert!(kf.x[0] - 5.0 < 0.01, "filter should converge to 5");
        assert!(last.is_finite());
    }

    #[test]
    fn ukf_equals_kalman_on_linear_model() {
        // Same linear problem through both paths must agree on the posterior.
        let mut kf = KalmanFilterImpl::new(vec![1.0, 1.0], vec![vec![1.0, 0.0], vec![0.0, 1.0]], StateFilterKind::Kalman);
        let mut ukf = KalmanFilterImpl::new(vec![1.0, 1.0], vec![vec![1.0, 0.0], vec![0.0, 1.0]], StateFilterKind::Unscented);
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![0.1, 0.0], vec![0.0, 0.1]];
        let z = vec![2.0, 3.0];
        kf.predict(&eye(2), &vec![vec![1e-3, 0.0], vec![0.0, 1e-3]]).unwrap();
        ukf.predict(&eye(2), &vec![vec![1e-3, 0.0], vec![0.0, 1e-3]]).unwrap();
        let ok = kf.update_linear(&z, &h, &r, 1e-9).unwrap();
        let ou = ukf.update_linear(&z, &h, &r, 1e-9).unwrap();
        for i in 0..2 {
            assert!((kf.x[i] - ukf.x[i]).abs() < 1e-2, "UKF≈KF posterior on linear model");
        }
        assert!((ok.mahalanobis - ou.mahalanobis).abs() < 1e-2);
    }

    #[test]
    fn innovation_stats_rejects_singular() {
        // A rank-deficient S must still yield a finite (ridge-protected) answer.
        let s = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let (z, d) = innovation_stats(&[0.5, -0.5], &s, 1e-6).unwrap();
        assert!(z.iter().all(|v| v.is_finite()));
        assert!(d.is_finite() && d >= 0.0);
    }
}