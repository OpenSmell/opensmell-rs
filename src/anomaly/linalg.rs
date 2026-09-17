/// Small dense linear-algebra helpers used by the Kalman filters.
///
/// The crate deliberately avoids a LAPACK dependency; channel counts are small
/// (≤ ~16) so hand-rolled, Cholesky-based routines are cheap and — unlike the
/// legacy Gauss-Jordan inverters — keep every solve on a positive-definite
/// matrix, which is what the dual-Kalman design relies on.
use crate::{Result, OpenSmellError};

pub fn zero_mat(n: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; n]; n]
}

pub fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = zero_mat(n);
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

pub fn diag(d: &[f64]) -> Vec<Vec<f64>> {
    let n = d.len();
    let mut m = zero_mat(n);
    for i in 0..n {
        m[i][i] = d[i];
    }
    m
}

pub fn mat_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(&x, &y)| x * y).sum())
        .collect()
}

pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let p = b.len();
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0.0;
            for k in 0..p {
                acc += a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = if n == 0 { 0 } else { a[0].len() };
    let mut out = vec![vec![0.0; n]; m];
    for i in 0..n {
        for j in 0..m {
            out[j][i] = a[i][j];
        }
    }
    out
}

pub fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut out = zero_mat(n);
    for i in 0..n {
        for j in 0..n {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

pub fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|&v| v * s).collect())
        .collect()
}

pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

pub fn outer(v: &[f64], w: &[f64]) -> Vec<Vec<f64>> {
    let n = v.len();
    let mut out = zero_mat(n);
    for i in 0..n {
        for j in 0..n {
            out[i][j] = v[i] * w[j];
        }
    }
    out
}

/// Cholesky decomposition: returns lower-triangular L with A = L·Lᵀ.
/// Errors on a non-positive-definite matrix (kept for diagnostics/tests).
pub fn cholesky(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = zero_mat(n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 1e-14 {
                    return Err(OpenSmellError::AnomalyDetection(format!(
                        "Cholesky failed: non-positive diagonal {sum:.3e} at row {i}"
                    )));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

/// Solve L·x = b for lower-triangular L.
fn lower_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut acc = b[i];
        for j in 0..i {
            acc -= l[i][j] * x[j];
        }
        x[i] = acc / l[i][i];
    }
    x
}

/// Solve Lᵀ·x = b for a lower-triangular L.
fn upper_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            acc -= l[j][i] * x[j];
        }
        x[i] = acc / l[i][i];
    }
    x
}

/// Solve the symmetric positive-definite system A·x = b (Cholesky-based).
/// The caller is responsible for A being PD; a tiny ridge keeps borderline
/// matrices from blowing up, matching how the innovation covariance is built.
pub fn solve_pd(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let l = cholesky(a)?;
    let y = lower_solve(&l, b);
    Ok(upper_solve(&l, &y))
}

/// Invert a positive-definite matrix via Cholesky (n small).
pub fn invert_pd(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    let l = cholesky(a)?;
    let mut inv = zero_mat(n);
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let y = lower_solve(&l, &e);
        let xj = upper_solve(&l, &y);
        for (i, &v) in xj.iter().enumerate() {
            inv[i][j] = v;
        }
    }
    Ok(inv)
}

/// Add `ridge` to the main diagonal (keeps solves well-conditioned).
pub fn add_ridge(a: &[Vec<f64>], ridge: f64) -> Vec<Vec<f64>> {
    let mut out = a.to_vec();
    for i in 0..out.len() {
        out[i][i] += ridge;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_pd_matches_invert() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];
        let x = solve_pd(&a, &b).unwrap();
        // Verify A x = b
        let ax = mat_vec(&a, &x);
        for (got, want) in ax.iter().zip(b.iter()) {
            assert!((got - want).abs() < 1e-9, "A·x should equal b");
        }
        let inv = invert_pd(&a).unwrap();
        let identity_check = mat_mul(&a, &inv);
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((identity_check[i][j] - want).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_cholesky_rejects_npd() {
        let a = vec![vec![1.0, 0.0], vec![0.0, -1.0]];
        assert!(cholesky(&a).is_err());
    }
}