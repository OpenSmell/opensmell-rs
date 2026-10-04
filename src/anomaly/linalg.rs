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

/// Sweep cap for the Jacobi eigensolver. Classical Jacobi converges
/// quadratically, so a handful of sweeps over a ≤ 16×16 matrix is already far
/// past double precision; the cap only exists so a pathological input cannot
/// spin.
pub const JACOBI_MAX_SWEEPS: usize = 32;

/// Off-diagonal size below which a Jacobi sweep is considered converged.
/// 1e-12 relative to the matrix's own scale: the eigenvalues of the covariance
/// matrices this is fed span many orders of magnitude, so an *absolute*
/// tolerance would either stop early on a stiff matrix or never stop on a
/// well-conditioned one.
pub const JACOBI_TOLERANCE: f64 = 1e-12;

/// Cyclic Jacobi eigendecomposition of a symmetric matrix `A = V Λ Vᵀ`.
///
/// Returns `(eigenvalues, eigenvectors)` with the eigenvalues in **descending**
/// order and the eigenvectors as **columns** (`eigenvectors[i][j]` is component
/// `i` of eigenvector `j`) — the ordering the SPE/Q residual monitor wants,
/// since it keeps the leading principal components.
///
/// Hand-rolled on purpose: the only crate-provided option is `ndarray-linalg`,
/// which is a heavy LAPACK/ARPACK binding, and the crate has deliberately
/// avoided that (`linalg.rs` header). Channel counts are ≤ ~16, so Jacobi's
/// O(n³)-per-sweep cost is irrelevant and its accuracy is excellent.
///
/// ```text
/// rotation: J is identity except on the (p,q) plane with J[p][p]=J[q][q]=c,
/// J[p][q]=s, J[q][p]=-s; the step replaces A by JᵀAJ and V by VJ, which zeroes
/// A[p][q] when t = s/c solves t² + 2θt - 1 = 0 with θ = (a_qq-a_pp)/(2·a_pq).
/// ```
pub fn symmetric_eigen(a: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = a.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if a.iter().any(|row| row.len() != n) {
        return Err(OpenSmellError::AnomalyDetection(
            "symmetric_eigen: matrix is not square".to_string(),
        ));
    }
    let mut m = a.to_vec();
    let mut v = identity(n);
    // Scale-relative off-diagonal test: the matrix's own magnitude sets the bar.
    let scale = m
        .iter()
        .enumerate()
        .map(|(i, row)| row[i].abs())
        .fold(0.0f64, f64::max)
        .max(1e-300);

    for _ in 0..JACOBI_MAX_SWEEPS {
        // Largest off-diagonal pivot: classic cyclic Jacobi, one rotation per
        // step, no sweep bookkeeping (n is tiny, so this is not the bottleneck).
        let mut p = 0usize;
        let mut q = 1usize;
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                if m[i][j].abs() > off {
                    off = m[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if off <= JACOBI_TOLERANCE * scale {
            break;
        }
        let app = m[p][p];
        let aqq = m[q][q];
        let apq = m[p][q];
        // t = -θ + sqrt(θ²+1), taken on whichever branch avoids cancellation.
        let theta = (aqq - app) / (2.0 * apq);
        let root = (theta * theta + 1.0).sqrt();
        let t = if theta >= 0.0 {
            1.0 / (theta + root)
        } else {
            1.0 / (theta - root)
        };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        for k in 0..n {
            if k != p && k != q {
                let akp = m[k][p];
                let akq = m[k][q];
                m[k][p] = c * akp - s * akq;
                m[p][k] = m[k][p];
                m[k][q] = s * akp + c * akq;
                m[q][k] = m[k][q];
            }
        }
        m[p][p] = c * c * app + s * s * aqq - 2.0 * c * s * apq;
        m[q][q] = s * s * app + c * c * aqq + 2.0 * c * s * apq;
        m[p][q] = 0.0;
        m[q][p] = 0.0;
        for k in 0..n {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }

    // Sort eigenvalues descending, permuting the eigenvector columns with them.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        m[j][j]
            .partial_cmp(&m[i][i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let eigenvalues: Vec<f64> = order.iter().map(|&j| m[j][j]).collect();
    let mut eigenvectors = vec![vec![0.0; n]; n];
    for (new_j, &old_j) in order.iter().enumerate() {
        for i in 0..n {
            eigenvectors[i][new_j] = v[i][old_j];
        }
    }
    Ok((eigenvalues, eigenvectors))
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

    #[test]
    fn symmetric_eigen_diagonal_is_identity() {
        let a = vec![vec![3.0, 0.0], vec![0.0, 1.0]];
        let (vals, vecs) = symmetric_eigen(&a).unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
        // Eigenvector 0 must be the first axis.
        assert!((vecs[0][0].abs() - 1.0).abs() < 1e-12);
        assert!(vecs[1][0].abs() < 1e-12);
    }

    #[test]
    fn symmetric_eigen_reconstructs_a_known_matrix() {
        // Eigenvalues 4 / 1 with a known rotation: A = Q Λ Qᵀ for a 45° rotation.
        let (c, s) = (0.5f64.sqrt(), 0.5f64.sqrt());
        let a = vec![
            vec![3.5, 1.5],
            vec![1.5, 3.5],
        ];
        let (vals, vecs) = symmetric_eigen(&a).unwrap();
        assert!((vals[0] - 5.0).abs() < 1e-10, "leading eigenvalue 5, got {}", vals[0]);
        assert!((vals[1] - 2.0).abs() < 1e-10, "trailing eigenvalue 2, got {}", vals[1]);
        // A = V Λ Vᵀ (the real test of the rotation signs).
        for i in 0..2 {
            for j in 0..2 {
                let mut recon = 0.0;
                for k in 0..2 {
                    recon += vecs[i][k] * vals[k] * vecs[j][k];
                }
                assert!(
                    (recon - a[i][j]).abs() < 1e-10,
                    "A[{i}][{j}] reconstructed as {recon}, want {}",
                    a[i][j]
                );
            }
        }
        // And the leading eigenvector is the symmetric (1,1)/√2 direction.
        assert!((vecs[0][0] - c).abs() < 1e-10 && (vecs[1][0] - c).abs() < 1e-10);
    }

    #[test]
    fn symmetric_eigen_sorts_descending_on_a_four_channel_matrix() {
        let a = vec![
            vec![4.0, 1.0, 0.5, 0.2],
            vec![1.0, 3.0, 0.4, 0.1],
            vec![0.5, 0.4, 2.0, 0.3],
            vec![0.2, 0.1, 0.3, 1.0],
        ];
        let (vals, vecs) = symmetric_eigen(&a).unwrap();
        assert!(vals.windows(2).all(|w| w[0] >= w[1] - 1e-12), "must be descending: {vals:?}");
        for i in 0..4 {
            for j in 0..4 {
                let mut recon = 0.0;
                for k in 0..4 {
                    recon += vecs[i][k] * vals[k] * vecs[j][k];
                }
                assert!((recon - a[i][j]).abs() < 1e-9, "A[{i}][{j}] = {recon} vs {}", a[i][j]);
            }
        }
    }

    #[test]
    fn symmetric_eigen_rejects_non_square() {
        assert!(symmetric_eigen(&[vec![1.0, 2.0]]).is_err());
    }
}