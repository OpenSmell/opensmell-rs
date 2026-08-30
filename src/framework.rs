//! Framework window features — pure-Rust port of `opensmell/opensmell/mox/features.py`
//! `extract_all_framework_features` plus the deterministic sorted ordering used by
//! `Osmograph/viz/paradigm_features.py::compute_framework_features`
//! (`keys = sorted(feat_dict.keys())`).
//!
//! For a 6-channel device this is a 187-dimensional vector:
//!   28 features per channel (abs×4, saturation×1, da×6, decay×6, health×4, hw×3,
//!   temp×4) → 168, plus 4 global metrics and 15 cross-channel selectivity ratios.
//!
//! The multi-exponential decay constants require a non-linear least-squares fit
//! (scipy `curve_fit` → MINPACK LM). We port an in-house Levenberg–Marquardt fitter.
//! For well-conditioned recovery curves the fitted parameters match scipy closely;
//! for collinear (near-identical) exponentials the solution is not unique and may
//! differ slightly (a documented limitation shared with the reference itself).

// ------------------------------------------------------------ helpers

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

fn population_std(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        let mu = mean(x);
        (x.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
    }
}

/// Median of a slice (numpy `np.median`). Does not sort in place; returns 0.0 for empty.
fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f64> = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// Composite trapezoid integral with unit spacing (numpy `np.trapezoid(y, dx=1)`).
fn trapezoid(y: &[f64]) -> f64 {
    match y.len() {
        0 | 1 => 0.0,
        2 => (y[0] + y[1]) / 2.0,
        n => (y[0] + y[n - 1]) / 2.0 + y[1..n - 1].iter().sum::<f64>(),
    }
}

/// Linear least-squares detrend (scipy `signal.detrend`, type='linear').
fn detrend(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mean_t = mean(&t);
    let mean_y = mean(y);
    let denom = t.iter().map(|&ti| (ti - mean_t).powi(2)).sum::<f64>();
    let numer: f64 = t
        .iter()
        .zip(y.iter())
        .map(|(&ti, &yi)| (ti - mean_t) * (yi - mean_y))
        .sum();
    let slope = if denom > 0.0 { numer / denom } else { 0.0 };
    let intercept = mean_y - slope * mean_t;
    y.iter()
        .zip(t.iter())
        .map(|(&yi, &ti)| yi - (slope * ti + intercept))
        .collect()
}

// ------------------------------------------------------------- FFT (Bluestein)

/// Radix-2 FFT (in place), iterative Cooley–Tukey. `n` must be a power of two.
/// `invert=true` performs the inverse (unnormalized) transform.
fn fft_radix2(re: &mut [f64], im: &mut [f64], invert: bool) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = 2.0 * std::f64::consts::PI / len as f64
            * if invert { 1.0 } else { -1.0 };
        let (wre, wim) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let mut cur_re = 1.0f64;
            let mut cur_im = 0.0f64;
            for k in 0..(len / 2) {
                let u_re = re[i + k];
                let u_im = im[i + k];
                let v_re = re[i + k + len / 2] * cur_re - im[i + k + len / 2] * cur_im;
                let v_im = re[i + k + len / 2] * cur_im + im[i + k + len / 2] * cur_re;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + len / 2] = u_re - v_re;
                im[i + k + len / 2] = u_im - v_im;
                let nc_re = cur_re * wre - cur_im * wim;
                let nc_im = cur_re * wim + cur_im * wre;
                cur_re = nc_re;
                cur_im = nc_im;
            }
            i += len;
        }
        len <<= 1;
    }
    if invert {
        for v in re.iter_mut() {
            *v /= n as f64;
        }
        for v in im.iter_mut() {
            *v /= n as f64;
        }
    }
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Bluestein FFT for arbitrary length N. Returns real/imag of the DFT bins.
fn fft_bluestein(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x_re.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n & (n - 1) == 0 {
        let mut re = x_re.to_vec();
        let mut im = x_im.to_vec();
        fft_radix2(&mut re, &mut im, false);
        return (re, im);
    }
    let m = next_pow2(2 * n - 1);
    let pi = std::f64::consts::PI;
    // w[k] = exp(-i pi k^2 / N)
    let w: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let ang = -pi * (k as f64 * k as f64) / n as f64;
            (ang.cos(), ang.sin())
        })
        .collect();
    let mut a_re = vec![0.0; m];
    let mut a_im = vec![0.0; m];
    for k in 0..n {
        a_re[k] = x_re[k] * w[k].0 - x_im[k] * w[k].1;
        a_im[k] = x_re[k] * w[k].1 + x_im[k] * w[k].0;
    }
    let mut b_re = vec![0.0; m];
    let mut b_im = vec![0.0; m];
    // b[k] = conj(w[k]); b[m-k] = conj(w[k])
    for k in 0..n {
        b_re[k] = w[k].0; // conj(w): real part same
        b_im[k] = -w[k].1;
        if k > 0 {
            b_re[m - k] = w[k].0;
            b_im[m - k] = -w[k].1;
        }
    }
    fft_radix2(&mut a_re, &mut a_im, false);
    fft_radix2(&mut b_re, &mut b_im, false);
    for k in 0..m {
        let ar = a_re[k];
        let ai = a_im[k];
        let br = b_re[k];
        let bi = b_im[k];
        a_re[k] = ar * br - ai * bi;
        a_im[k] = ar * bi + ai * br;
    }
    fft_radix2(&mut a_re, &mut a_im, true);
    let mut out_re = vec![0.0; n];
    let mut out_im = vec![0.0; n];
    for k in 0..n {
        out_re[k] = a_re[k] * w[k].0 - a_im[k] * w[k].1;
        out_im[k] = a_re[k] * w[k].1 + a_im[k] * w[k].0;
    }
    (out_re, out_im)
}

/// One-sided (positive-frequency) power spectral density matching scipy
/// `signal.periodogram(x, fs)` with the default boxcar window. Returns
/// `(freqs, psd)` of length `n // 2 + 1`.
fn periodogram(x: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let (re, im) = fft_bluestein(x, &vec![0.0; n]);
    let num_freqs = n / 2 + 1;
    let mut freqs = Vec::with_capacity(num_freqs);
    let mut psd = Vec::with_capacity(num_freqs);
    for k in 0..num_freqs {
        let f = (k as f64) * fs / n as f64;
        let mag_sq = re[k] * re[k] + im[k] * im[k];
        // scaling: 2/(N*fs) for interior, 1/(N*fs) for DC and Nyquist
        let scale = if k == 0 || (n % 2 == 0 && k == n / 2) {
            1.0 / (n as f64 * fs)
        } else {
            2.0 / (n as f64 * fs)
        };
        freqs.push(f);
        psd.push(mag_sq * scale);
    }
    (freqs, psd)
}

/// numpy `np.convolve(a, v, mode="valid")`.
fn convolve_valid(a: &[f64], v: &[f64]) -> Vec<f64> {
    let (n, m) = (a.len(), v.len());
    let out_len = if n >= m { n - m + 1 } else { 0 };
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let mut acc = 0.0;
        for j in 0..m {
            acc += a[i + j] * v[j];
        }
        out.push(acc);
    }
    out
}

fn argmax_abs(x: &[f64]) -> usize {
    let mut best = 0usize;
    for i in 1..x.len() {
        if x[i].abs() > x[best].abs() {
            best = i;
        }
    }
    best
}

/// Reference `_r0_from_contract`: explicit R0, else median of first `r0_samples`
/// finite samples; guards fall back to mean of positives, then 1.0.
fn r0_from_contract(series: &[f64], r0_samples: usize, r0: Option<f64>) -> f64 {
    let finite: Vec<f64> = series.iter().copied().filter(|v| v.is_finite()).collect();
    let mut r0v: f64 = match r0 {
        Some(v) => v,
        None => {
            if r0_samples > 0 {
                let lim = r0_samples.min(finite.len());
                median(&finite[..lim])
            } else {
                median(&finite)
            }
        }
    };
    if !r0v.is_finite() || r0v <= 0.0 {
        let positives: Vec<f64> = finite.iter().copied().filter(|&v| v > 0.0).collect();
        r0v = if positives.is_empty() { 1.0 } else { mean(&positives) };
    }
    if r0v > 0.0 { r0v } else { 1.0 }
}

// -------------------------------------------------------- per-channel blocks

#[derive(Clone, Copy)]
struct DeviceAgnostic {
    relative_amplitude: f64,
    direction: i64,
    rise_time: f64,
    decay_time: f64,
    auc: f64,
    endpoint_delta: f64,
    r0: f64,
    peak_idx: usize,
    is_dead: bool,
}

impl DeviceAgnostic {
    fn empty() -> Self {
        DeviceAgnostic {
            relative_amplitude: -1.0,
            direction: 0,
            rise_time: -1.0,
            decay_time: -1.0,
            auc: -1.0,
            endpoint_delta: -1.0,
            r0: 0.0,
            peak_idx: 0,
            is_dead: true,
        }
    }
}

fn first_cross(series: &[f64], thresh: f64, dir_: i64) -> Option<usize> {
    let cond: Vec<bool> = if dir_ > 0 {
        series.iter().map(|&s| s >= thresh).collect()
    } else {
        series.iter().map(|&s| s <= thresh).collect()
    };
    for (idx, &c) in cond.iter().enumerate() {
        if c {
            if idx >= series.len() - 1 {
                return None;
            }
            return Some(idx);
        }
    }
    None
}

fn compute_channel_device_agnostic(series: &[f64], r0_samples: usize, sr: f64, r0: Option<f64>) -> DeviceAgnostic {
    if series.len() < r0_samples + 2 {
        return DeviceAgnostic::empty();
    }
    let r0 = r0_from_contract(series, r0_samples, r0);
    let finite: Vec<f64> = series.iter().copied().filter(|v| v.is_finite()).collect();
    let std_ratio = if finite.is_empty() {
        f64::INFINITY
    } else {
        population_std(&finite) / r0
    };
    let dead = finite.len() < 2 || std_ratio < 0.001;
    if dead {
        return DeviceAgnostic {
            relative_amplitude: 0.0,
            direction: 0,
            rise_time: -1.0,
            decay_time: -1.0,
            auc: 0.0,
            endpoint_delta: 0.0,
            r0,
            peak_idx: 0,
            is_dead: true,
        };
    }

    let norm: Vec<f64> = series.iter().map(|&s| (s - r0) / r0).collect();
    let max_val = series.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_val = series.iter().copied().fold(f64::INFINITY, f64::min);
    let delta_max = max_val - r0;
    let delta_min = min_val - r0;
    let (delta_raw, direction) = if delta_max.abs() >= delta_min.abs() {
        (delta_max, 1i64)
    } else {
        (delta_min, -1i64)
    };

    let relative_amplitude = delta_raw.abs() / r0;
    let full_span = delta_raw.abs();

    let mut threshold_10 = r0 + 0.1 * full_span * direction as f64;
    let mut threshold_90 = r0 + 0.9 * full_span * direction as f64;
    threshold_10 = threshold_10.clamp(min_val, max_val);
    threshold_90 = threshold_90.clamp(min_val, max_val);

    let mut rise_time = -1.0;
    let idx_10 = first_cross(series, threshold_10, direction);
    let idx_90 = first_cross(series, threshold_90, direction);
    if let (Some(i10), Some(i90)) = (idx_10, idx_90) {
        rise_time = (i90 as f64 - i10 as f64).abs() / sr;
    }

    let mut decay_time = -1.0;
    let peak_idx = argmax_abs(&norm);
    let post_peak = &series[peak_idx..];
    if post_peak.len() > 2 {
        let desorb_dir = -direction;
        let cond_start: Vec<bool> = if desorb_dir > 0 {
            post_peak.iter().map(|&s| s >= threshold_90).collect()
        } else {
            post_peak.iter().map(|&s| s <= threshold_90).collect()
        };
        let cond_end: Vec<bool> = if desorb_dir > 0 {
            post_peak.iter().map(|&s| s >= threshold_10).collect()
        } else {
            post_peak.iter().map(|&s| s <= threshold_10).collect()
        };
        let start_idx = cond_start.iter().position(|&c| c);
        if let Some(si) = start_idx {
            let end_candidates: Vec<usize> = cond_end[si..]
                .iter()
                .enumerate()
                .filter(|(_, &c)| c)
                .map(|(i, _)| i)
                .collect();
            if !end_candidates.is_empty() {
                decay_time = (si + end_candidates[0]) as f64 / sr;
            }
        }
    }

    let absnorm: Vec<f64> = norm.iter().map(|v| v.abs()).collect();
    let auc = trapezoid(&absnorm);
    let endpoint_delta = (series[series.len() - 1] - r0) / r0;

    DeviceAgnostic {
        relative_amplitude,
        direction,
        rise_time,
        decay_time,
        auc,
        endpoint_delta,
        r0,
        peak_idx,
        is_dead: false,
    }
}

fn compute_channel_absolute(series: &[f64], r0: Option<f64>, a_const: f64, b_const: f64) -> (f64, f64, f64, f64) {
    let r0v = match r0 {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => r0_from_contract(series, 15, None),
    };
    let raw_resistance = if series.len() >= 10 {
        let start = series.len() - 10;
        mean(&series[start..])
    } else {
        mean(series)
    };
    let voltage = raw_resistance;
    let rr_ratio = if r0v > 0.0 { raw_resistance / r0v } else { 0.0 };
    let calib_conc = if b_const != 0.0 && rr_ratio > 0.0 {
        (rr_ratio / a_const.max(0.001)).powf(1.0 / b_const)
    } else {
        0.0
    };
    (raw_resistance, r0v, voltage, calib_conc)
}

fn compute_channel_temporal(series: &[f64], sr: f64) -> (f64, f64, f64, f64) {
    if series.len() < 5 {
        return (0.0, 0.0, 0.0, -1.0);
    }
    let diffs: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let hf_transient = if diffs.is_empty() { 0.0 } else { diffs.iter().map(|v| v.abs()).sum::<f64>() / diffs.len() as f64 };

    let detrended = detrend(series);
    let (mut osc_freq, mut osc_amp) = (0.0, 0.0);
    if detrended.len() > 20 {
        let (freqs, psd) = periodogram(&detrended, sr);
        let peak_idx = if psd.len() > 2 {
            let mut best = 1usize;
            for i in 2..psd.len() {
                if psd[i] > psd[best] {
                    best = i;
                }
            }
            if psd[best] < 0.0 { 0 } else { best }
        } else {
            0
        };
        if peak_idx > 0 {
            osc_freq = freqs[peak_idx];
            osc_amp = psd[peak_idx].sqrt();
        }
    }

    let mut response_latency = -1.0;
    let threshold = if series.len() >= 10 {
        population_std(&series[..10]) * 3.0
    } else {
        population_std(series) * 3.0
    };
    let baseline_mean = if series.len() >= 10 {
        mean(&series[..10])
    } else {
        series[0]
    };
    for i in 10..series.len() {
        if (series[i] - baseline_mean).abs() > threshold {
            response_latency = i as f64 / sr;
            break;
        }
    }
    (hf_transient, osc_freq, osc_amp, response_latency)
}

fn compute_channel_health(series: &[f64], r0_samples: usize, r0: Option<f64>) -> (f64, f64, f64, f64) {
    if series.len() < r0_samples + 5 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let r0 = r0_from_contract(series, r0_samples, r0);
    let drift_rate = if series.len() >= 10 {
        let start = series.len() - 10;
        (mean(&series[start..]) - r0) / r0
    } else {
        0.0
    };
    let sensitivity_decay = 0.0;
    let noise_floor = if r0 > 0.0 {
        let lim = r0_samples.min(series.len());
        population_std(&series[..lim]) / r0
    } else {
        0.0
    };

    let peak_idx = argmax_abs_series(series, r0);
    let mut hysteresis = 0.0;
    if peak_idx < series.len() - 5 && peak_idx > 5 {
        let ads_curve = &series[..=peak_idx];
        let des_curve = &series[peak_idx..];
        let ads_path: Vec<f64> = ads_curve.iter().map(|&s| (s - r0).abs()).collect();
        let des_path: Vec<f64> = des_curve.iter().map(|&s| (s - r0).abs()).collect();
        let ads = trapezoid(&ads_path);
        let des = trapezoid(&des_path);
        hysteresis = (ads - des).abs() / ads.max(1e-10);
    }
    (drift_rate, sensitivity_decay, noise_floor, hysteresis)
}

fn argmax_abs_series(series: &[f64], r0: f64) -> usize {
    let mut best = 0usize;
    for i in 1..series.len() {
        if (series[i] - r0).abs() > (series[best] - r0).abs() {
            best = i;
        }
    }
    best
}

fn compute_channel_hardware(series: &[f64]) -> (f64, f64, f64) {
    if series.len() < 2 {
        return (0.0, 0.0, 0.0);
    }
    let circuit_response = mean(series);
    let thermal_profile = population_std(series);
    let mut adc_noise = 0.0;
    if series.len() >= 10 {
        let win = vec![1.0 / 5.0; 5];
        let smooth = convolve_valid(series, &win);
        let residues: Vec<f64> = (0..smooth.len()).map(|i| series[2 + i] - smooth[i]).collect();
        adc_noise = if residues.is_empty() { 0.0 } else { population_std(&residues) };
    }
    (circuit_response, thermal_profile, adc_noise)
}

fn compute_saturation_index(series: &[f64], r0_samples: usize, r0: Option<f64>) -> f64 {
    if series.len() < r0_samples + 5 {
        return 0.0;
    }
    let r0 = r0_from_contract(series, r0_samples, r0);
    let norm: Vec<f64> = series.iter().map(|&s| (s - r0).abs() / r0).collect();
    let current_response = norm.iter().copied().fold(0.0f64, f64::max);
    let lim = r0_samples.min(norm.len());
    let noise_floor = population_std(&norm[..lim]);
    if current_response < noise_floor * 2.0 {
        return 0.0;
    }
    let denom = current_response + noise_floor * 10.0;
    if denom > 0.0 {
        (current_response / denom).min(1.0)
    } else {
        0.0
    }
}

// -------------------------------------------------- multi-exp decay (LM fit)

const DECAY_FAIL: (f64, f64, f64, f64, f64, f64, f64) = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0);

/// Levenberg–Marquardt least-squares fit with per-parameter diagonal scaling
/// (mirrors MINPACK's `lmder`/`lmdif` trust-region philosophy). Returns
/// `Some(parameters)` on convergence, `None` on non-convergence (matched to
/// `scipy.optimize.curve_fit` raising `RuntimeError`/`ValueError`).
///
/// `model(p, t)` yields the predicted y, `jac(p, t)` yields the Jacobian rows
/// (residual derivative wrt each parameter). `p0` is the initial guess.
/// `bounds` optionally clamps each parameter to `[lo, hi]` after every step.
fn lm_fit(
    t: &[f64],
    y: &[f64],
    p0: &[f64],
    maxfev: usize,
    bounds: Option<&[(f64, f64)]>,
    model: &dyn Fn(&[f64], &[f64]) -> Vec<f64>,
    jac: &dyn Fn(&[f64], &[f64]) -> Vec<Vec<f64>>,
) -> Option<Vec<f64>> {
    let n = y.len();
    let m = p0.len();
    if n < m {
        return None;
    }
    let mut p = p0.to_vec();
    let consume_bounds = |pnew: &mut Vec<f64>| {
        if let Some(b) = bounds {
            for j in 0..m {
                let (lo, hi) = b[j];
                pnew[j] = pnew[j].clamp(lo, hi);
            }
        }
    };
    consume_bounds(&mut p);

    let cost_of = |p: &[f64]| -> f64 {
        let pred = model(p, t);
        let mut c = 0.0;
        for i in 0..n {
            let r = pred[i] - y[i];
            c += r * r;
        }
        c
    };

    let mut resid: Vec<f64> = {
        let pred = model(&p, t);
        (0..n).map(|i| pred[i] - y[i]).collect()
    };
    let mut cost = cost_of(&p);

    let dot = |x: &[f64], y: &[f64]| -> f64 { x.iter().zip(y.iter()).map(|(a, b)| a * b).sum() };

    let mut delta = 1.0f64; // trust-region radius
    let mut lambda = 1.0e-3;
    let mut info = 0i32;
    let mut prev_cost = cost;
    let mut iter = 0usize;
    let mut damp_work: Vec<Vec<f64>> = Vec::with_capacity(m);
    while info == 0 && iter < maxfev {
        // Build A = J^T J and g = J^T r at current p.
        let j = jac(&p, t);
        let mut a = vec![vec![0.0f64; m]; m];
        let mut g = vec![0.0f64; m];
        for jj in 0..m {
            for i in 0..n {
                g[jj] += j[i][jj] * resid[i];
            }
            for kk in 0..m {
                let mut s = 0.0;
                for i in 0..n {
                    s += j[i][jj] * j[i][kk];
                }
                a[jj][kk] = s;
            }
        }
        if g.iter().all(|&gi| gi.abs() < 1e-9) {
            info = 1;
            break;
        }
        // Find a step with ||dp||_2 <= delta (raise lambda until it fits the region).
        let mut dp_opt: Option<Vec<f64>> = None;
        let mut lam = lambda;
        for _ in 0..40 {
            match solve_damped_inplace(&a, &g, lam, &mut damp_work) {
                Some(dp) => {
                    let norm2: f64 = dp.iter().map(|v| v * v).sum();
                    if norm2.sqrt() <= delta || lam > 1e12 {
                        dp_opt = Some(dp);
                        break;
                    }
                }
                None => {
                    // Singular: keep raising the damping so the system becomes
                    // solvable (Levenberg behavior in degenerate/collinear regions)
                    // instead of aborting.
                }
            }
            lam *= 10.0;
        }
        let dp = match dp_opt {
            Some(d) => d,
            None => {
                info = 5;
                break;
            }
        };
        let pnew_ = p.clone();
        let mut pnew = pnew_;
        for jj in 0..m {
            pnew[jj] += dp[jj];
        }
        consume_bounds(&mut pnew);
        let cost_new = cost_of(&pnew);

        // Gain ratio: predicted vs actual reduction.
        let adp = |dp: &[f64], ap: &[Vec<f64>]| -> Vec<f64> {
            let mut v = vec![0.0; m];
            for r in 0..m {
                for c in 0..m {
                    v[r] += ap[r][c] * dp[c];
                }
            }
            v
        };
        let adp_v = adp(&dp, &a);
        let pred_reduction = -dot(&g, &dp) - 0.5 * dot(&dp, &adp_v);
        let actual = cost - cost_new;
        let rho = if pred_reduction.abs() > 1e-300 {
            actual / pred_reduction
        } else {
            -1.0
        };

        if rho > 0.0 {
            // Accept the step.
            let mut rel_change = 0.0f64;
            for j in 0..m {
                let denom = p[j].abs() + 1e-12;
                rel_change = rel_change.max(dp[j].abs() / denom);
            }
            p = pnew;
            cost = cost_new;
            let pred = model(&p, t);
            for i in 0..n {
                resid[i] = pred[i] - y[i];
            }
            // MINPACK-style ftol: relative change in residual sum of squares.
            let cost_rel = (cost - prev_cost).abs() / cost.abs().max(1e-300);
            if cost_rel < 1e-9 {
                info = 1;
                break;
            }
            prev_cost = cost;
            if rel_change < 1e-9 {
                info = 1;
                break;
            }
            // Explosion guard: for degenerate (near-collinear) recovery curves the
            // amplitude/time-constant can diverge; bound the search deterministically
            // so the fit returns finite, reproducible parameters quickly.
            if p.iter().any(|&v| !v.is_finite() || v.abs() > 1e2) {
                info = 1;
                break;
            }
            // Expand trust region when the model tracked reality well.
            if rho > 0.75 {
                let step_norm = (dp.iter().map(|v| v * v).sum::<f64>()).sqrt();
                delta = delta.max(2.0 * step_norm);
                lambda = (lambda * 0.3).max(1e-14);
            } else if rho < 0.25 {
                lambda = (lambda * 3.0).min(1e12);
                let step_norm = (dp.iter().map(|v| v * v).sum::<f64>()).sqrt();
                delta = delta.max(0.5 * step_norm);
            } else {
                lambda = (lambda * 0.7).max(1e-14);
            }
            let max_dp = dp.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            if cost.abs() < 1e-6 && max_dp < 1e-8 {
                info = 1;
                break;
            }
        } else {
            // Reject: shrink the trust region and raise lambda.
            let step_norm = (dp.iter().map(|v| v * v).sum::<f64>()).sqrt();
            delta = 0.5 * step_norm;
            lambda = (lambda * 10.0).min(1e13);
            if delta < 1e-12 {
                info = 5;
                break;
            }
        }
        iter += 1;
    }
    if info > 4 {
        return None;
    }
    consume_bounds(&mut p);
    Some(p)
}

/// Solve `(a + lam*I) x = -g` in place, writing the augmented matrix into the
/// caller-provided `work` buffer (reused across damping retries to avoid
/// per-attempt allocation). Uses Gaussian elimination with partial pivoting and
/// the same pivot floor (1e-14) as the previous solver, so numerically
/// equivalent results.
fn solve_damped_inplace(
    a: &[Vec<f64>],
    g: &[f64],
    lam: f64,
    work: &mut Vec<Vec<f64>>,
) -> Option<Vec<f64>> {
    let n = a.len();
    // Build augmented matrix [a + lam*I | -g] into work, reusing its rows.
    work.resize_with(n, || Vec::with_capacity(n + 1));
    for r in 0..n {
        let row = &mut work[r];
        row.clear();
        row.reserve(n + 1);
        for c in 0..n {
            let v = a[r][c] + if r == c { lam } else { 0.0 };
            row.push(v);
        }
        row.push(-g[r]);
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = work[col][col].abs();
        for r in (col + 1)..n {
            let v = work[r][col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        work.swap(col, pivot);
        for r in (col + 1)..n {
            let factor = work[r][col] / work[col][col];
            if factor == 0.0 {
                continue;
            }
            for c in col..(n + 1) {
                work[r][c] -= factor * work[col][c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut sum = work[r][n];
        for c in (r + 1)..n {
            sum -= work[r][c] * x[c];
        }
        x[r] = sum / work[r][r];
    }
    Some(x)
}

/// Fit bi-exponential recovery and return `(tau1, tau2, tau3, a1, a2, a3, cost)`.
/// Mirrors `compute_multi_exp_decay` with `n_components=2` (the default): the
/// fast/slow two-component model, falling back to a single-exponential fit when
/// the bi model fails.
fn compute_multi_exp_decay_bi(
    t: &[f64],
    y: &[f64],
    a0: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let mut out = DECAY_FAIL;
    // single-exp fallback
    if y.len() > 0 {
        let p0 = vec![a0, 3.0, 0.0];
        if let Some(popt) = lm_fit(
            t,
            y,
            &p0,
            100,
            None,
            &|p, t| t.iter().map(|&ti| p[0] * (-ti / p[1]).exp() + p[2]).collect(),
            &|p, t| {
                t.iter()
                    .map(|&ti| {
                        let e = (-ti / p[1]).exp();
                        vec![e, p[0] * e * ti / (p[1] * p[1]), 1.0]
                    })
                    .collect()
            },
        ) {
            let cost = t
                .iter()
                .zip(y.iter())
                .map(|(&ti, &yi)| {
                    let pred = popt[0] * (-ti / popt[1]).exp() + popt[2];
                    (pred - yi) * (pred - yi)
                })
                .sum::<f64>();
            out = (popt[1].abs(), -1.0, -1.0, popt[0], -1.0, -1.0, cost);
        }
    }
    // bi-exp
    let p0 = vec![a0 * 0.7, 2.0, a0 * 0.3, 10.0, 0.0];
    // Decay time constants tau1 (p[1]) and tau2 (p[3]) are physically positive.
    let bound = [(f64::NEG_INFINITY, f64::INFINITY), (1e-4, f64::INFINITY),
                 (f64::NEG_INFINITY, f64::INFINITY), (1e-4, f64::INFINITY),
                 (f64::NEG_INFINITY, f64::INFINITY)];
    if let Some(popt) = lm_fit(
        t,
        y,
        &p0,
        200,
        Some(&bound),
        &|p, t| t.iter().map(|&ti| p[0] * (-ti / p[1]).exp() + p[2] * (-ti / p[3]).exp() + p[4]).collect(),
        &|p, t| {
            t.iter()
                .map(|&ti| {
                    let e1 = (-ti / p[1]).exp();
                    let e2 = (-ti / p[3]).exp();
                    vec![
                        e1,
                        p[0] * e1 * ti / (p[1] * p[1]),
                        e2,
                        p[2] * e2 * ti / (p[3] * p[3]),
                        1.0,
                    ]
                 })
                .collect()
        },
    ) {
        let cost = t
            .iter()
            .zip(y.iter())
            .map(|(&ti, &yi)| {
                let pred = popt[0] * (-ti / popt[1]).exp() + popt[2] * (-ti / popt[3]).exp() + popt[4];
                (pred - yi) * (pred - yi)
            })
            .sum::<f64>();
        out = (popt[1].abs(), popt[3].abs(), -1.0, popt[0], popt[2], -1.0, cost);
    }
    out
}

/// Public decay entry: `compute_multi_exp_decay` default (`n_components=2`).
pub fn compute_multi_exp_decay(
    series: &[f64],
    peak_idx: Option<usize>,
    sr: f64,
    r0: Option<f64>,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    if series.len() < 20 {
        return DECAY_FAIL;
    }
    let peak_idx = match peak_idx {
        Some(pk) => pk,
        None => {
            let rs = 15.min(series.len() / 3);
            let r0_est = r0_from_contract(series, rs, r0);
            argmax_abs_series(series, r0_est)
        }
    };
    let peak_idx = peak_idx.max(5).min(series.len().saturating_sub(10));
    let recovery = &series[peak_idx..];
    if recovery.len() < 10 {
        return DECAY_FAIL;
    }
    let t: Vec<f64> = (0..recovery.len()).map(|i| i as f64 / sr).collect();
    let last = recovery[recovery.len() - 1];
    let y: Vec<f64> = recovery.iter().map(|&v| v - last).collect();
    let y = y[..t.len()].to_vec();

    if y.iter().all(|&v| v == 0.0) || population_std(&y) < 1e-8 {
        return DECAY_FAIL;
    }
    let a0 = y[0];
    compute_multi_exp_decay_bi(&t, &y, a0)
}

// ------------------------------------------------------- framework assembly

/// Number of framework features for a given channel count:
/// `28*n + n*(n-1)/2 + 4` (abs4 + sat1 + da6 + decay6 + health4 + hw3 + temp4).
pub fn framework_feature_len(n_channels: usize) -> usize {
    28 * n_channels + n_channels.saturating_sub(1) * n_channels / 2 + 4
}

/// Compute the full framework feature vector in the deterministic sorted-name
/// order used by the reference runtime (see module docs). Returns `None` if the
/// window is empty or channel count is zero.
pub fn framework_window_features(window: &[Vec<f64>], r0_samples: usize, sr: f64) -> Option<Vec<f64>> {
    if window.is_empty() {
        return None;
    }
    let n_ch = window[0].len();
    if n_ch == 0 {
        return None;
    }
    let per_channel = 28usize;
    let global_start = per_channel * n_ch; // 168 for 6ch
    let sel_start = global_start + 4; // 172 for 6ch
    let total = framework_feature_len(n_ch);
    let mut feats = vec![0.0; total];

    // per-channel series
    let series_per_ch: Vec<Vec<f64>> = (0..n_ch)
        .map(|c| {
            let mut ch: Vec<f64> = window.iter().map(|s| s[c]).collect();
            for v in ch.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
            ch
        })
        .collect();

    let mut device_agnostic: Vec<DeviceAgnostic> = Vec::with_capacity(n_ch);

    for c in 0..n_ch {
        let series = &series_per_ch[c];
        let base = per_channel * c;
        let r0 = r0_from_contract(series, r0_samples, None);

        let da = compute_channel_device_agnostic(series, r0_samples, sr, Some(r0));
        let r0_used = da.r0;
        device_agnostic.push(da);

        // abs block
        let (raw_resistance, baseline, voltage, calib) =
            compute_channel_absolute(series, Some(r0_used), 1.0, -0.5);
        feats[base + 0] = baseline; // ch_abs_baseline_resistance
        feats[base + 1] = calib; // ch_abs_calibrated_concentration
        feats[base + 2] = raw_resistance; // ch_abs_raw_resistance
        feats[base + 3] = voltage; // ch_abs_voltage

        // saturation index
        feats[base + 4] = compute_saturation_index(series, r0_samples, Some(r0_used));

        // da block (6): auc, decay_time, direction, endpoint_delta, relative_amplitude, rise_time
        feats[base + 5] = da.auc;
        feats[base + 6] = da.decay_time;
        feats[base + 7] = da.direction as f64;
        feats[base + 8] = da.endpoint_delta;
        feats[base + 9] = da.relative_amplitude;
        feats[base + 10] = da.rise_time;

        // decay block (6): a1, a2, a3, tau1, tau2, tau3
        let dec = compute_multi_exp_decay(series, Some(da.peak_idx), sr, Some(r0_used));
        feats[base + 11] = dec.3; // a1
        feats[base + 12] = dec.4; // a2
        feats[base + 13] = dec.5; // a3
        feats[base + 14] = dec.0; // tau1
        feats[base + 15] = dec.1; // tau2
        feats[base + 16] = dec.2; // tau3

        // health block (4): drift_rate, hysteresis, noise_floor, sensitivity_decay
        let (drift, sens_decay, noise_floor, hysteresis) =
            compute_channel_health(series, r0_samples, Some(r0_used));
        feats[base + 17] = drift;
        feats[base + 18] = hysteresis;
        feats[base + 19] = noise_floor;
        feats[base + 20] = sens_decay;

        // hw block (3): adc_noise, circuit_response, thermal_profile
        let (circuit, thermal, adc) = compute_channel_hardware(series);
        feats[base + 21] = adc;
        feats[base + 22] = circuit;
        feats[base + 23] = thermal;

        // temp block (4): hf_transient, oscillation_amp, oscillation_freq, response_latency
        let (hf, osc_freq, osc_amp, latency) = compute_channel_temporal(series, sr);
        feats[base + 24] = hf;
        feats[base + 25] = osc_amp;
        feats[base + 26] = osc_freq;
        feats[base + 27] = latency;
    }

    // global metrics (4)
    let active_dr: Vec<f64> = (0..n_ch)
        .filter(|&i| !device_agnostic[i].is_dead)
        .map(|i| device_agnostic[i].relative_amplitude)
        .collect();
    let max = active_dr.iter().copied().fold(0.0f64, f64::max);
    feats[global_start + 0] = if active_dr.is_empty() { 0.0 } else { max };
    feats[global_start + 1] = if active_dr.is_empty() { 0.0 } else { mean(&active_dr) };
    feats[global_start + 2] = active_dr.len() as f64;
    let total_auc: f64 = (0..n_ch)
        .filter(|&i| !device_agnostic[i].is_dead)
        .map(|i| device_agnostic[i].auc)
        .sum();
    feats[global_start + 3] = total_auc;

    // selectivity ratios (15 for 6ch): dr_i / dr_j among *active* channels only
    {
        let mut sel_idx = sel_start;
        for ci in 0..n_ch {
            for cj in (ci + 1)..n_ch {
                let dr_i = device_agnostic[ci].relative_amplitude;
                let dr_j = device_agnostic[cj].relative_amplitude;
                // Ratio only computed if BOTH channels are active (non-dead, rel_amp>0).
                let active_i = !device_agnostic[ci].is_dead && dr_i > 0.0;
                let active_j = !device_agnostic[cj].is_dead && dr_j > 0.0;
                let val = if active_i && active_j && dr_j > 0.0 {
                    dr_i / dr_j
                } else {
                    0.0
                };
                feats[sel_idx] = val;
                sel_idx += 1;
            }
        }
    }

    Some(feats)
}

// --------------------------------------------------------------- tests

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Confirm the deterministic block layout offsets match the documented
    /// sorted order for a 6-channel device (187 dims).
    #[test]
    fn framework_len_is_187_for_six_channels() {
        assert_eq!(framework_feature_len(6), 187);
        assert_eq!(framework_feature_len(3), 28 * 3 + 3 * 2 / 2 + 4); // 84+3+4=91
        assert_eq!(framework_feature_len(4), 28 * 4 + 4 * 3 / 2 + 4); // 112+6+4=122
    }

    #[test]
    fn empty_window_returns_none() {
        assert!(framework_window_features(&[], 15, 10.0).is_none());
        assert!(framework_window_features(&[vec![]], 15, 10.0).is_none());
    }

    #[test]
    fn deterministic_constant_window() {
        // All-constant window -> dead channels, zero features where applicable.
        let window: Vec<Vec<f64>> = (0..40).map(|_| vec![100.0; 6]).collect();
        let f = framework_window_features(&window, 15, 10.0).unwrap();
        assert_eq!(f.len(), 187);
        // dead: relative amplitude 0, direction 0, decay tau -1
        assert_eq!(f[9], 0.0); // ch0 da relative_amplitude
        assert_eq!(f[7], 0.0); // ch0 da direction
        assert_eq!(f[14], -1.0); // ch0 decay tau1
        // sel ratios all zero (all dead)
        for i in 172..187 {
            assert_eq!(f[i], 0.0);
        }
        // global n active = 0
        assert_eq!(f[170], 0.0);
    }

    #[test]
    fn nonzero_signal_produces_active_features() {
        let window: Vec<Vec<f64>> = (0..100)
            .map(|t| {
                let x = -((t as f64 / 100.0) - 0.2).powi(2) * 200.0;
                vec![500.0 + 100.0 * x.exp(); 6]
            })
            .collect();
        let f = framework_window_features(&window, 15, 10.0).unwrap();
        assert_eq!(f.len(), 187);
        // relative amplitude positive
        assert!(f[9] > 0.0);
        // global n active = 6
        assert_eq!(f[170], 6.0);
        // sel ratios nonzero
        assert!(f[172] > 0.0);
    }
}
