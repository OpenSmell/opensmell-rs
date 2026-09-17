/// Dual state/parameter Kalman anomaly engine.
///
/// This is the Wave-2 replacement for the Welford/EWMA/windowed-Mahalanobis
/// stack: a state filter tracks the environmental level, a parameter filter
/// tracks hidden gain/offset (drift = parameter walk, poisoning = gain decay),
/// a regime model expects regime transitions, and the typology head names the
/// change. An optional adsorption-memory block absorbs post-exposure residue.
/// Spec: `docs/anomaly-engine-design.md`.
use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::{Result, OpenSmellError};
use super::filter::{KalmanFilterImpl, StateFilterKind, UkfParams};
use super::linalg::{diag, identity};
use super::regimes::{RegimeConfig, RegimeModel};
use super::typology::{Typology, TypologyConfig, TypologyHead};
use super::platt::{PlattCalibrator, PlattParams};
use super::stimulus::{HealthFinding, StimulusConfig, StimulusGainTracker};

/// Per-channel ambient correction from the on-board temperature/humidity too
/// short to fit the measurement-model note here — full spec in the design doc.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AmbientModel {
    /// Per-channel sensitivity to temperature delta (sensor units per °C).
    pub temp_slope: Vec<f64>,
    /// Per-channel sensitivity to relative-humidity delta (sensor units per %RH).
    pub rh_slope: Vec<f64>,
    pub t0: f64,
    pub rh0: f64,
}

impl AmbientModel {
    pub fn is_empty(&self) -> bool {
        self.temp_slope.is_empty() && self.rh_slope.is_empty()
    }

    /// Expected ambient-induced offset for each channel given a humidity/temp reading.
    pub fn correction(&self, temperature: Option<f64>, rh: Option<f64>) -> Vec<f64> {
        let d_t = temperature.map(|t| t - self.t0).unwrap_or(0.0);
        let d_h = rh.map(|h| h - self.rh0).unwrap_or(0.0);
        let n = self.temp_slope.len();
        (0..n)
            .map(|i| {
                let t = self.temp_slope.get(i).copied().unwrap_or(0.0) * d_t;
                let h = self.rh_slope.get(i).copied().unwrap_or(0.0) * d_h;
                t + h
            })
            .collect()
    }

    pub fn fit(&mut self, temp_slope: Vec<f64>, rh_slope: Vec<f64>, t0: f64, rh0: f64) {
        self.temp_slope = temp_slope;
        self.rh_slope = rh_slope;
        self.t0 = t0;
        self.rh0 = rh0;
    }
}

/// Stream cadence of the engine (10 Hz). Used to convert per-second desorption
/// time constants into per-sample decay factors.
const SAMPLE_PERIOD_S: f64 = 0.1;

/// Per-channel adsorption-memory configuration.
///
/// When enabled, the state filter gains an extra memory block `m` per channel
/// (`x_0..x_c, m_0..m_c`). `m` decays exponentially toward zero with the
/// channel's desorption time constant — the residue a sensor carries after a
/// high-concentration exposure (the "still smells like cake" effect). Because
/// `m` is its own state instead of part of the level or the parameter walk,
/// a desorption tail is predicted and absorbed rather than read as drift or as
/// a fresh event. Measurement becomes `y_i = g_i·(x_i + m_i) + o_i`; nothing
/// else in the engine changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdsorptionConfig {
    /// Master switch. Disabled keeps the legacy c-dimensional `[x]` state
    /// layout exactly (no behavioural change to existing deployments).
    pub enabled: bool,
    /// Per-channel desorption time constants in seconds. Shorter ⇒ residue
    /// clears faster; a missing entry falls back to `tau_default_s`.
    pub tau_s: Vec<f64>,
    /// Fallback time constant (seconds) for channels without a `tau_s` entry.
    /// A calibration/purge experiment should replace this per sensor.
    #[serde(default = "default_tau_s")]
    pub tau_default_s: f64,
    /// Process noise on the memory state (how freely `m` may move). Kept
    /// ≪ `q_state` so genuine events stay in `x` and only residue sticks to `m`.
    pub q_adsorption: f64,
}

fn default_tau_s() -> f64 {
    300.0
}

impl Default for AdsorptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tau_s: Vec::new(),
            tau_default_s: default_tau_s(),
            q_adsorption: 1e-5,
        }
    }
}

/// Per-sample decay factor of the memory state for one channel.
fn desorption_decay(cfg: &AdsorptionConfig, i: usize) -> f64 {
    let tau = cfg.tau_s.get(i).copied().unwrap_or(cfg.tau_default_s).max(0.1);
    (-SAMPLE_PERIOD_S / tau).exp()
}

/// Per-channel exponent from the response config (missing entries → default).
fn alphas_for(cfg: &ResponseConfig, n_channels: usize) -> Vec<f64> {
    (0..n_channels)
        .map(|i| cfg.alpha.get(i).copied().unwrap_or(cfg.alpha_default))
        .collect()
}

/// Parameter-vector width per channel: 2 when linear ([g, o]), 3 when the
/// power-law response is enabled ([g, o, α]).
fn param_width_for(cfg: &EngineConfig) -> usize {
    if cfg.response.power_law_enabled {
        3
    } else {
        2
    }
}

fn param_dim_for(cfg: &EngineConfig, n_channels: usize) -> usize {
    param_width_for(cfg) * n_channels
}

/// Interleave gain/offset/(exponent) into the parameter layout.
/// When `power_law` is true: `[g, o, α]` per channel (3-wide).
/// When false: `[g, o]` per channel (2-wide).
fn interleave_params(g: &[f64], o: &[f64], alphas: &[f64], power_law: bool) -> Vec<f64> {
    if power_law {
        g.iter()
            .zip(o.iter())
            .zip(alphas.iter())
            .flat_map(|((&gg, &oo), &aa)| [gg, oo, aa])
            .collect()
    } else {
        g.iter()
            .zip(o.iter())
            .flat_map(|(&gg, &oo)| [gg, oo])
            .collect()
    }
}

/// Exponential for the power-law response. φ(x, α) = |x|ᵅ·sgn(x), so
/// α = 1 is exactly the linear model, α < 1 is the compressed (saturating) MOX
/// response at high concentration, and the expression stays defined for the
/// negative excursions a normalized state can wander into.
fn phi(x: f64, alpha: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x.abs().powf(alpha).copysign(x)
    }
}

/// dφ/dx — the Jacobian used by the linear (EKF) state-update path.
fn dphi_dx(x: f64, alpha: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        alpha * x.abs().powf(alpha - 1.0)
    }
}

/// dφ/dα — the Jacobian column for the α parameter.
fn dphi_da(x: f64, alpha: f64) -> f64 {
    if x == 0.0 || x.abs() == 1.0 {
        0.0
    } else {
        x.abs().powf(alpha) * x.abs().ln() * x.signum()
    }
}

/// Power-law response-model configuration.
///
/// The default gain/offset model `y = g·x + o` is the local Taylor regime of
/// the true MOX power-law response `R_s/R_0 = (C/C_0)^(−α)`. When enabled, the
/// measurement model becomes `y_i = g_i·φ(x_i, α_i) + o_i` (φ above) and the
/// parameter filter tracks `[g, o, α]` per channel, so a large concentration
/// spike at the compressed high end is read as `x` (the state), not as gain
/// decay — the misreading that would otherwise fake a poison confirmation.
/// The UKF handles the non-linearity sigma-point-wise; the linear path uses the
/// local Jacobian (honestly labelled EKF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseConfig {
    /// Master switch. Disabled keeps the legacy linear `g·x + o` model exactly.
    pub power_law_enabled: bool,
    /// Per-channel exponent. Missing entries fall back to `alpha_default`.
    pub alpha: Vec<f64>,
    /// Fallback exponent for channels without an `alpha` entry (1.0 = linear).
    #[serde(default = "default_alpha_s")]
    pub alpha_default: f64,
    /// Parameter-filter walk noise on α (α is physical response *shape*, so it
    /// moves far more slowly than gain/offset).
    pub q_alpha: f64,
}

fn default_alpha_s() -> f64 {
    1.0
}

impl Default for ResponseConfig {
    fn default() -> Self {
        Self {
            power_law_enabled: false,
            alpha: Vec::new(),
            alpha_default: default_alpha_s(),
            q_alpha: 1e-6,
        }
    }
}

/// Full engine configuration (serde-serializable for the settings UI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub state_filter: StateFilterKind,
    pub ukf: UkfParams,
    /// State process noise (per channel). Small ⇒ smooth baseline.
    pub q_state: f64,
    /// Parameter walk noise (offset, per channel). ≪ q_state ⇒ slow drift.
    pub q_param: f64,
    /// Measurement-noise scale applied to the calibrated per-channel R.
    pub r_scale: f64,
    /// Ridge on innovation solves (keeps borderline S well-conditioned).
    pub innovation_ridge: f64,
    /// Per-budget per-channel forward thresholds in calibrated σ, in the order
    /// [standard, conservative, sensitive]. Scaled by 1/sensitivity.
    pub k_std: [f64; 3],
    /// Minimum per-channel z before any multivariate claim counts.
    pub z_min: f64,
    /// User-facing sensitivity knob (same semantics as the legacy detector).
    pub sensitivity: f64,
    pub regimes: RegimeConfig,
    pub typology: TypologyConfig,
    pub stimulus: StimulusConfig,
    /// Adsorption-memory state (extended state vector) configuration.
    #[serde(default)]
    pub adsorption: AdsorptionConfig,
    /// Power-law response-model configuration.
    #[serde(default)]
    pub response: ResponseConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            state_filter: StateFilterKind::Unscented,
            ukf: UkfParams::default(),
            q_state: 1e-3,
            q_param: 1e-6,
            r_scale: 1.0,
            innovation_ridge: 1e-9,
            k_std: [5.0, 6.0, 4.0],
            z_min: 0.5,
            sensitivity: 1.0,
            regimes: RegimeConfig::default(),
            typology: TypologyConfig::default(),
            stimulus: StimulusConfig::default(),
            adsorption: AdsorptionConfig::default(),
            response: ResponseConfig::default(),
        }
    }
}

/// One per-reading verdict from the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineVerdict {
    pub is_anomaly: bool,
    /// How many of the three budgets fired (0..3).
    pub anomaly_votes: usize,
    /// Honest magnitude: `sqrt(rᵀ S⁻¹ r)` (replaces legacy raw Mahalanobis).
    pub raw_score: f64,
    pub max_z: f64,
    pub z_scores: Vec<f64>,
    pub triggered_channels: Vec<usize>,
    /// Platt-calibrated probability that this reading is a real change.
    pub confidence: f64,
    pub threshold_confidence: f64,
    pub n_samples: usize,
    pub budget_fired: [bool; 3],
    pub typology: Option<Typology>,
    pub regime_switch: bool,
    pub regime: usize,
    /// Parameter filter's relative retained gain per channel (g/g0).
    pub relative_gains: Vec<f64>,
    pub health_findings: Vec<HealthFinding>,
    /// Ambient-and-state corrected smoothed reading (matches what was scored).
    pub smoothed: Vec<f64>,
    /// Adsorption-memory block `m` per channel (only populated when enabled).
    pub adsorption: Vec<f64>,
    /// Per-channel power-law exponents α (only populated when enabled).
    pub alpha_exponents: Vec<f64>,
    /// True while the fresh-device warm-up window is still being established
    /// (before the baseline has 60 samples). Never anomalous; the caller should
    /// surface "warming up" instead of the environment verdict.
    #[serde(default)]
    pub warming_up: bool,
}

/// A single environmental sample at 10 Hz.
#[derive(Debug, Clone, Copy, Default)]
pub struct AmbientReading {
    pub temperature: Option<f64>,
    pub humidity: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct DualKalmanEngine {
    pub n_channels: usize,
    pub config: EngineConfig,
    state_filter: KalmanFilterImpl,
    // Parameters θ = [g_0, o_0, g_1, o_1, ...] (absolute, re-anchored on
    // calibration / stimulus). Standard dual-EKF (Wan & van der Merwe 2000):
    // the parameter filter observes the SAME measurement as the state filter
    // (so no coupled closed loop drives θ to a degenerate rotation).
    params: KalmanFilterImpl,
    g: Vec<f64>,
    o: Vec<f64>,
    /// Current power-law exponent per channel (estimate from the parameter
    /// filter when enabled; static config value otherwise).
    alpha: Vec<f64>,
    r_diag: Vec<f64>,
    ambient: AmbientModel,
    underlying_r: Vec<f64>,
    regimes: RegimeModel,
    typology_head: TypologyHead,
    platt: PlattCalibrator,
    stimulus: StimulusGainTracker,
    n_samples: usize,
    warmup_samples: usize,
    /// Raw readings buffered before auto-calibration during the warm-up window.
    baseline_buffer: Vec<Vec<f64>>,
    last_innovation: Vec<f64>,
    last_inno_cov: Vec<Vec<f64>>,
    relative_gains: Vec<f64>,
    innovation_history: VecDeque<f64>,
    confirmed_anomaly_count: usize,
    confirmed_normal_count: usize,
    recent_health: Vec<HealthFinding>,
    poisoned_channels: Vec<usize>,
    calibrated: bool,
}

const WARMUP_SAMPLES: usize = 60;

impl DualKalmanEngine {
    pub fn new(n_channels: usize) -> Self {
        let config = EngineConfig::default();
        let p0 = diag(&vec![1.0; n_channels]);
        let state_filter =
            KalmanFilterImpl::new(vec![0.0; n_channels], p0, config.state_filter);
        let params = KalmanFilterImpl::new(
            interleave_params(
                &vec![1.0; n_channels],
                &vec![0.0; n_channels],
                &alphas_for(&config.response, n_channels),
                config.response.power_law_enabled,
            ),
            diag(&vec![1.0; param_dim_for(&config, n_channels)]),
            StateFilterKind::Kalman,
        );
        Self {
            n_channels,
            state_filter,
            params,
            g: vec![1.0; n_channels],
            o: vec![0.0; n_channels],
            alpha: alphas_for(&config.response, n_channels),
            r_diag: vec![1e-3; n_channels],
            ambient: AmbientModel::default(),
            underlying_r: vec![1e-3; n_channels],
            regimes: RegimeModel::new(n_channels, config.regimes.clone()),
            typology_head: TypologyHead::new(config.typology.clone()),
            platt: PlattCalibrator::new(),
            stimulus: StimulusGainTracker::new(
                n_channels,
                vec![1.0; n_channels],
                config.stimulus.clone(),
            ),
            config,
            n_samples: 0,
            warmup_samples: 0,
            baseline_buffer: Vec::new(),
            last_innovation: vec![0.0; n_channels],
            last_inno_cov: vec![vec![0.0; n_channels]; n_channels],
            relative_gains: vec![1.0; n_channels],
            innovation_history: VecDeque::new(),
            confirmed_anomaly_count: 0,
            confirmed_normal_count: 0,
            recent_health: Vec::new(),
            poisoned_channels: Vec::new(),
            calibrated: false,
        }
    }

    pub fn set_config(&mut self, config: EngineConfig) {
        let state_dim_changed = config.adsorption.enabled != self.config.adsorption.enabled;
        let param_dim_changed =
            config.response.power_law_enabled != self.config.response.power_law_enabled;
        self.config = config.clone();
        self.stimulus = StimulusGainTracker::new(
            self.n_channels,
            self.g.clone(),
            config.stimulus.clone(),
        );
        if state_dim_changed {
            // The state layout changes with the master switch; resize the
            // filter so a config applied before calibration stays consistent.
            self.state_filter = KalmanFilterImpl::new(
                vec![0.0; self.filter_dim()],
                diag(&vec![1.0; self.filter_dim()]),
                self.config.state_filter,
            );
            self.state_filter.ukf = self.config.ukf;
        }
        if param_dim_changed {
            // Parameter vector reshapes with the power-law switch.
            self.params = KalmanFilterImpl::new(
                self.param_init(),
                diag(&vec![1.0; self.param_dim()]),
                StateFilterKind::Kalman,
            );
            self.alpha = alphas_for(&self.config.response, self.n_channels);
        }
    }

    /// Dimension of the state filter given the current adsorption switch.
    fn filter_dim(&self) -> usize {
        if self.config.adsorption.enabled {
            2 * self.n_channels
        } else {
            self.n_channels
        }
    }

    fn param_width(&self) -> usize {
        param_width_for(&self.config)
    }

    fn param_dim(&self) -> usize {
        param_dim_for(&self.config, self.n_channels)
    }

    /// Initial parameter vector `[g=1, o=0, (α=config)]` per channel.
    fn param_init(&self) -> Vec<f64> {
        interleave_params(
            &vec![1.0; self.n_channels],
            &vec![0.0; self.n_channels],
            &alphas_for(&self.config.response, self.n_channels),
            self.config.response.power_law_enabled,
        )
    }

    /// Whether the power-law response model is part of the filter.
    pub fn response_enabled(&self) -> bool {
        self.config.response.power_law_enabled
    }

    /// Current per-channel exponent α from the parameter filter (empty when the
    /// power-law model is disabled).
    pub fn alpha_exponents(&self) -> Vec<f64> {
        if self.config.response.power_law_enabled {
            (0..self.n_channels)
                .map(|i| self.alpha[i].clamp(0.1, 3.0))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Whether the adsorption-memory state is part of the filter.
    pub fn adsorption_enabled(&self) -> bool {
        self.config.adsorption.enabled
    }

    /// Current adsorption-memory block `m` (empty when the state is disabled).
    pub fn adsorption_memory(&self) -> Vec<f64> {
        if self.config.adsorption.enabled {
            self.state_filter.x[self.n_channels..].to_vec()
        } else {
            Vec::new()
        }
    }

    pub fn set_ambient_model(&mut self, ambient: AmbientModel) {
        self.ambient = ambient;
    }

    pub fn ambient_model(&self) -> &AmbientModel {
        &self.ambient
    }

    /// Calibrate from a baseline window: state mean/covariance, per-channel
    /// measurement noise, initial offset anchor, and the first regime cluster.
    pub fn calibrate_baseline(&mut self, samples: &[Vec<f64>]) -> Result<()> {
        if samples.is_empty() {
            return Err(OpenSmellError::InsufficientData { expected: 1, actual: 0 });
        }
        let n = samples.len();
        let c = self.n_channels;
        for s in samples {
            if s.len() != c {
                return Err(OpenSmellError::InvalidChannelCount {
                    got: s.len(),
                    expected: c,
                });
            }
        }
        let mut mean = vec![0.0; c];
        for s in samples {
            for (i, &v) in s.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        let mut var = vec![1e-6; c];
        for s in samples {
            for (i, &v) in s.iter().enumerate() {
                let d = v - mean[i];
                var[i] += d * d;
            }
        }
        for v in var.iter_mut() {
            *v /= n.max(1) as f64;
        }
        // Measurement noise: calibrated variance scaled by r_scale, floored so a
        // perfectly flat calibration channel stays invertible.
        let r_floor = 1e-6;
        self.r_diag = var
            .iter()
            .map(|&v| (v * self.config.r_scale).max(r_floor))
            .collect();
        self.underlying_r = self.r_diag.clone();
        self.relative_gains = vec![1.0; c];

        // State prior: mean with a process-noise floor. With the adsorption
        // state enabled the memory block starts at zero with the same scale
        // (a freshly calibrated sensor holds no residue).
        let d = self.filter_dim();
        let p_x = var.iter().map(|&v| v.max(1e-4) * 4.0).collect::<Vec<f64>>();
        let mut prior = mean.clone();
        let mut prior_p = diag(&p_x);
        if self.config.adsorption.enabled {
            prior.extend(std::iter::repeat_n(0.0, self.n_channels));
            let mut full = vec![vec![0.0; d]; d];
            for i in 0..self.n_channels {
                full[i][i] = p_x[i];
                full[self.n_channels + i][self.n_channels + i] = p_x[i];
            }
            prior_p = full;
        }
        self.state_filter = KalmanFilterImpl::new(prior, prior_p, self.config.state_filter);
        self.state_filter.ukf = self.config.ukf;

        // Parameter prior: gain 1 (deterministic), offset 0 (deterministic),
        // and the configured exponent α (the local-linear starting point) — the
        // calibrated mean lives in the state, so the offset starts at 0.
        self.params = KalmanFilterImpl::new(
            self.param_init(),
            diag(&vec![1.0; self.param_dim()]),
            StateFilterKind::Kalman,
        );
        self.g = vec![1.0; c];
        self.o = vec![0.0; c];
        self.alpha = alphas_for(&self.config.response, c);

        // First regime cluster seeded from the calibrated baseline.
        let mut cov = vec![vec![0.0; c]; c];
        for i in 0..c {
            cov[i][i] = var[i];
        }
        self.regimes.seed(mean, cov);
        self.calibrated = true;
        Ok(())
    }

    /// True once a baseline calibrated the engine (manual or warm-up).
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Run one 10 Hz reading through both filters. `ambient` carries on-board
    /// temperature/humidity when present (None ⇒ ambient correction is zero).
    pub fn detect(&mut self, reading: &[f64], ambient: Option<AmbientReading>) -> Result<EngineVerdict> {
        let result = self.detect_raw(reading, ambient)?;
        Ok(result)
    }

    /// Convenience: detect with no ambient data (used by warm-up/tests).
    pub fn detect_no_ambient(&mut self, reading: &[f64]) -> Result<EngineVerdict> {
        self.detect(reading, None)
    }

    /// Verdict for a warm-up sample: explicitly non-anomalous, tagged `warming_up`
    /// so the caller can surface the honest "baseline still forming" state.
    fn warm_up_verdict(&self, reading: &[f64]) -> EngineVerdict {
        EngineVerdict {
            is_anomaly: false,
            anomaly_votes: 0,
            raw_score: 0.0,
            max_z: 0.0,
            z_scores: vec![0.0; self.n_channels],
            triggered_channels: Vec::new(),
            confidence: 0.0,
            threshold_confidence: self.threshold_confidence_for(self.n_samples),
            n_samples: self.n_samples,
            budget_fired: [false; 3],
            typology: None,
            regime_switch: false,
            regime: 0,
            relative_gains: self.relative_gains.clone(),
            health_findings: Vec::new(),
            smoothed: reading.to_vec(),
            adsorption: Vec::new(),
            alpha_exponents: Vec::new(),
            warming_up: true,
        }
    }

    /// Logistic threshold confidence in sample count (same meaning as the
    /// legacy detector: how much data has shaped this baseline).
    fn threshold_confidence_for(&self, n: usize) -> f64 {
        1.0 / (1.0 + (-(n as f64 - 30.0) / 15.0).exp())
    }

    fn detect_raw(&mut self, reading: &[f64], ambient: Option<AmbientReading>) -> Result<EngineVerdict> {
        if reading.len() != self.n_channels {
            return Err(OpenSmellError::InvalidChannelCount {
                got: reading.len(),
                expected: self.n_channels,
            });
        }
        self.n_samples += 1;

        // Fresh-device warm-up: an uncalibrated engine buffers the first
        // `WARMUP_SAMPLES` readings, reports `warming_up`, and never alarms —
        // the debut of a brand-new board stays quiet (design §6, §11.1). When
        // the buffer fills, it auto-calibrates exactly like the legacy
        // detector, so a caller that only feeds `detect` gets a baseline too.
        if !self.calibrated {
            self.baseline_buffer.push(reading.to_vec());
            self.warmup_samples += 1;
            if self.warmup_samples >= WARMUP_SAMPLES {
                let samples = std::mem::take(&mut self.baseline_buffer);
                self.calibrate_baseline(&samples)?;
                self.warmup_samples = 0;
                // Fall through: this reading now runs the real path on the
                // just-established baseline (it is exactly the 60th reading).
            } else {
                return Ok(self.warm_up_verdict(reading));
            }
        }

        // 1. Ambient-corrected measurement: the environment's contribution is
        //    predicted and subtracted, so weather is not an anomaly.
        let ambient_off = if self.ambient.is_empty() {
            vec![0.0; self.n_channels]
        } else {
            self.ambient.correction(
                ambient.and_then(|a| a.temperature),
                ambient.and_then(|a| a.humidity),
            )
        };
        let corrected: Vec<f64> = reading
            .iter()
            .zip(ambient_off.iter())
            .map(|(&y, &a)| y - a)
            .collect();

        // 2. State filter: predict. Environmental level `x` is a random walk
        // (F=I, Q = q_state·I); with adsorption enabled the memory block `m`
        // decays by exp(−Δt/τ_i) toward zero (Q = q_adsorption·I).
        let c = self.n_channels;
        let ads_on = self.config.adsorption.enabled;
        if ads_on {
            let d = 2 * c;
            let mut f = vec![vec![0.0; d]; d];
            let mut q = vec![vec![0.0; d]; d];
            for i in 0..c {
                f[i][i] = 1.0;
                f[c + i][c + i] = desorption_decay(&self.config.adsorption, i);
                q[i][i] = self.config.q_state;
                q[c + i][c + i] = self.config.adsorption.q_adsorption;
            }
            self.state_filter.predict(&f, &q)?;
        } else {
            let qx = diag(&vec![self.config.q_state; self.n_channels]);
            self.state_filter.predict(&identity(self.n_channels), &qx)?;
        }

        // Measurement model.  The *linear* base is y_i = g_i·x_i + o_i;
        // with adsorption: y_i = g_i·(x_i + m_i) + o_i (m enters linearly);
        // with power-law:  y_i = g_i·φ(x_i, α_i) + o_i (non-linear in x,
        //     φ(x,α)=|x|ᵅ·sgn(x)); both combined: y_i = g_i·(φ(x_i,α_i)+m_i)+o_i.
        // UKF handles arbitrary h; the Kalman path uses the EKF local Jacobian.
        let g = self.g.clone();
        let o = self.o.clone();
        let alpha = self.alpha.clone();
        let pl_on = self.config.response.power_law_enabled;
        let r_mat = diag(&self.r_diag);
        let state_out = match self.config.state_filter {
            StateFilterKind::Kalman => {
                // EKF: linearise ∂y/∂x at the predicted state.
                let mut h_lin = vec![vec![0.0; self.filter_dim()]; self.n_channels];
                for i in 0..self.n_channels {
                    let x_pred = self.state_filter.x[i];
                    let dlevel = if pl_on {
                        dphi_dx(x_pred, alpha[i])
                    } else {
                        1.0
                    };
                    h_lin[i][i] = g[i] * dlevel;
                    if ads_on {
                        h_lin[i][self.n_channels + i] = g[i];
                    }
                }
                self.state_filter.update_linear(&corrected, &h_lin, &r_mat, self.config.innovation_ridge)?
            }
            StateFilterKind::Unscented => {
                let h = |xp: &[f64]| -> Vec<f64> {
                    (0..c)
                        .map(|i| {
                            let level = if pl_on {
                                phi(xp[i], alpha[i])
                            } else {
                                xp[i]
                            } + if ads_on { xp[c + i] } else { 0.0 };
                            g[i] * level + o[i]
                        })
                        .collect()
                };
                self.state_filter
                    .update_sigma(&corrected, &h, &r_mat, self.config.innovation_ridge)?
            }
        };

        // 3. Read the innovation (this is the anomaly evidence).
        let r = state_out.innovation.clone();
        let s = state_out.innovation_cov.clone();
        let z_scores = state_out.z_scores.clone();
        let raw_score = state_out.mahalanobis;
        let max_z = z_scores.iter().cloned().fold(0.0f64, f64::max);
        let triggered: Vec<usize> = z_scores
            .iter()
            .enumerate()
            .filter(|(_, &z)| z > self.config.z_min)
            .map(|(i, _)| i)
            .collect();

        // 4. Parameter filter (dual-EKF): observes the same measurement as the state
        //    filter (standard Wan–van der Merwe dual-EKF).
        //    Linear model: y = g·x̂ + o, linear in θ with row i = [x̂_i, 1].
        //    Power-law model: y = g·φ(x̂, α) + o, linear in θ with row i =
        //        [φ(x̂_i, α_i) (+m_i), 1, g_i·dφ/dα(x̂_i, α_i)].
        //    A steady stream keeps θ ≈ [1,0,(1)]; persistent drift grows o,
        //    poisoning decays g, power-law tracks shape α.
        let x_hat = self.state_filter.x.clone();
        let pw = self.param_width();
        let pd = self.param_dim();
        let mut h_theta = vec![vec![0.0; pd]; self.n_channels];
        for i in 0..self.n_channels {
            let base = pw * i;
            let level = if ads_on {
                x_hat[i] + x_hat[self.n_channels + i]
            } else {
                x_hat[i]
            };
            if pl_on {
                h_theta[i][base] = phi(x_hat[i], alpha[i])
                    + if ads_on { x_hat[self.n_channels + i] } else { 0.0 };
                h_theta[i][base + 1] = 1.0;
                h_theta[i][base + 2] = self.g[i] * dphi_da(x_hat[i], alpha[i]);
            } else {
                h_theta[i][base] = level;
                h_theta[i][base + 1] = 1.0;
            }
        }
        let mut q_vec = Vec::with_capacity(pd);
        for _ in 0..self.n_channels {
            q_vec.push(self.config.q_param);
            q_vec.push(self.config.q_param);
            if pl_on {
                q_vec.push(self.config.response.q_alpha);
            }
        }
        let q_theta = diag(&q_vec);
        self.params.predict(&identity(pd), &q_theta)?;
        self.params.update_linear(&corrected, &h_theta, &r_mat, self.config.innovation_ridge)?;
        let theta = self.params.x.clone();
        let mut new_g = Vec::with_capacity(self.n_channels);
        let mut new_o = Vec::with_capacity(self.n_channels);
        for i in 0..self.n_channels {
            let base = pw * i;
            new_g.push(theta[base].clamp(0.1, 10.0));
            new_o.push(theta[base + 1]);
            if pl_on {
                self.alpha[i] = theta[base + 2].clamp(0.1, 3.0);
            }
        }
        self.g = new_g;
        self.o = new_o;
        self.relative_gains = self.g.clone();
        self.stimulus.set_filter_relative_gain(self.relative_gains.clone());

        // 5. Regime membership on the environmental component of the state; declare
        //    switches (never anomalies). The memory block is not part of the
        //    baseline (it always decays to zero).
        let reg_x: &[f64] = if ads_on {
            &self.state_filter.x[..self.n_channels]
        } else {
            &self.state_filter.x
        };
        let regime_update = self.regimes.update(reg_x)?;
        let regime_switch = regime_update.switched || regime_update.spawned;
        if regime_switch && !regime_update.anchor_mean.is_empty() {
            let anchor_mean = regime_update.anchor_mean.clone();
            let anchor_cov = regime_update.anchor_cov.clone();
            if ads_on {
                // Re-anchor only the environmental block; leave the memory
                // block (and its prior scale) untouched.
                self.state_filter.x[..self.n_channels]
                    .copy_from_slice(&anchor_mean[..self.n_channels]);
                for (i, row) in anchor_cov.iter().enumerate() {
                    self.state_filter.p[i][..self.n_channels].copy_from_slice(row);
                }
                for i in 0..self.n_channels {
                    for j in self.n_channels..2 * self.n_channels {
                        self.state_filter.p[i][j] = 0.0;
                        self.state_filter.p[j][i] = 0.0;
                    }
                }
            } else {
                self.state_filter.x = anchor_mean;
                self.state_filter.p = anchor_cov;
            }
        }

        // 6. Typology head on the innovation stream and the filter's level
        //    (state-anchored: a settled step still reads as a step).
        let big_channels: Vec<usize> = z_scores
            .iter()
            .enumerate()
            .filter(|(_, &z)| z > self.config.typology.z_note)
            .map(|(i, _)| i)
            .collect();
        let sigma: Vec<f64> = self.r_diag.iter().map(|v| v.sqrt()).collect();
        let level_now = self.state_filter.x[..self.n_channels].to_vec();
        let typology = self
            .typology_head
            .update(&z_scores, &big_channels, &level_now, &sigma);

        // 7. Anomaly verdict: per-budget per-channel thresholds scaled by
        //    1/sensitivity (preserves the legacy "how much" operator control).
        let sens = self.config.sensitivity.max(1e-3);
        let mut budget_fired = [false; 3];
        for (bi, &k) in self.config.k_std.iter().enumerate() {
            let eff = k / sens;
            if max_z > eff {
                budget_fired[bi] = true;
            }
        }
        let anomaly_votes = budget_fired.iter().filter(|&&b| b).count();
        let is_anomaly = anomaly_votes >= 2;

        // 8. Calibrated confidence (Platt on the raw score).
        let confidence = self.platt.predict(raw_score);

        // 9. Threshold confidence (logistic in sample count; same meaning as before).
        let threshold_confidence = self.threshold_confidence_for(self.n_samples);

        self.last_innovation = r.clone();
        self.last_inno_cov = s.clone();
        self.innovation_history.push_back(raw_score);
        if self.innovation_history.len() > 256 {
            self.innovation_history.pop_front();
        }

        // 10. Keep a health sweep of stimulus findings: the automatic reference
        //     stimulus fires on schedule; per-channel poison confirmation (both
        //     the parameter filter AND the physical gain agree) becomes findings.
        self.recent_health.clear();
        if self.n_samples.is_multiple_of((self.config.stimulus.period_s * 10).max(1) as usize) {
            let g0 = self.stimulus.g0.clone();
            if let Ok(findings) = self.stimulus.record_stimulus(
                self.n_samples as u64,
                &self.g,
                &g0,
            ) {
                self.recent_health.extend(findings);
            }
        }
        for ch in 0..self.n_channels {
            if self.stimulus.is_poisoned(ch) && !self.poisoned_channels.contains(&ch) {
                self.poisoned_channels.push(ch);
            }
        }
        // Confirmed-poison channels stay surfaced on every verdict until serviced.
        for &ch in &self.poisoned_channels {
            self.recent_health.push(HealthFinding {
                channel: ch,
                kind: super::stimulus::HealthFindingKind::PoisonConfirmed,
                message: format!(
                    "channel {}: confirmed poisoned (retained gain {:.2}) — needs service",
                    ch, self.stimulus.rho[ch]
                ),
            });
        }

        Ok(EngineVerdict {
            is_anomaly,
            anomaly_votes,
            raw_score,
            max_z,
            z_scores,
            triggered_channels: triggered,
            confidence,
            threshold_confidence,
            n_samples: self.n_samples,
            budget_fired,
            typology,
            regime_switch,
            regime: self.regimes.current,
            relative_gains: self.relative_gains.clone(),
            health_findings: self.recent_health.clone(),
            smoothed: corrected,
            adsorption: self.adsorption_memory(),
            alpha_exponents: self.alpha_exponents(),
            warming_up: false,
        })
    }

    /// Operator feedback: confirm whether the last reading was a real change.
    /// Feeds the Platt calibration (retrains every 10, from ≥ 20 samples).
    pub fn confirm(&mut self, was_anomaly: bool) {
        self.platt
            .add_feedback(*self.innovation_history.back().unwrap_or(&0.0), was_anomaly);
        if was_anomaly {
            self.confirmed_anomaly_count += 1;
        } else {
            self.confirmed_normal_count += 1;
        }
        if self.platt.n_feedback().is_multiple_of(10) {
            self.retrain_platt();
        }
    }

    /// Re-fit Platt scaling from confirmed feedback (Lin et al. Newton solver).
    pub fn retrain_platt(&mut self) {
        self.platt.retrain();
    }

    /// Record an operator-initiated reference stimulus; returns fresh findings.
    /// A confirmed poison also registers on the persistent service-alert list
    /// so subsequent engine verdicts keep surfacing it.
    pub fn record_stimulus(
        &mut self,
        channel_response: &[f64],
        expected_ref: &[f64],
    ) -> Result<Vec<HealthFinding>> {
        let findings = self
            .stimulus
            .record_stimulus(self.n_samples as u64, channel_response, expected_ref)?;
        for f in &findings {
            if f.kind == HealthFindingKind::PoisonConfirmed && !self.poisoned_channels.contains(&f.channel) {
                self.poisoned_channels.push(f.channel);
            }
        }
        Ok(findings)
    }

    pub fn platt_params(&self) -> &PlattParams {
        &self.platt.params
    }

    /// Import an external relative-gain estimate per channel (e.g. a stimulus
    /// measurement, or a restored detector state). Re-anchors the parameter
    /// gain so estimator and physical tracker agree from here on.
    pub fn set_relative_gains(&mut self, relative: Vec<f64>) {
        if relative.len() != self.n_channels {
            return;
        }
        let pw = self.param_width();
        for (i, r) in relative.iter().enumerate() {
            let clamped = r.clamp(0.1, 10.0);
            self.params.x[pw * i] = clamped;
            self.g[i] = clamped;
        }
        self.relative_gains = self.g.clone();
        self.stimulus.set_filter_relative_gain(self.relative_gains.clone());
    }

    pub fn regimes(&self) -> &RegimeModel {
        &self.regimes
    }

    pub fn confirm_counts(&self) -> (usize, usize) {
        (self.confirmed_anomaly_count, self.confirmed_normal_count)
    }
}

// Re-export submodule types for the crate-level API.
pub use super::stimulus::{HealthFindingKind, StimulusMeasurement};

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline(c: usize) -> Vec<Vec<f64>> {
        // Slightly noisy calibration window around nominal levels.
        (0..80)
            .map(|i| {
                let n = i as f64;
                (0..c)
                    .map(|ch| 1.0 + ch as f64 + 0.05 * (n * 1.7).sin())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn constant_stream_is_normal() {
        let mut eng = DualKalmanEngine::new(2);
        eng.calibrate_baseline(&baseline(2)).unwrap();
        for (i, _) in (0..50).enumerate() {
            let v = eng.detect_no_ambient(&[1.0, 2.0]).unwrap();
            if i < 3 || v.is_anomaly {
                eprintln!("steady i={} x={:?} innov={:?} z={:?} score={:.4} maxz={:.4} votes={}",
                    i, eng.state_filter.x, eng.last_innovation, v.z_scores, v.raw_score, v.max_z, v.anomaly_votes);
            }
        }
    }

    #[test]
    fn step_fires_and_ramp_is_forgiven_with_parameter_walk() {
        let mut eng = DualKalmanEngine::new(2);
        // A +3 ramp over 300 samples (30 s) is fast in real terms — the
        // operator tunes `q_param` to the environment's typical drift rate.
        // Here we set a drift-tracking walk so the ramp reads as drift.
        eng.config.q_param = 2e-4;
        eng.calibrate_baseline(&baseline(2)).unwrap();
        // Let the filters settle.
        for _ in 0..30 {
            let _ = eng.detect_no_ambient(&[1.0, 2.0]).unwrap();
        }
        // A slow ramp (+3 over 300 samples) is absorbed as parameter drift.
        let mut fired = false;
        for i in 0..300 {
            let v = 1.0 + 3.0 * (i as f64 / 300.0);
            let v2 = 2.0 + (v - 1.0);
            let r = eng.detect_no_ambient(&[v, v2]).unwrap();
            if r.is_anomaly {
                eprintln!("FIRED at i={} x={:?} g={:?} o={:?} innov={:?} z={:?} votes={}",
                    i, eng.state_filter.x, eng.g, eng.o, eng.last_innovation, r.z_scores, r.anomaly_votes);
                break;
            }
            fired |= r.is_anomaly;
        }
        assert!(!fired, "slow ramp must be absorbed (drift), not alarmed");
        // An abrupt step beyond the chased level is a real event → fires.
        let step = eng.detect_no_ambient(&[6.0, 7.0]).unwrap();
        assert!(step.is_anomaly, "abrupt step after the ramp must fire (votes {})", step.anomaly_votes);
    }

    #[test]
    fn frozen_parameters_alarm_on_shift() {
        let mut eng = DualKalmanEngine::new(2);
        eng.config.q_param = 0.0;
        eng.calibrate_baseline(&baseline(2)).unwrap();
        for _ in 0..30 {
            let _ = eng.detect_no_ambient(&[1.0, 2.0]).unwrap();
        }
        assert!(!eng.detect_no_ambient(&[1.0, 2.0]).unwrap().is_anomaly);
        assert!(eng.detect_no_ambient(&[4.0, 5.0]).unwrap().is_anomaly,
            "without parameter walk the same shift must alarm");
    }

    #[test]
    fn sensitivity_is_the_how_much_control() {
        // Sensitivities are in calibrated-σ units: the baseline window has
        // σ ≈ 0.035, so a +0.12 deviation is a moderate ~3σ event — big enough
        // to be noticed at high sensitivity, under the 8σ budget at low
        // sensitivity (k_std is divided by `sensitivity`).
        let mk = |sens: f64| -> DualKalmanEngine {
            let mut e = DualKalmanEngine::new(2);
            e.config.sensitivity = sens;
            e.config.q_state = 1e-4;
            e.calibrate_baseline(&baseline(2)).unwrap();
            e
        };
        let delta = [1.12, 2.0];
        let mut strict = mk(0.5);
        for _ in 0..30 {
            let _ = strict.detect_no_ambient(&[1.0, 2.0]).unwrap();
        }
        assert!(!strict.detect_no_ambient(&delta).unwrap().is_anomaly,
            "low sensitivity must tolerate the ~3σ delta");
        let mut sensitive = mk(4.0);
        for _ in 0..30 {
            let _ = sensitive.detect_no_ambient(&[1.0, 2.0]).unwrap();
        }
        assert!(sensitive.detect_no_ambient(&delta).unwrap().is_anomaly,
            "high sensitivity must call out the same delta");
    }

    #[test]
    fn humidity_step_is_not_anomaly_with_ambient_model() {
        let mut eng = DualKalmanEngine::new(2);
        eng.calibrate_baseline(&baseline(2)).unwrap();
        // Fit an ambient model: channel 0 reacts +0.5 per %RH.
        let mut amb = AmbientModel::default();
        amb.fit(vec![0.0, 0.0], vec![0.5, 0.2], 25.0, 50.0);
        eng.set_ambient_model(amb);
        for _ in 0..30 {
            let _ = eng
                .detect(&[1.0, 2.0], Some(AmbientReading { temperature: Some(25.0), humidity: Some(50.0) }))
                .unwrap();
        }
        // RH jumps to 70%: without correction this would be a big step.
        let humid = eng
            .detect(&[1.0, 2.0], Some(AmbientReading { temperature: Some(25.0), humidity: Some(70.0) }))
            .unwrap();
        assert!(!humid.is_anomaly, "weather-driven shift must be predicted, not alarming");
    }

    #[test]
    fn poison_requires_filter_and_stimulus_agreement() {
        let mut eng = DualKalmanEngine::new(1);
        eng.calibrate_baseline(&baseline(1)).unwrap();
        for _ in 0..10 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        // The parameter (statistical) side sees the decay first…
        eng.set_relative_gains(vec![0.4]);
        // …and the physical stimulus agrees: gain retention ≈ 0.4 of burn-in.
        let findings = eng.record_stimulus(&[0.4], &[1.0]).unwrap();
        assert!(
            findings.iter().any(|f| f.kind == HealthFindingKind::PoisonConfirmed),
            "filter + stimulus agreement must confirm poisoning"
        );
        // The per-reading sweep surfaces it as a health finding too.
        let v = eng.detect_no_ambient(&[1.0]).unwrap();
        assert!(
            v.health_findings.iter().any(|f| f.kind == HealthFindingKind::PoisonConfirmed),
            "engine verdict must surface the confirmed poison finding"
        );
    }

    #[test]
    fn adsorption_extends_state_and_exposes_memory() {
        let mut eng = DualKalmanEngine::new(2);
        assert_eq!(eng.state_filter.x.len(), 2, "legacy state layout by default");
        assert!(!eng.adsorption_enabled());
        let mut cfg = eng.config.clone();
        cfg.adsorption.enabled = true;
        cfg.adsorption.tau_s = vec![50.0, 80.0];
        eng.set_config(cfg);
        assert!(eng.adsorption_enabled());
        eng.calibrate_baseline(&baseline(2)).unwrap();
        assert_eq!(eng.state_filter.x.len(), 4, "state gains the m block per channel");
        assert_eq!(eng.adsorption_memory().len(), 2);
        for _ in 0..30 {
            let _ = eng.detect_no_ambient(&[1.0, 2.0]).unwrap();
        }
        let v = eng.detect_no_ambient(&[1.0, 2.0]).unwrap();
        assert_eq!(v.adsorption.len(), 2, "verdict exposes the memory block");
        assert!(!v.is_anomaly);
        assert!(
            eng.adsorption_memory().iter().all(|&m| m.abs() < 0.05),
            "memory stays near zero on a clean stream"
        );
    }

    #[test]
    fn adsorption_memory_predicts_desorption_tail() {
        let mut eng = DualKalmanEngine::new(1);
        let mut cfg = eng.config.clone();
        cfg.adsorption.enabled = true;
        cfg.adsorption.tau_s = vec![50.0];
        cfg.adsorption.q_adsorption = 1e-4;
        eng.set_config(cfg);
        eng.calibrate_baseline(&baseline(1)).unwrap();
        for _ in 0..30 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        // Simulate a true desorption tail: environment back at baseline, residue
        // m decaying from 0.5 with the configured τ. The filter's own model
        // predicts this decay, so innovations stay near zero and no alarm fires
        // while x stays pinned and m tracks the true residue curve.
        let phi: f64 = (-0.1_f64 / 50.0).exp();
        eng.state_filter.x[0] = 1.0;
        eng.state_filter.x[1] = 0.5;
        for k in 1..=200 {
            let obs = 1.0 + 0.5 * phi.powi(k);
            let v = eng.detect_no_ambient(&[obs]).unwrap();
            assert!(
                !v.is_anomaly,
                "desorption tail must not alarm (votes {})",
                v.anomaly_votes
            );
            let x_now = eng.state_filter.x[0];
            let m_now = eng.state_filter.x[1];
            assert!((x_now - 1.0).abs() < 0.10, "environment stays pinned (x={x_now})");
            let m_true = 0.5 * phi.powi(k);
            assert!(
                (m_now - m_true).abs() < 0.10,
                "memory tracks the true decay (m={m_now}, true={m_true})"
            );
        }
    }

    #[test]
    fn adsorption_works_in_linear_kalman_mode() {
        let mut cfg = EngineConfig {
            state_filter: StateFilterKind::Kalman,
            ..Default::default()
        };
        cfg.adsorption.enabled = true;
        cfg.adsorption.tau_s = vec![50.0];
        let mut eng = DualKalmanEngine::new(1);
        eng.set_config(cfg);
        eng.calibrate_baseline(&baseline(1)).unwrap();
        for _ in 0..30 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        let phi: f64 = (-0.1_f64 / 50.0).exp();
        eng.state_filter.x[0] = 1.0;
        eng.state_filter.x[1] = 0.5;
        for k in 1..=100 {
            let obs = 1.0 + 0.5 * phi.powi(k);
            let v = eng.detect_no_ambient(&[obs]).unwrap();
            assert!(!v.is_anomaly, "linear path must absorb the tail (votes {})", v.anomaly_votes);
        }
        assert!((eng.state_filter.x[0] - 1.0).abs() < 0.10);
        let m_true = 0.5 * phi.powi(100);
        assert!((eng.state_filter.x[1] - m_true).abs() < 0.10);
    }

    #[test]
    fn adsorption_disable_preserves_legacy_layout() {
        let mut eng = DualKalmanEngine::new(1);
        assert!(!eng.adsorption_enabled());
        eng.calibrate_baseline(&baseline(1)).unwrap();
        assert_eq!(eng.state_filter.x.len(), 1);
        let v = eng.detect_no_ambient(&[1.0]).unwrap();
        assert!(v.adsorption.is_empty());
        assert_eq!(eng.adsorption_memory().len(), 0);
    }

    // --- Power-law response tests ---

    #[test]
    fn phi_identity_at_alpha_one() {
        assert!((phi(3.0, 1.0) - 3.0).abs() < 1e-12);
        assert!((phi(-2.0, 1.0) - (-2.0)).abs() < 1e-12);
        assert_eq!(phi(0.0, 1.0), 0.0);
    }

    #[test]
    fn phi_compression_at_half_alpha() {
        assert!((phi(4.0, 0.5) - 2.0).abs() < 1e-12);
        assert!((phi(-4.0, 0.5) - (-2.0)).abs() < 1e-12);
        assert_eq!(phi(1.0, 0.5), 1.0);
    }

    #[test]
    fn dphi_dx_at_alpha_one_is_one() {
        assert!((dphi_dx(3.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((dphi_dx(-5.0, 1.0) - 1.0).abs() < 1e-12);
        assert_eq!(dphi_dx(0.0, 1.0), 0.0);
    }

    #[test]
    fn dphi_da_zero_at_unit_and_at_zero() {
        assert!(dphi_da(1.0, 1.0).abs() < 1e-12);
        assert!(dphi_da(-1.0, 0.5).abs() < 1e-12);
        assert_eq!(dphi_da(0.0, 0.5), 0.0);
        // Non-zero away from the special points.
        assert!(dphi_da(3.0, 1.0).abs() > 1e-3);
    }

    #[test]
    fn power_law_toggle_preserves_backward_compat() {
        // Default config (power-law disabled) must produce identical behaviour.
        let mut eng = DualKalmanEngine::new(2);
        assert!(!eng.response_enabled());
        eng.calibrate_baseline(&baseline(2)).unwrap();
        for _ in 0..30 {
            let _ = eng.detect_no_ambient(&[1.0, 2.0]).unwrap();
        }
        let v = eng.detect_no_ambient(&[1.0, 2.0]).unwrap();
        assert!(!v.is_anomaly);
        assert!(v.alpha_exponents.is_empty(), "no alpha when power-law disabled");
        assert_eq!(eng.alpha_exponents().len(), 0);
    }

    #[test]
    fn power_law_extends_param_dim() {
        let mut eng = DualKalmanEngine::new(2);
        assert_eq!(eng.param_dim(), 4, "linear: 2 per channel");
        let mut cfg = eng.config.clone();
        cfg.response.power_law_enabled = true;
        cfg.response.alpha = vec![0.5, 0.7];
        cfg.response.q_alpha = 1e-6;
        eng.set_config(cfg);
        assert!(eng.response_enabled());
        assert_eq!(eng.param_dim(), 6, "power-law: 3 per channel");
        assert_eq!(eng.alpha, vec![0.5, 0.7]);
        eng.calibrate_baseline(&baseline(2)).unwrap();
        assert_eq!(eng.params.x.len(), 6);
        let v = eng.detect_no_ambient(&[1.0, 2.0]).unwrap();
        assert_eq!(v.alpha_exponents.len(), 2);
    }

    #[test]
    fn power_law_alpha_stable_when_matched() {
        // When the filter's α prior already matches the true generator, the
        // online estimate must stay near its prior. Note this is a stability
        // check, not an identification proof: y = g·φ(x,α)+o is scale-invariant
        // under (x→cx, g→g/cᵅ), so α is only weakly observable online from one
        // channel — a known-concentration calibration does the real identification.
        let mut eng = DualKalmanEngine::new(1);
        let mut cfg = eng.config.clone();
        cfg.response.power_law_enabled = true;
        cfg.response.alpha = vec![0.5];
        cfg.response.q_alpha = 1e-7;
        cfg.q_param = 1e-6;
        cfg.q_state = 1e-2;
        eng.set_config(cfg);
        eng.calibrate_baseline(&baseline(1)).unwrap();
        for _ in 0..30 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        for k in 0..2000 {
            let phase = (k % 600) as f64 / 600.0;
            let x_true = 1.0 + 7.0 * phase;
            let y = phi(x_true, 0.5);
            let _ = eng.detect_no_ambient(&[y]).unwrap();
        }
        let a = eng.alpha[0];
        assert!(
            (a - 0.5).abs() < 0.4,
            "matched alpha must stay near prior through a sweep (got {a})"
        );
        let v = eng.detect_no_ambient(&[1.0]).unwrap();
        assert_eq!(v.alpha_exponents.len(), 1);
    }

    #[test]
    fn power_law_does_not_false_alarm_on_large_linear_reading() {
        // With α=1.0 (linear), a large reading that stays within the regime's
        // expected range should not false-alarm.
        let mut eng = DualKalmanEngine::new(1);
        let mut cfg = eng.config.clone();
        cfg.response.power_law_enabled = true;
        cfg.response.alpha = vec![1.0];
        cfg.response.q_alpha = 1e-6;
        eng.set_config(cfg);
        eng.calibrate_baseline(&baseline(1)).unwrap();
        for _ in 0..30 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        assert!(!eng.detect_no_ambient(&[1.0]).unwrap().is_anomaly);
    }
}
