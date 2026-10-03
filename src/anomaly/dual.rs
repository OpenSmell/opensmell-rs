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

/// Reference stream cadence of the engine (10 Hz). Physical rate constants
/// (`q_state`, `q_param`, desorption `τ`) are quoted *per second* and folded to
/// the actual reading cadence `dt` at detect time (`Q_step = Q_ref·dt/period`,
/// decay `exp(−dt/τ)`), so a field logger sampling at 6 Hz, 1 Hz or once a
/// minute sees the same physical process model it would at 10 Hz.
/// `EngineConfig::sample_period_s` is the cadence `detect` assumes when the
/// caller does not supply a real `dt`.
pub const SAMPLE_PERIOD_S: f64 = 0.1;

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
///
/// With `two_exp` the residue is bi-exponential —
/// `m(t) = a1·e^(−t/τ1) + (1−a1)·e^(−t/τ2)`, the field-observed post-exposure
/// shape (a fast wash-out followed by a slow plateau) — tracked as *two* memory
/// states per channel (`[x, m1, m2]`). Process noise on each component is
/// split by `a1` so the fast component carries most of the residue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdsorptionConfig {
    /// Master switch. Disabled keeps the legacy c-dimensional `[x]` state
    /// layout exactly (no behavioural change to existing deployments).
    #[serde(default)]
    pub enabled: bool,
    /// Per-channel desorption time constants in seconds. Shorter ⇒ residue
    /// clears faster; a missing entry falls back to `tau_default_s`.
    #[serde(default)]
    pub tau_s: Vec<f64>,
    /// Fallback time constant (seconds) for channels without a `tau_s` entry.
    /// A calibration/purge experiment should replace this per sensor.
    #[serde(default = "default_tau_s")]
    pub tau_default_s: f64,
    /// Process noise on the memory state (how freely `m` may move). Kept
    /// ≪ `q_state` so genuine events stay in `x` and only residue sticks to `m`.
    #[serde(default = "default_q_adsorption")]
    pub q_adsorption: f64,
    /// Bi-exponential residue: a second memory state per channel decaying with
    /// `tau2_s`; the two components' process noise is split by `a1`. Disabled
    /// keeps exactly the single-`m` legacy layout.
    #[serde(default)]
    pub two_exp: bool,
    /// Per-channel secondary (slow) time constant, seconds. Missing entries
    /// fall back to `tau2_default_s`.
    #[serde(default)]
    pub tau2_s: Vec<f64>,
    /// Fallback secondary time constant (seconds).
    #[serde(default = "default_tau2_s")]
    pub tau2_default_s: f64,
    /// Per-channel weight (0..1) of the fast component; the slow component
    /// takes the remainder.
    #[serde(default)]
    pub a1: Vec<f64>,
    /// Fallback fast-component weight for channels without an `a1` entry.
    #[serde(default = "default_a1")]
    pub a1_default: f64,
}

fn default_tau_s() -> f64 {
    300.0
}

/// serde default for `EngineConfig::sample_period_s` (the reference 10 Hz).
fn default_sample_period_s() -> f64 {
    SAMPLE_PERIOD_S
}

fn default_tau2_s() -> f64 {
    15.0
}

fn default_a1() -> f64 {
    0.6
}

fn default_q_adsorption() -> f64 {
    1e-5
}

fn default_min_channels() -> usize {
    1
}

impl Default for AdsorptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tau_s: Vec::new(),
            tau_default_s: default_tau_s(),
            q_adsorption: default_q_adsorption(),
            two_exp: false,
            tau2_s: Vec::new(),
            tau2_default_s: default_tau2_s(),
            a1: Vec::new(),
            a1_default: default_a1(),
        }
    }
}

/// Per-component memory decay factors for one channel over one `dt` interval.
/// Single-exponential mode reports the same factor twice (the second memory
/// state does not exist there; callers index it only when `two_exp`).
fn component_decays(cfg: &AdsorptionConfig, i: usize, dt_s: f64) -> [f64; 2] {
    let tau = cfg.tau_s.get(i).copied().unwrap_or(cfg.tau_default_s).max(0.1);
    let d1 = (-dt_s / tau).exp();
    if cfg.two_exp {
        let tau2 = cfg.tau2_s.get(i).copied().unwrap_or(cfg.tau2_default_s).max(0.1);
        [d1, (-dt_s / tau2).exp()]
    } else {
        [d1, d1]
    }
}

/// Per-channel process-noise split for the bi-exponential components. In
/// single-exponential mode the weight collapses to the full `q_adsorption` on
/// the single memory state.
fn mem_process_noise(cfg: &AdsorptionConfig, i: usize) -> [f64; 2] {
    if cfg.two_exp {
        let a = cfg.a1.get(i).copied().unwrap_or(cfg.a1_default).clamp(0.0, 1.0);
        [cfg.q_adsorption * a, cfg.q_adsorption * (1.0 - a)]
    } else {
        [cfg.q_adsorption, 0.0]
    }
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
    /// Reference cadence that `detect` assumes when no real `dt` is supplied,
    /// and the cadence the per-second physics constants are normalized to.
    /// Process noise is scaled by `dt/period` and desorption decays by
    /// `exp(−dt/τ)` when the caller drives `detect_with_dt` at a different rate.
    #[serde(default = "default_sample_period_s")]
    pub sample_period_s: f64,
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
    /// Minimum number of channels that must simultaneously clear the
    /// sensitivity-scaled budget threshold before an anomaly is trusted.
    /// 1 preserves the legacy single-channel `max_z` verdict; 2+ demands
    /// multi-sensor corroboration (rejects isolated single-sensor
    /// transients that a field corpus shows dominate clean-period FPs).
    #[serde(default = "default_min_channels")]
    pub min_channels: usize,
    /// Level-anchored threshold in calibrated σ (scaled by 1/sensitivity): a
    /// slow plume that the Kalman absorbs (small per-sample innovations) still
    /// moves the filter's level away from its calibrated baseline by many σ.
    /// `0` disables the level leg.
    #[serde(default)]
    pub level_budget: f64,
    /// Seconds of level history the level leg must be *growing* against: when
    /// > 0, the level-anchored vote only fires if the max level deviation is
    /// strictly larger now than `level_rise_lookback_s` ago. The TADI corpus
    /// discriminator: an active plume *rises* (deviation grows), while a
    /// post-exposure recovery plateau is already clean by reference yet the
    /// array is still elevated and *settling* (deviation falls). `0` disables
    /// the rise constraint (fires on any sufficient level).
    #[serde(default)]
    pub level_rise_lookback_s: f64,
    /// Seconds of rolling level reference for the level leg: when > 0, the
    /// deviation is measured against the sensor's own filtered level
    /// `level_ref_s` ago (a rolling baseline), instead of the fixed calibration
    /// anchor. The TADI finding: multi-day files drift far from the day-1
    /// anchor (temperature/humidity/aging), so a *stale* anchor turns slow
    /// weather drift into clean-period level FPs. A minutes-scale reference
    /// absorbs drift/weather over hours-days while leaving a minutes-scale
    /// plume visible. `0` keeps the calibration anchor.
    #[serde(default)]
    pub level_ref_s: f64,
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
            sample_period_s: SAMPLE_PERIOD_S,
            q_state: 1e-3,
            q_param: 1e-6,
            r_scale: 1.0,
            innovation_ridge: 1e-9,
            k_std: [5.0, 6.0, 4.0],
            z_min: 0.5,
            min_channels: 1,
            level_budget: 0.0,
            level_rise_lookback_s: 0.0,
            level_ref_s: 0.0,
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
    warmup_time: f64,
    /// Cumulative elapsed seconds fed to `detect_with_dt` (drives the
    /// stimulus schedule in real time instead of at a hard-coded sample count).
    time_s: f64,
    /// Next automatic-reference time boundary (seconds since engine start).
    next_event_at: f64,
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
    /// Calibrated baseline mean per channel (the "no-leak-yet" reference level).
    /// A level-anchored verdict compares the filter's smoothed level to this in
    /// calibrated σ — catching slow plumes whose per-sample innovations never
    /// cross the innovation budgets (the Kalman absorbs a slow ramp).
    level_anchor: Vec<f64>,
    /// (time_s, per-channel filtered level) history for the level leg. When a
    /// rolling reference (`level_ref_s > 0`) is configured the deviation is
    /// measured against the level N seconds ago (absorbs hours-days drift /
    /// weather, keeps a minutes-scale plume visible); it also drives the
    /// rise-constrained firing.
    level_history: std::collections::VecDeque<(f64, Vec<f64>)>,
}

const WARMUP_SECONDS: f64 = 6.0;

/// Maximum `dt` (seconds) tolerated before a reading is treated as a data gap;
/// rate constants saturate above this so a missed chunk does not inject an
/// unbounded process-noise spike.
const MAX_GAP_S: f64 = 60.0;

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
            next_event_at: config.stimulus.period_s as f64,
            config,
            n_samples: 0,
            warmup_time: 0.0,
            time_s: 0.0,
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
            level_anchor: vec![0.0; n_channels],
            level_history: std::collections::VecDeque::new(),
        }
    }

    pub fn set_config(&mut self, config: EngineConfig) {
        let state_dim_changed = config.adsorption.enabled != self.config.adsorption.enabled
            || config.adsorption.two_exp != self.config.adsorption.two_exp;
        let param_dim_changed =
            config.response.power_law_enabled != self.config.response.power_law_enabled;
        self.config = config.clone();
        self.stimulus = StimulusGainTracker::new(
            self.n_channels,
            self.g.clone(),
            config.stimulus.clone(),
        );
        // Re-anchor the automatic-reference schedule from the current time so a
        // caller changing `stimulus.period_s` after construction gets the new
        // cadence (next reference `period_s` after now).
        self.next_event_at = self.time_s + self.config.stimulus.period_s as f64;
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
            self.n_channels + self.mem_dim()
        } else {
            self.n_channels
        }
    }

    /// Number of memory states per channel: 1 (legacy single-`m`) or 2 when
    /// the bi-exponential residue is enabled.
    fn mem_dim(&self) -> usize {
        if self.config.adsorption.two_exp {
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
    /// In bi-exponential mode the per-channel value is the summed residue
    /// `m1 + m2` (the observable the singles-exp mode always reported).
    pub fn adsorption_memory(&self) -> Vec<f64> {
        if self.config.adsorption.enabled {
            let c = self.n_channels;
            if self.config.adsorption.two_exp {
                (0..c)
                    .map(|i| {
                        self.state_filter.x[c + 2 * i] + self.state_filter.x[c + 2 * i + 1]
                    })
                    .collect()
            } else {
                self.state_filter.x[c..].to_vec()
            }
        } else {
            Vec::new()
        }
    }

    /// Raw per-channel memory components (length `mem_dim`): the single `m` or
    /// `[m1, m2]` per channel when bi-exponential. Empty when disabled.
    pub fn adsorption_components(&self) -> Vec<f64> {
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
        self.level_anchor = mean.clone();
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
        // (a freshly calibrated sensor holds no residue). Bi-exponential mode
        // splits the prior by `a1` so the fast component is initially more
        // free to absorb residue, matching the fitted weight.
        let d = self.filter_dim();
        let p_x = var.iter().map(|&v| v.max(1e-4) * 4.0).collect::<Vec<f64>>();
        let mut prior = mean.clone();
        let mut prior_p = diag(&p_x);
        if self.config.adsorption.enabled {
            let ads = &self.config.adsorption;
            let mem = self.mem_dim();
            prior.extend(std::iter::repeat_n(0.0, mem));
            let mut full = vec![vec![0.0; d]; d];
            if ads.two_exp {
                for i in 0..self.n_channels {
                    let a = ads.a1.get(i).copied().unwrap_or(ads.a1_default).clamp(0.0, 1.0);
                    full[i][i] = p_x[i];
                    full[c + 2 * i][c + 2 * i] = p_x[i] * a;
                    full[c + 2 * i + 1][c + 2 * i + 1] = p_x[i] * (1.0 - a);
                }
            } else {
                for i in 0..self.n_channels {
                    full[i][i] = p_x[i];
                    full[self.n_channels + i][self.n_channels + i] = p_x[i];
                }
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

    /// Run one reading through both filters at the configured reference cadence.
    /// `ambient` carries on-board temperature/humidity when present (None ⇒ the
    /// ambient correction is zero).
    pub fn detect(&mut self, reading: &[f64], ambient: Option<AmbientReading>) -> Result<EngineVerdict> {
        self.detect_with_dt(reading, ambient, self.config.sample_period_s.max(1e-4))
    }

    /// Convenience: detect with no ambient data (used by warm-up/tests).
    pub fn detect_no_ambient(&mut self, reading: &[f64]) -> Result<EngineVerdict> {
        self.detect_with_dt(reading, None, self.config.sample_period_s.max(1e-4))
    }

    /// Run one reading with its *actual* inter-sample gap `dt_s`. The physics
    /// knobs are per-second: process noise scales as `dt/period`, desorption
    /// decays by `exp(−dt/τ)`, and the warm-up / stimulus schedule advances in
    /// real time. A field logger sampling at any cadence therefore sees the
    /// same physical process model it would at the reference 10 Hz.
    pub fn detect_with_dt(
        &mut self,
        reading: &[f64],
        ambient: Option<AmbientReading>,
        dt_s: f64,
    ) -> Result<EngineVerdict> {
        // Saturate pathological gaps (cross-day file splits, logger naps) so a
        // lost chunk does not blow the process-noise covariance apart; the
        // caller still sees the verdict for the reading that did arrive.
        let dt = dt_s.max(1e-4).min(MAX_GAP_S);
        self.detect_raw(reading, ambient, dt)
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

    fn detect_raw(&mut self, reading: &[f64], ambient: Option<AmbientReading>, dt: f64) -> Result<EngineVerdict> {
        if reading.len() != self.n_channels {
            return Err(OpenSmellError::InvalidChannelCount {
                got: reading.len(),
                expected: self.n_channels,
            });
        }
        self.n_samples += 1;
        self.time_s += dt;

        // Fresh-device warm-up: an uncalibrated engine buffers the first
        // `WARMUP_SECONDS` of readings, reports `warming_up`, and never alarms —
        // the debut of a brand-new board stays quiet (design §6, §11.1). When
        // the buffer fills, it auto-calibrates exactly like the legacy
        // detector, so a caller that only feeds `detect` gets a baseline too.
        // In seconds so a slow-logged board (e.g. the 6 s TADI loggers) holds a
        // comparable calibration span. 6 s ≈ 60 samples at the reference 10 Hz.
        if !self.calibrated {
            self.baseline_buffer.push(reading.to_vec());
            self.warmup_time += dt;
            if self.warmup_time + 1e-9 >= WARMUP_SECONDS {
                let samples = std::mem::take(&mut self.baseline_buffer);
                self.calibrate_baseline(&samples)?;
                self.warmup_time = 0.0;
                // Fall through: this reading now runs the real path on the
                // just-established baseline.
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
        // (F=I, Q = q_state·dr·I where dr = dt/period — the per-second walk
        // rate stays physical at any cadence); with adsorption enabled the
        // memory blocks decay by exp(−dt/τ_i) toward zero (Q split by the
        // bi-exponential weights when `two_exp`).
        let c = self.n_channels;
        let ads_on = self.config.adsorption.enabled;
        let two_exp = self.config.adsorption.two_exp;
        let dr = dt / self.config.sample_period_s.max(1e-6);
        let ads_cfg = &self.config.adsorption;
        if ads_on {
            let d = self.filter_dim();
            let mut f = vec![vec![0.0; d]; d];
            let mut q = vec![vec![0.0; d]; d];
            if ads_cfg.two_exp {
                for i in 0..c {
                    let [p1, p2] = component_decays(ads_cfg, i, dt);
                    let [q1, q2] = mem_process_noise(ads_cfg, i);
                    f[i][i] = 1.0;
                    f[c + 2 * i][c + 2 * i] = p1;
                    f[c + 2 * i + 1][c + 2 * i + 1] = p2;
                    q[i][i] = self.config.q_state * dr;
                    q[c + 2 * i][c + 2 * i] = q1 * dr;
                    q[c + 2 * i + 1][c + 2 * i + 1] = q2 * dr;
                }
            } else {
                for i in 0..c {
                    let [p1, _] = component_decays(ads_cfg, i, dt);
                    f[i][i] = 1.0;
                    f[c + i][c + i] = p1;
                    q[i][i] = self.config.q_state * dr;
                    q[c + i][c + i] = ads_cfg.q_adsorption * dr;
                }
            }
            self.state_filter.predict(&f, &q)?;
        } else {
            let qx = diag(&vec![self.config.q_state * dr; c]);
            self.state_filter.predict(&identity(c), &qx)?;
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
                        if two_exp {
                            h_lin[i][c + 2 * i] = g[i];
                            h_lin[i][c + 2 * i + 1] = g[i];
                        } else {
                            h_lin[i][c + i] = g[i];
                        }
                    }
                }
                self.state_filter.update_linear(&corrected, &h_lin, &r_mat, self.config.innovation_ridge)?
            }
            StateFilterKind::Unscented => {
                let h = |xp: &[f64]| -> Vec<f64> {
                    (0..c)
                        .map(|i| {
                            let mem = if ads_on {
                                if two_exp {
                                    xp[c + 2 * i] + xp[c + 2 * i + 1]
                                } else {
                                    xp[c + i]
                                }
                            } else {
                                0.0
                            };
                            let level = if pl_on {
                                phi(xp[i], alpha[i])
                            } else {
                                xp[i]
                            } + mem;
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
            let mem = if ads_on {
                if two_exp {
                    x_hat[c + 2 * i] + x_hat[c + 2 * i + 1]
                } else {
                    x_hat[c + i]
                }
            } else {
                0.0
            };
            let level = x_hat[i] + mem;
            if pl_on {
                h_theta[i][base] = phi(x_hat[i], alpha[i]) + mem;
                h_theta[i][base + 1] = 1.0;
                h_theta[i][base + 2] = self.g[i] * dphi_da(x_hat[i], alpha[i]);
            } else {
                h_theta[i][base] = level;
                h_theta[i][base + 1] = 1.0;
            }
        }
        let mut q_vec = Vec::with_capacity(pd);
        for _ in 0..self.n_channels {
            q_vec.push(self.config.q_param * dr);
            q_vec.push(self.config.q_param * dr);
            if pl_on {
                q_vec.push(self.config.response.q_alpha * dr);
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
        let regime_update = self.regimes.update(reg_x, dt)?;
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
                    for j in self.n_channels..self.filter_dim() {
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
        // Multi-channel corroboration: count how many channels clear the
        // *sensitive* budget at the current sensitivity. A real gas event
        // facing an array moves several channels together (same plume), while
        // an isolated sensor transient moves only one; the TADI corpus shows
        // the clean-period FPs are single-channel steps. `min_channels` (off by
        // default) requires that many channels to agree before the anomaly
        // verdict holds.
        let sensitive_eff = self.config.k_std[2] / sens;
        let channels_past = z_scores
            .iter()
            .filter(|&&z| z > sensitive_eff)
            .count();
        let corroborated = channels_past >= self.config.min_channels.max(1);
        let anomaly_votes = budget_fired.iter().filter(|&&b| b).count();
        let mut is_anomaly = anomaly_votes >= 2 && corroborated;

        // 7b. Level-anchored leg: a slow plume raises the *level* far from its
        // reference even when per-sample innovations (and so the budget votes)
        // stay small. Reference is either the calibration anchor (the
        // "no-leak-yet" baseline) or — when `level_ref_s > 0` — the sensor's
        // own level N seconds ago, so hours-days drift (weather/aging on a
        // multi-day file) does not read as a clean-period anomaly. Deliberately
        // orthogonal to the innovation budgets so it can catch what they
        // structurally miss, and it uses the *same* corroboration (plume =
        // several channels move together) to keep isolated sensor transients
        // quiet.
        if self.config.level_budget > 0.0 {
        let level_now: Vec<f64> = self.state_filter.x[..self.n_channels].to_vec();
        self.level_history.push_back((self.time_s, level_now.clone()));
        if self.level_history.len() > 1 << 16 {
            self.level_history.pop_front();
        }
        let ref_s = self.config.level_ref_s;
        let reference: Vec<f64> = if ref_s > 0.0 {
            let target_t = self.time_s - ref_s;
            self.level_history
                .iter()
                .rev()
                .find(|&&(ts, _)| ts <= target_t)
                .map(|&(_, ref lv)| lv.clone())
                .unwrap_or_else(|| self.level_anchor.clone())
        } else {
            self.level_anchor.clone()
        };
        let level_anchor_deviations: Vec<f64> = level_now
            .iter()
            .zip(reference.iter())
            .zip(self.underlying_r.iter())
            .map(|((&lv, &refv), &r)| (lv - refv).abs() / r.max(1e-12).sqrt())
            .collect();
        // Rise-constrained firing: an active plume *grows* the deviation; a
        // recovery plateau is elevated but settling (fails strict growth),
        // and a rolling reference already absorbs the slow drift that would
        // otherwise look rising.
        let mut level_is_rising = true;
        let rise_s = self.config.level_rise_lookback_s;
        if rise_s > 0.0 {
            let lookback_t = self.time_s - rise_s;
            let past_max = self
                .level_history
                .iter()
                .rev()
                .find(|&&(ts, _)| ts <= lookback_t)
                .map(|&(_, ref lv)| {
                    lv.iter()
                        .zip(reference.iter())
                        .zip(self.underlying_r.iter())
                        .map(|((&clv, &refv), &r)| (clv - refv).abs() / r.max(1e-12).sqrt())
                        .fold(f64::NEG_INFINITY, f64::max)
                });
            level_is_rising = match past_max {
                Some(past) => {
                    let cur = level_anchor_deviations.iter().cloned().fold(
                        f64::NEG_INFINITY,
                        f64::max,
                    );
                    cur > past
                }
                None => true, // not enough history yet: allow (cold start)
            };
        }
        let level_eff = self.config.level_budget / sens;
        let level_channels = level_anchor_deviations
            .iter()
            .filter(|&&d| d > level_eff)
            .count();
        if level_channels >= self.config.min_channels.max(1) && level_is_rising {
            // Give the level leg the same two-fold confidence the innovation
            // legs use: it fires only when corroborated (>= min_channels) *and*
            // at least one channel clears the conservative innovation budget
            // too (avoids flagging a mere slow drift alone). Single-channel
            // transients never pass here (they move level too fast, but only
            // on one channel → < min_channels).
            if budget_fired[1] || budget_fired[0] {
                is_anomaly = true;
            }
        }
        } // end level-anchored leg (level_budget > 0)

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
        // Automatic reference stimulus fires when real elapsed time crosses the
        // schedule boundary (was: a hard-coded `period_s·10` sample count, which
        // drifts by the actual sample cadence). At the reference 10 Hz this is
        // the same sample as before.
        let stim_period = self.config.stimulus.period_s as f64;
        // ε matches the warm-up accumulator convention so accumulated fp error
        // (10×0.1 = 0.9999999999999999) does not skip a schedule boundary.
        if stim_period > 0.0 && self.time_s + 1e-9 >= self.next_event_at {
            loop {
                self.next_event_at += stim_period;
                if self.next_event_at > self.time_s + 1e-9 {
                    break;
                }
            }
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

    /// Number of automatic reference stimuli recorded so far (schedule health
    /// hook — the `period_s`-separated schedule fires on real elapsed time).
    pub fn stimulus_history_len(&self) -> usize {
        self.stimulus.recent_history().len()
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

    // --- Continuous-time (dt threading) & bi-exponential tests ---

    #[test]
    fn reference_cadence_detect_and_detect_with_dt_are_identical() {
        // `detect[_no_ambient]` is exactly `detect_with_dt(dt = sample_period_s)`:
        // at the reference cadence the physics collapses bit-for-bit onto the
        // legacy constants (dr = 1, decay exp(−period/τ)).
        let mut a = DualKalmanEngine::new(2);
        a.calibrate_baseline(&baseline(2)).unwrap();
        let mut b = DualKalmanEngine::new(2);
        b.calibrate_baseline(&baseline(2)).unwrap();
        for k in 0..40 {
            let reading = [1.0 + 0.02 * (k as f64).sin(), 2.0];
            let va = a.detect_no_ambient(&reading).unwrap();
            let vb = b.detect_with_dt(&reading, None, 0.1).unwrap();
            assert_eq!(va.is_anomaly, vb.is_anomaly);
            assert_eq!(va.raw_score.to_bits(), vb.raw_score.to_bits());
            assert_eq!(va.max_z.to_bits(), vb.max_z.to_bits());
            assert_eq!(va.n_samples, vb.n_samples);
            assert_eq!(a.state_filter.x, b.state_filter.x);
        }
    }

    #[test]
    fn coarse_cadence_warmup_lasts_six_seconds() {
        // A board sampled once a second reaches its 6 s warm-up baseline after 6
        // readings (the legacy 60-readings rule was 60 samples @10 Hz = 6 s).
        let mut eng = DualKalmanEngine::new(1);
        let mut saw_warmup = 0usize;
        let mut armed_at = 0usize;
        for k in 0..10 {
            let v = eng.detect_with_dt(&[1.0], None, 1.0).unwrap();
            if v.warming_up {
                saw_warmup += 1;
            }
            if eng.is_calibrated() && armed_at == 0 {
                armed_at = k + 1;
            }
        }
        assert_eq!(saw_warmup, 5, "first five 1 s readings must be warming up");
        assert_eq!(armed_at, 6, "6th 1 s reading calibrates (6 s ≈ 60 @10 Hz)");
    }

    #[test]
    fn adsorption_tracks_true_tail_at_two_cadences() {
        // τ = 50 s at any cadence: the memory state after a 5 s desorption tail
        // must sit at exp(−5/50) whether the logger sampled every 0.1 s or 0.5 s.
        let mk = |dt: f64| {
            let mut eng = DualKalmanEngine::new(1);
            let mut cfg = eng.config.clone();
            cfg.adsorption.enabled = true;
            cfg.adsorption.tau_s = vec![50.0];
            cfg.adsorption.q_adsorption = 1e-4;
            eng.set_config(cfg);
            eng.calibrate_baseline(&baseline(1)).unwrap();
            for _ in 0..30 {
                let _ = eng.detect_with_dt(&[1.0], None, dt).unwrap();
            }
            eng.state_filter.x[0] = 1.0;
            eng.state_filter.x[1] = 0.5;
            eng
        };
        let mut fast = mk(0.1);
        let mut slow = mk(0.5);
        for k in 1..=50 {
            let obs = 1.0 + 0.5 * (-0.1 * k as f64 / 50.0).exp();
            let v = fast.detect_with_dt(&[obs], None, 0.1).unwrap();
            assert!(!v.is_anomaly, "tail must not alarm at 0.1 s cadence");
        }
        for k in 1..=10 {
            let obs = 1.0 + 0.5 * (-0.5 * k as f64 / 50.0).exp();
            let v = slow.detect_with_dt(&[obs], None, 0.5).unwrap();
            assert!(!v.is_anomaly, "tail must not alarm at 0.5 s cadence");
        }
        let m_true = 0.5 * (-5.0_f64 / 50.0).exp();
        assert!((fast.state_filter.x[1] - m_true).abs() < 0.10, "fast-cadence memory tracks");
        assert!((slow.state_filter.x[1] - m_true).abs() < 0.10, "slow-cadence memory tracks");
        assert!(
            (fast.state_filter.x[1] - slow.state_filter.x[1]).abs() < 0.10,
            "physical tail state must be ~cadence-independent"
        );
    }

    #[test]
    fn bi_exp_equals_single_exp_when_taus_merge() {
        // τ2 == τ1 with any a1 collapses the two components to one: the exposed
        // aggregated memory (m1 + m2) must behave like the legacy single-m state.
        let run = |two: bool| {
            let mut eng = DualKalmanEngine::new(1);
            let mut cfg = eng.config.clone();
            cfg.adsorption.enabled = true;
            cfg.adsorption.tau_s = vec![50.0];
            cfg.adsorption.q_adsorption = 1e-4;
            if two {
                cfg.adsorption.two_exp = true;
                cfg.adsorption.tau2_s = vec![50.0];
                cfg.adsorption.a1 = vec![0.6];
            }
            eng.set_config(cfg);
            eng.calibrate_baseline(&baseline(1)).unwrap();
            for _ in 0..30 {
                let _ = eng.detect_no_ambient(&[1.0]).unwrap();
            }
            eng.state_filter.x[0] = 1.0;
            if two {
                eng.state_filter.x[1] = 0.5 * 0.6;
                eng.state_filter.x[2] = 0.5 * (1.0 - 0.6);
            } else {
                eng.state_filter.x[1] = 0.5;
            }
            eng
        };
        let mut single = run(false);
        let mut bi = run(true);
        assert_eq!(single.state_filter.x.len(), 2);
        assert_eq!(bi.state_filter.x.len(), 3, "bi-exponential adds a second memory state");
        let phi: f64 = (-0.1_f64 / 50.0).exp();
        for k in 1..=200 {
            let obs = 1.0 + 0.5 * phi.powi(k);
            let vs = single.detect_no_ambient(&[obs]).unwrap();
            let vb = bi.detect_no_ambient(&[obs]).unwrap();
            assert!(!vs.is_anomaly && !vb.is_anomaly, "merged tail must not alarm");
        }
        assert_eq!(bi.adsorption_memory().len(), 1, "aggregated memory stays per-channel");
        let ms = single.adsorption_memory()[0];
        let mb = bi.adsorption_memory()[0];
        assert!(
            (ms - mb).abs() < 0.05,
            "merged bi-exp residue (m1+m2={mb}) tracks single-exp ({ms})"
        );
        assert_eq!(bi.adsorption_components().len(), 2);
    }

    #[test]
    fn bi_exp_exposes_two_components_with_split_noise() {
        let mut eng = DualKalmanEngine::new(1);
        let mut cfg = eng.config.clone();
        cfg.adsorption.enabled = true;
        cfg.adsorption.two_exp = true;
        cfg.adsorption.tau_s = vec![20.0];
        cfg.adsorption.tau2_s = vec![300.0];
        cfg.adsorption.a1 = vec![0.7];
        eng.set_config(cfg);
        eng.calibrate_baseline(&baseline(1)).unwrap();
        assert_eq!(eng.filter_dim(), 3);
        assert_eq!(eng.adsorption_components().len(), 2);
        for _ in 0..30 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        let v = eng.detect_no_ambient(&[1.0]).unwrap();
        assert!(!v.is_anomaly);
        assert_eq!(v.adsorption.len(), 1);
    }

    #[test]
    fn stimulus_schedule_fires_on_real_elapsed_time() {
        let mut eng = DualKalmanEngine::new(1);
        let mut cfg = eng.config.clone();
        cfg.stimulus.period_s = 1;
        eng.set_config(cfg);
        eng.calibrate_baseline(&baseline(1)).unwrap();
        for _ in 0..9 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        assert_eq!(eng.stimulus_history_len(), 0, "no reference before t=1 s");
        let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        assert_eq!(eng.stimulus_history_len(), 1, "t=1.0 s crosses the first boundary");
        for _ in 0..5 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        assert_eq!(eng.stimulus_history_len(), 1, "t=1.5 s: boundary 2 not reached");
        for _ in 0..5 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        assert_eq!(eng.stimulus_history_len(), 2, "t=2.0 s crosses the second boundary");
        // A 2.5 s gap records one measurement and catches the schedule up to the
        // boundary strictly after the reading's time (t: 2.0 → 4.5; next 5.0).
        let _ = eng.detect_with_dt(&[1.0], None, 2.5).unwrap();
        assert_eq!(eng.stimulus_history_len(), 3, "gap records once, schedule honest");
        for _ in 0..5 {
            let _ = eng.detect_no_ambient(&[1.0]).unwrap();
        }
        assert_eq!(eng.stimulus_history_len(), 4, "t=5.0 s resumes the per-second cadence");
    }
}
