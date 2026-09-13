# Anomaly Engine Design — Dual State/Parameter Kalman

**Status.** Wave-1 design document. This is the spec Wave 2 implements in
`opensmell-rs`. Companion documents: `master-architecture.md` (roadmap,
ontology) and `anomaly-detection.md` (the current shipped stack this replaces —
keep it as the record of what was shipped and why it is being retired).

The engineer implementing this doc needs nothing else: the math, the data
flow, the interfaces, the configuration, and the tests are all specified here.
Where the current code has a relevant hook, the file and symbol are named.

---

## 1. Goals and non-goals

Goals:

1. **One honest deviation score.** Replace the Welford + EWMA-chase + windowed-
   Mahalanobis stack with a single, well-conditioned innovation metric whose
   covariance is never near-singular by construction.
2. **Physically-grounded drift and health.** Drift = slow parameter walk; poison
   = parameter *decay*, confirmed by a physical gain-per-stimulus measurement.
3. **Expected change is not an alarm.** Multi-regime normal baseline and
   humidity/temperature covariates remove the two largest false-alarm sources in
   real deployments (regime switches and weather).
4. **Typology, not just a flag.** Report *what kind* of change: spike, step,
   ramp, or pulse.
5. **Public contract stability.** `FailSafeSystem` and the apps keep their
   verdict/confidence/alert-level interface; the engine behind them changes.

Non-goals (explicitly deferred): gas identification, absolute concentration,
cross-device comparability of raw readings, and batch classification. See the
ontology section of `master-architecture.md`.

## 2. Terminology

| Term | Meaning |
|------|---------|
| Scalar reading `y_i(t)` | Normalized MOX channel `i` at time `t` (post-`preprocessing.rs`; per-channel relative units) |
| State `x(t)` | The true environmental "level" vector the sensors transduce |
| Parameters `θ(t)` | Hidden sensor properties: per-channel gain `g_i`, offset `o_i` |
| Covariates `c(t)` | Measured ambient `[T, RH]` (temperature, relative humidity) from the on-board sensor |
| Innovation `r(t)` | Measurement residual `y − ŷ` after filtering |
| Regime `k` | A distinct normal operating mode (idle / active fermenter / ...), each with its own reference state |
| Reference stimulus | A known, reproducible heater-pulse input used to measure gain per stimulus |

## 3. Measurement and process models

### 3.1 Measurement model (per channel)

```
y_i(t) = g_i(t) · h_i( x(t), c(t) ) + o_i(t) + v_i(t),   v ~ N(0, R_i)
```

Role of each term:

- `x(t)`: what actually changed in the environment (events + baseline).
- `c(t)`: humidity/temperature; enters `h_i` so weather-induced baseline shifts
  are *predicted* by the filter and produce small innovations, not alarms.
  Precisely: `h_i = f_i(x) · φ_i(c)` where `φ_i` is the per-channel ambient
  response curve (a low-order polynomial fit during calibration) and `f_i` is the
  gas response. The ambient correction and the gas response get cleanly separated.
- `g_i(t)`, `o_i(t)`: hidden gain/offset. `g` is the poison carrier (a decayed
  gain means a chemically dead channel), `o` is the electrical zero drift.
- `v_i(t)`: measurement noise, variance `R_i` estimated from a calibration slot.

During normal operation the true environmental state is a slowly moving baseline
plus occasional events. That is captured by the process model below — no
discontinuity enters except when an event arrives, which is exactly what we want
to call out.

### 3.2 Process model (state)

```
x(t)     = F(t) x(t−1) + w_x(t)          (random-walk level; F ≈ I, small Q_x)
w_x ~ N(0, Q_x)
```

When a regime transition is *declared* (Section 6), `x` is re-anchored to that
regime's reference; this is the one place the state is allowed a jump and, by
construction, it is not treated as an anomaly.

### 3.3 Process model (parameters)

```
θ(t) = θ(t−1) + w_θ(t)                   (slow random walk)
g_i(t+1) = γ(t) · g_i(t)  with γ ∈ (γ_min, 1]   (gain walk; decay is poisoning)
```

The parameters move slowly (process noise `Q_θ ≪ Q_x`). The gain carries the
multiplicative decay; the offset drifts additively. Poisoning shows up as a
sustained, statistically significant `γ < 1` per channel — a *parameter-level*
finding that the ambient reading cannot fake.

## 4. Dual estimation — the two filters

Two interlaced estimators, updated per 10 Hz reading. The **state filter**
tracks `x` given the current parameters; the **parameter filter** tracks `θ`
given the current state. Each uses the other's latest estimate. See
[Wan & van der Merwe, 2000] for the canonical dual-EKF treatment, and
[Haykin, 2001, ch. 1] for dual estimation of neural inputs vs. weights.

### 4.1 State filter

**Recommended: UKF** (unscented transform) — `h_i` contains cross-sensitivity
and ambient terms, so a Jacobian-free transform avoids linearization error at
essentially the cost of the sigma-point spread. Spec:

```
Given θ̂(t−1), P_x(t−1):
  pred:   x̂⁻(t)      = F x̂(t−1)
          P_x⁻(t)     = F P_x(t−1) Fᵀ + Q_x
  sigma:  𝒳 = [x̂⁻, x̂⁻ ± (√((n+λ)P_x⁻))_j],  j = 1..n
  obs:    ŷ_j         = h_i(𝒳_j, θ̂(t−1), c(t))        (stacked over channels)
          ŷ(t)        = Σ_j W_m^j ŷ_j
          P_yy(t)     = Σ_j W_c^j (ŷ_j − ŷ)(ŷ_j − ŷ)ᵀ + R
          P_xy(t)     = Σ_j W_c^j (𝒳_j − x̂⁻)(ŷ_j − ŷ)ᵀ
  update: K(t)        = P_xy P_yy⁻¹
          x̂(t)        = x̂⁻(t) + K(t) r(t),       r(t) = y(t) − ŷ(t)
          P_x(t)      = P_x⁻(t) − K(t) P_yy K(t)ᵀ
```

`P_yy` is the innovation covariance `S(t)`; its inverse is always defined
because we are adding `R > 0` to a covariance built from a sigma-point spread —
this removes the near-singular `Σ` that forced the ridge term and Euclidean
fallback in the shipped Mahalanobis path. The Kalman update needs no matrix
variance of `P_yy`; `P_xy P_yy⁻¹` is a least-squares solve with positive-definite
`P_yy`.

If the channel model is kept linear for a deployment (no covariates fitted, no
cross-terms), the filter degenerates to a plain Kalman filter; ship
`NewKalmanTrait` with `ekf/ukf/kf` implementations behind one `filter` trait so
the linear, cheap case is available and UKF is the default.

### 4.2 Parameter filter

The parameters enter almost linearly (gain multiplies `h_i`, offset adds), so an
**EKF with a 2-parameter-per-channel state** is sufficient and cheap:

```
θ = [g_1, o_1, g_2, o_2, …]           (2·c dimensional)
pred:   θ̂⁻(t) = θ̂(t−1)              (random walk; Q_θ small)
        H_θ    = ∂y/∂θ  evaluated at (x̂(t), θ̂⁻, c(t)):
                ∂y_i/∂g_i = h_i(x̂, c)          ∂y_i/∂o_i = 1
update: same EKF equations; P_θ ← P_θ⁻ − K_θ S_θ K_θᵀ
```

The measurement used by the parameter filter is the **residual of the state
filter** (the innovation that the state filter could not explain). This is what
makes the two filters see different things: fast changes go to the state (event),
slow persistent residual bias goes to the parameters (drift). If a channel's
relative gain `g_i/g_i(0)` drops monotonically and crosses
`θ_poison_relative_gain = 0.5`, the parameter filter raises a **poisoning health
finding** (Section 8) — independent of any ambient reading.

### 4.3 The differences that matter (vs shipped stack)

| Aspect | Shipped (`adaptive.rs`) | Target (this spec) |
|--------|-----------------------|--------------------|
| Baseline | Welford mean+z·σ per channel; EWMA chase | UKF state + coupled param walk |
| Multivariate | Windowed Mahalanobis, ridge 1e-6, Euclidean fallback | Innovation `P_yy` positive-definite by construction |
| Drift | `drift_alpha` EWMA on normal readings | Parameter random-walk; drift is `θ` change |
| Poison | Unused module; no consensus wiring | Gain decay in `θ` + stimulus confirmation → consensus |
| Ambient | Ignored | `c(t)` in `h_i` ⇒ weather is predicted, not alarming |
| Regime | Single Gaussian | Multi-regime clusters (Section 6) |
| Event shape | Flag only | Typology head (Section 7) |

## 5. Innovation and the anomaly verdict

Per reading:

```
r(t)   = y(t) − ŷ(t)                    innovation vector (state filter)
d²(t)  = rᵀ S(t)⁻¹ r                    innovation Mahalanobis (chi-square, dim c)
z_i(t) = |r_i(t)| / sqrt(S_ii(t))       per-channel standardized residual
```

- `d²(t)` replaces `mahalanobis_distance` and `raw_score`. Because `S(t)` comes
  from the filter (always PD) there is no ridge hack and no condition-number
  trap. The `sqrt(d²)` is the honest "how many baseline-standard-deviations
  away" magnitude for the operator, exactly as today's `raw_score` meant to be.
- **Verdict rule.** Anomaly iff `d²(t) > χ²_{c}(1−α)` for the ensemble's most
  sensitive budget AND `max_i z_i(t) > z_min` — the second clause keeps a weird
  single channel from firing the whole detector (mirrors today's
  "confidence override requires a per-channel vote").
- Threshold confidence keeps its current meaning and logistic form (sample
  count → 0..1), because the operator still needs "how much data shaped this
  baseline". The count now feeds `Q_x`/`Q_θ` shrinkage instead of a `z` offset.

## 6. Multi-regime normal baseline

The single-regime assumption is the biggest false-alarm source in real use
(e.g. a fermenter idling → active). Implementation:

- **Online clustering (sequential k-means with decay).** Keep up to
  `K_max = 3` cluster centers `x̄_k` + covariance `P_k`, updated online with a
  forgetting factor `β`. On reading `t`: compute the innovation distance
  `d_k²(t) = (x̂(t) − x̄_k)ᵀ P_k⁻¹ (x̂(t) − x̄_k)` for each cluster.
  - `min d_k² < θ_regime_join` ⇒ nearest cluster absorbs the reading
    (roll the center, scale the covariance, forget older history).
  - else if `K < K_max` and the reading is *persistent* (sustained for
    `θ_regime_min_samples`, default 300 = 30 s) ⇒ spawn a new cluster.
  - else ⇒ not normal in any regime ⇒ that alone is not an anomaly: the verdict
    still requires the Section-5 rule.
- **Transfer probability.** Maintain a small Markov transition matrix
  `Π(k′|k)` over regimes. A *declared* switch `k → k′` re-anchors `x̂` to `x̄_k′`
  with `P_x ← P_k′` and is reported as `event=regime_switch`, **not** an anomaly.
- **Warm start.** The calibration/warm-up window (`WARMUP_SAMPLES = 60`)
  seeds cluster 1 exactly as today's baseline does — the debut of a fresh device
  remains quiet.

Regime changes and the anomaly rule interact as follows: a step *within* a
regime still fires (it is a genuine change); the *same* step that coincides with
a declared transition does not. That is the intended and testable distinction.

## 7. Typology head

Classify the *recent innovation stream* on a sliding window (default
`W = 60` samples @ 10 Hz = 6 s, plus a `W_long = 600` for ramps). Features:

```
a   = max_j z_j(t−(W−1)..t)             peak standardized innovation
τ   = samples with z_j > z_hold         hold time above threshold
m   = per-channel post/post mean shift   (persists after transient?)
s   = least-squares slope of r over W_long (normalized by innovation std)
u   = filtered level change Δx̂ across W (state change attributable to event)
```

Decision tree (tie-break by order shown):

| Branch | Test | Typology |
|--------|------|----------|
| τ < 3 AND a large | single/few-sample transient, returns | **spike** |
| m significant AND τ ≥ 3, no recovery | persistent level change | **step** |
| s significant on W_long | monotone trend, no step | **ramp** |
| m significant; then recovers to ~0 within W_long | rise then return | **pulse** |
| τ < 3 AND a small | below-threshold jitter | none (no typology emitted) |

Each emission is a `Typology` with parameters: `kind`, `channel(s)`, `onset`,
`amplitude`, `duration`. This is the future "status line": *"step on channel 2,
+0.8, started ~40 s ago"*. The UI shows the single most-recent typology plus the
current innovation level — two time horizons, zero knobs.

## 8. Poison and health — gain per reference stimulus

`poisoning.rs` becomes the *physical confirmation* of the parameter filter, not
a parallel unexplained detector.

- **Reference stimulus.** On a schedule (`θ_stimulus_period`, default 1/h) or
  on explicit trigger, the device applies a controlled heater-pulse / reference
  exposure and measures the per-channel response `Δy_i`.
- **Gain per stimulus.** `ĝ_i = Δy_i / Δh_i^ref` where `Δh_i^ref` is the
  transducer's expected response to that reference (from burn-in). The ratio
  `ρ_i = ĝ_i / g_i(0)` is the *physical* gain retention.
- **Reconciliation.** `ρ_i` must agree with the parameter filter's relative gain
  `g_i/g_i(0)`. If `ρ_i` and the filter disagree by more than
  `θ_gain_disagreement = 20%`, report `health = WARNING` (the filter needs
  re-anchoring); if both agree that `ρ_i < θ_poison_relative_gain`, report
  `FAILED` (poisoned channel).
- **Consensus wiring.** `FailSafeSystem::detect` change:
  - any `FAILED`/`CRITICAL` channel ⇒ single detector fires (extend the existing
    degraded-sensor override, which today keys only on `assess_channel_health`).
  - a *confirmed* poison finding (filter + stimulus agree) escalates
    `alert_level` regardless of the anomaly vote, as a **service** alert: "this
    sensor is done; it needs service", separate from the **environment** alert
    stream.

The two independent sources (filtered gain decay + physical stimulus) are what
make the poison claim credible to operators and reviewers: one is statistics,
the other is a measurement.

## 9. Confidence calibration — Platt via Lin et al.

Current state: `retrain_platt_scaling` does a 3×3 grid search in `(a, b)`
around `(1, 0)` on the negative log-likelihood. This doc replaces it.

- **Data.** Pairs `(s_i, y_i)`: `s_i` is the anomaly evidence (use `d` or
  `sqrt(d²)` — spec picks `d`), `y_i ∈ {0,1}` is the operator's confirmed
  feedback ("did this matter?"), stored on the event. Keep today's guards:
  retrain every 10 feedback samples, only from ≥ 20.
- **Objective.** Minimize summed cross-entropy with a small prior on `b`
  (empirically keeps `P ≈ 0.5` when `s ≈ s_mean`):

```
J(a, b) = Σ_i [ log(1 + exp(a s_i + b)) − (1−y_i)(a s_i + b) ]
          + tiny·(b − b₀)²
```

- **Solver.** Newton's method with backtracking line search. The gradient and
  Hessian are analytic:

```
p_i = σ(a s_i + b)
∇J  = Σ_i (p_i − y_i) s_i ,   Σ_i (p_i − y_i)
H   = Σ_i p_i(1−p_i) [ s_i²   s_i;   s_i   1 ]
```

- **Numerical care.** When the two classes are cleanly separable, the maximum
  is at infinite slope — this is precisely the case [Lin et al., 2007] fixes.
  The design adopts their device: fit a slightly *regularized* sigmoid (targets
  `t⁺ = (N⁺+1)/(N⁺+2)`, `t⁻ = 1/(N⁻+2)` in the probabilistic-output formulation,
  i.e. never allow exact 1/0 targets), and if the Hessian is not positive-definite
  use a gradient step with line search. Behavior to pin in tests: predictions are
  monotone in `s`, cross-entropy never increases, and the solver terminates in a
  bounded number of iterations on toy separable/noisy data.
- Keep the exposed names (`retrain_platt_scaling`, `calibrated_confidence`) and
  update the honesty notes in `anomaly-detection.md` once merged.

## 10. Data flow and interfaces (Rust)

### 10.1 New types (added to the crate; public contract unchanged)

```
Engine modules (suggested layout):
  src/anomaly/
    mod.rs            // re-exports, Verdict/Typology types, consensus
    filter.rs         // KalmanTrait: KalmanFilter | ExKalmanFilter | UnscentedKalmanFilter
    dual.rs           // DualKalmanEngine: state+parameter interlacing
    regimes.rs        // RegimeModel: sequential-k-means + transition matrix
    typology.rs       // TypologyHead: windowed innovation classifier
    platt.rs          // PlattCalibrator: Lin et al. Newton solver
    stimulus.rs       // StimulusGainTracker: gain-per-reference + reconciliation
```

```
pub enum TypologyKind { Spike, Step, Ramp, Pulse }
pub struct Typology {
    pub kind: TypologyKind,
    pub channels: Vec<usize>,
    pub onset_sample: u64,
    pub amplitude: f64,
    pub duration_samples: usize,
}
pub struct Verdict {
    pub is_anomaly: bool,
    pub confidence: f64,          // Platt(calibrated_confidence)
    pub threshold_confidence: f64, // logistic in sample count (unchanged meaning)
    pub typology: Option<Typology>,
    pub regime_switch: Option<usize>,
    pub alert_level: AlertLevel,   // existing enum
    pub health_findings: Vec<HealthFinding>, // includes PoisonConfirmed
}
```

`DualKalmanEngine::detect(&mut self, y: &[f64], c: Option<Ambient>, t: u64) -> Verdict`
is the 10 Hz entry; `FailSafeSystem::detect` keeps its signature — it now feeds
`y` and `c` into `DualKalmanEngine` and applies the same consensus/vote rules on
the verdict (updating the degraded-sensor override per Section 8).

### 10.2 Configuration (one `DetectionConfig`, JSON round-trip)

| Key | Default | Meaning |
|-----|---------|---------|
| `use_ukf` | true | UKF vs linear KF (only if no covariates/cross-terms) |
| `q_state` | 1e-3 | state process noise |
| `q_param` | 1e-6 | parameter walk noise (≪ `q_state`; drift rate) |
| `r_scale` | 1.0 | measurement noise scale applied to calibrated `R` |
| `alpha_ukf`/`beta_ukf`/`kappa_ukf` | 0.001 / 2.0 / 0.0 | UKF sigma-point constants |
| `k_max` | 3 | regime count cap |
| `regime_join` | as fitted | chi-square join threshold per cluster |
| `regime_min_samples` | 300 | persistence before a new cluster spawns |
| `window` | 60 | typology window (samples) |
| `window_long` | 600 | ramp window (samples) |
| `platt_min_feedback` / `platt_retrain_every` | 20 / 10 | as today |
| `stimulus_period_s` | 3600 | reference-stimulus cadence |
| `poison_relative_gain` | 0.5 | `g/g(0)` or `ρ` below ⇒ poisoned |
| `gain_disagreement` | 0.20 | filter vs stimulus disagreement ⇒ WARNING |
| `sensitivity` | 1.0 | kept: scales the verdict threshold the same way today |

The three legacy knobs (`smoothing_alpha`, `drift_alpha`) are **removed**;
their jobs are subsumed by `q_state`/`q_param` and the process models. Tuning
guidance: `sensitivity` remains the operator-facing single knob.

## 11. Validation plan

### 11.1 Existing behaviors that must still hold (pin in tests)

Every row of the current test matrix in `anomaly-detection.md` §Validation must
pass against the new engine:

| Scenario | Required result |
|----------|-----------------|
| Single-sample spike (+6, heavy smoothing) | Not an anomaly (typology: spike suppressed below threshold) |
| Sustained step | Anomaly within ~40 samples (typology: step) |
| Slow ramp (+3 over 300 samples), drift on | Normal; baseline chased (parameters absorb; no alarm) |
| Abrupt step after ramp | Anomaly |
| Same shift, drift off (q_param→0) | Anomaly |
| Small delta, low vs high sensitivity | No vs yes |
| First 59 readings of fresh device | Warming up, never anomaly |
| Baseline zeros | No screaming during warm-up |

### 11.2 New behaviors (new tests, Wave-2 gate)

| Scenario | Required result |
|----------|-----------------|
| Regime switch (fermenter idle→active) | `regime_switch` event, **no** anomaly |
| Same magnitude step *inside* one regime | Anomaly (step) |
| Humidity step (RH +20%) feeding `c(t)` | Small innovation, **no** anomaly |
| Humidity sensor missing/saturated | Fallback to `φ_i = 0` correction, no crash |
| Poisoned channel (gain↘ in measurements) | Parameter gain crosses 0.5 ⇒ `PoisonConfirmed`; consensus degrades to single-vote |
| Stimulus before/after poison | `ρ` agrees with filter within 20% |
| Typology: synthesize spike/step/ramp/pulse | Right `TypologyKind` labels |
| Platt: separable and noisy toy data | Monotone in `s`; cross-entropy non-increasing; bounded iterations |

### 11.3 Wave-3 harness interface

Expose `replay_dataset<'a>(&mut engine, samples: impl Iterator<Item = &'a Sample>)`
so the Wörner et al. (2025) 12-month dataset and the Monte-Carlo synthetic sweep
can drive the engine end-to-end and emit per-sample verdicts for the
TPR/FPR/PPV/latency report. The harness lives in `src/tests/replay/` and must
not require hardware.

## 12. Implementation milestones (Wave 2 order)

1. `filter.rs`: three filter implementations + trait, with Cholesky-based
   least-squares solve; unit tests vs analytic Kalman on a 1D random walk.
2. `dual.rs`: interlacing loop; regression on the existing synthetic matrix.
3. `regimes.rs`: clusters + transition matrix + regime-switch re-anchor.
4. `typology.rs`: windowed classifier on the innovation stream.
5. `platt.rs`: Lin et al. Newton solver; swap into `retrain_platt_scaling`.
6. `stimulus.rs` + consensus wiring in `FailSafeSystem::detect`.
7. Whole-engine Wave-2 gate: sections 11.1–11.2 green; `cargo test` clean;
   `DetectionConfig` JSON round-trip preserved.

## References

1. Julier, S. J., & Uhlmann, J. K. (1997). *A new extension of the Kalman filter
   to nonlinear systems.* Proc. SPIE 3068, Signal Processing, Sensor Fusion,
   and Target Recognition VI. doi:10.1117/12.280797
2. Wan, E. A., & van der Merwe, R. (2000). *The unscented Kalman filter for
   nonlinear estimation.* Proc. IEEE Adaptive Systems for Signal Processing,
   Communications, and Control Symposium. doi:10.1109/ASSPCC.2000.882463
3. Haykin, S. (ed.) (2001). *Kalman Filtering and Neural Networks.* Wiley.
   (Dual extended Kalman filtering: ch. 1.)
4. Lin, H.-T., Lin, C.-J., & Weng, R. C. (2007). *A note on Platt's probabilistic
   outputs for support vector machines.* Machine Learning, 68(3), 267–276.
   doi:10.1007/s10994-007-5018-6
5. Platt, J. C. (1999). *Probabilistic outputs for support vector machines...*
   Advances in Large Margin Classifiers, 61–74. MIT Press.
6. Wörner, J., Eimler, J., & Pein-Hackelbusch, M. (2025). *Long-term drift
   behavior in metal oxide gas sensor arrays...* Scientific Data, 12, 1628.
   doi:10.1038/s41597-025-05993-8 (Wave-3 replay corpus)
7. Roberts, S. W. (1959); Lucas, J. M., & Saccucci, M. S. (1990). *EWMA control
   schemes* (retained context for the smoothing questions the Kalman process
   model generalizes). Technometrics 1(3), 239–250; 32(1), 1–12.