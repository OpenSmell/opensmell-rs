# Anomaly Detection in OpenSmell — Research & Design Notes

Research and design notes for the live anomaly-detection subsystem: the problem it
solves, the math behind each detector, how the pieces combine, how to calibrate
and validate it, and where it is still limited.

**Status.** The pipeline described here is implemented and shipped. The primary
implementations live in `opensmell-rs/src/adaptive.rs` (live streaming detector)
and `opensmell-rs/src/anomaly/mod.rs` (baseline-fit detector). Anything marked
*planned* is not yet implemented; where the code uses an approximation of a
published method, this document says so explicitly.

## Overview

OpenSmell monitors a small array of metal-oxide (MOX) gas sensors over long
deployments. The hard problem is not detecting a big response — it is deciding,
sample after sample, whether the current reading differs from "normal" in a way
that matters, while:

- the sensors **drift** slowly (ageing, poisoning, temperature and humidity);
- the environment **changes** (a warm-up fermenter, a background that creeps);
- a reading can be an **event** (a real change worth alerting on) or **noise**;
- a sensor can **fail** (stuck at zero, flatline, unplugged).

The subsystem answers three questions per 10 Hz reading:

| Question | Answer produced |
|----------|-----------------|
| "Is this reading different from baseline?" | Per-channel deviation score + Mahalanobis distance |
| "By how much do we trust the current threshold?" | Threshold confidence (logistic in sample count) |
| "Should the operator be told?" | Anomaly verdict + calibrated probability + escalating alert level |

## Pipeline

```
OSM line @ 10 Hz
  → EWMA smoothing (one weight per sample)
  → score per channel:  |x_i − μ_i|   vs   adaptive threshold_i
  → multivariate score:  Mahalanobis distance from baseline
  → calibrated confidence (Platt sigmoid on Mahalanobis score)
  → three detectors (target FPR 0.05 / 0.01 / 0.10)
  → consensus (majority vote; confidence override; degraded-sensor override)
  → on normal readings only: drift-correction EWMA chases the baseline
  → escalation (warning / critical / emergency) and alert on the watch
```

Entry points: `AdaptiveAnomalyDetector::detect_drift_corrected` for the live
path, `FailSafeSystem::detect` for the ensemble, and `AnomalyDetector::detect`
for the batch/baseline-fit detector.

## Building Blocks and the Math

### 3.1 Adaptive per-channel threshold (Welford's online algorithm)

Each channel keeps a streaming estimate of the mean and summed squared deviation
of its absolute deviation scores using Welford's recurrence (`AdaptiveThreshold::update`):

```
delta     = x_n − mean
mean     += delta / n
delta2    = x_n − mean
m2       += delta * delta2          # corrected sum of squares
```

This avoids the catastrophic cancellation of the naive "sum of squares minus
correction term" form; each value is used exactly once and need not be stored
[Welford, 1962]. After more than 10 samples the threshold is placed at

```
threshold = mean + z · std ,   z = Φ⁻¹(1 − target_fpr),   std = sqrt(m2 / (n−1))
```

clamped to `[min_threshold, max_threshold]` (1.0 → 0.1…10.0 default). Because
scores are non-negative and roughly right-skewed, the mean+z·std rule is used as
a practical control limit, not as an assertion of normality.

### 3.2 Inverse normal CDF (the z-score source)

`normal_ppf` implements P. J. Acklam's rational approximation of the standard
normal quantile function: three piecewise rational regions (lower tail, central,
upper tail) with break-points at p = 0.02425, giving relative error below
1.15×10⁻⁹ across (0, 1) [Acklam, 2003]. The source comment credits
Abramowitz & Stegun; the coefficients are Acklam's minimax rational
approximation (of the same classical family as A&S §26.2.23), and the comment
should be read as a loose reference to that family.

Practical implication: `target_fpr = 0.01` ⇔ z ≈ 2.33, `0.05` ⇔ z ≈ 1.64,
`0.10` ⇔ z ≈ 1.28. The `score()` path divides the threshold by the operator's
`sensitivity`, which is the "by how much" control.

### 3.3 Mahalanobis distance and matrix inversion

The multivariate score is the Mahalanobis distance of the reading from the
baseline mean (`mahalanobis_distance`):

```
D_M(x) = sqrt( (x − μ)ᵀ Σ⁻¹ (x − μ) )
```

- The covariance is estimated from the calibration/warm-up sample set
  (`calibrate_baseline`).
- A ridge term `1e-6` is added to the diagonal (also in `AnomalyDetector::fit`)
  so a near-singular matrix stays invertible — MOX channels are strongly
  correlated, so this matters.
- Inversion uses Gauss-Jordan elimination with partial pivoting; a singular
  matrix returns an error rather than producing garbage (`invert_matrix`).
- If the calibration set is not larger than the channel count (n ≤ channels),
  Σ⁻¹ is unavailable and the score falls back to the Euclidean distance.

The distance is reported as `raw_score` — the honest, unit-free "how many
baseline standard deviations away" magnitude for the operator
[Mahalanobis, 1936].

### 3.4 Confidence calibration (Platt scaling)

Scores are not probabilities. OpenSmell maps the Mahalanobis score to a
probability via Platt scaling [Platt, 1999]:

```
P(anomaly | x) = 1 / (1 + exp(a · s + b))
```

where `s` is the raw score and `(a, b)` are fit by minimising the
cross-entropy on user-confirmed feedback (`retrain_platt_scaling`). Retraining
runs every 10 feedback samples, and only once at least 20 exist; before that the
defaults `a = 1, b = 0` are used.

**Honesty note.** The current minimiser is a coarse 3×3 local search over `(a, b)`
against the negative log-likelihood, not Platt's Levenberg–Marquardt solver nor
the improved Newton method of Lin et al. This is adequate for calibration drift
tracking but is *planned* to be replaced by the Lin et al. algorithm, which
provably converges and handles the small-sample numerical cases [Lin et al., 2007].

### 3.5 Threshold confidence vs calibrated confidence

Do not confuse the two numbers the detector exposes:

- `threshold_confidence` — logistic in the number of samples seen
  (`AdaptiveThreshold::confidence`): 0 at n=0, 0.5 at n=30, ≈0.95 at n=100.
  It says *"how much data has gone into this threshold"*, not *"how anomalous is
  the current reading"*.
- `calibrated_confidence` — the Platt probability for the *current reading*
  ("we are ~92% confident this is a real change").

The UI shows both deliberately; the first answers "should I trust it yet?",
the second answers "how different is this right now?".

## Smoothing and Drift — the EWMA Loops

Two EWMA loops in `detect_drift_corrected` encode the design principle
**"forgive slow drift, call out real change".**

1. **Smoothing EWMA.** Each fresh reading is blended one step toward the prior
   smoothed state: `s_t = smoothing_alpha · x_t + (1 − smoothing_alpha) · s_t−1`.
   With the default `smoothing_alpha = 0.6` a single-sample spike is heavily
   damped (a +6 spike on channel 0 with α=0.1 reaches the scorer as +0.6 and does
   not trip a ~1.0 threshold), while a sustained step pushes the smoother upward
   and trips within a few samples. This is the geometric moving average of
   SPC [Roberts, 1959; Lucas & Saccucci, 1990].

2. **Drift-correction EWMA.** Only on *normal* verdicts the baseline is chased
   toward the recent reading:
   `μ_i ← (1 − drift_alpha)·μ_i + drift_alpha · s_i`. With the default
   `drift_alpha = 0.002` a slow environmental ramp of +3 over 300 samples reads
   as normal, yet an abrupt step after the ramp still fires. With
   `drift_alpha = 0` the baseline is frozen and the same slow shift alarms —
   the codebase's unit tests pin both behaviours.

The approach is deliberately simpler than the classifier-ensemble drift
compensation of Vergara et al. [2012]: it assumes a single stationary normal
regime per deployment and adapts its reference point online, rather than
re-training classifiers across drifting batches. That is the right trade for a
10 Hz streaming alarm, and the wrong one for gas *classification* across long
data sets (see Validation).

A legacy time-decay term (`exp(−0.001 · seconds_since_calibration)` on the
reported threshold) is kept only for continuity with the original adaptive model;
the live correction is the EWMA baseline chase above.

## Ensemble of Three Detectors

`FailSafeSystem` runs three independent `AdaptiveAnomalyDetector`s with different
false-positive budgets, then combines them:

| Detector | target_fpr | z ≈ | Character |
|----------|-----------|-----|-----------|
| Standard | 0.05 | 1.64 | Default operating point |
| Conservative | 0.01 | 2.33 | Fewer alarms; bigger differences |
| Sensitive | 0.10 | 1.28 | Earliest warning; more alarms |

Combination rules (in `FailSafeSystem::detect`):

- **Majority vote:** anomaly if ≥ 2 of 3 detectors fire.
- **Confidence override:** if ≥ 1 detector fires *and* its Platt confidence
  exceeds 0.9, fire. This exists because a Mahalanobis distance can blow up on a
  poorly-conditioned covariance; requiring at least one per-channel verdict
  prevents the confidence path vetoing an otherwise all-clear result.
- **Degraded-sensor override:** if any sensor health score < 0.5, a single
  detector firing is enough — a suspect sensor is treated as already on edge.

## Warm-up and Baseline Establishment

A freshly attached device must not scream ANOMALY while its mean/covariance are
still unknown. `FailSafeSystem` buffers the first `WARMUP_SAMPLES = 60` readings
(≈6 s at 10 Hz), then calibrates every detector at once and reports `warming_up`
with `baseline_progress` until ready (`adaptive.rs`). An explicit/manual
calibration marks the system ready immediately and skips the deferred warm-up.

## Sensor-Health Integration

Detection and health are separate but coupled:

- The FailSafe detector flags `stuck_zero` channels (`value ≈ 0`) as critical
  failures.
- The fleet-health assessment (`assess_channel_health` in the desktop app) scores
  each channel over a rolling ~30 s window (≤300 readings @ 10 Hz) using:
  - coefficient of variation (baseline stability),
  - a first-difference RMS **noise floor** (temporal resolution),
  - a half-window **drift rate** (relative change, front half vs back half),
  - explicit **stuck** detection: standard deviation ≤ 1e-9 ⇒ `FAILED`.

| Condition | Score | Status |
|-----------|-------|--------|
| std ≤ 1e-9 (flatline) | 0.0 | FAILED — unresponsive |
| cv ≥ 0.3 or drift > 15% | 0.1 | CRITICAL — recalibrate/replace |
| cv ≥ 0.2 or drift > 8% | 0.4 | WARNING — baseline refresh soon |
| cv ≥ 0.1 | 0.7 | WARNING — elevated noise floor |
| otherwise | 1.0 | OK |

A perfectly flat trace is *not* a healthy sensor — it is unresponsive. A healthy
score decays the consensus requirement from 2 votes to 1, so detection remains
useful while one channel is degraded.

## Escalation and Alerting

`FailSafeSystem` turns momentary verdicts into a persistent, watch-sized state:

| Alert level | Trigger | Meaning |
|-------------|---------|---------|
| normal | — | no alarm |
| warning | 2 consecutive anomalies | first sign of change |
| critical | 5 consecutive anomalies | sustained change; pay attention |
| emergency | 10 consecutive anomalies | intervention needed |

The level resets to normal after 20 consecutive normal readings. The UI shows
one escalating state plus the calibrated confidence, rather than a stream of
pop-ups.

## Calibration Protocol (Operator)

No statistics background required:

1. **Expose the sensors to clean air** (or the deployment's reference
   environment) and let the device stream. The automatic warm-up establishes the
   baseline in ~6 s; for a careful deployment run an explicit calibration over
   ~30 s of steady readings and trigger it from the Calibration panel.
2. **Trust the numbers, not the labels.** Wait until `threshold_confidence` is
   meaningfully high (n > 30 readings is 0.5; n ≥ 100 is ≈0.95) before deciding
   false alarms are false.
3. **Tune one knob, `sensitivity`** (effective threshold = threshold /
   sensitivity). Start at 1.0; if readings that should alarm are silent, raise
   it; if background changes alarm, lower it. `smoothing_alpha` and
   `drift_alpha` have sane defaults and are usually left alone.
4. **Confirm anomalies when asked.** Each confirmed "yes/no" reading refits the
   Platt calibration, so confidence numbers become meaningful per deployment.
5. **Watch the Fleet health tab.** A FAILED/CRITICAL channel explains both
   silence (stuck sensor) and chatter (drifting sensor) before the alarm is
   blamed.

## Validation Methodology

### Synthetic ground truth (recommended, and partly automated already)

Generate a controlled stream and assert the detector's behaviour against it. The
codebase tests already pin the core behaviours:

| Injected scenario | Expected (unit-tested) |
|-------------------|------------------------|
| Single-sample spike (+6, heavy smoothing) | Not an anomaly |
| Sustained step | Anomaly within ~40 samples |
| Slow ramp (+3 over 300 samples, drift on) | Normal; baseline chased |
| Abrupt step after the ramp | Anomaly |
| Same shift with drift off | Anomaly (proves drift correction caused the forgiveness) |
| Same small delta at low vs high sensitivity | No vs yes |
| First 59 readings of a fresh device | Warming up, never anomaly |
| Baseline mean is zeros | No screaming anomalies during warm-up |

For a numeric report, generate a labeled stream per channel of the form
`normal ~ N(μ, σ)` plus known events, run `FailSafeSystem::detect` offline, and
report:

- **FPR** — fraction of normal samples flagged (target ≈ the ensemble's 0.05
  operating point, and well below the best detector's);
- **TPR / detection latency** — fraction of event samples flagged within k
  samples of onset (measure both; escalation depends on consecutive counts);
- **PPV / false-alarm-to-real ratio** — how many alerts were real, which is what
  the operator actually experiences.

Keep the Welford/scoring decoupled from the health window so the two can be
validated independently.

### Public drift datasets (optional, with caveats)

- **UCI Gas Sensor Array Drift Dataset** — 13,910 measurements, 16 MOX sensors,
  6 gases, 36 months, 10 batches [Vergara et al., 2012; UCI]. The standard
  benchmark for drift-compensated *classification*.
- **Wörner et al. (2025)** — a 12-month, 62-sensor e-nose data set with raw
  time series and features, aimed explicitly at drift detection and
  compensation; reproduces long-term CVs of 25–41% per sensor [Wörner et al., 2025].

**Caveat.** Those benchmarks evaluate batch classification accuracy across
drifting batches. OpenSmell's detector is a streaming, single-normal-regime
alarm; it does not classify gases and does not keep a classifier to doctor
across batches. Applying the batch benchmark protocol to it would misreport its
performance. To evaluate drift behaviour specifically, use its EWMA/chase
properties with synthetic ramps, and the reproduction of *long-term* MOX CVs
(25–41% [Wörner et al., 2025]) as the outside bound of what `drift_alpha` should
accommodate.

## Failure Modes and Mitigations

| Mode | Symptom | Mitigation |
|------|---------|------------|
| Near-singular covariance | Mahalanobis score blows up | 1e-6 ridge; Gauss-Jordan singular check; confidence override requires a per-channel vote |
| Stuck / flatlined sensor | Constant reading reads as "stable" | Noise-floor zero ⇒ FAILED; `stuck_zero` critical failure |
| Plug-in transient | First readings far from anything | Warm-up buffer; baseline not ready ⇒ no anomaly |
| Single-sample spike | Jitter on one channel | Smoothing EWMA damps it |
| Slow real drift misclassified | A decline is absorbed as normal | By design (drift correction); health tab's drift rate shows it and recommends recalibration |
| Too few feedback samples | Platt defaults (a=1, b=0) | Confidence numbers are soft until ≥20 confirmed samples; threshold confidence makes this visible |
| Covariance needs n > channels | Euclidean fallback | Documented; calibrate with more than `n_channels` samples for full multivariate behaviour |

## Tuning Knobs

`DetectionConfig` — three plain floats (round-trip through JSON for the desktop
settings UI):

| Field | Default | Meaning | Guidance |
|-------|---------|---------|----------|
| `smoothing_alpha` | 0.6 | EWMA weight per sample before scoring | Lower ⇒ more damping of spikes; 1.0 passes raw readings through |
| `drift_alpha` | 0.002 | EWMA rate the baseline chases normal readings | 0.0 freezes baseline (only recalibration moves it); raise to absorb slow environmental drift |
| `sensitivity` | 1.0 | effective threshold = threshold / sensitivity | >1 flags smaller differences; <1 requires larger ones |

## Open Questions / Future Work

- **Isolation Forest and LOF.** `AnomalyMethod::IsolationForest` and `::LOF`
  exist as enum variants but are unimplemented; `Ensemble` currently reduces to
  Mahalanobis. Adding distance-based LOF over a rolling buffer is the natural
  next detector.
- **Platt solver.** Replace the 3×3 grid search with the convex Newton method
  [Lin et al., 2007].
- **Long-term validation.** Replay a months-long drift series (e.g. [Wörner et
  al., 2025]) through the streaming path and publish TPR/FPR-over-time curves.
- **Cross-device calibration transfer** — reuse a baseline across identical
  boards (Rodríguez-Luján et al. show minimal-experiment array calibration is
  feasible for batch classification), and online CUSUM as a second drift
  detector beside the EWMA chase.

## References

1. Welford, B. P. (1962). *Note on a Method for Calculating Corrected Sums of
   Squares and Products.* Technometrics, 4(3), 419–420.
   doi:10.1080/00401706.1962.10490022
2. Acklam, P. J. (2003). *An algorithm for computing the inverse normal
   cumulative distribution function.* (oral/traditional publication; coefficients
   and error bound widely reproduced).
3. Abramowitz, M., & Stegun, I. A. (1964). *Handbook of Mathematical Functions*
   (Eq. 26.2.23). National Bureau of Standards.
4. Mahalanobis, P. C. (1936). *On the generalised distance in statistics.*
   Proceedings of the National Institute of Sciences of India, 2(1), 49–55
   (reprint: Sankhya A 80(S1), 1–7, 2018, doi:10.1007/s13171-019-00164-5).
5. Platt, J. C. (1999). *Probabilistic outputs for support vector machines and
   comparisons to regularized likelihood methods.* In Advances in Large Margin
   Classifiers, 61–74. MIT Press.
6. Lin, H.-T., Lin, C.-J., & Weng, R. C. (2007). *A note on Platt's probabilistic
   outputs for support vector machines.* Machine Learning, 68(3), 267–276.
   doi:10.1007/s10994-007-5018-6
7. Roberts, S. W. (1959). *Control chart tests based on geometric moving
   averages.* Technometrics, 1(3), 239–250.
   doi:10.1080/00401706.1959.10489860
8. Lucas, J. M., & Saccucci, M. S. (1990). *Exponentially weighted moving
   average control schemes: properties and enhancements.* Technometrics, 32(1),
   1–12. doi:10.2307/1269835
9. Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., & Huerta, R.
   (2012). *Chemical gas sensor drift compensation using classifier ensembles.*
   Sensors and Actuators B: Chemical, 166–167, 320–329.
   doi:10.1016/j.snb.2012.01.074
10. Rodríguez-Luján, I., Fonollosa, J., Vergara, A., Homer, M., & Huerta, R.
    (2014). *On the calibration of sensor arrays for pattern recognition using
    the minimal number of experiments.* Chemometrics and Intelligent Laboratory
    Systems. (Dataset also at: UCI Machine Learning Repository, *Gas Sensor Array
    Drift Dataset*.)
11. Wörner, J., Eimler, J., & Pein-Hackelbusch, M. (2025). *Long-term drift
    behavior in metal oxide gas sensor arrays: a one-year dataset from an
    electronic nose.* Scientific Data, 12, 1628.
    doi:10.1038/s41597-025-05993-8
12. Romain, A.-C., & Nicolas, J. (2009). *Long term stability of metal
    oxide-based gas sensors for e-nose environmental applications: an overview.*
    Sensors and Actuators B: Chemical, 146(2), 502–506.
    doi:10.1016/j.snb.2009.12.027