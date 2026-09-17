# Sensor Memory, Identifiability, and Feature Transfer — Wave 4

**Status.** All four benchmarks verified (Sep 16). Results in
`reports/bench_dynamic_memory.json`, `reports/bench_detector_memory_fp.json`,
`reports/bench_identifiability.json`, `reports/bench_features_audit.json`.
Reproducible via `python benchmarks/run_all.py`. Companion documents:
`anomaly-engine-design.md` (§11.12), `reports/README.md` (wave-4 table).

---

## 1. Motivation

The wave-3 audit established that drift, environment, and mixture nonlinearity
are first-order corruptors for MOX arrays (see `reports/README.md` findings
4–6, 13). Three related questions remained open:

1. **Memory**: when an exposure ends, how fast does the sensor truly return to
   baseline, and does the *history* of what came before change the *next*
   response?
2. **Identifiability**: given the measured noise and memory, how long a
   recording window does a consumer need to tell two gas conditions apart?
3. **Feature transfer**: of the 272 features the SDK and primitive libraries
   compute, which ones carry *the same meaning* across devices, environments,
   and protocols — and which ones are bound to one specific setup?

Wave 4 answers all three with controlled, data-driven measurements on the
dynamic-mixtures (2×12 h, ground-truth schedule) and turbulent-mixtures
(180 recordings, T/RH controlled) archives.

---

## 2. W1 — Dynamic Memory (`bench_dynamic_memory.py`)

### What it measures

A history-dependence audit on the UCI dynamic-mixtures ground-truth schedule.
Three controlled questions, all at 1 Hz with per-second ground-truth ppm:

| Question | Method |
|----------|--------|
| Does the clean gap before an exposure change the next response? | Response-vs-gap correlation for identical (gas,ppm) configs with gap as the only varying factor |
| Does *self*-priming differ from *cross*-priming? | `A→A→A` vs `A→B→A` (same gas twice vs different gas in between), same ppm |
| How many timescales does recovery need? | AIC comparison of 1-exponential vs 2-exponential fit on the air tail after each exposure |

### Results

**Gap modulation** (response-vs-gap correlation):

| File | Configs | Seconds | frac_configs_negative_corr | median_corr | pooled_slope |
|------|---------|---------|--------------------------|-------------|--------------|
| ethylene_CO.txt | 277 | 42,082 | 0.75 | −0.361 | negative |
| ethylene_methane.txt | 247 | 41,785 | 0.615 | −0.414 | negative |

Interpretation: across repeated identical (gas,ppm) configs the only thing
that varies is the length of the clean gap before the exposure. The
consistently negative correlation means **shorter gap ⇒ lower next
response** in most device families. This is not a small perturbation — the
effect size (median |r| ≈ 0.39) is comparable to the device-identity effect
on same-type pairs.

**Priming** (self vs cross):

| Metric | ethylene_CO | ethylene_methane |
|--------|-------------|------------------|
| frac groups where self > cross | 0.75 | 0.938 |
| median delta (self − cross) | +0.152 | +0.152 |

Self-priming *raised* the next response in 27/32 config-channel groups
overall. But the sign is gas/device-specific in the remaining groups, so
**priming cannot be collapsed to a single scalar** — any anomaly engine
that models memory must track it per-channel.

**Recovery timescales** (bi-exponential fit quality):

| Family | File 1 frac_bi_exp | File 2 frac_bi_exp | median τ_fast | median τ_slow |
|--------|--------------------|--------------------|---------------|---------------|
| TGS2600 | 1.000 | 1.000 | 15–25 s | 60 s |
| TGS2602 | 0.889 | 0.897 | 25 s | 60 s |
| TGS2610 | 0.994 | 1.000 | 8 s | 45 s |
| TGS2620 | 1.000 | 1.000 | 15 s | 45 s |

**97 % of fitted tails require bi-exponential.** A single-exponential
model (`h(t) = A·exp(−t/τ)`) is wrong for this array. The fast component
(τ ≈ 8–25 s) dominates the first ~30 s; the slow component (τ ≈ 45–60 s)
dominates beyond ~60 s. This is the timescale the anomaly engine's memory
model must capture.

### Design implications

- The `next-safe-gap` rule must be per-channel, not a global constant.
- The recovery memory model in `dual.rs` should use at least two exponential
  components; the measured τ_s (45–60 s) is the minimum "wait" for the slow
  tail.
- Priming direction cannot be assumed; the engine must learn it per-channel
  from baseline statistics.

---

## 3. W2 — Detector FP Decomposition (`bench_detector_memory_fp.py`)

### What it measures

How much of a real anomaly detector's clean-air false-alarm rate is caused by
**memory residue** (physical sensor state from a recent exposure) versus the
algorithm's own noise floor. The production residual axes (Kalman innovation,
latent-Δ, EWMA level, fused max-|z|) are run on the dynamic-mixtures
ground-truth schedule and every air alarm is binned by time since the
previous exposure ended.

Two floors are measured:

| Floor | Definition | Meaning |
|-------|------------|---------|
| Deep-clean (≥60 s air) | All air seconds after ≥60 s of continuous air | FP rate once the fast recovery component has decayed |
| Truly-recovered (≥300 s air) | All air seconds after ≥300 s of continuous air | FP rate once even the slow component has decayed |

Memory excess = FP_rate(deep-clean) − FP_rate(truly-recovered).

### Results

**Fused detector (max-|z|, threshold 3σ) aggregate:**

| Metric | ethylene_CO | ethylene_methane | Median |
|--------|-------------|------------------|--------|
| Deep-clean FP floor (≥60 s) | 0.471 | 0.249 | **0.36** |
| Truly-recovered FP floor (≥300 s) | 0.104 | 0.0034 | **0.054** |
| Memory excess vs 300 s | +0.324 | +0.107 | **+0.216** |

**62 % of residual FP rate is memory residue**, not algorithm noise.

**FP rate by time-since-exposure (fused detector, ethylene_CO):**

| Gap (s) | FP rate | Air seconds |
|---------|---------|-------------|
| 0–10 | 0.428 | 880 |
| 10–30 | 0.713 | 1,760 |
| 30–60 | 1.000 | 2,640 |
| 60–120 | 0.632 | 3,862 |
| 120–300 | 0.372 | 2,424 |
| 300+ | 0.131 | 743 |

FP rate **peaks at 30–60 s** post-exposure (when the slow recovery component
is still dominant) and collapses to the algorithm floor only at ≥300 s.
This sets the **minimum safe inter-event spacing** for any detector that
uses residual axes: events closer than 300 s apart cannot be independently
assessed.

### Per-axis false-positive rates (ethylene_CO, fused threshold)

| Axis | FP in air | TP in gas |
|------|-----------|-----------|
| Kalman innovation | 0.368 | 0.441 |
| Fused max-|z| | 0.755 | 0.652 |

The Kalman innovation axis has a lower FP floor but also lower sensitivity;
the fused max-|z| is more sensitive but pays for it in memory-driven FPs.

### Design implications

- The anomaly engine should report a **memory-confidence flag** when the
  time since the last event is <300 s. Alerts within this window should be
  down-weighted or suppressed.
- The 60 s "deep-clean" floor is useful for fast-turnover applications but
  the true FPR is 6× better at 300 s — deployment FPR claims must state the
  inter-event gap assumption.

---

## 4. W3 — Identifiability (`bench_identifiability.py`)

### What it measures

For every pairwise comparison of gas conditions (dose within a gas, and gas
identity at matched dose), how long a single 8-channel MOX recording must be
to achieve ≥95 % classification accuracy, given the measured replicate
scatter and the W1-calibrated memory residue of the previous exposure.

Method: closed-form Gaussian accuracy `Φ(d_k/2)` where `d_k` is the
Mahalanobis-2 separation at window length `k`. Memory residue is modeled as
an exponential decay with time constants τ = {15 s, 55 s} (fast/slow, from
W1 medians) and carryover amplitude 11 % (from A1).

### Noise calibration

Per-channel within-level σ (8 turbulent-mixture channels):

```
σ_level = [0.0548, 0.0476, 0.0699, 0.0789, 0.0909, 0.0984, 0.1015, 0.1039]
σ_sample = [0.0663, 0.0663, 0.0663, 0.0663, 0.0663, 0.0663, 0.0663, 0.0663]
```

### Dose resolution (within-gas)

| Pair | Acc ceiling (k=∞) | Clean min-k (s) | Median-mem min-k | Worst-mem min-k (gap 1 s) | Separation at k=1 |
|------|-------------------|-----------------|-------------------|---------------------------|-------------------|
| CO-H vs CO-L | 0.997 | 0.5 | 0.5 | 0.5 | 5.28 |
| CO-L vs CO-M | 0.779 | null | null | null | 1.46 |
| Me-H vs Me-L | 1.000 | 0.5 | 0.5 | 0.5 | 6.41 |
| Me-L vs Me-M | 0.976 | 0.5 | 10 | null | 3.68 |
| ethylene-H vs ethylene-L | 0.999 | 0.5 | 0.5 | 0.5 | 6.10 |
| ethylene-L vs ethylene-M | 0.963 | 1.0 | null | null | 3.40 |

Key findings:
- **CO-L vs CO-M is impossible** (ceiling 0.779, d/2 < 0.75) — the
  power-law compresses these low-dose responses below the noise floor.
  No window length helps.
- **Me-L vs Me-M needs 10 s** under median memory; under worst-case memory
  it becomes impossible.
- High-dose pairs (H vs L) resolve in **0.5 s** regardless of memory.

### Gas identity (cross-gas, matched dose)

| Pair | Acc ceiling | Clean min-k | Separation at k=1 |
|------|-------------|-------------|-------------------|
| CO-L vs Me-L | 0.915 | null | 2.65 |
| CO-L vs ethylene-L | 0.999 | 0.5 | 5.81 |
| Me-L vs ethylene-L | 1.000 | 0.5 | 7.65 |
| CO-M vs Me-M | 0.977 | 0.5 | 3.79 |
| CO-M vs ethylene-M | 0.994 | 0.5 | 4.84 |
| Me-M vs ethylene-M | 0.999 | 0.5 | 6.18 |
| CO-H vs Me-H | 1.000 | 0.5 | 6.85 |
| CO-H vs ethylene-H | 1.000 | 0.5 | 7.58 |
| Me-H vs ethylene-H | 1.000 | 0.5 | 8.03 |

All matched-dose cross-gas pairs except CO vs Me at low dose resolve in
**0.5 s** clean windows. The hardest cross-gas pair (CO-L vs Me-L) has
ceiling 0.915 — difficult but not impossible.

**Impossible pairs under clean conditions: 2** (CO-L vs CO-M, CO-L vs Me-L).
**Impossible under worst-case memory (gap 1 s): 4** (add Me-L vs Me-M,
ethylene-L vs ethylene-M).

### Design implications

- Substance identification from a single 8-channel recording is **fast
  (0.5 s) for most gas pairs** but **fundamentally bounded for low-dose
  same-gas discrimination** by the power-law ceiling.
- Memory residue at gap 1 s is sufficient for cross-gas identification
  (only 0.5 s needed), but same-gas dose discrimination requires ≥10 s
  or more.
- The 300 s safe-gap from W2 is conservative for identification purposes;
  the identification task is more robust to memory than the anomaly-detection
  task.

---

## 5. W5 — Feature Audit (`bench_features_audit.py`)

### What it measures

Every feature in the 187-dim SDK vector and the 85-dim physical-primitive
vector is classified along four axes:

1. **Catalog dimension** — which slot in `feature_catalog/CATALOG.md` the
   feature belongs to (absolute, advanced, da_, direction, global, health,
   hw_, temp_).
2. **Physics-semantic category** — a 10-way regex-based taxonomy: dose &
   amplitude, direction & selectivity, rise kinetics, recovery/decay memory,
   baseline & absolute offset, calibrated absolute, health & drift,
   dynamics & noise, hardware & transduction, saturation & nonlinearity.
3. **Invariance census + interop verdict** — from `existing_mapping.py`
   invariance labels, classified into transferable / device_bound /
   requires_calibration / protocol_confounded.
4. **Compute robustness** — `nan_rate` measured on 24 real SmellNet windows;
   features with `nan_rate > 0` are flagged as frequently undefined.

### Results summary

| Vector | Transferable | Device-bound | Requires calibration | Protocol-confounded | Total |
|--------|-------------|--------------|---------------------|--------------------|----|
| Framework (187-dim) | 34 | 39 | 42 | 72 | 187 |
| Primitives (85-dim) | 67 | 6 | — | 12 | 85 |
| **Total** | **101** | **45** | **42** | **84** | **272** |

After intersecting with `nan_rate == 0` (fully defined on real windows):
**98 / 272 (36 %) are both transferable and fully-defined.**

### Category rollup (framework)

| Category | n | Transferable | Device-bound | Requires calib. | Protocol-confounded |
|----------|---|-------------|--------------|-----------------|---------------------|
| 1 dose & amplitude | 22 | 22 | 0 | 0 | 0 |
| 2 direction & selectivity | 21 | 6 | 15 | 0 | 0 |
| 3 rise kinetics | 12 | 0 | 0 | 0 | 12 |
| 4 recovery / decay memory | 42 | 0 | 0 | 0 | 42 |
| 5 baseline & absolute offset | 18 | 0 | 0 | 18 | 0 |
| 6 calibrated absolute | 6 | 0 | 0 | 6 | 0 |
| 7 health & drift | 24 | 0 | 24 | 0 | 0 |
| 8 dynamics & noise | 24 | 0 | 0 | 6 | 18 |
| 9 hardware & transduction | 12 | 0 | 0 | 12 | 0 |
| 10 saturation & nonlinearity | 6 | 6 | 0 | 0 | 0 |

### Category rollup (primitives)

| Category | n | Transferable | Device-bound | Protocol-confounded |
|----------|---|-------------|--------------|---------------------|
| 1 dose & amplitude | 12 | 12 | 0 | 0 |
| 2 direction & selectivity | 55 | 55 | 0 | 0 |
| 3 rise kinetics | 6 | 0 | 0 | 6 |
| 4 recovery / decay memory | 6 | 0 | 0 | 6 |
| 5 baseline & absolute offset | 6 | 0 | 6 | 0 |

### Surviving categories

Only three categories have any transferable features:
1. **Dose & amplitude** (`_da_` family + `phys_` amplitude primitives) —
   22 framework + 12 primitives = 34 features.
2. **Direction & selectivity** (6 framework `_da_`-direction + 15 framework
   selectivity ratios + 55 primitive direction/logratio/covariance features)
   — 76 features.
3. **Saturation & nonlinearity** (6 `advanced_saturation_index` features) —
   6 features.

### Protocol-confounded features

Rise/decay kinetics (54 framework + 12 primitives = 66 total) and dynamics
& noise features (18 framework) are **protocol-confounded**: their meaning
depends on flow rate, dead volume, and purge protocol. These must never be
shared raw across different measurement setups.

### Design implications

- The shareable core is: `_da_` amplitude/dose features, selectivity ratios,
  saturation index, and the `phys_` primitive vector (normalized
  log-response, direction, covariance).
- Kinetic/decay features should be kept as **local-only** diagnostics; they
  cannot be standardized across protocols without a normalization convention.
- Hardware, health, and calibration-bound features are by-design
  device-specific and should not appear in interoperability schemas.

---

## 6. Cross-cutting conclusions

| Finding | Implication |
|---------|-------------|
| Recovery is bi-exponential (τ_fast ≈ 15–25 s, τ_slow ≈ 45–60 s) | Memory model in the anomaly engine needs ≥2 exponential components; single-exponential is wrong |
| FP peaks at 30–60 s post-exposure; ≥300 s to fully recover | Minimum safe inter-event gap for residual-axis detectors is ~5 min; report memory-confidence flag below this |
| 62 % of residual FP is memory, not algorithm noise | Improving the algorithm alone will not fix FPR; inter-event spacing or memory compensation is required |
| CO-L vs CO-L is impossible at any window (ceiling 0.779) | Low-dose same-gas discrimination is fundamentally power-law-limited; ship as a known limitation |
| Cross-gas identification needs only 0.5 s clean | Fast screening is feasible; the bottleneck is dose discrimination, not gas identity |
| 36 % of 272 features are transferable + fully defined | Only the `_da_` family, selectivity ratios, saturation index, and `phys_` primitives survive interoperability |
| Rise/decay features are protocol-confounded | Never share kinetic features raw across different measurement setups |
