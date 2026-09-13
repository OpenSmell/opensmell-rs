# OpenSmell — Master Architecture & Roadmap

The north-star design document for the OpenSmell project: what the system is
for, what it can honestly claim, how the pieces fit together, and the wave
plan that turns today's implementation into a 10-year-relevant product.

**Status.** Living document. Wave-2+ items are *planned* unless marked shipped.
Mirror of doc set: `anomaly-detection.md` (current shipped detectors),
`anomaly-engine-design.md` (the target dual-Kalman engine).

---

## 1. Value proposition

Osmograph — the OpenSmell device + software stack — monitors a small array of
metal-oxide (MOX) gas sensors and tells an operator, sample after sample,
**whether the environment has changed in a way that matters, and what kind of
change it is** (a spike, a step, a slow drift, a pulse), with an honest
probability and an escalating alert level, on a device that needs almost zero
configuration.

That is the wedge. It is deliberately *not* "name the gas" or "give me the
concentration": MOX sensors are low-cost, cross-sensitive, relative,
drifting transducers. Every inference that ignores those four facts collapses
in the field. Deviation-from-baseline is the only inference that survives the
physics on a single device, so that is the honest ontology — and the deferred,
compounding one is **the fleet**: a quality-gated body of `.osmell` recordings
with labels, which is what finally makes calibration transfer, gas identification,
and concentration plausible.

Three properties the project optimizes:

| Property | Meaning in practical terms |
|----------|----------------------------|
| High truth | Every number exposed is calibrated and honest; no borrowed claims; real-data TPR/FPR/PPV, not synthetic-only |
| High utility | Answers that change the operator's next action: status, typology, alert level, health |
| Easy to use | One status line, two time horizons, zero visible knobs; "did this matter?" is the only feedback loop |

10-year relevance means the architecture must survive better sensors, larger
fleets, and new question types — so the core design separates **detection
(physics-honest)**, **classification (data-hungry, deferred)**, and
**interoperability (the standard)**.

---

## 2. Ontology — what OpenSmell claims and does not claim

### 2.1 Claims that hold today (shipped)

- **Features.** A standardized feature vector per recording
  (`28c + c(c−1)/2 + 4` where `c` = sensor count): 6 dev-agnostic + 4 absolute +
  4 temporal + 4 health + 3 hardware + 1 saturation + 6 decay per channel.
  Contract: Python (reference) ≡ JS (TypeScript port), verified byte-for-byte on
  synthetic exposures.
- **Quality.** A 7-factor MOX scorer producing subscores (e.g. response speed,
  stability, saturation integrity) and a total 0–100 grade with verbal bands
  (e.g. `Poor`). Based on the open MOX "smellability" assessment chain.
- **Smellability.** The multi-stage reduction (normalize → preprocessing →
  features → quality → smell index) that turns raw channels into a decision.
- **Health.** Per-channel fleet health: `OK / WARNING / CRITICAL / FAILED`
  from coefficient-of-variation, noise floor, drift rate, and explicit
  stuck/flatline detection (`std ≤ 1e-9 ⇒ FAILED`).
- **Anomaly (current).** Per-channel adaptive thresholds (Welford mean+z·σ) and
  a Mahalanobis distance against a calibrated baseline, ensembled across three
  false-positive budgets with confidence calibration (Platt) and escalation
  (warning/critical/emergency). Shipped in `opensmell-rs`.
- **Format.** `.osmell` container v1.1.0 (sensor/calibration/sample/events
  sections) read and written identically in Python and JS; JSON-events stream
  for interop.

### 2.2 Intentional non-claims (honest deferrals)

- **Gas identity.** We do not name gases. Cross-sensitivity + drift makes
  single-device classification fragile; it becomes defensible only fleet-wise.
- **Absolute concentration.** MOX output is relative; concentration needs
  per-device calibration models (power law, inverse concentration) which are
  *planned* to ship as an owned module — until then we report deviation, not ppm.
- **Cross-device comparability** of raw readings. Only after calibration transfer
  (Wave 5) do two devices agree; the *labels* and *quality gates* already travel.

### 2.3 The physics that forces the ontology

MOX sensors: response ≈ `gain · f(analyte) + offset`, with gain and offset
drifting over weeks-to-months, responding to temperature and humidity, and
degrading permanently under poisoning. Consequences:

1. Raw readings are not comparable across days — only deviations from a local
   baseline survive.
2. The sensor's *state* (its gain/offset) is a hidden variable that must be
   tracked; detecting poison = detecting a *change in hidden gain*, not a change
   in the reading.
3. The environment has discrete normal regimes (e.g. fermenter idle vs active),
   not one Gaussian blob — a single-regime baseline misfires at regime switches,
   so the normal model must be multi-regime.

These three consequences are exactly what the target engine (Section 4) is
built to address.

---

## 3. System architecture

Three parallel SDKs sharing one contract, plus applications, plus a data layer.

```
                .osmell v1.1.0  (the contract)
                     │
      ┌──────────────┼─────────────────────────┐
      │              │                         │
opensmell (Py)   opensmell-rs            opensmell-js (TS)
reference impl   streaming DAQ           browser/edge parity
docs @ xyz       anomaly engine          docs @ github/OpenSmell
      │              │                         │
      └──────┬───────┴─────────┬───────────────┘
             │                 │
     osmograph-desktop    osmograph-web
     (Tauri 2 + Rust + TS) (Next.js @ mox.opensmell.xyz)
             │                 │
             └───────┬─────────┘
                     ▼
              Data Hub / HF sync  (fleet commons)
```

| Layer | Repo | Role |
|-------|------|------|
| `opensmell` (Python) | local | Canonical reference, docs site backend, calibration science |
| `opensmell-rs` | github.com/OpenSmell/opensmell-rs | Fast streaming detection on the device: dual-Kalman engine, health, consensus, protocol |
| `opensmell-js` | github.com/OpenSmell/opensmell-js | TypeScript port with exact Python parity for browser/edge + tooling |
| `.osmell` format | spec | Interoperability backbone; conformance suite (planned) gates every writer |
| `osmograph-desktop` | local | Tauri 2 desktop: live DAQ, burn-in, quality, calibration, Data Hub sync |
| `osmograph-web` | local | Web status surface at `mox.opensmell.xyz` |
| Data commons | planned | Quality-gated corpus + cross-device calibration transfer (fleet bootstrapping) |

Division of labour: the **Python** package is the reference and the calibration
science; the **Rust** crate is where streaming *truth* is produced at 10 Hz;
the **JS** package guarantees the ecosystem shares one brain; the **apps** are
where it becomes easy to use; the **commons** is where the value compounds.

---

## 4. Detection architecture — target (dual-Kalman engine)

Summary for the roadmap; complete spec in `anomaly-engine-design.md`.

The current shipped stack (`adaptive.rs`: Welford online thresholds + EWMA drift
chase + windowed Mahalanobis; `anomaly/mod.rs`: batch baseline fit) is replaced
by a **dual extended Kalman / unscented Kalman** engine with a cleaner
separation of *state* (moves quickly) and *parameters* (move slowly — that is
drift):

- **State filter.** Tracks the working sensor levels and baseline. Stale/spiky
  readings produce *innovations* (measurement residuals) which are the anomaly
  score. Covariates: **humidity and temperature** enter the measurement model so
  weather-caused variation is *expected*, not anomalous.
- **Parameter filter.** Tracks hidden gain/offset per channel — drift is a
  parameter *walk*; **poisoning is a gain *decay***. This is the physical reading
  of health.
- **Multi-regime baseline.** Online clustering of 2–3 normal states replaces the
  single Gaussian; a regime switch is an *expected* transition, not an alarm.
- **Typology head.** The innovation stream is classified into
  **spike / step / ramp / pulse** so the operator learns *what happened*, not
  just that something happened.
- **Calibration.** Platt probability via a convex Newton solver (Lin et al.
  2007), replacing the current 3×3 grid search, trained on operator feedback.
- **Poison stimulus.** Health is confirmed by a physical reference: gain-per-
  heater-pulse before/after a stimulus reports baseline trust independent of the
  ambient reading.

This design single-sources every "higher-level" claim (drift, poison, health,
regime) and remains honest under humidity/temperature interference.

---

## 5. Roadmap — five waves

Each wave has an exit criterion (shown under "Done when"). Waves 1–3 are the
scientific/engineering core; Wave 4 makes the ecosystem consistent; Wave 5
makes it a product.

### Wave 1 — Architecture & design (this doc set)

Deliverables:

- `master-architecture.md` (this file): value prop, ontology, architecture, roadmap.
- `anomaly-engine-design.md`: complete spec for the dual-Kalman engine
  (math, data flow, interfaces, calibration, validation plan).

Done when: both documents are committed to `opensmell-rs`, and an engineer with
no context can implement Wave 2 from the design doc alone.

### Wave 2 — Engine implementation (opensmell-rs)

Replace the detection core with the dual-Kalman design:

1. Dual EKF/UKF state+parameter filters (`adaptive.rs`), keeping the public
   verdict/confidence/alert contract stable so apps don't break mid-wave.
2. Multi-regime baseline via online clustering; humidity/temperature covariates
   in the measurement model.
3. Typology head (spike/step/ramp/pulse) on the innovation stream.
4. Lin et al. Platt solver (replace 3×3 grid search in `retrain_platt_scaling`).
5. Stimulus-based poison/health detection (gain-per-reference) wired into the
   consensus path (`poisoning.rs` → `FailSafeSystem`).

Done when: the replaced modules pass the existing unit-test matrix
(synthetic spike/step/ramp behaviours must still hold), plus new tests for
regime switches (no alarm), humidity-tracking (no alarm), and poison (degraded
gain ⇒ health drop, not a false "event").

### Wave 3 — Validation (papers-grade numbers)

- Replay harness: run the Wörner et al. (2025) ~12-month, 62-sensor dataset
  through the streaming path.
- Monte-Carlo calibration sweep across synthetic scenarios to publish
  **TPR / FPR / PPV** and detection-latency curves.
- Report the long-term MOX CV reproduction (25–41%) as the outside bound the
  drift model must accommodate.

Done when: there is a published (repo + docs site) TPR/FPR/PPV report on
real data — the number papers and operators actually require.

### Wave 4 — JS parity & interoperability guarantee

- Port `calibration.py` to `opensmell-js` (power-law, inverse concentration,
  leave-one-out cross-validation).
- Port the smellability chain (~4.5k lines) to `opensmell-js`.
- Conformance suite: a hard-to-vary compliance gate for `.osmell` v1.1.0 writers
  + a reference corpus committed to the spec repo.

Done when: any `.osmell` file produced by a conformant writer round-trips
identically across Python/Rust/JS, verified by CI.

### Wave 5 — Product & data commons

- Frontend status/typology view (desktop + web): one status line, two time
  horizons (now = step/spike, hours = ramp/drift), "did this matter?" feedback.
- Data commons: quality-gated corpus, schema registry, Data Hub / HF sync,
  cross-device calibration transfer (fleet bootstrapping: a calibrated board
  seeds its uncalibrated siblings).

Done when: a new board reaches a usable baseline in minutes via fleet transfer,
and the fleet dataset grows with exported `.osmell` + label feedback.

---

## 6. Guiding principles (do not regress)

1. **No borrowed claims.** Every number exposed is derived from shipped code and,
   where it calls a published method, documented as such — including where we
   approximate (anomaly-detection.md is explicit about the 3×3 Platt search).
2. **Truth over sales.** Papers and journals reject detectors with only synthetic
   curves. Real-data TPR/FPR/PPV is a Wave-3 gate, not a nice-to-have.
3. **The contract is the castle.** `.osmell` + feature names + quality scoring
   are byte-verified across Python/JS; the conformance suite (Wave 4) makes that
   a promise instead of an accident.
4. **Zero visible knobs for operators.** Configuration is engineering; the
   operator's only loop is "did this matter?"
5. **Defer what physics can't support.** Gas ID and ppm are fleet-scale claims,
   not single-device ones; the roadmap sequences them accordingly rather than
   pretending the device can do them today.