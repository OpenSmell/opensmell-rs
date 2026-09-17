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
- **Anomaly (current).** The dual-Kalman engine (`src/anomaly/dual.rs`): a state
  filter for working levels coupled to a parameter filter for hidden gain/offset,
  multi-regime clustering with humidity/temperature covariates, a typology head,
  Platt calibration, and stimulus-confirmed poison health findings — all behind
  the same verdict/confidence/alert contract. The legacy Welford/EWMA/Mahalanobis
  stack (`adaptive.rs`) is retained with `#[deprecated]` knobs for a clean
  cutover. Shipped in `opensmell-rs`.
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

The shipped engine (Wave 2/2.5) goes one step deeper on the measurement model:
an optional adsorption-memory state `m` per channel absorbs desorption residue,
and the default linear `g·x + o` can be replaced with the power-law
`g·φ(x, α) + o` (tracking α per channel) so concentration compression is read as
state, never as fake gain decay. Both default off for a drop-in legacy config.

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

### Wave 2 — Engine implementation (opensmell-rs) — shipped

Replace the detection core with the dual-Kalman design:

**Status.** All five items are shipped in `opensmell-rs` (spec: `anomaly-engine-design.md`).
The public verdict/confidence/alert contract stayed stable so apps did not break.

1. Dual EKF/UKF state+parameter filters — **shipped** as `src/anomaly/dual.rs`
   (state filter linear- or unscented-Kalman per `state_filter`; parameter filter
   tracks `[g, o]` per channel).
2. Multi-regime baseline via online clustering (`regimes.rs`) and
   humidity/temperature covariates in the measurement model — **shipped**.
3. Typology head (spike/step/ramp/pulse) on the innovation stream
   (`typology.rs`) — **shipped**.
4. Lin et al. Platt solver (`platt.rs`, replacing the 3×3 grid search) — **shipped**.
5. Stimulus-based poison/health detection (`stimulus.rs`) wired into the
   consensus path — **shipped**.

Done when: the replaced modules pass the existing unit-test matrix
(synthetic spike/step/ramp behaviours must still hold), plus new tests for
regime switches (no alarm), humidity-tracking (no alarm), and poison (degraded
gain ⇒ health drop, not a false "event"). — Done (139 tests green, incl.
`tests/wave2_gate.rs` pinning the full §11.1–11.2 matrix and the replay
harness).

### Wave 2.5 — Measurement physics (adsorption, power-law, auto-tune) — shipped

Squeezed in after the engine core so the *measurement model* matches the MOX
physics the engine was designed around:

1. **Adsorption memory** (`AdsorptionConfig`). A per-channel memory state `m`
   that decays with a desorption time constant `τ` absorbs the "still smells
   like cake" residue after a heavy exposure — so a desorption tail is
   *predicted* by the filter and absorbed, not read as drift or a fresh event.
   Adsorption/memory effects get their own transient state; they are not a
   spike/ramp/drift classification.
2. **Power-law response** (`ResponseConfig`). The default `y = g·x + o` is only
   the local Taylor regime of the true MOX power-law law. When enabled the
   measurement model becomes `y_i = g_i·φ(x_i, α_i) + o_i` with
   `φ(x, α) = |x|ᵅ·sgn(x)`, and the parameter filter tracks
   `[g, o, α]` per channel — so a large concentration spike at the compressed
   high end is read as the state `x`, not as gain decay (the misreading that
   would otherwise fake a poison confirmation).
3. **Auto-tune calibration** (`calibration::AutoTune`). `q_state`, `q_param`,
   the desorption `τ`, and the exponent `α` are *derived from measurements*
   (baseline variance, drift first-differences, desorption tails, log-log
   response fits) instead of engineer-guessed constants.

**Backward compatibility.** Both configs default to off/linear with
`#[serde(default)]`, so a legacy `EngineConfig` JSON round-trips unchanged.

### Wave 3 — Validation (papers-grade numbers)

**Status.** Real-data TPR/FPR/PPV + latency gate **closed on the UCI dynamic
mixtures corpus** (`src/bin/realdata_eval.rs`, §11.4, §11.6 of the engine design
doc): full 11.6–11.7 h causal replays score 89/89 and 95/98 events at
`sensitivity = 3` with clean FPR ≈ 1e-3, holdout-selected (fixed rule, no
retrofitting), with Wilson/bootstrap 95% CIs (`rs_validate.py`; archived JSON
at `e-nose-evals/u2_gas_leak/results/rs_realdata_*.json`). The Wörner et al.
(2025) long-horizon drift corpus remains the 12-month extension. The second
generalization corpus (UCI home activity, `src/bin/indoor_eval.rs`, §11.5) was
run and resolves as an **honest negative**: 6–7/68 stimuli at clean FPR
0.0065–0.0130 with no admissible operating point, driven by low per-induction
stimulus strength near the ambient home-activity noise floor (documented with a
stimulus/drift audit; archived at
`e-nose-evals/u2_gas_leak/results/rs_realdata_indoor_sens{2,3,4}.json`). The
negative is partly algorithm-limited: an EWMA control chart (`src/anomaly/ewma.rs`,
`--baseline ewma`) recovers 18–22/68 at admissible FPR on indoor-air but
fails the dynamic-mixtures recordings — no single detector+params is
admissible on both corpora yet (§11.7, the open cross-regime item). The
recommended operating point is unchanged; the negative and headroom are
published rather than tuned away. A third real-data gate added and **closed**
(§11.9, `src/bin/tadi_eval.rs`): the TADI-2019 field corpus (Zenodo 8399829,
controlled CH4 releases at an industrial site, six Figaro TGS MOS loggers with
CRDS ground truth) scores 86% release detection (54/63) at sens=3 — Logger_H,
closest to the release point, 100% at sens≥2.5 — with FA/month ~1,900 at sens=3
measured during intermittent-plume gaps (a strict overestimate of background
FA). The sensitivity/FPR tradeoff proves stable across all three real corpora,
and the real-noise gap narrows to ~1 order on outdoor MOS data (archived at
`e-nose-evals/u2_gas_leak/results/tadi_field_sweep.json`). A fourth gate
**closed** (§11.10, `worner_eval.py`): the Wörner et al. (2025) 12-month /
39-day / 62-channel MOX drift corpus shows the clean-air baseline drifts
11–65σ/channel (log R) over the year — a slow additive offset that kills any
static baseline within ~2 days (Mahalanobis FPR=1.000 from Day 3) but is
absorbed by adaptive tracking: the EWMA with static Day-1 calibration and
continuous causal replay detects 682/700 (97.4%) exposures with stage-1 FPR
flat ~0.002–0.004 across all 40 days, bounding the additive drift term §11.8
left open (archived at
`e-nose-evals/u2_gas_leak/results/worner_ewma_a05_t5_continuous.json`).
Separation quality of the tracked baseline is measured separately (§11.11,
`reports/bench_separability_margin.json`): median event/clean margin +0.78σ,
95% of windows positively separated, 0.2% clean FPR, 10 s onset latency.

- Replay harness: run the Wörner et al. (2025) ~12-month, 62-sensor dataset
  through the streaming path. — **harness shipped** (`replay_dataset`,
  `ReplayMetrics`); the Wörner ingestion is the open extension, the UCI
  dynamic-mixtures recordings already score end-to-end.
- Real-data operating curve: `realdata_eval` replays the full UCI recordings at
  10 Hz, calibrates on the earliest clean window, and reports per-event
detection/latency + clean FPR vs the `--sensitivity` knob. — **done** (§11.4);
   causal-calibration variant and holdout/CIs in §11.6.
- Monte-Carlo calibration sweep across synthetic scenarios to publish
  **TPR / FPR / PPV** and detection-latency curves. — **done** (§11.8,
  `src/bin/mc_sweep.rs`, archived under
  `e-nose-evals/u2_gas_leak/results/mc_sweep_{ewma_full,dual_probe}.json`):
  fixes the deployment budget FA ≤ 1/month ⇔ FPR ≲ 4e-7, shows both detectors
  reach it on controlled noise where they target a regime (dual. sens ≤ 1.0,
  EWMA α 0.02 thr ≥ 4 — TPR 1.0, 0 FA/month on squares and bursts), quantifies
  the real-corpus FA excess as a 2–4-order additive device-noise term the
  Wörner corpus must bound, and exposes the sens>1.0 FA cliff, the smooth-ramp
  invisibility of the EWMA, and impulse-contamination FA leak. Synthetic
  curves are NOT a substitute for real-device TPR/FPR/PPV.
- Report the long-term MOX CV reproduction (25–41%) as the outside bound the
  drift model must accommodate. (pending)
- TADI-2019 field corpus (controlled industrial methane releases). — **done**
  (§11.9 `src/bin/tadi_eval.rs`): closes the deployment-corpus gap with a
  third independent real dataset; the additive real-noise term on outdoor MOS
  data is ~1 order above the algorithmic ceiling (tightening the §11.8 2–4
  order estimate, which was inflated by the dynamic-mixtures corpus's
  high-channel/high-cadence controlled environment).

Done when: there is a published (repo + docs site) TPR/FPR/PPV report on
real data — the number papers and operators actually require. — **partially
met**: real-data numbers are published in §11.4 and §11.9 (three independent
corpora: dynamic-mixtures, home-activity, field methane); the docs-site page
and the 12-month Wörner extension remain.

### Wave 4 — JS parity & interoperability guarantee

- Port `calibration.py` to `opensmell-js` (power-law, inverse concentration,
  leave-one-out cross-validation).
- Port the smellability chain (~4.5k lines) to `opensmell-js`.
- Conformance suite: a hard-to-vary compliance gate for `.osmell` v1.1.0 writers
  + a reference corpus committed to the spec repo.

Done when: any `.osmell` file produced by a conformant writer round-trips
identically across Python/Rust/JS, verified by CI.

### Wave 5 — Product & data commons

**Status.** Domain-adapter foundation shipped (`src/anomaly/adapters.rs`); the
frontend and data commons remain open.

- Frontend status/typology view (desktop + web): one status line, two time
  horizons (now = step/spike, hours = ramp/drift), "did this matter?" feedback.
  — **adapter foundation shipped**: a general `ProcessAdapter` trait plus a
  `FermentationAdapter` that maps the engine's regimes onto fermentation stages
  (`idle / lag / exponential / stationary / decline`) and turns verdicts into
  process events (`stage transition`, rapid/slow change, sensor fault) with a
  one-line `summarize`. The engine stays general-purpose; domain interpretation
  lives in the plugin.
- Data commons: quality-gated corpus, schema registry, Data Hub / HF sync,
  cross-device calibration transfer (fleet bootstrapping: a calibrated board
  seeds its uncalibrated siblings). (pending)

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