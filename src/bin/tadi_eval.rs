//! Whole-engine real-data evaluation on the TADI-2019 field corpus
//! (TotalEnergies Anomaly Detection Initiative, Zenodo 8399829).
//!
//! October 2019, 41 controlled CH4 releases at 0.15–150 g/s at an industrial
//! mock site near Pau (FR). Six low-cost Figaro TGS MOS loggers (2611C, 2600,
//! 2611E) sampled ~6 s while a high-precision CRDS analyser logged the true
//! CH4 mole fraction. This is the closest public proxy to our deployment
//! scenario: real outdoor leaks, advected plumes, sensor aging/weather present,
//! and an independent reference for the ground truth.
//!
//! Each logger CSV contains only release-measurement windows (no Release=0
//! background). Each release is a 15-min cycle: ~5 min compressed-air baseline
//! → ~5 min sample-air exposure → ~5 min compressed-air recovery.
//!
//! Evaluation mirrors a single-box deployment:
//!   1. Calibrate once on the earliest low-CH4 samples across the whole
//!      stream (the operator asserts "no leak yet" at startup).
//!   2. Replay every sample causally; record alarms.
//!   3. Per release block, score whether an alarm fires during the
//!      plume-present portion (CH4 >= --thr ppm).
//!   4. FPR aggregated over all non-plume samples (CH4 < --thr).
//!
//! Usage:
//!   tadi_eval <logger.csv> [--sensitivity <v>] [--baseline dual|ewma]
//!             [--alpha <a>] [--thr <t>] [--min-votes <v>]
//!             [--confirm-s <seconds>] [--out out.json]
//!
//!   --confirm-s <seconds> (EWMA baseline only): require a raw >=min-votes
//!   verdict to persist that many seconds of accumulated wall-clock before a
//!   release may be marked detected (decision-layer time gate; 0 disables).
use opensmell::anomaly::ewma::EwmaConfig;
use opensmell::anomaly::{DualKalmanEngine, EngineConfig, StreamDetector};
use opensmell::calibration::AutoTune;
use serde::Serialize;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

const MIN_CAL_SAMPLES: usize = 30;

#[derive(Clone)]
struct Record {
    reading: Vec<f64>,
    ch4: f64,
    release: u32,
    t: f64,
}

#[derive(Serialize, Clone)]
struct ReleaseResult {
    release: u32,
    n_rows: u64,
    event_rows: u64,
    peak_ch4: f64,
    detected: bool,
    latency_s: Option<f64>,
    event_alarms: u64,
    /// Peak per-channel sensor reading during the plume-present portion.
    peak_sensor: Vec<f64>,
    /// Per-channel reading in the last clean sample before the plume starts
    /// (the "no-leak-yet" reference the engine is calibrated against).
    pre_sensor: Vec<f64>,
    /// Peak innovation score (max_z) the engine produced during the event.
    peak_maxz: f64,
    /// Max number of channels simultaneously past the sensitive budget during
    /// the event (the corroboration count the verdict saw).
    peak_channels_past: usize,
    /// Max number of budgets fired in a single sample during the event.
    peak_votes: usize,
}

#[derive(Clone)]
struct CleanAlarm {
    t: f64,
    release: u32,
    ch4: f64,
}

#[derive(Serialize, Clone)]
struct AdsorptionMetrics {
    enabled: bool,
    two_exp: bool,
    tau_s: f64,
    tau2_s: f64,
    a1: f64,
}

#[derive(Serialize, Default)]
struct Metrics {
    file: String,
    n_channels: usize,
    sample_dt_s: f64,
    ch4_threshold_ppm: f64,
    sensitivity: f64,
    baseline: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ewma_config: Option<EwmaConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    adsorption: Option<AdsorptionMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mem_gate_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dual_confirm_s: Option<f64>,
    min_channels: usize,
    autotune: bool,
    gas_anchor: bool,
    n_rows: u64,
    n_releases: u64,
    detected_releases: u64,
    detection_rate: f64,
    clean_seconds: f64,
    clean_alarm_seconds: f64,
    clean_fpr: f64,
    fa_per_month: f64,
    median_latency_s: Option<f64>,
    typologies: BTreeMap<String, u64>,
    details: Vec<ReleaseResult>,
    runtime_s: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn parse_datetime_s(tok: &str) -> f64 {
    let t = tok.trim().replace('T', " ");
    let t = t.trim_start_matches('"').trim_end_matches('"');
    let nums: Vec<f64> = t
        .split(|c: char| c == '-' || c == ':' || c == ' ' || c == '.')
        .filter_map(|p| p.parse::<f64>().ok())
        .collect();
    if nums.len() < 4 {
        return 0.0;
    }
    let (y, mo, d) = (nums[0] as u32, nums[1] as u32, nums[2] as u32);
    let days = civil_days(d, mo, y) as f64;
    (days * 86400.0) + nums[3] * 3600.0 + nums.get(4).copied().unwrap_or(0.0) * 60.0
        + nums.get(5).copied().unwrap_or(0.0)
}

fn civil_days(day: u32, month: u32, year: u32) -> i64 {
    let y = year as i64;
    let m = month as i64;
    let d = day as i64;
    let y0 = if m <= 2 { y - 1 } else { y };
    let era = if y0 >= 0 { y0 } else { y0 - 399 } / 400;
    let yoe = y0 - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn parse_csv(path: &str) -> (Vec<String>, Vec<Record>, f64) {
    let f = File::open(path).expect("open csv");
    let mut r = BufReader::new(f);
    let mut header = String::new();
    let _ = r.read_line(&mut header);
    let cols: Vec<&str> = header.trim_end().split(',').collect();
    let skip = |c: &str| {
        c == "time" || c == "CH4" || c == "Release" || c.starts_with("RH_")
            || c.starts_with("T_") || c.starts_with("P_")
    };
    let ch_idx: Vec<usize> = cols
        .iter()
        .enumerate()
        .filter(|(_, c)| !skip(c))
        .map(|(i, _)| i)
        .collect();
    let ch_names: Vec<String> = ch_idx.iter().map(|&i| cols[i].to_string()).collect();
    let ch4_idx = cols.iter().position(|c| *c == "CH4").expect("CH4 column");
    let time_idx = cols.iter().position(|c| *c == "time").expect("time column");
    let rel_idx = cols.iter().position(|c| *c == "Release").expect("Release column");

    let mut recs: Vec<Record> = Vec::new();
    let mut t_first = None;
    let mut t_last = None;
    let mut line = String::new();
    while let Ok(n) = r.read_line(&mut line) {
        if n == 0 {
            break;
        }
        if line.trim().is_empty() {
            line.clear();
            continue;
        }
        let toks: Vec<&str> = line.trim_end().split(',').collect();
        if toks.len() < cols.len() {
            line.clear();
            continue;
        }
        let reading: Vec<f64> = ch_idx
            .iter()
            .map(|&i| toks[i].trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        if reading.iter().any(|v| !v.is_finite()) {
            line.clear();
            continue;
        }
        let ch4: f64 = toks[ch4_idx].trim().parse().unwrap_or(0.0);
        let rel: u32 = toks[rel_idx].trim().parse().unwrap_or(0);
        let t = parse_datetime_s(toks[time_idx]);
        if t_first.is_none() {
            t_first = Some(t);
        }
        t_last = Some(t);
        recs.push(Record {
            reading,
            ch4,
            release: rel,
            t,
        });
        line.clear();
    }
    let dt = if recs.len() > 1 {
        let diffs: Vec<f64> = recs.windows(2).map(|w| w[1].t - w[0].t).filter(|d| *d > 0.0 && *d < 600.0).collect();
        let mut sorted = diffs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !sorted.is_empty() {
            sorted[sorted.len() / 2]
        } else {
            0.0
        }
    } else {
        0.0
    };
    (ch_names, recs, dt.max(1.0))
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = env::args().collect();
    let data_path = args
        .get(1)
        .expect("usage: tadi_eval <logger.csv> [--sensitivity <v>] [--out out.json]");
    let sensitivity = args
        .iter()
        .rposition(|a| a == "--sensitivity")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(3.0)
        .max(1e-3);
    let ch4_thr = args
        .iter()
        .rposition(|a| a == "--thr")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(5.0)
        .max(0.1);
    let out_path = args.iter().rposition(|a| a == "--out").map(|i| args[i + 1].clone());
    let baseline = args
        .iter()
        .position(|a| a == "--baseline")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("dual")
        .to_string();
    let mut ewma_cfg = EwmaConfig {
        alpha: args
            .iter()
            .rposition(|a| a == "--alpha")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.05),
        threshold_sigma: args
            .iter()
            .rposition(|a| a == "--thr-sigma")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(5.0),
        min_votes: args
            .iter()
            .rposition(|a| a == "--min-votes")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2),
        sample_period_s: 0.1,
        confirm_window_s: args
            .iter()
            .rposition(|a| a == "--confirm-s")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0)
            .max(0.0),
    };

    // Phase-1.1 memory levers (all default-off, preserving prior behaviour):
    //   --adsorption     enable the engine's adsorption-memory state (residue
    //                    absorbed into a decaying memory, not scored as drift)
    //   --two-exp        bi-exponential residue (fast wash-out + slow plateau,
    //                    from the W1 dynamic-memory finding); implies adsorption
    //   --tau-s <s>      per-channel desorption time constant (default 300 s)
    //   --tau2-s <s>     slow-component tau for --two-exp (default 15 s)
    //   --a1 <0..1>      fast-component weight for --two-exp (default 0.6)
    //   --mem-gate-s <s> post-event wall-clock cooldown: suppress alarms for
    //                    this many seconds after the previous alarm. Levels are
    //                    never suppressed for the *current* release's first hit,
    //                    and detection latency is measured pre-gate.
    //   --min-channels <n> multi-sensor corroboration: require ≥n channels to
    //                    clear the sensitive budget before an anomaly is
    //                    trusted (default 1 = single-channel max_z verdict).
    //   --autotune       per-array physics from measurements: fit q_state and
    //                    baseline variance from the logger's own calibration
    //                    window (AutoTune::from_baseline) instead of the
    //                    engine defaults; drift-derived q_param from the
    //                    per-minute baseline path.
    //   --gas-anchor     per-gas response anchor: fit the per-channel power-law
    //                    exponent α from the log-log slope of (CH4 reference,
    //                    sensor reading) over the calibration window and enable
    //                    the power-law measurement model (the Wörner response
    //                    shape, but fitted to this logger's own sensors).
    //   --physics-out <f> write the fitted EngineConfig (tuned physics +
    //                    calibration R/mean) to JSON after calibration.
    //   --physics-from <f> load an EngineConfig fitted on another device and
    //                    use its *physics* (q_state, q_param, τ, α, sensitivity,
    //                    adsorption, confirm budget) while still calibrating
    //                    this device's own baseline mean/R. This is the
    //                    Phase-1.3 independent-device holdout: does a tuned
    //                    physics transfer across loggers?
    let physics_out = args
        .iter()
        .rposition(|a| a == "--physics-out")
        .map(|i| args[i + 1].clone());
    let physics_from = args
        .iter()
        .rposition(|a| a == "--physics-from")
        .map(|i| args[i + 1].clone());
    // Phase-1.3 independent-device holdout: load physics tuned on another
    // device. The EngineConfig is deserialized and its physics applied; the
    // baseline itself (mean/R) is still calibrated from THIS device's own
    // calibration window (as an honest deployment must).
    let loaded_physics: Option<EngineConfig> = physics_from.as_ref().and_then(|p| {
        let txt = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&txt).ok()
    });
    let autotune = args.iter().position(|a| a == "--autotune").is_some();
    let gas_anchor = args.iter().position(|a| a == "--gas-anchor").is_some();
    // Shared-physics override (Phase-1.3b): a fleet/class-level walk budget (a
    // sensor-class property) applied on every device while R stays per-device.
    let q_state_ov = args
        .iter()
        .rposition(|a| a == "--q-state")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| *v > 0.0);
    let q_param_ov = args
        .iter()
        .rposition(|a| a == "--q-param")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| *v > 0.0);
    let level_budget = args
        .iter()
        .rposition(|a| a == "--level-budget")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        .max(0.0);
    let level_rise_s = args
        .iter()
        .rposition(|a| a == "--level-rise")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        .max(0.0);
    let level_ref_s = args
        .iter()
        .rposition(|a| a == "--level-ref")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        .max(0.0);
    let adsorption = args
        .iter()
        .position(|a| a == "--adsorption")
        .is_some()
        || args.iter().position(|a| a == "--two-exp").is_some();
    let two_exp = args.iter().position(|a| a == "--two-exp").is_some();
    let tau_s = args
        .iter()
        .rposition(|a| a == "--tau-s")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(300.0)
        .max(0.1);
    let tau2_s = args
        .iter()
        .rposition(|a| a == "--tau2-s")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(15.0)
        .max(0.1);
    let a1 = args
        .iter()
        .rposition(|a| a == "--a1")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.6)
        .clamp(0.0, 1.0);
    let mem_gate_s = args
        .iter()
        .rposition(|a| a == "--mem-gate-s")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        .max(0.0);
    // Pre-verdict wall-clock persistence for the DUAL baseline (mirrors the
    // EWMA confirmation window): a raw anomaly only asserts after it has
    // persisted `confirm_s` accumulated seconds; any normal sample resets the
    // accumulator. 0.0 = a single-sample vote fires immediately.
    let dual_confirm_s = args
        .iter()
        .rposition(|a| a == "--confirm-s")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        .max(0.0);
    let min_channels = args
        .iter()
        .rposition(|a| a == "--min-channels")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);

    let (ch_names, records, dt_s) = parse_csv(data_path);
    // Drive the detectors at the logger's real cadence (TADI samples ~every
    // 6 s, not the engine's reference 10 Hz). The EWMA smoothing constant is
    // quoted at this cadence; per-sample `detect_with_dt` handles the gaps.
    //
    // `--force-dt <s>` overrides every inter-sample gap (a legacy-simulation
    // knob for the published re-run: feeding a 6 s logger as 10 Hz was the old
    // field bug, and this reproduces it for an honest before/after table).
    let force_dt = args
        .iter()
        .rposition(|a| a == "--force-dt")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|d| *d > 0.0);
    ewma_cfg.sample_period_s = if let Some(fd) = force_dt {
        fd.max(1e-6)
    } else {
        dt_s.max(1e-6)
    };
    let n_ch = ch_names.len();

    // Find the global calibration window: earliest contiguous CH4 < ch4_thr
    // samples, spanning at least MIN_CAL.
    let mut cal_end = 0usize;
    for (i, rec) in records.iter().enumerate() {
        if rec.ch4 >= ch4_thr {
            cal_end = i;
            break;
        }
        cal_end = i + 1;
    }
    if cal_end < MIN_CAL_SAMPLES {
        eprintln!(
            "no calibration window found (<{} pre-event samples in first {} rows); skipping",
            MIN_CAL_SAMPLES,
            records.len()
        );
        return;
    }
    let cal_start = 0;
    let cal_samples: Vec<Vec<f64>> = records[cal_start..cal_end]
        .iter()
        .map(|r| r.reading.clone())
        .collect();
    // `--cal-cap <seconds>` shortens the *warm-up* used for the baseline/AutoTune
    // fit (Phase-2 protocol question: how little clean data can a deployment
    // calibrate from and still verify to 0 FA?); verification still runs over
    // the entire held-out clean period, so a pass here is honest.
    let cal_cap = args
        .iter()
        .rposition(|a| a == "--cal-cap")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .map(|s| (s / dt_s).max(1.0) as usize)
        .unwrap_or(usize::MAX);
    let use_cal = cal_cap.min(cal_samples.len());
    let fit_samples: Vec<Vec<f64>> = cal_samples[..use_cal].to_vec();

    // Per-sample duration in seconds (median cadence where the gap is out of
    // range, so cross-day file splits don't inject absurd process noise).
    let mut dt_per = vec![dt_s; records.len()];
    for i in 1..records.len() {
        let gap = records[i].t - records[i - 1].t;
        if gap > 0.0 && gap < 600.0 {
            dt_per[i] = gap;
        }
    }

    if let Some(fd) = force_dt {
        for d in dt_per.iter_mut() {
            *d = fd;
        }
    }

    // Single global deployment: calibrate once, replay the whole stream.
    let mut det = if baseline == "ewma" {
        StreamDetector::ewma(n_ch, ewma_cfg.clone())
    } else if adsorption || min_channels > 1 || autotune || gas_anchor || loaded_physics.is_some() || level_budget > 0.0 {
        // Phase-1.1/1.2 levers:
        //   - adsorption memory (residue decays in a dedicated state block)
        //   - multi-sensor corroboration (--min-channels)
        //   - per-array AutoTune: q_state / q_param / baseline_std fitted from
        //     the logger's own calibration window (measurement-derived, not
        //     engine defaults)
        //   - per-gas response anchor: power-law α fitted from the log-log
        //     slope of (CRDS CH4 reference, sensor reading) and the power-law
        //     measurement model enabled.
        //   - --physics-from: a config tuned on another device supplies the
        //     physics (q_state, q_param, τ, α, sensitivity, adsorption, budget),
        //     applied on top of this device's own baseline calibration.
        let mut e = DualKalmanEngine::new(n_ch);
        if let Some(src) = &loaded_physics {
            // Adopt the source device's physics wholesale (its tuned noise
            // budgets, response shape, adsorption memory), but force THIS
            // device's decision knobs from the CLI flags so a holdout compares
            // the same verdict layer.
            e.config.sensitivity = src.sensitivity;
            e.config.q_state = src.q_state;
            e.config.q_param = src.q_param;
            e.config.adsorption = src.adsorption.clone();
            e.config.response = src.response.clone();
            e.config.min_channels = min_channels.max(src.min_channels);
        }
        e.config.sensitivity = sensitivity;
        e.config.adsorption.enabled = adsorption;
        e.config.adsorption.two_exp = two_exp;
        e.config.adsorption.tau_default_s = tau_s;
        e.config.adsorption.tau2_default_s = tau2_s;
        e.config.adsorption.a1_default = a1;
        e.config.min_channels = min_channels;
        if autotune {
            // Fit the physics knobs from the same calibration window that
            // `calibrate_baseline` uses (steady, pre-event CH4 < --thr).
            if let Ok(tuned) = AutoTune::from_baseline(&fit_samples) {
                tuned.apply(&mut e.config);
            }
        }
        if let Some(q) = q_state_ov {
            e.config.q_state = q;
        }
        if let Some(q) = q_param_ov {
            e.config.q_param = q;
        }
        e.config.level_budget = level_budget;
        e.config.level_rise_lookback_s = level_rise_s;
        e.config.level_ref_s = level_ref_s;
        if gas_anchor {
            // Fit the per-channel response exponent against the reference CH4
            // column over the *plume-present* portion (CH4 >= --thr): that is
            // where the sensor is actually driven, so the log-log slope is the
            // real response shape. The clean-window fit is degenerate (CH4 is
            // ~constant ~2 ppm there, so a slope has nothing to span).
            let c = ch_names.len();
            let mut pairs: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
            for rec in records.iter() {
                if rec.ch4 < ch4_thr {
                    continue;
                }
                let x = vec![rec.ch4; c];
                let mut y = rec.reading.clone();
                y.resize(c, 0.0);
                pairs.push((x, y));
            }
            let mut tuned = AutoTune {
                baseline_std: vec![1.0; c],
                ..Default::default()
            };
            if let Ok(t) = tuned.with_response(&pairs) {
                let _ = t.apply(&mut e.config);
                e.config.response.power_law_enabled = true;
            }
        }
        e.set_config(e.config.clone());
        StreamDetector::Dual(e)
    } else {
        StreamDetector::dual(n_ch, sensitivity)
    };
    // Sensitive-budget threshold (k_std[2]/sens) the corroboration gate uses —
    // needed to replay the verdict's count of channels past budget.
    let self_k_std2 = det
        .engine_config()
        .map(|c| c.k_std[2])
        .unwrap_or(3.0);
    let calibrated = fit_samples.len() >= n_ch && det.calibrate(&fit_samples).is_ok();
    if !calibrated {
        eprintln!("calibration failed (n_ch={n_ch}, cal_samples={})", cal_samples.len());
        return;
    }
    // Warm up on calibration samples at their real cadence.
    for (i, r) in records[cal_start..cal_end].iter().enumerate() {
        let _ = det.detect_with_dt(&r.reading, dt_per[cal_start + i]);
    }

    // Snapshot the fitted engine config (physics + calibrated baseline) for a
    // Phase-1.3 holdout: another device adopts this physics while calibrating
    // its own baseline.
    if let Some(path) = &physics_out {
        if let Some(cfg) = det.engine_config() {
            let txt = serde_json::to_string_pretty(&cfg).expect("serialize physics");
            let mut f = File::create(path).expect("create physics-out");
            f.write_all(txt.as_bytes()).expect("write physics-out");
        } else {
            eprintln!("--physics-out only supported on the dual baseline");
        }
    }

    // Replay the entire stream from cal_end onward, recording alarms. Each
    // sample drives the engine at its own inter-sample gap, so the ~6 s TADI
    // cadence maps to the same physical process model the engine is tuned for
    // (Q scaled by dt/period, desorption decay exp(−dt/τ), real warm-up time).
    let mut alarms = vec![false; records.len()];
    let mut kinds: BTreeMap<String, u64> = BTreeMap::new();
    // Optionally dump each clean-period alarm (t, ch4, release) for locating
    // the handful of isolated FPs that dominate false-alarm seconds.
    let clean_dump = args
        .iter()
        .rposition(|a| a == "--clean-dump")
        .map(|i| args[i + 1].clone());
    let mut clean_alarms: Vec<CleanAlarm> = Vec::new();
    // Wall-clock memory cooldown: once an alarm has fired, hold the decision
    // layer quiet for `mem_gate_s` (default 0 = off) to ride out the
    // post-exposure residue spike the W2 bench measured (FP peaks 30-60 s
    // after an exposure and only collapses past ~300 s). Detection latency is
    // scored against the pre-gate alarm stream so the gate cannot inflate it.
    let mut last_alarm_t = -1e300;
    // Pre-verdict persistence accumulator for the dual baseline (decision-layer
    // only, mirrors the EWMA confirmation window).
    let mut dual_hold_s = 0.0f64;
    // Per-sample verdict internals (replay diagnostics): peak innovation,
    // corroboration count, and budget votes as the engine saw them.
    let mut peak_z_per_sample = vec![0.0f64; records.len()];
    let mut corrob_per_sample = vec![0usize; records.len()];
    let mut votes_per_sample = vec![0usize; records.len()];
    for (i, rec) in records.iter().enumerate().skip(cal_end) {
        let mut kind = None;
        let mut anom = if let Ok(v) = det.detect_with_dt(&rec.reading, dt_per[i]) {
            kind = v.kind.clone();
            peak_z_per_sample[i] = v.max_z;
            corrob_per_sample[i] = v
                .z_scores
                .iter()
                .filter(|&&z| z > self_k_std2 / sensitivity)
                .count();
            votes_per_sample[i] = v.anomaly_votes;
            v.is_anomaly
        } else {
            false
        };
        if baseline != "ewma" && dual_confirm_s > 0.0 {
            if anom {
                dual_hold_s += dt_per[i].max(1e-6);
                anom = dual_hold_s + 1e-9 >= dual_confirm_s;
            } else {
                dual_hold_s = 0.0;
            }
        }
        if anom && mem_gate_s > 0.0 && records[i].t - last_alarm_t < mem_gate_s {
            anom = false;
        }
        if anom {
            last_alarm_t = records[i].t;
        }
        alarms[i] = anom;
        if anom {
            *kinds
                .entry(kind.clone().unwrap_or_else(|| "none".to_string()))
                .or_insert(0) += 1;
        }
        if anom && rec.ch4 < ch4_thr {
            clean_alarms.push(CleanAlarm {
                t: rec.t,
                release: rec.release,
                ch4: rec.ch4,
            });
        }
    }

    // Locate the isolated clean-period FPs that dominate false-alarm seconds.
    if let Some(path) = &clean_dump {
        let mut out = String::from("t,release,ch4\n");
        for ca in &clean_alarms {
            out.push_str(&format!("{},{},{}\n", ca.t, ca.release, ca.ch4));
        }
        let mut f = File::create(path).expect("create clean dump");
        f.write_all(out.as_bytes()).expect("write clean dump");
    }

    // Build release blocks (in order of first appearance).
    let mut by_release: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, rec) in records.iter().enumerate() {
        by_release.entry(rec.release).or_default().push(i);
    }

    let mut clean_s = 0.0f64;
    let mut clean_ala = 0.0f64;
    let mut latencies_ok: Vec<f64> = Vec::new();
    let mut detected_count = 0u64;
    let mut details: Vec<ReleaseResult> = Vec::new();

    for (&rel, rows) in by_release.iter() {
        let n_rows = rows.len();
        let peak = rows.iter().map(|&i| records[i].ch4).fold(0.0f64, f64::max);
        let mut peak_sensor = vec![0.0f64; n_ch];
        let mut pre_sensor = vec![0.0f64; n_ch];
        let mut init_pre = false;
        let mut det_r = ReleaseResult {
            release: rel,
            n_rows: n_rows as u64,
            event_rows: 0,
            peak_ch4: peak,
            detected: false,
            latency_s: None,
            event_alarms: 0,
            peak_sensor,
            pre_sensor,
            peak_maxz: 0.0,
            peak_channels_past: 0,
            peak_votes: 0,
        };
        let first_event = rows.iter().position(|&i| records[i].ch4 >= ch4_thr);
        let first_event_time = first_event.map(|fi| records[rows[fi]].t);
        let mut fired = false;
        let mut event_s = 0.0f64;
        let mut event_a = 0u64;
        let mut clean_s_local = 0.0f64;
        let mut clean_aa_local = 0.0f64;
        for &i in rows {
            if i < cal_end {
                continue;
            }
            let is_event = records[i].ch4 >= ch4_thr;
            if is_event {
                det_r.event_rows += 1;
                event_s += dt_per[i];
                for (c, &v) in records[i].reading.iter().enumerate() {
                    if v > det_r.peak_sensor[c] {
                        det_r.peak_sensor[c] = v;
                    }
                }
                if peak_z_per_sample[i] > det_r.peak_maxz {
                    det_r.peak_maxz = peak_z_per_sample[i];
                }
                if corrob_per_sample[i] > det_r.peak_channels_past {
                    det_r.peak_channels_past = corrob_per_sample[i];
                }
                if votes_per_sample[i] > det_r.peak_votes {
                    det_r.peak_votes = votes_per_sample[i];
                }
                if alarms[i] {
                    det_r.event_alarms += 1;
                    event_a += 1;
                }
            } else {
                if !init_pre && !det_r.pre_sensor.is_empty() {
                    det_r.pre_sensor = records[i].reading.clone();
                    init_pre = true;
                }
                // Alarm seconds, not samples: each clean alarm contributes the
                // wall-clock it covered (dt; ~6 s for TADI), so FPR is a
                // per-second probability, not an under-scaled sample ratio.
                clean_s_local += dt_per[i];
                if alarms[i] {
                    clean_aa_local += dt_per[i];
                }
            }
            if !fired && is_event && alarms[i] && first_event_time.is_some() {
                fired = true;
                det_r.detected = true;
                det_r.latency_s = Some(records[i].t - first_event_time.unwrap());
            }
        }
        clean_s += clean_s_local;
        clean_ala += clean_aa_local;
        if det_r.detected {
            detected_count += 1;
            latencies_ok.push(det_r.latency_s.unwrap_or(0.0));
        }
        details.push(det_r);
    }

    let n_releases = by_release.len() as u64;
    let mut lat_sorted = latencies_ok.clone();
    lat_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let m = Metrics {
        file: data_path.clone(),
        n_channels: n_ch,
        sample_dt_s: force_dt.unwrap_or(dt_s),
        ch4_threshold_ppm: ch4_thr,
        sensitivity,
        baseline: baseline.clone(),
        ewma_config: if baseline == "ewma" {
            Some(ewma_cfg.clone())
        } else {
            None
        },
        adsorption: if adsorption {
            Some(AdsorptionMetrics {
                enabled: true,
                two_exp,
                tau_s,
                tau2_s,
                a1,
            })
        } else {
            None
        },
        mem_gate_s: if mem_gate_s > 0.0 {
            Some(mem_gate_s)
        } else {
            None
        },
        dual_confirm_s: if dual_confirm_s > 0.0 {
            Some(dual_confirm_s)
        } else {
            None
        },
        min_channels,
        autotune,
        gas_anchor,
        n_rows: records.len() as u64,
        n_releases,
        detected_releases: detected_count,
        detection_rate: if n_releases > 0 {
            detected_count as f64 / n_releases as f64
        } else {
            0.0
        },
        clean_seconds: clean_s,
        clean_alarm_seconds: clean_ala,
        clean_fpr: if clean_s > 0.0 {
            // Honest per-second false-alarm probability (alarm-seconds over
            // clean-seconds), not a sample-count ratio.
            clean_ala / clean_s
        } else {
            0.0
        },
        fa_per_month: if clean_s > 0.0 {
            (clean_ala / clean_s) * 86400.0 * 30.0
        } else {
            0.0
        },
        median_latency_s: percentile(&lat_sorted, 0.5).into(),
        typologies: kinds,
        details,
        runtime_s: t0.elapsed().as_secs_f64(),
    };

    let json = serde_json::to_string_pretty(&m).expect("serialize");
    if let Some(op) = &out_path {
        let mut f = File::create(op).expect("create output");
        f.write_all(json.as_bytes()).expect("write output");
        println!("wrote {op}");
    } else {
        println!("{json}");
    }
}
