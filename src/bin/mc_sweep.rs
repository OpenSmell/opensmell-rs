//! Monte-Carlo TPR/FPR/PPV sweep (open §12 Wave-3 item).
//!
//! Synthetic replay stress test. Generates multi-channel MOX-like streams at
//! 10 Hz from a parameterised model — AR(1) per-channel noise, a slow per-
//! channel drift random walk, optional impulse contamination, and either of
//! the two event regimes observed in the real corpora: square pulses
//! (the dynamic-mixtures regime) and weak slowly rising ramps (the home-
//! activity regime). Each stream is then replayed through the SAME scoring
//! pipeline as `realdata_eval`/`indoor_eval` (causal calibration on the first
//! clean window, 120 s detection window, 60 s recovery margins) for every
//! `StreamDetector` configuration in the grid, and results are averaged over
//! seeded realizations so all configs see identical streams.
//!
//! Decision-theoretic readout: every (scenario × config) cell is scored with
//! TPR, clean FPR, false alarms per day / per month at 24/7 operation, and
//! PPV at the operator leak rates 0.1/1/5 events per day. A cell qualifies for
//! risky deployment when false alarms stay ≤ 1/month at TPR ≥ 0.9 — the FA
//! budget a real gas alarm needs to avoid alarm fatigue.
//!
//! Honesty note: synthetic stimuli characterise the detectors' operating
//! curves under controlled ground truth and a stated FPR budget. They are NOT
//! a substitute for long-term drift data (the Wörner corpus gap): one array,
//! one day, no physical sensor ageing — the same fixed-device limitation all
//! real corpora here share.
//!
//! Usage: mc_sweep [--out results.json] [--seeds N] [--scenario <substr>] [--quick]
//! `--scenario` filters by name substring; `--quick` = 2 seeds for fast dev.
use opensmell::anomaly::ewma::EwmaConfig;
use opensmell::anomaly::StreamDetector;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const N_CH: usize = 16;
const OUT_HZ: usize = 10;
const STARTUP_S: usize = 120;
const CAL_MIN_S: usize = 300;
const CAL_FLOOR_S: usize = 120;
const DETECT_WINDOW_S: usize = 120;
const RECOVERY_S: usize = 60;
const PRE_EVENT_S: usize = 60;
const DETECT_WINDOW: usize = DETECT_WINDOW_S * OUT_HZ;

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Regime {
    Square,
    Ramp,
    Clean,
    Burst,
}

struct Scenario {
    name: &'static str,
    regime: Regime,
    duration_s: usize,
    n_events: usize,
    strength: f64,
    burst: f64,
    drift60: f64,
    spike_p: f64,
    spike_amp: f64,
    noise_rho: f64,
    seed_base: u64,
}

struct Sample {
    ch: [f64; N_CH],
    truth: bool,
}

#[derive(Clone)]
enum DetCfg {
    Dual(f64),
    Ewma(EwmaConfig),
}

#[derive(Default, Serialize, Clone, Copy)]
struct CellAgg {
    seed: u64,
    n_events: u64,
    detected: u64,
    tpr: Option<f64>,
    median_latency_s: Option<f64>,
    p75_latency_s: Option<f64>,
    clean_seconds: u64,
    clean_alarms: u64,
    fpr: f64,
    fa_per_day: f64,
    fa_per_month: f64,
    ppv_at_01: f64,
    ppv_at_1: f64,
    ppv_at_5: f64,
    deploy_admissible: bool,
}

fn gauss(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn schedule(rng: &mut StdRng, regime: &Regime, n: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let mut t = 600.0; // first event comfortably after STARTUP + CAL_MIN = 420 s
    for _ in 0..n {
        match regime {
            Regime::Square | Regime::Clean => {
                let dur = 80.0 + rng.gen::<f64>() * 40.0;
                out.push((t, t + dur));
                t = t + dur + 60.0 + rng.gen::<f64>() * 60.0;
            }
            Regime::Ramp | Regime::Burst => {
                // 120 s rise, 180 s hold, 120 s fall, ~13 min between onsets
                // (6 events fit in 2 h).
                let dur = 420.0;
                out.push((t, t + dur));
                t = t + dur + 780.0;
            }
        }
    }
    out
}

fn envelope(regime: &Regime, es: f64, ee: f64, sec: f64) -> f64 {
    match regime {
        Regime::Square | Regime::Clean | Regime::Burst => {
            let edge = if matches!(regime, Regime::Burst) {
                2.0
            } else {
                0.5
            };
            if sec < es || sec > ee {
                0.0
            } else if sec < es + edge {
                (sec - es) / edge
            } else if sec > ee - edge {
                (ee - sec) / edge
            } else {
                1.0
            }
        }
        Regime::Ramp => {
            let rise = (ee - es).min(120.0);
            let fall = (ee - es).min(120.0);
            if sec < es || sec > ee {
                0.0
            } else if sec < es + rise {
                (sec - es) / rise
            } else if sec > ee - fall {
                (ee - sec) / fall
            } else {
                1.0
            }
        }
    }
}

impl Scenario {
    /// Deterministically generate one 10 Hz realization.
    fn generate(&self, seed: u64) -> Vec<Sample> {
        let mut rng = StdRng::seed_from_u64(self.seed_base ^ seed);
        let n = self.duration_s * OUT_HZ;
        let events = schedule(&mut rng, &self.regime, self.n_events);

        let mut which = [0.0f64; N_CH];
        for w in which.iter_mut() {
            *w = if rng.gen::<f64>() < 0.6 {
                0.5 + rng.gen::<f64>()
            } else {
                rng.gen::<f64>() * 0.2
            };
        }

        let mut samples = Vec::with_capacity(n);
        let mut drift = [0.0f64; N_CH];
        let mut noise = [0.0f64; N_CH];
        let mut next_drift = 600usize;
        let mut ev_idx = 0usize;
        for t in 0..n {
            if t == next_drift {
                next_drift += 600;
                for d in drift.iter_mut() {
                    *d += gauss(&mut rng) * self.drift60;
                }
            }
            let sec = t as f64 / OUT_HZ as f64;
            while ev_idx < events.len() && events[ev_idx].1 < sec {
                ev_idx += 1;
            }
            let cur = events.get(ev_idx).copied();
            let truth = cur.is_some_and(|(es, ee)| sec >= es && sec <= ee);
            let env_val = cur.map_or(0.0, |(es, ee)| envelope(&self.regime, es, ee, sec));
            let amplitude = env_val * self.strength;

            let mut ch = [0.0f64; N_CH];
            for c in 0..N_CH {
                // Burst regime emulates second-scale stochastic excitation: on
                // responsive channels the event inflates the noise innovation,
                // so EWMA sees transient 5σ+ multi-channel spikes at onset
                // (the mechanism behind its home-activity detections).
                let burst_factor = if self.burst > 0.0 && env_val > 0.0 {
                    1.0 + self.burst * which[c] * env_val
                } else {
                    1.0
                };
                noise[c] = self.noise_rho * noise[c]
                    + (1.0 - self.noise_rho * self.noise_rho).sqrt()
                        * gauss(&mut rng)
                        * burst_factor;
                let mut x = drift[c] + noise[c];
                if amplitude > 0.0 {
                    x += amplitude * which[c];
                }
                if rng.gen::<f64>() < self.spike_p {
                    x += gauss(&mut rng) * self.spike_amp;
                }
                ch[c] = x;
            }
            samples.push(Sample { ch, truth });
        }
        samples
    }
}

fn find_cal_window(samples: &[Sample]) -> Option<(usize, usize)> {
    let start0 = STARTUP_S * OUT_HZ;
    if start0 >= samples.len() {
        return None;
    }
    let mut i = start0;
    while i + CAL_FLOOR_S * OUT_HZ <= samples.len() {
        for &period in [CAL_MIN_S * OUT_HZ, CAL_FLOOR_S * OUT_HZ].iter() {
            if i + period <= samples.len() && samples[i..i + period].iter().all(|s| !s.truth) {
                return Some((i, i + period));
            }
        }
        i += OUT_HZ;
    }
    None
}

fn find_events(samples: &[Sample]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < samples.len() {
        if samples[i].truth {
            let start = i;
            while i < samples.len() && samples[i].truth {
                i += 1;
            }
            out.push((start, i.saturating_sub(1)));
        } else {
            i += 1;
        }
    }
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for &(s, e) in &out {
        if let Some(last) = merged.last_mut() {
            if s <= last.1 + 5 {
                last.1 = e;
                continue;
            }
        }
        merged.push((s, e));
    }
    merged
}

fn ppv(r: f64, tpr: Option<f64>, fa_day: f64) -> f64 {
    match tpr {
        Some(t) => {
            let d = r * t;
            d / (d + fa_day)
        }
        None => 0.0,
    }
}

/// Replay one stream through one configuration, mirroring the evaluators'
/// windowing and margins exactly.
fn evaluate(seed: u64, cfg: &DetCfg, samples: &[Sample]) -> CellAgg {
    let n = samples.len();
    let cal = find_cal_window(samples).expect("cal window");
    let (cal_start, cal_end) = cal;

    let mut det = match cfg {
        DetCfg::Dual(s) => StreamDetector::dual(N_CH, *s),
        DetCfg::Ewma(c) => StreamDetector::ewma(N_CH, c.clone()),
    };
    let cal_samples: Vec<Vec<f64>> = samples[cal_start..cal_end]
        .iter()
        .map(|s| s.ch.to_vec())
        .collect();
    det.calibrate(&cal_samples).expect("calibrate");
    for s in &samples[..cal_end] {
        let _ = det.detect(&s.ch);
    }

    let mut alarms = vec![false; n];
    for (i, s) in samples.iter().enumerate().skip(cal_end) {
        if let Ok(v) = det.detect(&s.ch) {
            alarms[i] = v.is_anomaly;
        }
    }

    let events = find_events(samples);
    let mut latencies: Vec<f64> = Vec::new();
    let mut detected = 0u64;
    let mut scored = 0u64;
    for &(s, _e) in &events {
        if s < cal_end {
            continue;
        }
        scored += 1;
        let window_end = (s + DETECT_WINDOW).min(n);
        if let Some(off) = alarms[s..window_end].iter().position(|a| *a) {
            detected += 1;
            latencies.push(off as f64 / OUT_HZ as f64);
        }
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut clean_seconds = 0u64;
    let mut clean_alarms = 0u64;
    for (i, &alarmed) in alarms.iter().enumerate().skip(cal_end) {
        let in_event = events.iter().any(|&(s, e)| i >= s && i <= e);
        let after_event = events.iter().any(|&(_s, e)| i > e && i <= e + RECOVERY_S * OUT_HZ);
        let before_event = events
            .iter()
            .any(|&(s, _e)| i < s && s - i <= PRE_EVENT_S * OUT_HZ);
        if in_event {
            // event seconds: not counted in clean FPR
        } else if after_event || before_event {
            // recovery margins: excluded from clean FPR
        } else {
            clean_seconds += 1;
            if alarmed {
                clean_alarms += 1;
            }
        }
    }

    let fpr = if clean_seconds > 0 {
        clean_alarms as f64 / clean_seconds as f64
    } else {
        0.0
    };
    let tpr = if scored > 0 {
        Some(detected as f64 / scored as f64)
    } else {
        None
    };
    let fa_day = fpr * 86400.0;
    let fa_month = fpr * 2_592_000.0;

    CellAgg {
        seed,
        n_events: scored,
        detected,
        tpr,
        median_latency_s: percentile(&latencies, 0.5),
        p75_latency_s: percentile(&latencies, 0.75),
        clean_seconds,
        clean_alarms,
        fpr,
        fa_per_day: fa_day,
        fa_per_month: fa_month,
        ppv_at_01: ppv(0.1, tpr, fa_day),
        ppv_at_1: ppv(1.0, tpr, fa_day),
        ppv_at_5: ppv(5.0, tpr, fa_day),
        deploy_admissible: fa_month <= 1.0 && tpr.is_some_and(|t| t >= 0.9),
    }
}

#[derive(Serialize)]
struct RawCell {
    scenario: String,
    config: String,
    #[serde(flatten)]
    agg: CellAgg,
}

#[derive(Default, Serialize)]
struct AggCell {
    scenario: String,
    config: String,
    n_seeds: u64,
    n_events: u64,
    detected: u64,
    tpr: Option<f64>,
    median_latency_s: Option<f64>,
    p75_latency_s: Option<f64>,
    fpr: f64,
    fa_per_day: f64,
    fa_per_month: f64,
    ppv_at_01: f64,
    ppv_at_1: f64,
    ppv_at_5: f64,
    deploy_meets_bar: bool,
}

impl AggCell {
    fn from_cells(scenario: &str, config: &str, cells: &[CellAgg]) -> Self {
        let n = cells.len() as f64;
        let sums = |f: fn(&CellAgg) -> f64| cells.iter().map(f).sum::<f64>() / n;
        let tpr_avg = {
            let vals: Vec<f64> = cells.iter().filter_map(|c| c.tpr).collect();
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().sum::<f64>() / vals.len() as f64)
            }
        };
        let med = |pick: fn(&CellAgg) -> Option<f64>, p: f64| {
            let mut vals: Vec<f64> = cells.iter().filter_map(pick).collect();
            if vals.is_empty() {
                return None;
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            percentile(&vals, p)
        };
        let deploy = cells.iter().all(|c| c.deploy_admissible);
        Self {
            scenario: scenario.to_string(),
            config: config.to_string(),
            n_seeds: cells.len() as u64,
            n_events: cells.iter().map(|c| c.n_events).sum(),
            detected: cells.iter().map(|c| c.detected).sum(),
            tpr: tpr_avg,
            median_latency_s: med(|c| c.median_latency_s, 0.5),
            p75_latency_s: med(|c| c.p75_latency_s, 0.75),
            fpr: sums(|c| c.fpr),
            fa_per_day: sums(|c| c.fa_per_day),
            fa_per_month: sums(|c| c.fa_per_month),
            ppv_at_01: sums(|c| c.ppv_at_01),
            ppv_at_1: sums(|c| c.ppv_at_1),
            ppv_at_5: sums(|c| c.ppv_at_5),
            deploy_meets_bar: deploy,
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    Some(sorted[idx.min(sorted.len() - 1)])
}

#[derive(Serialize)]
struct Archive {
    note: String,
    seeds: u64,
    scenarios: Vec<String>,
    configs: Vec<String>,
    raw: Vec<RawCell>,
    aggregates: Vec<AggCell>,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "sq_06",
            regime: Regime::Square,
            duration_s: 5400,
            n_events: 20,
            strength: 6.0,
            burst: 0.0,
            drift60: 0.1,
            spike_p: 0.0,
            spike_amp: 0.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9,
        },
        Scenario {
            name: "sq_10",
            regime: Regime::Square,
            duration_s: 5400,
            n_events: 20,
            strength: 10.0,
            burst: 0.0,
            drift60: 0.1,
            spike_p: 0.0,
            spike_amp: 0.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 1,
        },
        Scenario {
            name: "sq_20",
            regime: Regime::Square,
            duration_s: 5400,
            n_events: 20,
            strength: 20.0,
            burst: 0.0,
            drift60: 0.1,
            spike_p: 0.0,
            spike_amp: 0.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 2,
        },
        Scenario {
            name: "sq_10_drift",
            regime: Regime::Square,
            duration_s: 5400,
            n_events: 20,
            strength: 10.0,
            burst: 0.0,
            drift60: 0.8,
            spike_p: 0.0,
            spike_amp: 0.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 3,
        },
        Scenario {
            name: "sq_10_spikes",
            regime: Regime::Square,
            duration_s: 5400,
            n_events: 20,
            strength: 10.0,
            burst: 0.0,
            drift60: 0.1,
            spike_p: 1e-5,
            spike_amp: 10.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 4,
        },
        Scenario {
            name: "rm_6",
            regime: Regime::Ramp,
            duration_s: 7200,
            n_events: 6,
            strength: 6.0,
            burst: 0.0,
            drift60: 0.1,
            spike_p: 0.0,
            spike_amp: 0.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 5,
        },
        Scenario {
            name: "rm_10",
            regime: Regime::Ramp,
            duration_s: 7200,
            n_events: 6,
            strength: 10.0,
            burst: 0.0,
            drift60: 0.1,
            spike_p: 0.0,
            spike_amp: 0.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 6,
        },
        Scenario {
            name: "rm_16",
            regime: Regime::Ramp,
            duration_s: 7200,
            n_events: 6,
            strength: 16.0,
            burst: 0.0,
            drift60: 0.1,
            spike_p: 0.0,
            spike_amp: 0.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 7,
        },
        Scenario {
            name: "clean",
            regime: Regime::Clean,
            duration_s: 5400,
            n_events: 0,
            strength: 10.0,
            burst: 0.0,
            drift60: 0.1,
            spike_p: 0.0,
            spike_amp: 0.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 8,
        },
        Scenario {
            name: "clean_drift",
            regime: Regime::Clean,
            duration_s: 5400,
            n_events: 0,
            strength: 10.0,
            burst: 0.0,
            drift60: 0.8,
            spike_p: 0.0,
            spike_amp: 0.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 9,
        },
        Scenario {
            name: "clean_spikes",
            regime: Regime::Clean,
            duration_s: 5400,
            n_events: 0,
            strength: 10.0,
            burst: 0.0,
            drift60: 0.1,
            spike_p: 1e-5,
            spike_amp: 10.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 10,
        },
        Scenario {
            name: "bst_8",
            regime: Regime::Burst,
            duration_s: 7200,
            n_events: 6,
            strength: 2.0,
            burst: 8.0,
            drift60: 0.1,
            spike_p: 0.0,
            spike_amp: 0.0,
            noise_rho: 0.6,
            seed_base: 0x9E37_79B9 + 11,
        },
    ]
}

fn configs() -> Vec<(String, DetCfg)> {
    let mut out: Vec<(String, DetCfg)> = Vec::new();
    for s in [0.75, 1.0, 1.5, 2.5] {
        out.push((format!("dual_{s:.2}"), DetCfg::Dual(s)));
    }
    for a in [0.02, 0.05, 0.1] {
        for t in [4.0, 5.0, 6.0] {
            out.push((
                format!("ewma_a{a}_t{t}_v2"),
                DetCfg::Ewma(EwmaConfig {
                    alpha: a,
                    threshold_sigma: t,
                    min_votes: 2,
                }),
            ));
        }
    }
    out
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let seeds: usize = args
        .iter()
        .rposition(|a| a == "--seeds")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick { 2 } else { 6 });
    let out_path = args
        .iter()
        .rposition(|a| a == "--out")
        .map(|i| args[i + 1].clone());
    let filter = args
        .iter()
        .rposition(|a| a == "--scenario")
        .map(|i| args[i + 1].clone());
    let cfg_filter = args
        .iter()
        .rposition(|a| a == "--config")
        .map(|i| args[i + 1].clone());
    let no_dual = args.iter().any(|a| a == "--no-dual");

    let sfilter: Option<Vec<String>> =
        filter.map(|f| f.split(',').map(|s| s.to_string()).collect());
    let cfilter: Option<Vec<String>> =
        cfg_filter.map(|f| f.split(',').map(|s| s.to_string()).collect());

    let scens: Vec<Scenario> = scenarios()
        .into_iter()
        .filter(|s| {
            sfilter
                .as_ref()
                .is_none_or(|fs| fs.iter().any(|f| s.name.contains(f.as_str())))
        })
        .collect();
    let cfgs: Vec<(String, DetCfg)> = configs()
        .into_iter()
        .filter(|(n, _)| {
            let dual = n.starts_with("dual");
            if no_dual && dual {
                return false;
            }
            cfilter
                .as_ref()
                .is_none_or(|fs| fs.iter().any(|f| n.contains(f.as_str())))
        })
        .collect();

    let mut cache: HashMap<(usize, u64), Vec<Sample>> = HashMap::new();
    let mut raw: Vec<RawCell> = Vec::new();
    let mut aggs: Vec<AggCell> = Vec::new();

    for (si, sc) in scens.iter().enumerate() {
        for (cfg_name, cfg) in &cfgs {
            let mut cells: Vec<CellAgg> = Vec::new();
            for seed in 0..seeds as u64 {
                let samples = cache.entry((si, seed)).or_insert_with(|| sc.generate(seed));
                cells.push(evaluate(seed, cfg, samples));
            }
            for c in &cells {
                raw.push(RawCell {
                    scenario: sc.name.to_string(),
                    config: cfg_name.clone(),
                    agg: *c,
                });
            }
            let a = AggCell::from_cells(sc.name, cfg_name, &cells);
            println!(
                "{:<14} {:<18} {:<4} {:>6.3} {:<6} {:<5.0}/d {:<6.2}/mo {:>6.0}  {}",
                sc.name,
                cfg_name,
                a.n_events,
                a.tpr.unwrap_or(f64::NAN),
                fmt_fpr(a.fpr),
                a.fa_per_day,
                a.fa_per_month,
                a.ppv_at_1,
                if a.deploy_meets_bar { "OK" } else { "" },
            );
            aggs.push(a);
        }
    }
    eprintln!("elapsed {:.1} s", t0.elapsed().as_secs_f64());

    if let Some(op) = &out_path {
        let archive = Archive {
            note: "Monte-Carlo TPR/FPR/PPV sweep. Synthetic stimuli — NOT a substitute for long-term drift corpora (Wörner gap). deploy_meets_bar = FA<=1/month && TPR>=0.9 on every realization. FPR is per-second (10 Hz verdicts); FA/day,FA/month assume 24/7. PPV at operator leak rates 0.1/1/5 events/day.".to_string(),
            seeds: seeds as u64,
            scenarios: scens.iter().map(|s| s.name.to_string()).collect(),
            configs: cfgs.iter().map(|(n, _)| n.clone()).collect(),
            raw,
            aggregates: aggs,
        };
        let json = serde_json::to_string_pretty(&archive).expect("serialize");
        let mut f = File::create(op).expect("create output");
        f.write_all(json.as_bytes()).expect("write output");
        println!("wrote {op}");
    }
}

fn fmt_fpr(f: f64) -> String {
    if f == 0.0 {
        "0".to_string()
    } else {
        format!("{f:.2e}")
    }
}
