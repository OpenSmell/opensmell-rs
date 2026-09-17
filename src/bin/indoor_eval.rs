//! Whole-engine real-data evaluation on the UCI home-activity corpus
//! (Gas sensors for home activity monitoring, archive.ics.uci.edu #362).
//!
//! 100 independent inductions (36 wine, 33 banana, 31 background) from a
//! residential e-nose: 8 MOX resistances + temperature + humidity at 1 Hz,
//! with ~1 h of background activity recorded before and after each stimulus.
//! Each induction is treated as its own deployment: calibrate during the
//! pre-stimulus background (the operator asserts clean air), then replay and
//! score whether an alarm fires in the stimulus window. Background inductions
//! act as pure false-positive test beds.
//!
//! This exercises the exact claim the dynamic-mixtures corpus cannot:
//! temperature and humidity co-vary with the target signal, and the sessions
//! span real home months (the prototype ran ~2 years in one author's home).
//!
//! Usage:
//!   indoor_eval <dataset.dat> <metadata.dat> [--sensitivity <v>] [--baseline dual|ewma]
//!              [--alpha <a>] [--thr <t>] [--min-votes <v>] [--out out.json]
use opensmell::anomaly::ewma::EwmaConfig;
use opensmell::anomaly::StreamDetector;
use serde::Serialize;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

const N_CH: usize = 8;
const STARTUP_S: usize = 120;
const CAL_MIN_S: usize = 300;
const CAL_FLOOR_S: usize = 120;
const DETECT_WINDOW_S: usize = 120;
const RECOVERY_S: usize = 60;
const PRE_EVENT_S: usize = 60;

#[derive(Clone)]
struct Record {
    reading: [f64; N_CH],
    gas: bool,
}

#[derive(Clone, Copy)]
struct Induction {
    id: u32,
    target: bool,
    dt_h: f64,
}

#[derive(Serialize)]
struct InductionResult {
    id: u32,
    target: bool,
    dt_h: f64,
    samples: u64,
    scored: bool,
    detected: bool,
    latency_s: Option<f64>,
    cal_start_s: u64,
    cal_end_s: u64,
    clean_seconds: u64,
    clean_alarms: u64,
    event_seconds: u64,
    event_alarms: u64,
    recovery_excluded: u64,
}

#[derive(Serialize, Default)]
struct Metrics {
    file: String,
    n_channels: usize,
    sensitivity: f64,
    baseline: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ewma_config: Option<EwmaConfig>,
    inductions: u64,
    target_inductions: u64,
    scored_inductions: u64,
    detected_inductions: u64,
    detection_rate: f64,
    median_latency_s: Option<f64>,
    p75_latency_s: Option<f64>,
    latencies_s: Vec<f64>,
    clean_seconds: u64,
    clean_alarms: u64,
    clean_fpr: f64,
    event_seconds: u64,
    event_alarms: u64,
    event_seconds_alarmed: f64,
    recovery_excluded_seconds: u64,
    missed: Vec<u32>,
    typologies: BTreeMap<String, u64>,
    details: Vec<InductionResult>,
    runtime_s: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn parse_metadata(path: &str) -> Vec<Induction> {
    let f = File::open(path).expect("open metadata");
    let mut r = BufReader::new(f);
    let mut header = String::new();
    let _ = r.read_line(&mut header);
    let mut out = Vec::new();
    let mut line = String::new();
    while let Ok(n) = r.read_line(&mut line) {
        if n == 0 {
            break;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 5 {
            line.clear();
            continue;
        }
        let id: u32 = toks[0].parse().unwrap_or(0);
        let class = toks[2];
        let dt_h: f64 = toks[4].parse().unwrap_or(0.0);
        out.push(Induction {
            id,
            target: class != "background",
            dt_h,
        });
        line.clear();
    }
    out
}

fn find_cal_window(records: &[Record]) -> Option<(usize, usize)> {
    let start0 = STARTUP_S.min(records.len());
    if start0 >= records.len() {
        return None;
    }
    // Causal: an operator asserts "clean air now"; only past (recovery) margins
    // are available, never the next event.
    let mut i = start0;
    while i + CAL_FLOOR_S <= records.len() {
        for &period in [CAL_MIN_S, CAL_FLOOR_S].iter() {
            if i + period <= records.len()
                && records[i..i + period].iter().all(|r| !r.gas)
                && (records[..i]
                    .iter()
                    .rev()
                    .position(|r| r.gas)
                    .unwrap_or(usize::MAX)
                    >= RECOVERY_S)
            {
                return Some((i, i + period));
            }
        }
        i += 1;
    }
    None
}

fn find_events(records: &[Record]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < records.len() {
        if records[i].gas {
            let start = i;
            while i < records.len() && records[i].gas {
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

/// Detector selection for a replay deployment.
struct DetectorSpec {
    baseline: String,
    sensitivity: f64,
    ewma_cfg: EwmaConfig,
}

fn run_deployment(
    id: u32,
    target: bool,
    dt_h: f64,
    records: &[Record],
    spec: &DetectorSpec,
    kinds: &mut BTreeMap<String, u64>,
) -> (InductionResult, Vec<f64>) {
    let mut detail = InductionResult {
        id,
        target,
        dt_h,
        samples: records.len() as u64,
        scored: false,
        detected: false,
        latency_s: None,
        cal_start_s: 0,
        cal_end_s: 0,
        clean_seconds: 0,
        clean_alarms: 0,
        event_seconds: 0,
        event_alarms: 0,
        recovery_excluded: 0,
    };
    let mut lat: Vec<f64> = Vec::new();
    let cal = match find_cal_window(records) {
        Some(c) => c,
        None => return (detail, lat),
    };
    let (cal_start, cal_end) = cal;
    let cal_samples: Vec<Vec<f64>> = records[cal_start..cal_end]
        .iter()
        .map(|r| r.reading.to_vec())
        .collect();
    if cal_samples.is_empty() || cal_samples.len() < N_CH {
        return (detail, lat);
    }
    detail.cal_start_s = cal_start as u64;
    detail.cal_end_s = cal_end as u64;

    let mut det = if spec.baseline == "ewma" {
        StreamDetector::ewma(N_CH, spec.ewma_cfg.clone())
    } else {
        StreamDetector::dual(N_CH, spec.sensitivity)
    };
    let ok = det.calibrate(&cal_samples).is_ok();
    if !ok {
        return (detail, lat);
    }
    for r in &records[..cal_end] {
        let _ = det.detect(&r.reading);
    }

    let mut alarms = vec![false; records.len()];
    for (i, rec) in records.iter().enumerate().skip(cal_end) {
        if let Ok(v) = det.detect(&rec.reading) {
            alarms[i] = v.is_anomaly;
            if v.is_anomaly {
                *kinds.entry(v.kind.clone().unwrap_or_else(|| "none".to_string())).or_insert(0) += 1;
            }
        }
    }

    let events = find_events(records);
    let mut clean_seconds = 0u64;
    let mut clean_alarms = 0u64;
    let mut event_seconds = 0u64;
    let mut event_alarms = 0u64;
    let mut recovery_excluded = 0u64;
    for i in cal_end..records.len() {
        let in_event = events.iter().any(|&(s, e)| i >= s && i <= e);
        let after_event = events.iter().any(|&(_s, e)| i > e && i <= e + RECOVERY_S);
        let before_event = events.iter().any(|&(s, _e)| i < s && s - i <= PRE_EVENT_S);
        if in_event {
            event_seconds += 1;
            if alarms[i] {
                event_alarms += 1;
            }
        } else if after_event || before_event {
            recovery_excluded += 1;
        } else {
            clean_seconds += 1;
            if alarms[i] {
                clean_alarms += 1;
            }
        }
    }
    detail.clean_seconds = clean_seconds;
    detail.clean_alarms = clean_alarms;
    detail.event_seconds = event_seconds;
    detail.event_alarms = event_alarms;
    detail.recovery_excluded = recovery_excluded;

    if let Some(&(s, e)) = events.first() {
        if s >= cal_end {
            detail.scored = true;
            let window_end = (s + DETECT_WINDOW_S).min(records.len());
            if let Some(off) = alarms[s..window_end].iter().position(|a| *a) {
                detail.detected = true;
                detail.latency_s = Some(off as f64);
                lat.push(off as f64);
            }
        }
        let _ = e;
    }
    (detail, lat)
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = env::args().collect();
    let data_path = args
        .get(1)
        .expect("usage: indoor_eval <dataset.dat> <metadata.dat> [--sensitivity <v>] [--out out.json]");
    let meta_path = args
        .get(2)
        .expect("usage: indoor_eval <dataset.dat> <metadata.dat> [--sensitivity <v>] [--out out.json]");
    let sensitivity = args
        .iter()
        .rposition(|a| a == "--sensitivity")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(3.0)
        .max(1e-3);
    let out_path = args.iter().rposition(|a| a == "--out").map(|i| args[i + 1].clone());

    let baseline = args
        .iter()
        .position(|a| a == "--baseline")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("dual")
        .to_string();
    let ewma_cfg = EwmaConfig {
        alpha: args
            .iter()
            .rposition(|a| a == "--alpha")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.05),
        threshold_sigma: args
            .iter()
            .rposition(|a| a == "--thr")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(5.0),
        min_votes: args
            .iter()
            .rposition(|a| a == "--min-votes")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2),
    };

    let inductions = parse_metadata(meta_path);

    // Single pass over the dataset, bucketing rows by induction id (~60 MB).
    let f = File::open(data_path).expect("open dataset");
    let mut r = BufReader::new(f);
    let mut header = String::new();
    let _ = r.read_line(&mut header);
    let mut by_id: BTreeMap<u32, Induction> = BTreeMap::new();
    for ind in inductions {
        by_id.insert(ind.id, ind);
    }
    let mut in_scope: BTreeMap<u32, Vec<Record>> = BTreeMap::new();
    let mut line = String::new();
    while let Ok(n) = r.read_line(&mut line) {
        if n == 0 {
            break;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 11 {
            line.clear();
            continue;
        }
        let id: u32 = toks[0].parse().unwrap_or(u32::MAX);
        if let Some(ind) = by_id.get(&id) {
            let t_h: f64 = toks[1].parse().unwrap_or(0.0);
            let mut reading = [0.0f64; N_CH];
            for (i, tk) in toks[2..2 + N_CH].iter().enumerate() {
                if let Ok(v) = tk.parse::<f64>() {
                    reading[i] = v;
                }
            }
            let gas = ind.target && t_h >= 0.0 && t_h < ind.dt_h;
            in_scope.entry(id).or_default().push(Record { reading, gas });
        }
        line.clear();
    }

    let mut kinds: BTreeMap<String, u64> = BTreeMap::new();
    let mut details: Vec<InductionResult> = Vec::new();
    let mut latencies: Vec<f64> = Vec::new();
    let mut detected = 0u64;
    let mut scored = 0u64;
    let mut targets = 0u64;
    let mut total_clean_seconds = 0u64;
    let mut total_clean_alarms = 0u64;
    let mut total_event_seconds = 0u64;
    let mut total_event_alarms = 0u64;
    let mut total_recovery = 0u64;
    let mut missed: Vec<u32> = Vec::new();

    let spec = DetectorSpec {
        baseline: baseline.clone(),
        sensitivity,
        ewma_cfg: ewma_cfg.clone(),
    };
    for (id, records) in in_scope.iter() {
        let ind = by_id[id];
        if ind.target {
            targets += 1;
        }
        let (detail, lat) = run_deployment(*id, ind.target, ind.dt_h, records, &spec, &mut kinds);
        total_clean_seconds += detail.clean_seconds;
        total_clean_alarms += detail.clean_alarms;
        total_event_seconds += detail.event_seconds;
        total_event_alarms += detail.event_alarms;
        total_recovery += detail.recovery_excluded;
        if detail.scored {
            scored += 1;
            if detail.detected {
                detected += 1;
                latencies.extend(lat);
            } else {
                missed.push(*id);
            }
        }
        details.push(detail);
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let m = Metrics {
        file: data_path.to_string(),
        n_channels: N_CH,
        sensitivity,
        baseline: baseline.clone(),
        ewma_config: if baseline == "ewma" {
            Some(ewma_cfg.clone())
        } else {
            None
        },
        inductions: in_scope.len() as u64,
        target_inductions: targets,
        scored_inductions: scored,
        detected_inductions: detected,
        detection_rate: if scored > 0 { detected as f64 / scored as f64 } else { 0.0 },
        median_latency_s: percentile(&latencies, 0.5).into(),
        p75_latency_s: percentile(&latencies, 0.75).into(),
        latencies_s: latencies,
        clean_seconds: total_clean_seconds,
        clean_alarms: total_clean_alarms,
        clean_fpr: if total_clean_seconds > 0 {
            total_clean_alarms as f64 / total_clean_seconds as f64
        } else {
            0.0
        },
        event_seconds: total_event_seconds,
        event_alarms: total_event_alarms,
        event_seconds_alarmed: if total_event_seconds > 0 {
            total_event_alarms as f64 / total_event_seconds as f64
        } else {
            0.0
        },
        recovery_excluded_seconds: total_recovery,
        missed: missed,
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