//! Real-data whole-engine evaluation (Wave-3 gate, §11.3).
//!
//! Replays a *real* recorded MOX stream through `DualKalmanEngine` and scores
//! the alarms against the recording's ground truth. Targets the UCI dynamic
//! gas-mixtures recordings (`e-nose-evals/data/dynamic-mixtures/*.txt`): 12 h
//! of 16 real Figaro sensor readings at 100 Hz with the true CO/ethylene
//! concentration squared-wave every 80–120 s as ground truth.
//!
//! Pipeline:
//!   1. stream-parse rows → per-channel raw sensor response `S` (the UCI
//!      recordings are already normalized responses; a resistance transform
//!      collapses the halftime channels and was rejected on data),
//!   2. decimate 100 Hz → 10 Hz (per-0.1 s per-channel median),
//!   3. label `gas_present = CO + ethylene ≥ threshold`,
//!   4. calibrate the engine on the first long clean-air window,
//!   5. replay the whole 10 Hz series, then score per-event detection rate and
//!      latency, per-sample clean FPR, alarm PPV, and typology mix.
//!
//! Events that end before the replay starts (i.e. before calibration finished)
//! are marked `scored: false` and excluded from the detection rate and latency:
//! they predate the system going online.
//!
//! Usage:
//!   realdata_eval <file.txt> [limit_seconds] [threshold_ppm] [--sensitivity <v>] [--out results.json]
//! The optional `limit_seconds` truncates the reading for fast dev iterations.
//! `--sensitivity` is the engine operator knob (the budget thresholds k_std are
//! divided by it); sweep it to trace the detection/false-positive operating curve.
//! `--causal-cal` restricts calibration to past information only (an operator
//! asserting "clean air now"); without it, the window search may also use the
//! next event as a margin. `--baseline dual|ewma` selects the detector
//! (DualKalman engine or the EWMA control chart; default dual), with
//! `--alpha/--thr/--min-votes` tuning the EWMA.
use opensmell::anomaly::ewma::EwmaConfig;
use opensmell::anomaly::StreamDetector;
use serde::Serialize;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

const N_CH: usize = 16;
const N_RAW_PER_SEC: usize = 100;
const N_RAW_PER_BUCKET: usize = 10;
const OUT_HZ: usize = 10;
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

#[derive(Default)]
struct Bucket {
    ch: [Vec<f64>; N_CH],
}

fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = s.len() / 2;
    if s.len() % 2 == 1 {
        Some(s[m])
    } else {
        Some((s[m - 1] + s[m]) / 2.0)
    }
}

fn parse_row(line: &str, out: &mut Bucket) -> Option<f64> {
    let toks: Vec<&str> = line.split_whitespace().collect();
    if toks.len() < 3 + N_CH {
        return None;
    }
    let t: f64 = toks[0].parse().ok()?;
    for (i, tk) in toks[3..3 + N_CH].iter().enumerate() {
        if let Ok(s) = tk.parse::<f64>() {
            out.ch[i].push(s);
        }
    }
    Some(t)
}

#[derive(Serialize)]
struct EventInfo {
    start_s: f64,
    end_s: f64,
    scored: bool,
    detected: bool,
    latency_s: Option<f64>,
}

#[derive(Serialize, Default)]
struct Metrics {
    file: String,
    n_channels: usize,
    sensitivity: f64,
    baseline: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ewma_config: Option<EwmaConfig>,
    causal_calibration: bool,
    decimated_seconds: u64,
    n_events: u64,
    detected_events: u64,
    detection_rate: f64,
    median_latency_s: Option<f64>,
    p75_latency_s: Option<f64>,
    latencies_s: Vec<f64>,
    events: Vec<EventInfo>,
    clean_seconds: u64,
    clean_alarms: u64,
    clean_fpr: f64,
    event_seconds: u64,
    event_alarms: u64,
    event_seconds_alarmed: f64,
    recovery_excluded_seconds: u64,
    typologies: BTreeMap<String, u64>,
    calibration_start_s: u64,
    calibration_end_s: u64,
    runtime_s: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .expect("usage: realdata_eval <file.txt> [limit_seconds] [threshold_ppm] [--out results.json]");
    let limit_sec = args.get(2).and_then(|s| s.parse::<usize>().ok());
    let threshold = args.get(3).and_then(|s| s.parse::<f64>().ok()).unwrap_or(5.0);
    let sensitivity = args
        .iter()
        .rposition(|a| a == "--sensitivity")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0)
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

    let file = File::open(path).expect("open dataset");
    let mut reader = BufReader::new(file);
    let mut header = String::new();
    let _ = reader.read_line(&mut header);

    let mut records: Vec<Record> = Vec::new();
    let mut bucket = Bucket::default();
    let mut n_in_bucket = 0usize;
    let mut last_reading = [1.0f64; N_CH];
    let mut n_lines = 0usize;
    let mut line = String::new();
    let begin = path.to_string();
    let raw_limit: Option<usize> = limit_sec.map(|s| s * N_RAW_PER_SEC);

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        n_lines += 1;
        if let Some(lim) = raw_limit {
            if n_lines > lim {
                break;
            }
        }
        if let Some(t) = parse_row(&line, &mut bucket) {
            n_in_bucket += 1;
            if n_in_bucket >= N_RAW_PER_BUCKET {
                let mut reading = [0.0f64; N_CH];
                for c in 0..N_CH {
                    match median(&bucket.ch[c]) {
                        Some(m) => {
                            reading[c] = m;
                            last_reading[c] = m;
                        }
                        None => reading[c] = last_reading[c],
                    }
                    bucket.ch[c].clear();
                }
                n_in_bucket = 0;
                let _ = t;
                records.push(Record { reading, gas: false });
            }
        }
    }

    // Second pass over the decimated series is unnecessary: gas truth comes from
    // the concentration columns, so re-parse with the truth recorded.
    let mut pass2 = String::new();
    let mut gas_reparse = Vec::with_capacity(records.len());
    {
        let file2 = File::open(path).expect("reopen dataset");
        let mut reader2 = BufReader::new(file2);
        let _ = reader2.read_line(&mut pass2);
        let mut gas_bucket_co = Vec::new();
        let mut gas_bucket_eth = Vec::new();
        let mut n2 = 0usize;
        let mut prev_co = 0.0f64;
        let mut prev_eth = 0.0f64;
        let mut line2 = String::new();
        let mut raw_count = 0usize;
        loop {
            line2.clear();
            match reader2.read_line(&mut line2) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
            raw_count += 1;
            if let Some(lim) = raw_limit {
                if raw_count > lim {
                    break;
                }
            }
            let toks: Vec<&str> = line2.split_whitespace().collect();
            if toks.len() < 3 + N_CH {
                continue;
            }
            let co = toks[1].parse::<f64>().unwrap_or(prev_co);
            let eth = toks[2].parse::<f64>().unwrap_or(prev_eth);
            prev_co = co;
            prev_eth = eth;
            gas_bucket_co.push(co);
            gas_bucket_eth.push(eth);
            n2 += 1;
            if n2 >= N_RAW_PER_BUCKET {
                let co_m = median(&gas_bucket_co).unwrap_or(0.0);
                let eth_m = median(&gas_bucket_eth).unwrap_or(0.0);
                gas_bucket_co.clear();
                gas_bucket_eth.clear();
                n2 = 0;
                gas_reparse.push(co_m + eth_m >= threshold);
            }
        }
    }
    if gas_reparse.len() == records.len() {
        for (r, g) in records.iter_mut().zip(gas_reparse.iter()) {
            r.gas = *g;
        }
    }

    for r in &mut records {
        let _ = &mut r.gas;
    }

    let n_sec = records.len() as u64;

    // Calibration window: the first long clean-air run after the startup spike.
    // In causal (deployment-realizable) mode the operator asserts "clean air
    // now"; the search may only use the past (no peeking at future events).
    let causal = args.iter().any(|a| a == "--causal-cal");
    let cal = find_cal_window(&records, causal);
    let (cal_start, cal_end) = match cal {
        Some((s, e)) => (s, e),
        None => {
            eprintln!("no clean-air calibration window found; aborting");
            return;
        }
    };

    let mut det = if baseline == "ewma" {
        StreamDetector::ewma(N_CH, ewma_cfg.clone())
    } else {
        StreamDetector::dual(N_CH, sensitivity)
    };
    let cal_samples: Vec<Vec<f64>> = records[cal_start..cal_end]
        .iter()
        .map(|r| r.reading.to_vec())
        .collect();
    if cal_samples.is_empty() || cal_samples.len() < N_CH {
        eprintln!("calibration window too small ({})", cal_samples.len());
        return;
    }
    det.calibrate(&cal_samples).expect("calibrate");

    let n_cal = crate_like_feed(&mut det, &records[..cal_end]);

    // Replay + collect flags.
    let mut alarms = vec![false; records.len()];
    let mut kinds: BTreeMap<String, u64> = BTreeMap::new();
    for (i, rec) in records.iter().enumerate().skip(cal_end) {
        let v = match det.detect(&rec.reading) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("detect error at sample {i}: {e}");
                continue;
            }
        };
        alarms[i] = v.is_anomaly;
        if v.is_anomaly {
            *kinds.entry(v.kind.clone().unwrap_or_else(|| "none".to_string())).or_insert(0) += 1;
        }
    }

    // True events: runs of gas-present (merge ≤ 5-sample gaps). Events that end
    // before the replay starts are unscored (they predate the system going
    // online at the end of calibration) and are excluded from the rate.
    let events = find_events(&records);
    let mut latencies: Vec<f64> = Vec::new();
    let mut detected = 0u64;
    let mut scored_events = 0u64;
    let mut events_detail: Vec<EventInfo> = Vec::new();
    for &(s, e) in &events {
        let scored = s >= cal_end;
        let window_end = (s + DETECT_WINDOW_S * OUT_HZ).min(records.len());
        let offset = if scored {
            alarms[s..window_end].iter().position(|a| *a)
        } else {
            None
        };
        if scored {
            scored_events += 1;
        }
        if let Some(off) = offset {
            detected += 1;
            latencies.push(off as f64 / OUT_HZ as f64);
        }
        events_detail.push(EventInfo {
            start_s: s as f64 / OUT_HZ as f64,
            end_s: e as f64 / OUT_HZ as f64,
            scored,
            detected: offset.is_some(),
            latency_s: offset.map(|off| off as f64 / OUT_HZ as f64),
        });
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Clean / recovery / event sample bookkeeping.
    let mut clean_seconds = 0u64;
    let mut clean_alarms = 0u64;
    let mut event_seconds = 0u64;
    let mut event_alarms = 0u64;
    let mut recovery_excluded = 0u64;
    for (i, &alarmed) in alarms.iter().enumerate().skip(cal_end) {
        let in_event = events.iter().any(|&(s, e)| i >= s && i <= e);
        let after_event = events.iter().any(|&(_s, e)| i > e && i <= e + RECOVERY_S * OUT_HZ);
        let before_event = events.iter().any(|&(s, _e)| {
            i < s && s - i <= PRE_EVENT_S * OUT_HZ
        });
        if in_event {
            event_seconds += 1;
            if alarmed {
                event_alarms += 1;
            }
        } else if after_event || before_event {
            recovery_excluded += 1;
        } else {
            clean_seconds += 1;
            if alarmed {
                clean_alarms += 1;
            }
        }
    }
    let event_seconds_alarmed = if event_seconds > 0 {
        event_alarms as f64 / event_seconds as f64
    } else {
        0.0
    };

    let m = Metrics {
        file: begin.clone(),
        n_channels: N_CH,
        sensitivity,
        baseline: baseline.clone(),
        ewma_config: if baseline == "ewma" {
            Some(ewma_cfg.clone())
        } else {
            None
        },
        causal_calibration: causal,
        decimated_seconds: n_sec,
        n_events: scored_events,
        detected_events: detected,
        detection_rate: if scored_events == 0 {
            0.0
        } else {
            detected as f64 / scored_events as f64
        },
        median_latency_s: percentile(&latencies, 0.5).into(),
        p75_latency_s: percentile(&latencies, 0.75).into(),
        latencies_s: latencies,
        events: events_detail,
        clean_seconds,
        clean_alarms,
        clean_fpr: if clean_seconds > 0 {
            clean_alarms as f64 / clean_seconds as f64
        } else {
            0.0
        },
        event_seconds,
        event_alarms,
        event_seconds_alarmed,
        recovery_excluded_seconds: recovery_excluded,
        typologies: kinds,
        calibration_start_s: (cal_start / OUT_HZ) as u64,
        calibration_end_s: (cal_end / OUT_HZ) as u64,
        runtime_s: t0.elapsed().as_secs_f64(),
    };
    let _ = n_cal;

    let json = serde_json::to_string_pretty(&m).expect("serialize");
    if let Some(op) = &out_path {
        let mut f = File::create(op).expect("create output");
        f.write_all(json.as_bytes()).expect("write output");
        println!("wrote {op}");
    } else {
        println!("{json}");
    }
}

fn crate_like_feed(det: &mut StreamDetector, recs: &[Record]) -> u64 {
    let mut n = 0u64;
    for r in recs {
        if det.detect(&r.reading).is_ok() {
            n += 1;
        }
    }
    n
}

fn cal_margins_ok(records: &[Record], i: usize, causal: bool) -> bool {
    let prev_event = records[..i]
        .iter()
        .rev()
        .position(|r| r.gas)
        .unwrap_or(usize::MAX);
    let after_recovery = prev_event == usize::MAX || prev_event >= RECOVERY_S * OUT_HZ;
    if causal {
        return after_recovery;
    }
    let next_event = records[i..].iter().position(|r| r.gas).unwrap_or(usize::MAX);
    after_recovery
        && (next_event == usize::MAX || next_event >= PRE_EVENT_S * OUT_HZ)
}

fn find_cal_window(records: &[Record], causal: bool) -> Option<(usize, usize)> {
    let start0 = STARTUP_S * OUT_HZ;
    if start0 >= records.len() {
        return None;
    }
    // Scan time-forward and accept the EARLIEST valid window (longest first at
    // that offset), so as many early events as possible fall inside the scored
    // replay instead of being pushed past a late 300 s window choice.
    let mut i = start0;
    while i + CAL_FLOOR_S * OUT_HZ <= records.len() {
        for &period in [CAL_MIN_S * OUT_HZ, CAL_FLOOR_S * OUT_HZ].iter() {
            if i + period <= records.len()
                && records[i..i + period].iter().all(|r| !r.gas)
                && cal_margins_ok(records, i, causal)
            {
                return Some((i, i + period));
            }
        }
        i += OUT_HZ;
    }
    None
}

fn find_events(records: &[Record]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
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
    // Merge gaps of ≤ 5 samples (50 s of truth jitter at the transitions).
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