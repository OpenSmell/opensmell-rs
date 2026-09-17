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
//!             [--alpha <a>] [--thr <t>] [--min-votes <v>] [--out out.json]
use opensmell::anomaly::ewma::EwmaConfig;
use opensmell::anomaly::StreamDetector;
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
    n_rows: u64,
    n_releases: u64,
    detected_releases: u64,
    detection_rate: f64,
    clean_seconds: u64,
    clean_alarms: u64,
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
    let ewma_cfg = EwmaConfig {
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
    };

    let (ch_names, records, dt_s) = parse_csv(data_path);
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

    // Single global deployment: calibrate once, replay the whole stream.
    let mut det = if baseline == "ewma" {
        StreamDetector::ewma(n_ch, ewma_cfg.clone())
    } else {
        StreamDetector::dual(n_ch, sensitivity)
    };
    let calibrated = cal_samples.len() >= n_ch && det.calibrate(&cal_samples).is_ok();
    if !calibrated {
        eprintln!("calibration failed (n_ch={n_ch}, cal_samples={})", cal_samples.len());
        return;
    }
    // Warm up on calibration samples.
    for r in &records[cal_start..cal_end] {
        let _ = det.detect(&r.reading);
    }

    // Replay the entire stream from cal_end onward, recording alarms.
    let mut alarms = vec![false; records.len()];
    let mut kinds: BTreeMap<String, u64> = BTreeMap::new();
    for (i, rec) in records.iter().enumerate().skip(cal_end) {
        if let Ok(v) = det.detect(&rec.reading) {
            alarms[i] = v.is_anomaly;
            if v.is_anomaly {
                *kinds
                    .entry(v.kind.clone().unwrap_or_else(|| "none".to_string()))
                    .or_insert(0) += 1;
            }
        }
    }

    // Build release blocks (in order of first appearance).
    let mut by_release: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, rec) in records.iter().enumerate() {
        by_release.entry(rec.release).or_default().push(i);
    }

    let mut clean_s = 0u64;
    let mut clean_a = 0u64;
    let mut latencies_ok: Vec<f64> = Vec::new();
    let mut detected_count = 0u64;
    let mut details: Vec<ReleaseResult> = Vec::new();
    // Per-sample duration in seconds (clamped to within 3x median cadence so
    // cross-day file gaps don't count as clean time).
    let mut dt_per = vec![dt_s; records.len()];
    for i in 1..records.len() {
        let gap = records[i].t - records[i - 1].t;
        if gap > 0.0 && gap < 600.0 {
            dt_per[i] = gap;
        }
    }

    for (&rel, rows) in by_release.iter() {
        let n_rows = rows.len();
        let peak = rows.iter().map(|&i| records[i].ch4).fold(0.0f64, f64::max);
        let mut det_r = ReleaseResult {
            release: rel,
            n_rows: n_rows as u64,
            event_rows: 0,
            peak_ch4: peak,
            detected: false,
            latency_s: None,
            event_alarms: 0,
        };
        let first_event = rows.iter().position(|&i| records[i].ch4 >= ch4_thr);
        let first_event_time = first_event.map(|fi| records[rows[fi]].t);
        let mut fired = false;
        let mut event_s = 0.0f64;
        let mut event_a = 0u64;
        let mut clean_s_local = 0.0f64;
        let mut clean_a_local = 0u64;
        for &i in rows {
            if i < cal_end {
                continue;
            }
            let is_event = records[i].ch4 >= ch4_thr;
            if is_event {
                det_r.event_rows += 1;
                event_s += dt_per[i];
                if alarms[i] {
                    det_r.event_alarms += 1;
                    event_a += 1;
                }
            } else {
                clean_s_local += dt_per[i];
                clean_a_local += 1; // count samples, not seconds
                if alarms[i] {
                    clean_a += 1;
                }
            }
            if !fired && is_event && alarms[i] && first_event_time.is_some() {
                fired = true;
                det_r.detected = true;
                det_r.latency_s = Some(records[i].t - first_event_time.unwrap());
            }
        }
        clean_s += clean_s_local as u64;
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
        sample_dt_s: dt_s,
        ch4_threshold_ppm: ch4_thr,
        sensitivity,
        baseline: baseline.clone(),
        ewma_config: if baseline == "ewma" {
            Some(ewma_cfg.clone())
        } else {
            None
        },
        n_rows: records.len() as u64,
        n_releases,
        detected_releases: detected_count,
        detection_rate: if n_releases > 0 {
            detected_count as f64 / n_releases as f64
        } else {
            0.0
        },
        clean_seconds: clean_s,
        clean_alarms: clean_a,
        clean_fpr: if clean_s > 0 {
            clean_a as f64 / clean_s as f64
        } else {
            0.0
        },
        fa_per_month: if clean_s > 0 {
            (clean_a as f64 / clean_s as f64) * 86400.0 * 30.0
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
