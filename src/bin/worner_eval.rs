//! worner_eval — engine-level replay of the Wörner 2025 corpus (62-channel MOS,
//! Diacetyl / EtOH / Phenylethanol, 40 experiment "days", 1 Hz, with real
//! T01/H01 per second and a Cycle_Stage 1/2/3 protocol marker).
//!
//! Answers the Phase-C question the threshold-level simulation (worner_drift_decision.py)
//! cannot: does the *adaptive dual-Kalman engine itself* — the thing a shipped
//! unit actually runs — survive 40 sessions of real baseline drift while still
//! detecting the exposures?
//!
//! Protocol (honest, mirrors tadi_eval's scoring so results are comparable):
//!   - calibration: the earliest file's Cycle_Stage==1 (pre-exposure) window
//!     (tadi_eval calibrates on the earliest CH4<5ppm window; stage-1 is the
//!     Wörner analog of "no-leak-yet"). Warm-up runs those same samples through.
//!   - replay: every sample of every file *in real timestamp order*, each at its
//!     own inter-sample gap (the engine saturates cross-session gaps at
//!     MAX_GAP_S = 60 s, exactly like tadi_eval).
//!   - clean = stage-1 rows of post-calibration files (FA s/mo measured here).
//!   - event = stage-2/3 rows of each post-calibration file; a release is
//!     "detected" if an alarm fires within that file's exposure/recovery window;
//!     latency = first alarm minus stage-2 onset.
//!
//! Run: ./target/debug/worner_eval <dir-of-csvs> [--sensitivity S] [--min-channels N]
//!      [--adsorption] [--two-exp] [--mem-gate-s S] [--out out.json]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use opensmell::anomaly::DualKalmanEngine;
use serde::Serialize;

#[derive(Clone)]
struct WRow {
    reading: Vec<f64>,
    stage: u8,
    t: f64,
}

#[derive(Clone)]
struct WFile {
    name: String,
    rows: Vec<WRow>,
    n_clean: usize,
    stage2_onset: f64,
    peak_stage2: f64,
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y0 = if m <= 2 { y - 1 } else { y };
    let era = if y0 >= 0 { y0 } else { y0 - 399 } / 400;
    let yoe = y0 - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn date_to_epoch(date_tok: &str, time_tok: &str) -> f64 {
    // Wörner: Date[yyyy-mm-dd] may appear as dd-mm-yy; Time[hh:mm:ss].
    let d: Vec<i64> = date_tok
        .split('-')
        .filter_map(|p| p.parse::<i64>().ok())
        .collect();
    let t: Vec<i64> = time_tok
        .split(':')
        .filter_map(|p| p.parse::<i64>().ok())
        .collect();
    if d.len() < 3 || t.len() < 3 {
        return 0.0;
    }
    // Date[yyyy-mm-dd] is given as a 2-digit century year, e.g. "24-02-16" =
    // 2024-02-16. Fields are [yy, mm, dd].
    let (y2, mm, dd) = (d[0], d[1], d[2]);
    let year = 2000 + y2;
    days_from_civil(year as i64, mm as i64, dd as i64) as f64 * 86400.0
        + t[0] as f64 * 3600.0
        + t[1] as f64 * 60.0
        + t[2] as f64
}

fn parse_csv(path: &Path, n_channels: usize) -> Option<WFile> {
    let txt = fs::read_to_string(path).ok()?;
    let mut lines = txt.lines();
    let header = lines.next()?;
    let cols: Vec<&str> = header.split_whitespace().collect();
    let ci = cols.iter().position(|c| *c == "Cycle_Stage")?;
    let ti = cols.iter().position(|c| c.starts_with("Time"))?;
    let di = cols.iter().position(|c| c.starts_with("Date"))?;
    let r_idx: Vec<usize> = cols
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            let rest = c.strip_prefix('R').unwrap_or("");
            let digits: String = rest.chars().take_while(|ch| ch.is_ascii_digit()).collect();
            !digits.is_empty()
        })
        .take(n_channels.max(1))
        .map(|(i, _)| i)
        .collect();
    if r_idx.is_empty() {
        return None;
    }
    let mut rows: Vec<WRow> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() <= ci.max(*r_idx.last().unwrap()) {
            continue;
        }
        let stage: u8 = match toks[ci].parse() {
            Ok(v) if (1..=3).contains(&v) => v,
            _ => continue,
        };
        let mut reading = vec![0.0f64; r_idx.len()];
        let mut bad = false;
        for (k, &ri) in r_idx.iter().enumerate() {
            if let Ok(v) = toks[ri].parse::<f64>() {
                if v.is_finite() && v > 0.0 {
                    reading[k] = v;
                } else {
                    bad = true;
                    break;
                }
            } else {
                bad = true;
                break;
            }
        }
        if bad {
            continue;
        }
        let t = date_to_epoch(toks[di], toks[ti]);
        rows.push(WRow { reading, stage, t });
    }
    if rows.len() < 10 {
        return None;
    }
    rows.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    let n_clean = rows.iter().filter(|r| r.stage == 1).count();
    let stage2_onset = rows
        .iter()
        .find(|r| r.stage >= 2)
        .map(|r| r.t)
        .unwrap_or(0.0);
    let peak_stage2 = rows
        .iter()
        .filter(|r| r.stage >= 2)
        .fold(-1e300_f64, |acc, r| acc.max(r.reading[0]));
    Some(WFile {
        name: path.file_name()?.to_string_lossy().to_string(),
        rows,
        n_clean,
        stage2_onset,
        peak_stage2,
    })
}

#[derive(Serialize)]
struct Metrics {
    file: String,
    n_files: usize,
    n_post_cal_files: usize,
    n_detected: usize,
    n_releases: usize,
    detection_rate: f64,
    clean_seconds: f64,
    clean_alarm_seconds: f64,
    fa_per_month: f64,
    median_latency_s: f64,
    cal_file: String,
    cal_rows: usize,
    config: BTreeMap<String, String>,
    per_release: Vec<ReleaseOut>,
}

#[derive(Serialize)]
struct ReleaseOut {
    file: String,
    n_clean: usize,
    clean_alarm_sec: f64,
    stage2_onset_t: f64,
    detected: bool,
    latency_s: f64,
    peak_ch0: f64,
    typology: Option<String>,
}

fn percentile(sorted: &mut Vec<f64>, p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn build_engine(n_ch: usize, args: &[String], sensitivity: f64, min_channels: usize) -> DualKalmanEngine {
    let adsorption = args.iter().any(|a| a == "--adsorption" || a == "--two-exp");
    let mut e = DualKalmanEngine::new(n_ch);
    e.config.sensitivity = sensitivity;
    e.config.min_channels = min_channels;
    if adsorption {
        e.config.adsorption.enabled = true;
        e.config.adsorption.two_exp = args.iter().any(|a| a == "--two-exp");
    }
    if let Some(b) = args
        .iter()
        .rposition(|a| a == "--level-budget")
        .map(|i| args[i + 1].parse::<f64>().ok())
        .flatten()
    {
        e.config.level_budget = b;
    }
    if let Some(r) = args
        .iter()
        .rposition(|a| a == "--level-ref")
        .map(|i| args[i + 1].parse::<f64>().ok())
        .flatten()
    {
        e.config.level_ref_s = r;
    }
    e.set_config(e.config.clone());
    e
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = env::args().collect();
    let data_dir = args
        .get(1)
        .cloned()
        .expect("usage: worner_eval <dir> [--sensitivity S] [--min-channels N]");
    let sensitivity = args
        .iter()
        .rposition(|a| a == "--sensitivity")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(3.0)
        .max(1e-3);
    let min_channels = args
        .iter()
        .rposition(|a| a == "--min-channels")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let out_path = args.iter().rposition(|a| a == "--out").map(|i| args[i + 1].clone());
    let n_channels = args
        .iter()
        .rposition(|a| a == "--n-channels")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(1, 62);

    let mut files: Vec<WFile> = fs::read_dir(&data_dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "csv").unwrap_or(false))
        .filter_map(|e| parse_csv(&e.path(), n_channels))
        .collect();
    files.sort_by(|a, b| {
        let ta = a.rows.first().map(|r| r.t).unwrap_or(0.0);
        let tb = b.rows.first().map(|r| r.t).unwrap_or(0.0);
        ta.partial_cmp(&tb).unwrap()
    });
    if files.is_empty() {
        eprintln!("no usable CSVs in {data_dir}");
        return;
    }

    // Calibration: earliest file's clean (stage-1) window.
    let cal = &files[0];
    let n_ch = cal.rows[0].reading.len();
    let cal_rows: Vec<Vec<f64>> = cal
        .rows
        .iter()
        .filter(|r| r.stage == 1)
        .map(|r| r.reading.clone())
        .collect();
    if cal_rows.len() < n_ch {
        eprintln!("calibration window too small");
        return;
    }

    let mut det = build_engine(n_ch, &args, sensitivity, min_channels);
    if det.calibrate_baseline(&cal_rows).is_err() {
        eprintln!("calibrate failed");
        return;
    }
    // Warm up on the calibration rows at their real cadence.
    let mut prev_cal_t: Option<f64> = None;
    for r in cal.rows.iter().filter(|rr| rr.stage == 1) {
        let dt = match prev_cal_t {
            Some(pt) => (r.t - pt).max(1e-3).min(60.0),
            None => 1.0,
        };
        let _ = det.detect_with_dt(&r.reading, None, dt);
        prev_cal_t = Some(r.t);
    }

    // Replay all files after calibration in chronological order.
    let mut clean_s = 0.0f64;
    let mut clean_ala = 0.0f64;
    let mut latencies: Vec<f64> = Vec::new();
    let mut n_detected = 0usize;
    let mut n_post = 0usize;
    let mut per_release: Vec<ReleaseOut> = Vec::new();
    let mut last_alarm_t = -1e300;
    let mut dual_hold_s = 0.0f64;
    let reanchor_daily = args.iter().any(|a| a == "--reanchor-daily");
    // Wall-clock helper: the calendar day (integer seconds since epoch / 86400)
    // so a new file on a new real day triggers a fresh calibration.
    let day_of = |t: f64| (t / 86400.0).floor() as i64;
    let mut last_cal_day: i64 = day_of(cal.rows[0].t);
    let mem_gate_s = args
        .iter()
        .rposition(|a| a == "--mem-gate-s")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        .max(0.0);
    let dual_confirm_s = args
        .iter()
        .rposition(|a| a == "--confirm-s")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        .max(0.0);

    for (_fi, file) in files.iter().enumerate().skip(1) {
        n_post += 1;
        // Daily re-anchor: on a new real calendar day, re-calibrate on this
        // file's own stage-1 (verified-clean) window and warm up, mirroring the
        // "verify-before-arm each morning" product flow. This file's clean rows
        // then serve as the calibration reference (not scored as FA).
        if reanchor_daily {
            let dd = day_of(file.rows.first().map(|r| r.t).unwrap_or(0.0));
            if dd != last_cal_day {
                let cal_rows: Vec<Vec<f64>> = file
                    .rows
                    .iter()
                    .filter(|r| r.stage == 1)
                    .map(|r| r.reading.clone())
                    .collect();
                if cal_rows.len() >= n_ch && det.calibrate_baseline(&cal_rows).is_ok() {
                    last_cal_day = dd;
                    let mut prev_cal_t: Option<f64> = None;
                    for r in &file.rows {
                        if r.stage != 1 {
                            continue;
                        }
                        let ddt = match prev_cal_t {
                            Some(pt) => (r.t - pt).max(1e-3).min(60.0),
                            None => 1.0,
                        };
                        let _ = det.detect_with_dt(&r.reading, None, ddt);
                        prev_cal_t = Some(r.t);
                    }
                    continue; // calibration file: not scored, not in clean_total
                }
            }
        }
        let mut file_clean_ala = 0.0f64;
        let mut detected = false;
        let mut lat = f64::NAN;
        let mut typology = None;
        let mut prev_t_global: Option<f64> = None;
        for r in &file.rows {
            // Inter-sample gap from the engine's point of view: sample time
            // minus the previous sample's time (both within and across files).
            let dt = match prev_t_global {
                Some(pt) => (r.t - pt).max(1e-3).min(60.0),
                None => 1.0,
            };
            let v = det.detect_with_dt(&r.reading, None, dt);
            let anom = v.is_ok() && v.as_ref().unwrap().is_anomaly;
            let kind = v.ok().and_then(|vv| vv.typology.as_ref().map(|t| t.kind.as_str().to_string()));
            if anom {
                dual_hold_s += dt;
                let confirmed = dual_hold_s + 1e-9 >= dual_confirm_s;
                let gated = mem_gate_s > 0.0 && r.t - last_alarm_t < mem_gate_s;
                if confirmed && !gated {
                    last_alarm_t = r.t;
                    typology = kind.clone();
                    if r.stage == 1 {
                        file_clean_ala += dt;
                    } else if !detected {
                        detected = true;
                        lat = (r.t - file.stage2_onset).max(0.0);
                    }
                }
            } else {
                dual_hold_s = 0.0;
            }
            if r.stage == 1 {
                clean_s += dt;
            }
            prev_t_global = Some(r.t);
        }
        clean_ala += file_clean_ala;
        if detected {
            n_detected += 1;
            latencies.push(lat);
        }
        per_release.push(ReleaseOut {
            file: file.name.clone(),
            n_clean: file.n_clean,
            clean_alarm_sec: file_clean_ala,
            stage2_onset_t: file.stage2_onset,
            detected,
            latency_s: if lat.is_finite() { lat } else { -1.0 },
            peak_ch0: file.peak_stage2,
            typology,
        });
    }

    let fa_per_month = if clean_s > 0.0 {
        (clean_ala / clean_s) * 86400.0 * 30.0
    } else {
        0.0
    };
    let detection_rate = if n_post > 0 {
        n_detected as f64 / n_post as f64
    } else {
        0.0
    };
    let mut lat_sorted = latencies.clone();
    lat_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let m = Metrics {
        file: data_dir.clone(),
        n_files: files.len(),
        n_post_cal_files: n_post,
        n_detected,
        n_releases: n_post,
        detection_rate,
        clean_seconds: clean_s,
        clean_alarm_seconds: clean_ala,
        fa_per_month,
        median_latency_s: percentile(&mut lat_sorted, 0.5),
        cal_file: cal.name.clone(),
        cal_rows: cal_rows.len(),
        config: BTreeMap::from([
            ("sensitivity".into(), format!("{sensitivity}")),
            ("min_channels".into(), format!("{min_channels}")),
            ("n_channels".into(), format!("{n_channels}")),
            ("reanchor_daily".into(), if reanchor_daily { "on".into() } else { "off".into() }),
            ("adsorption".into(), if args.iter().any(|a| a == "--adsorption" || a == "--two-exp") { "on".into() } else { "off".into() }),
            ("confirm_s".into(), format!("{dual_confirm_s}")),
            ("mem_gate_s".into(), format!("{mem_gate_s}")),
        ]),
        per_release,
    };
    let json = serde_json::to_string_pretty(&m).unwrap();
    if let Some(op) = &out_path {
        fs::write(op, &json).unwrap();
        println!("wrote {op}");
        println!("det={:.3} fa/mo={:.1} lat_med={:.1} (cal={}, {} files)", detection_rate, fa_per_month, m.median_latency_s, cal.name, n_post);
    } else {
        println!("{json}");
    }
    eprintln!("runtime {:.1}s", t0.elapsed().as_secs_f64());
}