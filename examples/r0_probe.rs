//! Cross-SDK R0 window resolution probe.
//!
//! Prints the resolved baseline window and the R0 / sensitivity_decay that each
//! SDK derives from the same window, for a set of cadences and window lengths.
//! The Python and JS halves of the same probe live in
//! /tmp/opencode/r0/ and are diffed against this output.

use opensmell::features::health;
use opensmell::{r0_window_samples, Baseline, R0_WINDOW_DEFAULT};

const CADENCES: [f64; 4] = [1.0, 2.0, 10.0, 100.0];
const LENGTHS: [f64; 4] = [5.0, 10.0, 60.0, 300.0];

fn window(fs: f64, duration_s: f64) -> Vec<Vec<f64>> {
    // 12 s clean-air plateau, then a monotone 50-unit exposure ramp over 10 s.
    let n = (duration_s * fs).round() as usize + 1;
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let ramp = ((t - 12.0) / 10.0).clamp(0.0, 1.0);
            vec![1000.0 + 50.0 * ramp]
        })
        .collect()
}

fn main() {
    println!("cadence_hz,duration_s,n_samples,window,r0,sensitivity_decay");
    for fs in CADENCES {
        for duration_s in LENGTHS {
            if duration_s < 12.0 {
                continue; // shorter than the plateau: R0 would read the ramp
            }
            let w = window(fs, duration_s);
            let resolved = r0_window_samples(w.len(), R0_WINDOW_DEFAULT);
            let bl = Baseline::from_samples(&w);
            // health::names order: [noise_floor, sensitivity_decay, drift_rate,
            // hysteresis].
            let h = health::extract_window(&w, &bl, fs).unwrap();
            println!(
                "{fs},{duration_s},{},{resolved},{},{}",
                w.len(),
                bl.r0[0],
                h[1]
            );
        }
    }
}
