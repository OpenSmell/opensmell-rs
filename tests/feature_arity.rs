//! Feature-arity proofs: every extraction path must return exactly as many
//! values per channel as `names()` declares.
//!
//! Found by audit, Track 1 (health feature specification). `health::extract()`
//! — the single-sample path — pushed three values per channel while the shared
//! `health::names()` declared four. Because the two were paired positionally,
//! column 1 was labelled `sensitivity_decay` but held raw sensitivity, column 2
//! was labelled `drift_rate` but held hysteresis, and `drift_rate` was absent
//! entirely. No test compared the two lengths, so it passed unnoticed.
//!
//! These assertions are deliberately structural rather than numeric: a silent
//! column shift corrupts every downstream consumer that trusts `names()`.

use opensmell::features::health;
use opensmell::Baseline;

fn baseline(n: usize) -> Baseline {
    let row = vec![0.0; n];
    Baseline::from_samples(&vec![row; 40])
}

#[test]
fn single_sample_path_matches_declared_names() {
    let n = 3;
    let norm = vec![0.5, -0.25, 0.75];
    let got = health::extract(&norm, &norm, &baseline(n)).expect("extract should succeed");
    let declared = health::names(n);

    assert_eq!(
        got.len(),
        declared.len(),
        "health::extract() returned {} features but names() declares {} \
         for {n} channels — columns would silently misalign",
        got.len(),
        declared.len()
    );
    assert_eq!(declared.len(), n * 4, "four health features per channel");
}

#[test]
fn window_path_matches_declared_names() {
    let n = 2;
    let sr = 10.0;
    // window is samples x channels: 6 samples, 2 channels
    let window = vec![
        vec![0.0, 0.0],
        vec![0.4, -0.3],
        vec![0.6, -0.5],
        vec![0.5, -0.4],
        vec![0.3, -0.2],
        vec![0.2, -0.1],
    ];
    let got =
        health::extract_window(&window, &baseline(n), sr).expect("extract_window should succeed");
    let declared = health::names(n);

    assert_eq!(
        got.len(),
        declared.len(),
        "health::extract_window() returned {} features but names() declares {} \
         for {n} channels",
        got.len(),
        declared.len()
    );
}

#[test]
fn both_paths_agree_on_layout() {
    let n = 1;
    let single = health::extract(&[0.5], &[0.5], &baseline(n)).expect("extract");
    let window =
        health::extract_window(&[vec![0.0], vec![0.5]], &baseline(n), 10.0).expect("extract_window");

    assert_eq!(
        single.len(),
        window.len(),
        "the single-sample and windowed paths disagree on feature count; \
         callers cannot switch between them without reindexing"
    );
}

#[test]
fn names_are_stable_and_ordered() {
    assert_eq!(
        health::names(1),
        vec![
            "ch0_noise_floor".to_string(),
            "ch0_sensitivity_decay".to_string(),
            "ch0_drift_rate".to_string(),
            "ch0_hysteresis".to_string(),
        ],
        "feature order is part of the wire contract; changing it silently \
         reshuffles every consumer's columns"
    );
}