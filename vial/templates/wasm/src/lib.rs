//! WASM component stub for a tier-2 Vial.
//!
//! Implement the `vial` WIT world in `wit/vial.wit` and export transforms here.
//! Build (component toolchain): `cargo build --target wasm32-unknown-unknown --release`
//! or `cargo component build`. Runs memory-isolated with capability-based access.

/// Deviation of a reading from its baseline — the simplest honest change detector.
pub fn transform(reading: f64, baseline: f64) -> f64 {
    // TODO: replace with your transform. Keep outputs dimensionless/relative unless
    // you have calibration; never return an absolute concentration from an index sensor.
    reading - baseline
}
