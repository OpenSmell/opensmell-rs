# OpenSmell Rust SDK

MOX e-nose feature extraction, anomaly detection, and calibration engine.

## Architecture

```
opensmell-rs/src/
  lib.rs              — Core types (SensorReading, Baseline, errors)
  features/           — 7 modular feature groups (130+ features per 10-sensor config)
  anomaly/mod.rs      — Mahalanobis anomaly detector with Gauss-Jordan matrix inversion
  adaptive.rs         — Adaptive thresholds (Welford's online variance), Platt scaling, fail-safe system, labeling
  poisoning.rs        — Sensor degradation detection (sensitivity decay, noise increase, recovery slowdown, baseline drift)
  calibration/mod.rs  — Zero-calibration, cartridge swaps, cross-device alignment
  health/mod.rs       — Fisher's discriminant ratio, similarity warnings, fleet health
  protocol/mod.rs     — OSM serial protocol parser, Arduino firmware generator
  preprocessing.rs    — CSV loading, baseline correction (EWMA/median/mean/percentile), signal filtering, windowing, validation
```

## Quick Start

```rust
use opensmell::*;

fn main() -> Result<()> {
    // 1. Load raw data from CSV
    let raw = RawData::from_csv("sensor_log.csv")?;
    println!("Loaded {} samples, {} channels", raw.n_samples(), raw.n_channels());

    // 2. Validate data quality
    let mut data = raw.clone();
    let warnings = DataValidator::default().validate(&mut data);
    if !warnings.is_empty() {
        eprintln!("Data quality warnings: {:?}", warnings);
    }

    // 3. Estimate baseline R0 (zero-calibration: 30 min in clean air)
    let baseline = BaselineCorrection {
        method: BaselineMethod::Ewma { alpha: 0.001 },
        ..Default::default()
    }.estimate_r0(&data)?;

    // 4. Normalize: (Rs - R0) / R0
    let normalized = baseline.normalize(&data.samples[0]);
    println!("Normalized reading: {:?}", normalized);

    // 5. Extract features (all 7 groups)
    let features = extract_features(
        &SensorReading::new(data.samples[0].clone(), 0.0),
        &baseline,
        &FeatureGroup::all(),
    )?;
    println!("Extracted {} features", features.len());

    // 6. Detect anomalies
    let mut detector = AdaptiveAnomalyDetector::new(data.n_channels(), 0.05);
    detector.calibrate_baseline(&data.samples)?;
    let result = detector.detect(&data.samples[100])?;
    println!("Anomaly: {}, confidence: {:.3}", result.is_anomaly, result.calibrated_confidence);

    // 7. Label and learn
    detector.update_with_feedback(&data.samples[100], true, "Gas leak detected")?;
    let improvement = detector.get_accuracy_improvement();
    println!("Accuracy improvement: {:.1}%", improvement.improvement * 100.0);

    Ok(())
}
```

## Feature Groups

Seven modular groups covering the full sensor lifecycle:

| Group | Features | Purpose |
|-------|----------|---------|
| `Anomaly` | 8 + N per reading; 8N + C(N,2) per window | Drift, stability, noise, sensitivity decay, hysteresis, saturation |
| `Classification` | 4N per reading; 5N per window | Resistance, normalized deviation, peak response, AUC |
| `Health` | 3N per reading; 4N per window | Noise floor, sensitivity decay, drift rate, hysteresis |
| `Kinetics` | 2N per reading; 6N per window | Rise time (10-90%), decay time (90-10%), tau_fast, tau_slow |
| `Selectivity` | C(N,2) per reading; 2C(N,2) per window | Cross-channel ratios for gas discrimination |
| `Temporal` | N per reading; 4N per window | HF transients, oscillation frequency, response latency |
| `Hardware` | 2 per reading; 2N per window | Circuit response, ADC noise |

**Example:** 10-sensor config produces 130 features per reading, 1,275 per window.

### Feature Count Formulas

- **Single reading:** `8N + N + 3N + 2N + C(N,2) + N + 2` = `14N + N(N-1)/2 + 2`
- **Window features:** `8N + C(N,2) + 5N + 4N + 6N + 2C(N,2) + 4N + 2N` = `29N + 3N(N-1)/2 + 2`

For N=10: 130 features (single), 1,275 features (window).

## Anomaly Detection

### Mahalanobis Detector

```rust
let detector = AnomalyDetector::fit(&baseline_samples, 1.0)?; // sensitivity=1.0
let score = detector.detect(&reading, AnomalyMethod::Mahalanobis)?;
// score.score: 0.0 (normal) to 1.0 (highly anomalous)
```

Uses Gauss-Jordan elimination for covariance matrix inversion. Threshold = 95th percentile of training distances × sensitivity.

### Adaptive Detector (Production)

```rust
let mut detector = AdaptiveAnomalyDetector::new(n_channels, 0.05); // 5% target FPR
detector.calibrate_baseline(&baseline_data)?;
let result = detector.detect(&reading)?;
// result.is_anomaly, result.calibrated_confidence (Platt-scaled)
// result.triggered_channels: which channels contributed most

// User feedback loop
detector.update_with_feedback(&reading, true, "confirmed anomaly")?;
let state = detector.export_state(); // serialize to JSON
```

**Key features:**
- **Welford's online variance** — single-pass, numerically stable statistics
- **Platt scaling** — calibrates raw distances to probabilities (gradient search every 10 feedback items)
- **Drift tracking** — EWMA of normal-sample deviations, flags when statistics diverge
- **Per-channel thresholds** — each sensor slot gets its own adaptive threshold

### Fail-Safe System

Three redundant detectors with majority vote:

| Detector | FPR Target | Sensitivity |
|----------|-----------|-------------|
| Standard | 5% | Normal |
| Conservative | 1% | Low (misses fewer normals) |
| Sensitive | 10% | High (catches more anomalies) |

```rust
let mut failsafe = FailSafeSystem::new(n_channels);
let result = failsafe.detect(&reading)?;
// result.alert_level: 0=normal, 1=warning, 2=critical, 3=emergency
// result.anomaly_votes: how many detectors triggered
// result.sensor_failures: degraded channels
```

**Escalation logic:**
- Level 1 (warning): 2+ consecutive anomalies
- Level 2 (critical): 5+ consecutive anomalies
- Level 3 (emergency): 10+ consecutive anomalies
- Reset to 0: 20 consecutive normal readings

## Labeling System

One-click labeling with CSV+JSON export for data-commons:

```rust
let mut labeler = LabelingSystem::new();
labeler.label_sample(&reading, true, "Gas leak at 42:15", 0.95)?;
labeler.batch_label(&readings, false, "Background air");

let stats = labeler.get_statistics();
// stats.total, stats.normal, stats.anomaly, stats.anomaly_ratio

let output_dir = labeler.export_for_commons(std::path::Path::new("./output"))?;
// Creates CSV + metadata JSON
```

## Sensor Poisoning Detection

Four degradation types with configurable thresholds:

| Degradation | Default Threshold | Indicator |
|-------------|------------------|-----------|
| Sensitivity decay | 5%/day | Catalyst poisoning (permanent) |
| Noise increase | 10%/day | Electrical degradation |
| Recovery slowdown | 15%/day | Surface contamination (reversible) |
| Baseline drift | 2%/day | Environmental drift |

```rust
let config = SensorHealthConfig {
    sensitivity_decay_threshold: 0.05,
    noise_increase_threshold: 0.10,
    recovery_time_threshold: 0.15,
    baseline_drift_threshold: 0.02,
    min_windows: 3,
    window_size_hours: 24.0,
    ..Default::default()
};

let mut detector = PoisoningDetector::new(n_channels, config);
let status = detector.update_channel(0, &channel_data)?;
// status.health_score: 0.0 (failed) to 1.0 (perfect)
// status.degradation_type: Some(DegradationType::SensitivityDecay)
// status.estimated_remaining_life_hours
```

## Calibration

### Zero-Calibration (v0 approach)

30 minutes in clean air establishes baseline R0:

```rust
let mut calibrator = Calibrator::zero_calibration("device-001".into(), 10);
calibrator.calibrate(&baseline_samples, timestamp)?;
let normalized = calibrator.normalize(&raw_reading);
```

### Cartridge Swaps

Razor-blade model: swap sensors without losing calibration context:

```rust
let swap = calibrator.swap_cartridge(
    channel=2,
    new_cartridge_id="MQ-135-2024-001".into(),
    timestamp=now,
)?;
// swap.transfer_success: true if verification passed
// swap.verification_score: 0.0-1.0

let status = calibrator.cartridge_status();
// Vec<CartridgeStatus> with age, health, replacement recommendations
```

### Cross-Device Alignment

Align readings from different devices to a common reference:

```rust
let calibrator = CrossDeviceCalibrator::new(reference_profile, target_profile)?;
let aligned_features = calibrator.align(&target_features)?;
let quality = calibrator.alignment_quality(); // 0.0-1.0
```

## Preprocessing

### Baseline Correction

```rust
let baseline_correction = BaselineCorrection {
    method: BaselineMethod::Ewma { alpha: 0.001 }, // THE critical parameter
    baseline_fraction: 0.15,
    min_baseline_samples: 30,
};

let r0 = baseline_correction.estimate_r0(&raw_data)?;
let normalized = baseline_correction.normalize(&raw_data, &r0);
```

**Methods:**
- `Median` — robust to outliers, default for clean environments
- `Mean` — simple, assumes normal distribution
- `Ewma { alpha }` — exponentially weighted, tracks slow drift (α=0.001 validated on 1M+ samples)
- `Percentile { p }` — resistant to sensor poisoning

### Signal Filters

```rust
let filter = SignalFilter {
    filter_type: FilterType::SavitzkyGolay { polynomial_order: 3 },
    window_size: 11,
};
let filtered = filter.apply(&signal);
```

**Filter types:**
- `Median` — removes spikes without affecting edges
- `MovingAverage` — smooths noise, delays transients
- `SavitzkyGolay` — preserves peak shape, good for kinetics
- `HighPass { cutoff_fraction }` — removes baseline drift

### Window Extraction

```rust
let extractor = WindowExtractor::new(window_size=100, stride=10);
let windows = extractor.extract_windows(&normalized_data);
let avg_features = extractor.extract_averaged_features(
    &normalized_data, &baseline, &FeatureGroup::all(),
)?;
```

## Health Monitoring

```rust
let mut monitor = HealthMonitor::new(n_channels=10, window_size=100);
monitor.set_baseline(baseline_r0);
let fleet_health = monitor.add_reading(&reading)?;
// fleet_health.overall_status: Healthy/Warning/Critical/Failed
// fleet_health.sensors: Vec<SensorHealth> with drift, noise, sensitivity, lifetime estimates
```

### Fisher's Discriminant Ratio

Measures class separability (how distinguishable two substances are):

```rust
let fdr = fisher_discriminant_ratio(&class_a_readings, &class_b_readings)?;
// Per-feature FDR values; >1.0 means separable, >10.0 means easily separable

let pairwise = pairwise_fdr(&[
    ("ethanol", ethanol_readings),
    ("acetone", acetone_readings),
    ("ammonia", ammonia_readings),
])?;
// Returns all class pairs with per-feature and mean FDR
```

### Similarity Warning

Flags when two substances are too similar to classify reliably:

```rust
let (should_warn, message) = similarity_warning(
    &class_a_mean, &class_b_mean,
    &class_a_std, &class_b_std,
    threshold=3.0,
)?;
```

## Protocol

### OSM Serial Protocol

```
OSM,<adc0>,<adc1>,...,<adcN>    — sensor data line
INFO,<device_id>,<fw_version>,<n_sensors>  — device identification
CAL,<channel>,<r0_value>         — calibration event
ERR,<code>,<message>             — error
PING                              — heartbeat
```

```rust
let protocol = OsmProtocol::new(expected_channels=10);
match protocol.parse_line("OSM,1024,2048,512", host_timestamp)? {
    OsmMessage::Data { channels, timestamp } => { /* process */ }
    OsmMessage::Info { device_id, firmware_version, n_sensors } => { /* register device */ }
    OsmMessage::Calibration { channel, r0_value } => { /* update baseline */ }
    OsmMessage::Error { code, message } => { /* handle error */ }
    OsmMessage::Ping => { /* respond pong */ }
    _ => {}
}
```

### Arduino Firmware Generator

```rust
let sketch = generate_arduino_sketch(
    sensor_pins=&[32, 33, 34, 35, 36, 39],
    wifi_ssid="OpenSmell-Dev",
    wifi_password="password123",
);
// Outputs complete .ino file with WiFi, mDNS, TCP server, 10Hz sampling
```

## Data Commons

Contribute labeled data to the shared knowledge base:

```rust
use data_commons::{VerificationPipeline, Metadata};

let pipeline = VerificationPipeline::new(std::path::Path::new("./commons_data"));
let contribution = pipeline.submit(
    std::path::Path::new("labeled_session.csv"),
    std::path::Path::new("metadata.json"),
)?;
// contribution.quality_score: 0-100
// contribution.status: AutoVerified if score >= 60, else Pending

// Human review
let approved = pipeline.approve(&contribution.id)?;
```

**Quality scoring (0-100):**
- Signal quality (30): SNR per channel
- Baseline stability (25): first 20% vs last 20% drift
- Metadata completeness (20): substance, device, date, sensor info, notes
- Session duration (15): 100-1000 samples optimal
- Novelty (10): how different from existing contributions

## Dependencies

```toml
ndarray = "0.15"         # Matrix operations
num-traits = "0.2"       # Generic numeric traits
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
csv = "1"
chrono = "0.4"
log = "0.4"
```

## Testing

```bash
cargo test                    # Run all 20 tests
cargo test -- --nocapture     # Show println! output
cargo test features           # Feature extraction tests only
cargo test adaptive           # Adaptive detector tests only
cargo test poisoning          # Poisoning detector tests only
```
