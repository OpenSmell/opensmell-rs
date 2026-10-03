//! Timing discipline for sequentially-sampled channels — mirror of
//! `opensmell/opensmell/mox/timing.py`.
//!
//! An ESP32 reads its ADC1 channels in a loop, so the values in one reported
//! frame were not acquired at the same instant. The frame is timestamped when it
//! is *reported*, so every channel in it is credited with the frame time and none
//! is credited with the microseconds it actually lagged by.
//!
//! Two consequences matter for anything that reasons about time:
//!
//! 1. Cross-channel comparisons during a sharp common-mode event are biased by the
//!    sweep. [`de_skew`] puts the channels back on a common time base.
//! 2. A reported cadence is a Nyquist statement about what is resolvable at all.
//!    [`min_detectable_duration`] reports the floor, so a caller can decline to
//!    claim an event shorter than the sampling can express.
//!
//! Measured ESP32 sweep is roughly 1-12 ms against a 500 ms sample period. That is
//! negligible for second-scale MOX chemistry, so de-skewing is optional and off by
//! default: it costs a tail of NaNs, and applying it unconditionally would discard
//! real samples in exchange for a correction two orders of magnitude below the
//! noise floor. It matters when a consumer wants channel coherence during a fast
//! transient, and for multi-channel arrays whose sweep is long relative to their
//! sample period.
//!
//! The source of truth for the underlying measurements is
//! `electronic-nose/SAMPLING_CONTRACT.md` ("Sequential ADC skew").

/// De-skewing beyond this fraction of the sample period is not a rounding fix, it
/// is a structural claim that the reported cadence does not describe the data.
pub const MAX_SKEW_FRACTION_OF_PERIOD: f64 = 0.5;

/// Below this many samples per channel there is not enough data to estimate a
/// sweep or to say anything about cadence variability.
pub const MIN_SAMPLES_FOR_TIMING: usize = 3;

/// Acquisition offset per channel, in seconds from the frame timestamp.
///
/// Channel 0 is read first and so carries no lag; channel `k` is read `k` sweeps
/// after the frame was stamped. Assumes a fixed sweep, which is what a bare
/// `analogRead` loop gives.
pub fn scan_offsets(n_channels: usize, sweep_s: f64) -> Result<Vec<f64>, String> {
    if n_channels == 0 {
        return Err(format!("n_channels must be >= 1, got {n_channels}"));
    }
    if !(sweep_s >= 0.0) {
        return Err(format!("sweep_s must be >= 0, got {sweep_s}"));
    }
    Ok((0..n_channels)
        .map(|k| k as f64 * sweep_s)
        .collect())
}

/// Shift each channel onto a common time base.
///
/// `series` is row-major `(n_samples, n_channels)`, indexed by frame. Channel
/// `k`'s sample at frame `i` was really acquired at `t_i + k * sweep_s`, so its
/// value belongs at common-time index `i - shift_k`. The shift is applied by
/// dropping a leading slice of each channel and leaving the tail as NaN, because
/// the data for those frames does not exist yet -- extrapolating them would invent
/// the samples that a cross-channel comparison is most sensitive to.
///
/// `sample_period_s` is required rather than inferred: the sample index carries no
/// time information, and guessing a period from a frame count is exactly the kind
/// of assumption this module exists to avoid.
///
/// The residual sub-sample part of the sweep is not corrected. With a 12 ms sweep
/// at 2 Hz that is 2.4% of a sample, below the converter's own quantisation;
/// [`TimingReport::residual_sub_sample_skew_s`] reports it so the caller can
/// confirm that rather than assume it.
pub fn de_skew(
    series: &[f64],
    n_samples: usize,
    n_channels: usize,
    sweep_s: f64,
    sample_period_s: f64,
) -> Result<Vec<f64>, String> {
    if n_channels == 0 {
        return Err("series must have at least one channel".to_string());
    }
    if series.len() != n_samples * n_channels {
        return Err(format!(
            "series has {} values but {n_samples}x{n_channels} was declared",
            series.len()
        ));
    }
    if !(sample_period_s > 0.0) {
        return Err(format!(
            "sample_period_s must be > 0, got {sample_period_s}"
        ));
    }
    if !(sweep_s >= 0.0) {
        return Err(format!("sweep_s must be >= 0, got {sweep_s}"));
    }

    let offsets = scan_offsets(n_channels, sweep_s)?;
    let mut out = vec![f64::NAN; series.len()];

    for ch in 0..n_channels {
        let raw_shift = (offsets[ch] / sample_period_s).floor();
        // Clamp so a sweep far longer than the recording cannot index out of range.
        let max_shift = if n_samples == 0 { 0 } else { n_samples - 1 };
        let shift = (raw_shift as i64).clamp(0, max_shift as i64) as usize;

        if shift == 0 {
            for i in 0..n_samples {
                out[i * n_channels + ch] = series[i * n_channels + ch];
            }
        } else {
            for i in shift..n_samples {
                out[(i - shift) * n_channels + ch] = series[i * n_channels + ch];
            }
        }
    }

    Ok(out)
}

/// Shortest event the sample sequence can resolve, in seconds.
///
/// An event occupying fewer than about two samples cannot be separated from the
/// sampling itself: it may fall entirely between two frames and leave no trace, or
/// be split across two frames in a way that is indistinguishable from noise. The
/// threshold is `2 dt`, not `dt`, because a one-sample feature has no way to
/// distinguish "the event happened between samples" from "the event did not
/// happen".
///
/// This is a floor on resolvability, not on detectability. A longer event can
/// still be undetectable if its amplitude is below the noise floor; that requires
/// the channel's noise level and is a separate question.
pub fn min_detectable_duration(sampling_rate_hz: f64) -> Result<f64, String> {
    if !(sampling_rate_hz > 0.0) {
        return Err(format!(
            "sampling_rate_hz must be > 0, got {sampling_rate_hz}"
        ));
    }
    Ok(2.0 / sampling_rate_hz)
}

/// Whether an event of `duration_s` is expressible at this cadence.
pub fn is_resolvable(duration_s: f64, sampling_rate_hz: f64) -> Result<bool, String> {
    if !(duration_s >= 0.0) {
        return Err(format!("duration_s must be >= 0, got {duration_s}"));
    }
    Ok(duration_s >= min_detectable_duration(sampling_rate_hz)?)
}

/// For callers that must not report a sub-resolution duration.
///
/// Returns `None` as a refusal to claim the event, which is safer than passing on
/// a number the sample sequence cannot support.
pub fn restrict_to_resolvable(
    duration_s: f64,
    sampling_rate_hz: f64,
) -> Result<Option<f64>, String> {
    Ok(if is_resolvable(duration_s, sampling_rate_hz)? {
        Some(duration_s)
    } else {
        None
    })
}

/// Median inter-sample gap, the cadence the timestamps actually show.
///
/// Measured beats declared: a firmware build constant can be wrong, and the
/// timestamps are what the analysis will divide by. Non-positive steps are dropped
/// rather than included, so a duplicated frame does not deflate the estimate.
pub fn measure_period(time: &[f64]) -> Result<f64, String> {
    if time.len() < 2 {
        return Err(format!("need at least 2 timestamps, got {}", time.len()));
    }
    let mut gaps: Vec<f64> = time
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|g| *g > 0.0)
        .collect();
    if gaps.is_empty() {
        return Err("timestamps contain no positive step".to_string());
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).expect("gap is finite"));
    let mid = gaps.len() / 2;
    Ok(if gaps.len() % 2 == 0 {
        (gaps[mid - 1] + gaps[mid]) / 2.0
    } else {
        gaps[mid]
    })
}

/// What the sample sequence can and cannot support.
#[derive(Debug, Clone, PartialEq)]
pub struct TimingReport {
    pub n_samples: usize,
    pub n_channels: usize,
    pub sampling_rate_hz: Option<f64>,
    pub sample_period_s: Option<f64>,
    /// `"timestamps"`, `"declared"`, or `"unavailable"`.
    pub rate_source: &'static str,
    pub min_detectable_duration_s: Option<f64>,
    pub sweep_s: Option<f64>,
    pub max_skew_s: Option<f64>,
    pub max_skew_fraction_of_period: Option<f64>,
    pub residual_sub_sample_skew_s: Option<f64>,
    pub flags: Vec<String>,
}

impl TimingReport {
    /// Whether an event of this duration can be expressed at this cadence.
    pub fn resolvable(&self, duration_s: f64) -> bool {
        match self.min_detectable_duration_s {
            Some(floor) => duration_s >= floor,
            None => false,
        }
    }

    /// Whether cross-channel coherence here requires de-skewing.
    ///
    /// False when the sweep is unknown or small against the sample period. The
    /// threshold is half a period: below that the skew cannot move a channel by a
    /// full sample, so it cannot reorder or misalign a comparison.
    pub fn needs_de_skew(&self) -> bool {
        match self.max_skew_fraction_of_period {
            Some(f) => f > MAX_SKEW_FRACTION_OF_PERIOD,
            None => false,
        }
    }
}

/// Describe the timing limits of a recording.
///
/// Cadence is taken from the timestamps when they are available and from
/// `sampling_rate_hz` otherwise, recording which in `rate_source`. Neither source
/// is trusted over the other when both exist: the cross-check belongs to the
/// ingestion gate in `electronic-nose/SAMPLING_CONTRACT.md`, which flags a
/// disagreement of more than 2x as a likely time-unit error.
pub fn timing_report(
    time: Option<&[f64]>,
    sampling_rate_hz: Option<f64>,
    n_channels: Option<usize>,
    sweep_s: Option<f64>,
) -> TimingReport {
    let mut flags: Vec<String> = Vec::new();

    let n_obs = time.map(|t| t.len()).unwrap_or(0);
    if n_obs > 0 && n_obs < MIN_SAMPLES_FOR_TIMING {
        flags.push("too_few_samples_to_characterise_timing".to_string());
    }

    let mut period_s: Option<f64> = None;
    let mut rate: Option<f64> = None;
    let mut source = "unavailable";

    if let Some(t) = time {
        if t.len() >= 2 {
            match measure_period(t) {
                Ok(p) => {
                    period_s = Some(p);
                    rate = Some(1.0 / p);
                    source = "timestamps";
                }
                Err(_) => flags.push("timestamps_contain_no_positive_step".to_string()),
            }
        }
    }

    if rate.is_none() {
        if let Some(declared) = sampling_rate_hz {
            if declared > 0.0 {
                rate = Some(declared);
                period_s = Some(1.0 / declared);
                if source == "unavailable" {
                    source = "declared";
                }
            }
        }
    }

    if source == "unavailable" {
        flags.push("cadence_unknown".to_string());
    }
    if let Some(declared) = sampling_rate_hz {
        if !(declared > 0.0) {
            flags.push("declared_rate_not_positive".to_string());
        }
    }
    if let Some(r) = rate {
        if r < 1.0 {
            flags.push("cadence_below_1hz".to_string());
        }
    }

    let channels = n_channels.unwrap_or(0);

    let mut max_skew = None;
    let mut skew_fraction = None;
    let mut residual = None;
    match sweep_s {
        Some(sweep) if channels > 1 => {
            let total = sweep * (channels - 1) as f64;
            max_skew = Some(total);
            if let Some(p) = period_s {
                if p > 0.0 {
                    skew_fraction = Some(total / p);
                    let offsets = scan_offsets(channels, sweep).unwrap_or_default();
                    let worst = offsets
                        .iter()
                        .map(|o| {
                            let frames = o / p;
                            (frames - frames.round()).abs() * p
                        })
                        .fold(0.0_f64, f64::max);
                    residual = Some(worst);
                    if total / p > MAX_SKEW_FRACTION_OF_PERIOD {
                        flags.push("channel_sweep_exceeds_half_sample_period".to_string());
                    }
                }
            }
        }
        Some(_) => flags.push("single_channel_so_sweep_is_irrelevant".to_string()),
        None => {}
    }

    let floor = rate.and_then(|r| min_detectable_duration(r).ok());

    TimingReport {
        n_samples: n_obs,
        n_channels: channels,
        sampling_rate_hz: rate,
        sample_period_s: period_s,
        rate_source: source,
        min_detectable_duration_s: floor,
        sweep_s,
        max_skew_s: max_skew,
        max_skew_fraction_of_period: skew_fraction,
        residual_sub_sample_skew_s: residual,
        flags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, channels: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(n * channels);
        for i in 0..n {
            for k in 0..channels {
                v.push(i as f64 + k as f64);
            }
        }
        v
    }

    fn at(series: &[f64], n_channels: usize, i: usize, ch: usize) -> f64 {
        series[i * n_channels + ch]
    }

    #[test]
    fn first_channel_has_no_lag() {
        assert_eq!(scan_offsets(6, 0.002).unwrap()[0], 0.0);
    }

    #[test]
    fn offsets_are_evenly_spaced() {
        let off = scan_offsets(6, 0.002).unwrap();
        for (k, o) in off.iter().enumerate() {
            assert!((o - k as f64 * 0.002).abs() < 1e-12, "channel {k}: {o}");
        }
    }

    #[test]
    fn zero_sweep_is_all_zero() {
        assert!(scan_offsets(6, 0.0).unwrap().iter().all(|o| *o == 0.0));
    }

    #[test]
    fn rejects_bad_scan_arguments() {
        assert!(scan_offsets(0, 0.001).is_err());
        assert!(scan_offsets(4, -0.001).is_err());
    }

    #[test]
    fn min_detectable_is_two_periods() {
        assert!((min_detectable_duration(2.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((min_detectable_duration(1.0).unwrap() - 2.0).abs() < 1e-12);
        assert!((min_detectable_duration(10.0).unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn rejects_non_positive_rate() {
        assert!(min_detectable_duration(0.0).is_err());
        assert!(min_detectable_duration(-1.0).is_err());
    }

    #[test]
    fn one_sample_is_not_resolvable() {
        assert!(!is_resolvable(0.5, 2.0).unwrap());
        assert!(is_resolvable(1.0, 2.0).unwrap());
    }

    #[test]
    fn boundary_is_inclusive() {
        assert!(is_resolvable(1.0, 2.0).unwrap());
        assert!(!is_resolvable(0.9999999, 2.0).unwrap());
    }

    #[test]
    fn restrict_refuses_below_floor() {
        assert_eq!(restrict_to_resolvable(0.4, 2.0).unwrap(), None);
        assert_eq!(restrict_to_resolvable(1.5, 2.0).unwrap(), Some(1.5));
    }

    #[test]
    fn de_skew_shifts_later_channels_earlier() {
        let (n, ch) = (10, 3);
        let out = de_skew(&ramp(n, ch), n, ch, 0.5, 0.5).unwrap();
        assert!((at(&out, ch, 0, 0) - 0.0).abs() < 1e-12);
        assert!((at(&out, ch, 0, 1) - 2.0).abs() < 1e-12);
        assert!((at(&out, ch, 0, 2) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn de_skew_tail_is_nan_not_extrapolated() {
        let (n, ch) = (10, 3);
        let out = de_skew(&ramp(n, ch), n, ch, 0.5, 0.5).unwrap();
        assert!(at(&out, ch, n - 1, 1).is_nan());
        assert!(at(&out, ch, n - 1, 2).is_nan());
        assert!(at(&out, ch, n - 2, 2).is_nan());
        assert!(!at(&out, ch, n - 1, 0).is_nan());
    }

    #[test]
    fn de_skew_zero_sweep_is_identity() {
        let (n, ch) = (10, 3);
        let src = ramp(n, ch);
        let out = de_skew(&src, n, ch, 0.0, 0.5).unwrap();
        for (a, b) in src.iter().zip(out.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn de_skew_sub_period_sweep_rounds_to_no_shift() {
        // 12 ms against a 500 ms period is 2.4% of a sample: below one frame.
        let (n, ch) = (10, 3);
        let src = ramp(n, ch);
        let out = de_skew(&src, n, ch, 0.012, 0.5).unwrap();
        for (a, b) in src.iter().zip(out.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn de_skew_shift_never_exceeds_frame_count() {
        let (n, ch) = (10, 3);
        let out = de_skew(&ramp(n, ch), n, ch, 100.0, 0.5).unwrap();
        assert_eq!(out.len(), n * ch);
        assert!(at(&out, ch, 1, 2).is_nan());
        assert!(at(&out, ch, 1, 1).is_nan());
    }

    #[test]
    fn de_skew_rejects_bad_arguments() {
        assert!(de_skew(&ramp(10, 3), 10, 3, 0.0, 0.0).is_err());
        assert!(de_skew(&ramp(10, 0), 10, 0, 0.0, 0.5).is_err());
        assert!(de_skew(&ramp(10, 3), 9, 3, 0.0, 0.5).is_err());
        assert!(de_skew(&ramp(10, 3), 10, 3, -1.0, 0.5).is_err());
    }

    #[test]
    fn measure_period_median_gap() {
        assert!((measure_period(&[0.0, 0.5, 1.0, 1.5]).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn duplicate_frames_do_not_deflate_estimate() {
        assert!(
            (measure_period(&[0.0, 0.0, 0.5, 1.0, 1.5]).unwrap() - 0.5).abs() < 1e-12
        );
    }

    #[test]
    fn measure_period_uses_median_not_mean() {
        let mut t = vec![0.0];
        for _ in 0..9 {
            t.push(t[t.len() - 1] + 0.5);
        }
        t.push(t[t.len() - 1] + 0.9);
        assert!((measure_period(&t).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn measure_period_rejects_insufficient_and_non_advancing() {
        assert!(measure_period(&[0.0]).is_err());
        assert!(measure_period(&[1.0, 1.0, 1.0]).is_err());
    }

    #[test]
    fn report_prefers_measured_cadence_over_declared() {
        let t: Vec<f64> = (0..4).map(|i| i as f64 * 0.5).collect();
        let rep = timing_report(Some(&t), Some(1.0), Some(6), None);
        assert_eq!(rep.rate_source, "timestamps");
        assert!((rep.sampling_rate_hz.unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn report_falls_back_to_declared_rate() {
        let rep = timing_report(None, Some(2.0), Some(6), None);
        assert_eq!(rep.rate_source, "declared");
        assert!((rep.min_detectable_duration_s.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unknown_cadence_is_flagged_not_guessed() {
        let rep = timing_report(None, None, Some(6), None);
        assert_eq!(rep.rate_source, "unavailable");
        assert!(rep.sampling_rate_hz.is_none());
        assert!(rep.min_detectable_duration_s.is_none());
        assert!(rep.flags.iter().any(|f| f == "cadence_unknown"));
        assert!(!rep.resolvable(1000.0));
    }

    #[test]
    fn tiny_recording_is_flagged() {
        let rep = timing_report(Some(&[0.0, 0.5]), None, Some(2), None);
        assert!(rep
            .flags
            .iter()
            .any(|f| f == "too_few_samples_to_characterise_timing"));
    }

    #[test]
    fn sub_1hz_cadence_is_flagged() {
        let rep = timing_report(Some(&[0.0, 3.0, 6.0]), None, Some(2), None);
        assert!(rep.flags.iter().any(|f| f == "cadence_below_1hz"));
    }

    #[test]
    fn measured_sweep_is_reported() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let rep = timing_report(Some(&t), None, Some(6), Some(0.012));
        assert!((rep.max_skew_s.unwrap() - 0.060).abs() < 1e-12);
        assert!((rep.max_skew_fraction_of_period.unwrap() - 0.12).abs() < 1e-12);
        assert!((rep.residual_sub_sample_skew_s.unwrap() - 0.06).abs() < 1e-12);
    }

    #[test]
    fn small_sweep_does_not_require_de_skew() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let rep = timing_report(Some(&t), None, Some(6), Some(0.012));
        assert!(!rep.needs_de_skew());
        assert!(!rep
            .flags
            .iter()
            .any(|f| f == "channel_sweep_exceeds_half_sample_period"));
    }

    #[test]
    fn sweep_over_half_a_period_requires_de_skew() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let rep = timing_report(Some(&t), None, Some(6), Some(0.06));
        assert!(rep.needs_de_skew());
        assert!(rep
            .flags
            .iter()
            .any(|f| f == "channel_sweep_exceeds_half_sample_period"));
    }

    #[test]
    fn single_channel_sweep_is_irrelevant() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let rep = timing_report(Some(&t), None, Some(1), Some(0.5));
        assert!(rep.flags.iter().any(|f| f == "single_channel_so_sweep_is_irrelevant"));
        assert!(!rep.needs_de_skew());
    }

    #[test]
    fn unknown_sweep_does_not_force_de_skew() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let rep = timing_report(Some(&t), None, Some(6), None);
        assert!(!rep.needs_de_skew());
        assert!(rep.max_skew_s.is_none());
    }

    #[test]
    fn contract_threshold_matches_sampling_contract() {
        // The SAMPLING_CONTRACT rule is "an event shorter than about 2 dt cannot
        // be resolved".
        assert!((min_detectable_duration(2.0).unwrap() - 2.0 * (1.0 / 2.0)).abs() < 1e-12);
    }

    #[test]
    fn measured_esp32_sweep_is_below_threshold() {
        // Worst measured sweep (12 ms x 5 gaps = 60 ms) at the nominal 2 Hz.
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
        let rep = timing_report(Some(&t), None, Some(6), Some(0.012));
        assert!((rep.max_skew_s.unwrap() - 0.060).abs() < 1e-12);
        assert!(rep.max_skew_fraction_of_period.unwrap() < MAX_SKEW_FRACTION_OF_PERIOD);
    }
}