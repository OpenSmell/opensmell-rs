//! MOX thermodynamic feasibility chain (Smellability).
//!
//! Pure-Rust port of `opensmell/opensmell/mox/smellability/` (itself a port of
//! `osmograph-web/lib/smellability/`) — the 4-step chain that grades whether a
//! substance is detectable by a MOX e-nose at room-temperature headspace:
//!
//!   identity -> volatility -> headspace concentration -> MOX redox check
//!
//! This module is a faithful port of the *deterministic core*: the transport
//! physics, band tables, and the chain engine, plus a small reference compound
//! dataset. It deliberately does NOT vendor the full volatile-organic catalogue
//! (~4800 entries) or the RDKit/SMARTS structure-in pipeline; those live in the
//! Python/TypeScript reference and are out of scope for this embedded port. The
//! `resolve_and_run` entry point therefore grades the bundled reference
//! compounds (and any `Chemical` constructed with the public types). The
//! physics and chain logic are numerically cross-checked against the Python
//! implementation in the tests.

use std::collections::BTreeMap;

// ----------------------------------------------------------------- constants

pub const AMBIENT_TEMP_C: f64 = 25.0;
pub const AMBIENT_TEMP_K: f64 = 298.15;
pub const DEFAULT_SENSOR_COUNT: usize = 6;
pub const DEFAULT_DISTANCE_M: f64 = 0.1;
pub const MOX_FLOOR_PPM: f64 = 1.0;
pub const REFERENCE_CHEMICAL_ID: &str = "ethanol";

pub const R: f64 = 8.314;
pub const N_A: f64 = 6.022e23;
pub const P_ATM: f64 = 101325.0;

/// Rated substance-capacity per sensor count (canonical spec table).
pub fn max_substances(sensor_count: usize) -> usize {
    match sensor_count {
        3 => 6,
        4 => 12,
        5 => 20,
        6 => 40,
        12 => 200,
        24 => 10_000,
        _ => 40,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataSource {
    Measured,
    Estimated,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Green,
    Yellow,
    Red,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VerdictConfidence {
    High,
    Medium,
    Low,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignalStrength {
    Strong,
    Moderate,
    Weak,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponseSpeed {
    Fast,
    Medium,
    Slow,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignalBand {
    Strong,
    Moderate,
    Weak,
    Marginal,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResolvedEntityKind {
    Chemical,
    Composite,
    Class,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
enum Ord3 {
    Green,
    Yellow,
    Red,
}

fn verdict_ord(v: Verdict) -> Ord3 {
    match v {
        Verdict::Green => Ord3::Green,
        Verdict::Yellow => Ord3::Yellow,
        Verdict::Red => Ord3::Red,
    }
}

/// The stricter (worse) of two verdicts.
pub fn worst_verdict(a: Verdict, b: Verdict) -> Verdict {
    if verdict_ord(a) >= verdict_ord(b) {
        a
    } else {
        b
    }
}

pub fn signal_score(strength: SignalStrength) -> f64 {
    match strength {
        SignalStrength::Strong => 1.0,
        SignalStrength::Moderate => 0.6,
        SignalStrength::Weak => 0.3,
        SignalStrength::None => 0.0,
    }
}

// ------------------------------------------------------------------- physics

/// Antoine saturated vapor pressure (Pa) from NIST constants (T in °C).
pub fn vapor_pressure_antoine(a: f64, b: f64, c: f64, temp_c: f64) -> f64 {
    let p_mmhg = 10f64.powf(a - b / (temp_c + c));
    p_mmhg * 133.322
}

/// Clausius–Clapeyron saturated vapor pressure (Pa) from a normal boiling point.
pub fn vapor_pressure_clausius_clapeyron(temp_k: f64, t_boil_k: f64, delta_h_vap: f64) -> f64 {
    P_ATM * (-(delta_h_vap / R) * (1.0 / temp_k - 1.0 / t_boil_k)).exp()
}

/// Trouton's-rule enthalpy of vaporization (J/mol).
pub fn delta_h_vap_trouton(t_boil_k: f64) -> f64 {
    88.0 * t_boil_k
}

/// Evaporation flux (mol/(m²·s))?
pub fn evaporation_flux(p_vap: f64, mol_weight_kg: f64, temp_k: f64) -> f64 {
    p_vap / (2.0 * std::f64::consts::PI * mol_weight_kg * R * temp_k).sqrt()
}

/// Fuller–Schettler–Giddings binary diffusion coefficient (m²/s).
pub fn diffusion_coefficient_fuller(mol_weight: f64, diffusion_volume: f64, temp_k: f64, pressure_atm: f64) -> f64 {
    let m_air = 28.97_f64;
    let v_air = 20.1_f64;
    let d_cm2 = (0.00143 * temp_k.powf(1.75))
        / (pressure_atm * (v_air.powf(1.0 / 3.0) + diffusion_volume.powf(1.0 / 3.0)).powi(2))
        * (1.0 / m_air + 1.0 / mol_weight).sqrt();
    d_cm2 * 1e-4
}

pub fn concentration_at_distance(evap_rate: f64, d: f64, distance_m: f64) -> f64 {
    evap_rate / (4.0 * std::f64::consts::PI * d * distance_m)
}

pub fn incident_flux(concentration: f64, mol_weight_kg: f64, temp_k: f64) -> f64 {
    concentration * ((R * temp_k) / (2.0 * std::f64::consts::PI * mol_weight_kg)).sqrt()
}

pub fn diffusion_volume_from_mw(mol_weight: f64) -> f64 {
    1.1 * mol_weight
}

/// A compound's inputs to the incident-flux model.
#[derive(Clone, Copy, Debug)]
pub struct IncidentFluxInput {
    pub vapor_pressure_pa: f64,
    pub mol_weight_kg: f64,
    pub diffusion_volume_cm3: f64,
}

pub fn incident_flux_proportional(input: IncidentFluxInput) -> f64 {
    let d = diffusion_coefficient_fuller(input.mol_weight_kg * 1000.0, input.diffusion_volume_cm3, 298.15, 1.0);
    input.vapor_pressure_pa / (input.mol_weight_kg * d)
}

pub fn signal_ratio_vs_ref(compound: IncidentFluxInput, reference: IncidentFluxInput) -> f64 {
    incident_flux_proportional(compound) / incident_flux_proportional(reference)
}

// -------------------------------------------------------------------- types

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Property {
    pub value: Option<f64>,
    pub source: DataSource,
    pub note: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AntoineCoeffs {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChemicalProperties {
    pub molecular_weight: Property,
    pub boiling_point: Property,
    pub vapor_pressure_25: Property,
    pub functional_groups: Vec<String>,
    pub redox_active: bool,
    pub non_redox: Option<bool>,
    pub gas: Option<bool>,
    pub odor_descriptor: Option<String>,
    pub antoine: Option<AntoineCoeffs>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Chemical {
    pub id: String,
    pub name: String,
    pub synonyms: Vec<String>,
    pub cas: Option<String>,
    pub smiles: Option<String>,
    pub props: ChemicalProperties,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChainValue {
    pub label: String,
    pub value: String,
    pub source: DataSource,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChainStep {
    pub id: String,
    pub label: String,
    pub verdict: Verdict,
    pub reason: String,
    pub detail: String,
    pub values: Vec<ChainValue>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConstituentVerdict {
    pub chemical_id: String,
    pub name: String,
    pub weight_fraction: f64,
    pub weight_source: DataSource,
    pub steps: Vec<ChainStep>,
    pub verdict: Verdict,
    pub signal_strength: SignalStrength,
    pub response_speed: ResponseSpeed,
    pub signal_score: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CrossCheck {
    pub sensor_count: usize,
    pub max_distinguishable: usize,
    pub library_substances: Vec<String>,
    pub confusable: Vec<String>,
    pub note: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeasibilityVerdict {
    pub entity_id: String,
    pub entity_name: String,
    pub kind: ResolvedEntityKind,
    pub verdict: Verdict,
    pub confidence: VerdictConfidence,
    pub signal_strength: SignalStrength,
    pub response_speed: ResponseSpeed,
    pub constituents: Vec<ConstituentVerdict>,
    pub steps: Vec<ChainStep>,
    pub exposure_guidance: String,
    pub dilution_guidance: String,
    /// ISO-8601 UTC timestamp.
    pub computed_at: String,
    pub sensor_count: usize,
    pub notes: Vec<String>,
    pub cross_check: Option<CrossCheck>,
}

#[derive(Clone, Debug, Default)]
pub struct ChainOptions {
    pub sensor_count: Option<usize>,
    pub library_substances: Option<Vec<String>>,
    pub temp_c: Option<f64>,
}

// -------------------------------------------------------------- band tables

fn volatility_label_impl(p_vap_pa: Option<f64>) -> &'static str {
    let Some(p) = p_vap_pa else {
        return "unknown";
    };
    if p >= 10_000.0 {
        "very high"
    } else if p >= 1000.0 {
        "high"
    } else if p >= 100.0 {
        "moderate"
    } else if p >= 1.0 {
        "low"
    } else {
        "negligible"
    }
}

fn headspace_ppm_band_impl(ppm: f64) -> SignalBand {
    if ppm >= 1000.0 {
        SignalBand::Strong
    } else if ppm >= 100.0 {
        SignalBand::Moderate
    } else if ppm >= 10.0 {
        SignalBand::Weak
    } else if ppm >= MOX_FLOOR_PPM {
        SignalBand::Marginal
    } else {
        SignalBand::None
    }
}

/// Signal-band label for a signal ratio relative to the reference.
pub fn signal_band_label(ratio: f64) -> SignalBand {
    if ratio >= 1.0 {
        SignalBand::Strong
    } else if ratio >= 0.1 {
        SignalBand::Moderate
    } else if ratio >= 0.01 {
        SignalBand::Weak
    } else if ratio >= 0.001 {
        SignalBand::Marginal
    } else {
        SignalBand::None
    }
}

// ----------------------------------------------------------- reference data

fn property(value: Option<f64>, source: DataSource) -> Property {
    Property { value, source, note: None }
}

fn reference_dataset() -> BTreeMap<String, Chemical> {
    let mut m = BTreeMap::new();
    for c in [
        Chemical {
            id: "ethanol".into(),
            name: "Ethanol".into(),
            synonyms: vec!["ethyl alcohol".into(), "alcohol".into(), "grain alcohol".into(), "drinking alcohol".into()],
            cas: Some("64-17-5".into()),
            smiles: None,
            props: ChemicalProperties {
                molecular_weight: property(Some(46.07), DataSource::Measured),
                boiling_point: property(Some(78.37), DataSource::Measured),
                vapor_pressure_25: property(Some(7870.0), DataSource::Measured),
                functional_groups: vec!["alcohol".into()],
                redox_active: true,
                non_redox: None,
                gas: None,
                odor_descriptor: Some("alcoholic, solvent".into()),
                antoine: Some(AntoineCoeffs { a: 8.20417, b: 1642.89, c: 230.3 }),
            },
        },
        Chemical {
            id: "acetone".into(),
            name: "Acetone".into(),
            synonyms: vec!["propanone".into(), "dimethyl ketone".into(), "nail polish remover".into()],
            cas: Some("67-64-1".into()),
            smiles: None,
            props: ChemicalProperties {
                molecular_weight: property(Some(58.08), DataSource::Measured),
                boiling_point: property(Some(56.05), DataSource::Measured),
                vapor_pressure_25: property(Some(30600.0), DataSource::Measured),
                functional_groups: vec!["ketone".into()],
                redox_active: true,
                non_redox: None,
                gas: None,
                odor_descriptor: Some("sweet, fruity, solvent".into()),
                antoine: Some(AntoineCoeffs { a: 7.11714, b: 1210.595, c: 229.664 }),
            },
        },
        Chemical {
            id: "water".into(),
            name: "Water".into(),
            synonyms: vec!["H2O".into(), "water vapor".into(), "humidity".into()],
            cas: Some("7732-18-5".into()),
            smiles: None,
            props: ChemicalProperties {
                molecular_weight: property(Some(18.02), DataSource::Measured),
                boiling_point: property(Some(100.0), DataSource::Measured),
                vapor_pressure_25: property(Some(3170.0), DataSource::Measured),
                functional_groups: vec!["inorganic".into()],
                redox_active: false,
                non_redox: Some(false),
                gas: None,
                odor_descriptor: Some("odorless (humidity response)".into()),
                antoine: None,
            },
        },
        Chemical {
            id: "limonene".into(),
            name: "Limonene".into(),
            synonyms: vec!["d-limonene".into(), "citrus terpene".into()],
            cas: Some("5989-27-5".into()),
            smiles: None,
            props: ChemicalProperties {
                molecular_weight: property(Some(136.23), DataSource::Measured),
                boiling_point: property(Some(176.0), DataSource::Measured),
                vapor_pressure_25: property(Some(270.0), DataSource::Estimated),
                functional_groups: vec!["terpene".into(), "alkene".into()],
                redox_active: true,
                non_redox: None,
                gas: None,
                odor_descriptor: Some("citrus, orange".into()),
                antoine: None,
            },
        },
        Chemical {
            id: "methane".into(),
            name: "Methane".into(),
            synonyms: vec!["natural gas".into(), "CH4".into()],
            cas: Some("74-82-8".into()),
            smiles: None,
            props: ChemicalProperties {
                molecular_weight: property(Some(16.04), DataSource::Measured),
                boiling_point: property(Some(-161.5), DataSource::Measured),
                vapor_pressure_25: property(None, DataSource::Unknown),
                functional_groups: vec!["alkane".into()],
                redox_active: true,
                non_redox: None,
                gas: Some(true),
                odor_descriptor: Some("odorless (odorized in supply)".into()),
                antoine: None,
            },
        },
    ] {
        m.insert(c.id.clone(), c);
    }
    m
}

// -------------------------------------------------------------------- chain

#[derive(Clone, Copy, Debug)]
struct EffectiveVaporPressure {
    pa: f64,
    source: DataSource,
}

#[derive(Clone, Copy, Debug)]
struct Guidance {
    exposure: &'static str,
    dilution: &'static str,
}

fn speed_from_volatility(pa: Option<f64>, gas: bool) -> ResponseSpeed {
    if gas || (pa.is_some() && pa.unwrap() >= 1000.0) {
        ResponseSpeed::Fast
    } else if pa.is_some() && pa.unwrap() >= 100.0 {
        ResponseSpeed::Medium
    } else if pa.is_some() && pa.unwrap() >= 1.0 {
        ResponseSpeed::Slow
    } else {
        ResponseSpeed::Unknown
    }
}

fn effective_vapor_pressure(c: &Chemical) -> EffectiveVaporPressure {
    if let Some(v) = c.props.vapor_pressure_25.value {
        return EffectiveVaporPressure { pa: v, source: c.props.vapor_pressure_25.source.clone() };
    }
    if let Some(an) = &c.props.antoine {
        return EffectiveVaporPressure {
            pa: vapor_pressure_antoine(an.a, an.b, an.c, AMBIENT_TEMP_C),
            source: DataSource::Measured,
        };
    }
    if c.props.gas == Some(true) {
        return EffectiveVaporPressure { pa: P_ATM, source: DataSource::Measured };
    }
    if let Some(bp) = c.props.boiling_point.value {
        let t_boil_k = bp + 273.15;
        let pa = vapor_pressure_clausius_clapeyron(AMBIENT_TEMP_K, t_boil_k, delta_h_vap_trouton(t_boil_k));
        return EffectiveVaporPressure { pa, source: DataSource::Estimated };
    }
    EffectiveVaporPressure { pa: 0.0, source: DataSource::Unknown }
}

fn reference_compound() -> Chemical {
    reference_dataset().remove(REFERENCE_CHEMICAL_ID).expect("reference ethanol present")
}

fn signal_ratio(c: &Chemical, reference: &Chemical) -> (f64, DataSource) {
    let vp = effective_vapor_pressure(c);
    if matches!(vp.source, DataSource::Unknown) || vp.pa <= 0.0 {
        return (0.0, vp.source);
    }
    let rvp = effective_vapor_pressure(reference);

    let input = |chem: &Chemical, pv: f64| IncidentFluxInput {
        vapor_pressure_pa: pv,
        mol_weight_kg: chem.props.molecular_weight.value.map(|mw| mw / 1000.0).unwrap_or(0.05),
        diffusion_volume_cm3: chem.props.molecular_weight.value.map(diffusion_volume_from_mw).unwrap_or(55.0),
    };

    let ratio = signal_ratio_vs_ref(input(c, vp.pa), input(reference, rvp.pa));
    let source = match (vp.source, rvp.source) {
        (DataSource::Measured, DataSource::Measured) => DataSource::Measured,
        _ => DataSource::Estimated,
    };
    (ratio, source)
}

fn headspace_ppm(c: &Chemical) -> (Option<f64>, bool, DataSource) {
    let vp = effective_vapor_pressure(c);
    if c.props.gas == Some(true) {
        return (None, true, DataSource::Measured);
    }
    if matches!(vp.source, DataSource::Unknown) || vp.pa <= 0.0 {
        return (None, false, DataSource::Unknown);
    }
    (Some((vp.pa / P_ATM) * 1e6), false, vp.source)
}

fn fmt_pa(pa: Option<f64>) -> String {
    let Some(pa) = pa else { return "unknown".into() };
    if pa >= 100_000.0 {
        format!("{:.0} kPa", pa / 1000.0)
    } else if pa >= 1000.0 {
        format!("{:.2} kPa", pa / 1000.0)
    } else {
        format!("{:.0} Pa", pa)
    }
}

fn fmt_ratio(ratio: f64) -> String {
    if ratio >= 10.0 {
        format!("{:.1}× ethanol", ratio)
    } else if ratio >= 1.0 {
        format!("{:.2}× ethanol", ratio)
    } else if ratio >= 0.1 {
        format!("{:.0}% of ethanol", ratio * 100.0)
    } else {
        format!("{:.1}% of ethanol", ratio * 100.0)
    }
}

fn fmt_ppm(ppm: Option<f64>) -> String {
    let Some(ppm) = ppm else { return "unknown".into() };
    if ppm >= 10_000.0 {
        format!("{:.0}k", ppm / 1000.0)
    } else if ppm >= 100.0 {
        format!("{}", ppm.round())
    } else {
        format!("{:.1}", ppm)
    }
}

fn run_constituent_chain(c: &Chemical, in_catalogue: bool) -> ConstituentVerdict {
    let vp = effective_vapor_pressure(c);
    let mut steps: Vec<ChainStep> = Vec::new();
    let mw = c.props.molecular_weight.value;
    let bp = c.props.boiling_point.value;
    let odour = c
        .props
        .odor_descriptor
        .as_ref()
        .map(|o| format!(" Odour: {}.", o))
        .unwrap_or_default();

    steps.push(ChainStep {
        id: "identity".into(),
        label: "Identity & properties".into(),
        verdict: Verdict::Green,
        reason: if in_catalogue {
            format!("{} resolved from the compound dictionary.", c.name)
        } else {
            format!("{} reconstructed from its SMILES structure.", c.name)
        },
        detail: format!(
            "{}{}. Molecular weight {}, boiling point {}.{}",
            c.name,
            c.cas.as_ref().map(|x| format!(" (CAS {})", x)).unwrap_or_default(),
            mw.map(|m| format!("{:.1} g/mol", m)).unwrap_or_else(|| "unknown".into()),
            bp.map(|b| format!("{:.1} °C", b)).unwrap_or_else(|| "unknown".into()),
            odour,
        ),
        values: vec![
            ChainValue {
                label: "Molecular weight".into(),
                value: mw.map(|m| format!("{:.1} g/mol", m)).unwrap_or_else(|| "unknown".into()),
                source: c.props.molecular_weight.source.clone(),
            },
            ChainValue {
                label: "Boiling point".into(),
                value: bp.map(|b| format!("{:.1} °C", b)).unwrap_or_else(|| "unknown".into()),
                source: c.props.boiling_point.source.clone(),
            },
            ChainValue {
                label: "Vapor pressure @ 25 °C".into(),
                value: fmt_pa(Some(vp.pa)),
                source: vp.source.clone(),
            },
        ],
    });

    let vol_source = if matches!(vp.source, DataSource::Unknown) { None } else { Some(vp.pa) };
    let vol_label = volatility_label_impl(vol_source);
    let (vol_verdict, vol_reason): (Verdict, String) = if c.props.gas == Some(true) {
        (
            Verdict::Green,
            format!("{} is a gas at room temperature — it is already in the vapor phase.", c.name),
        )
    } else if !matches!(vp.source, DataSource::Unknown) {
        if vol_label == "very high" || vol_label == "high" || vol_label == "moderate" {
            (
                Verdict::Green,
                format!("{} has {} volatility ({} at 25 °C) — it readily enters the headspace.", c.name, vol_label, fmt_pa(Some(vp.pa))),
            )
        } else if vol_label == "low" {
            (
                Verdict::Yellow,
                format!("{} has low volatility ({} at 25 °C) — expect a slow, weak headspace unless the sample is warmed.", c.name, fmt_pa(Some(vp.pa))),
            )
        } else {
            (
                Verdict::Red,
                format!("{} is effectively non-volatile at room temperature ({}) — it will not reach the sensor without heating.", c.name, fmt_pa(Some(vp.pa))),
            )
        }
    } else {
        (
            Verdict::Yellow,
            "Vapor pressure unknown — volatility cannot be assessed.".into(),
        )
    };
    steps.push(ChainStep {
        id: "volatility".into(),
        label: "Volatility".into(),
        verdict: vol_verdict,
        reason: vol_reason,
        detail: "Vapor pressure at 25 °C via Antoine equation where constants are curated, else Clausius-Clapeyron from the boiling point with Trouton's-rule enthalpy.".into(),
        values: vec![ChainValue {
            label: "Volatility class".into(),
            value: if c.props.gas == Some(true) { "gas".into() } else { vol_label.into() },
            source: vp.source.clone(),
        }],
    });

    let head = headspace_ppm(c);
    let ratio_info = signal_ratio(c, &reference_compound());
    let hs_band: Option<SignalBand> = if head.1 {
        Some(SignalBand::Strong)
    } else if let Some(h) = head.0 {
        Some(headspace_ppm_band_impl(h))
    } else {
        None
    };
    let mut signal_strength = SignalStrength::None;
    let (sig_verdict, sig_reason): (Verdict, String) = match hs_band {
        None => (
            Verdict::Yellow,
            "Headspace concentration unknown — signal strength cannot be assessed.".into(),
        ),
        Some(b) => match b {
            SignalBand::Strong | SignalBand::Moderate => {
                signal_strength = if b == SignalBand::Strong { SignalStrength::Strong } else { SignalStrength::Moderate };
                if head.1 {
                    (
                        Verdict::Green,
                        format!(
                            "{} is a gas — the vapor phase is available at full concentration, well above the ~{:.0} ppm MOX floor.",
                            c.name, MOX_FLOOR_PPM
                        ),
                    )
                } else {
                    (
                        Verdict::Green,
                        format!(
                            "Saturated headspace is ≈ {} ppm — far above the ~{:.0} ppm MOX floor.",
                            fmt_ppm(head.0), MOX_FLOOR_PPM
                        ),
                    )
                }
            }
            SignalBand::Weak => {
                signal_strength = SignalStrength::Weak;
                let mult = (head.0.unwrap_or(0.0) / MOX_FLOOR_PPM).round().max(1.0);
                (
                    Verdict::Yellow,
                    format!(
                        "Saturated headspace is ≈ {} ppm — detectable, but only {:.0}× the MOX floor. Warm the sample and maximize surface area.",
                        fmt_ppm(head.0), mult
                    ),
                )
            }
            SignalBand::Marginal | SignalBand::None => {
                signal_strength = if b == SignalBand::Marginal { SignalStrength::Weak } else { SignalStrength::None };
                let mult = (head.0.unwrap_or(0.0) / MOX_FLOOR_PPM).round().max(1.0);
                (
                    Verdict::Red,
                    format!(
                        "Saturated headspace is ≈ {} ppm — within {:.0}× of the ~{:.0} ppm floor and unlikely to give a usable response.",
                        fmt_ppm(head.0), mult, MOX_FLOOR_PPM
                    ),
                )
            }
        },
    };
    steps.push(ChainStep {
        id: "signal".into(),
        label: "Headspace concentration".into(),
        verdict: sig_verdict,
        reason: sig_reason,
        detail: "Saturated headspace is the mole fraction of the compound at its vapor pressure (p_vap / P_atm). It is the physical upper bound in an enclosed chamber and is compared against the practical MOX detection floor.".into(),
        values: vec![
            ChainValue {
                label: "Saturated headspace".into(),
                value: if head.1 {
                    "full vapor phase (gas)".into()
                } else if let Some(h) = head.0 {
                    format!("{} ppm", fmt_ppm(Some(h)))
                } else {
                    "unknown".into()
                },
                source: head.2.clone(),
            },
            ChainValue {
                label: "Relative to ethanol".into(),
                value: if matches!(ratio_info.1, DataSource::Unknown) { "unknown".into() } else { fmt_ratio(ratio_info.0) },
                source: ratio_info.1.clone(),
            },
        ],
    });

    let groups = if c.props.functional_groups.is_empty() {
        "no recognized functional groups".to_string()
    } else {
        c.props.functional_groups.join(", ")
    };
    let (react_verdict, react_reason): (Verdict, String) = if c.props.non_redox == Some(true) {
        (
            Verdict::Red,
            format!("{} is not redox-active at MOX operating temperatures — it will not produce the surface reduction MOX sensors detect.", c.name),
        )
    } else if c.props.redox_active {
        (
            Verdict::Green,
            format!("Contains {}; these are oxidized at the ~350 °C sensor surface, producing the resistance change MOX arrays detect.", groups),
        )
    } else if c.id == "water" {
        (
            Verdict::Yellow,
            "Water is not a reducing VOC, but humidity strongly modulates MOX baseline resistance — expect a baseline shift rather than an analyte response.".into(),
        )
    } else {
        (
            Verdict::Yellow,
            format!("{} is not a reducing gas; any response is indirect (e.g. humidity/matrix effects).", c.name),
        )
    };
    steps.push(ChainStep {
        id: "reactivity".into(),
        label: "MOX reactivity".into(),
        verdict: react_verdict,
        reason: react_reason,
        detail: "MOX sensors respond to gases that undergo surface redox at operating temperature. Functional-group chemistry determines this; see the MOX boundaries in the science docs.".into(),
        values: vec![ChainValue {
            label: "Functional groups".into(),
            value: groups,
            source: if c.props.functional_groups.is_empty() { DataSource::Estimated } else { DataSource::Measured },
        }],
    });

    let verdict = steps.iter().fold(Verdict::Green, |acc, s| worst_verdict(acc, s.verdict));
    let speed = speed_from_volatility(vol_source, c.props.gas == Some(true));

    ConstituentVerdict {
        chemical_id: c.id.clone(),
        name: c.name.clone(),
        weight_fraction: 1.0,
        weight_source: DataSource::Measured,
        steps,
        verdict,
        signal_strength,
        response_speed: speed,
        signal_score: signal_score(signal_strength),
    }
}

fn confidence_of(constituents: &[ConstituentVerdict]) -> VerdictConfidence {
    let mut has_unknown = false;
    let mut has_estimated = false;
    for c in constituents {
        for s in &c.steps {
            for v in &s.values {
                match v.source {
                    DataSource::Unknown => has_unknown = true,
                    DataSource::Estimated => has_estimated = true,
                    DataSource::Measured => {}
                }
            }
        }
    }
    if has_unknown {
        VerdictConfidence::Low
    } else if has_estimated {
        VerdictConfidence::Medium
    } else {
        VerdictConfidence::High
    }
}

fn build_cross_check(sensor_count: usize, library: &[String], name: &str, synonyms: &[String]) -> CrossCheck {
    let max_distinguishable = max_substances(sensor_count);
    let lower_name = name.to_lowercase();
    let lower_syns: Vec<String> = synonyms.iter().map(|s| s.to_lowercase()).collect();

    let confusable: Vec<String> = library
        .iter()
        .filter(|label| {
            let l = label.to_lowercase();
            l == lower_name || lower_syns.contains(&l) || l == lower_name || lower_name.contains(&l)
        })
        .cloned()
        .collect();

    let note = if library.is_empty() {
        format!(
            "At {} sensors the array is rated to resolve roughly {} distinct substances. Cross-sensitivity to your library is unknown until you add labeled sessions.",
            sensor_count, max_distinguishable
        )
    } else if !confusable.is_empty() {
        format!(
            "At {} sensors the array is rated to resolve roughly {} distinct substances. \"{}\" in your library may overlap with this substance's response — verify with a labeled exposure.",
            sensor_count, max_distinguishable, confusable.join("\", \"")
        )
    } else {
        format!(
            "At {} sensors the array is rated to resolve roughly {} distinct substances. No exact label overlap found in your library.",
            sensor_count, max_distinguishable
        )
    };

    CrossCheck {
        sensor_count,
        max_distinguishable,
        library_substances: library.to_vec(),
        confusable,
        note,
    }
}

fn guidance(signal: SignalStrength, speed: ResponseSpeed) -> Guidance {
    let _base = "Capture a 30-60 s clean-air baseline first; record the exposure, then a recovery window.";
    match (signal, speed) {
        (SignalStrength::Strong, ResponseSpeed::Fast) => Guidance {
            exposure: concat!(
                "Capture a 30-60 s clean-air baseline first; record the exposure, then a recovery window. ",
                "Signal is expected fast and strong — keep exposures short (10-30 s) and use an enclosed chamber or gentle airflow for repeatability."
            ),
            dilution: "Start diluted (≈1:10 in clean air) and reduce dilution only if the response is small.",
        },
        (SignalStrength::Strong, _) => Guidance {
            exposure: concat!(
                "Capture a 30-60 s clean-air baseline first; record the exposure, then a recovery window. ",
                "Strong signal expected — an enclosed chamber and moderate exposure (20-40 s) will keep you out of saturation."
            ),
            dilution: "A mild dilution (≈1:5) helps stay in the linear response region.",
        },
        (SignalStrength::Moderate, _) => Guidance {
            exposure: concat!(
                "Capture a 30-60 s clean-air baseline first; record the exposure, then a recovery window. ",
                "Moderately detectable — allow 30-60 s of exposure; a small chamber or gentle airflow improves repeatability."
            ),
            dilution: "A mild dilution (≈1:3) may help stay in the linear region.",
        },
        (SignalStrength::Weak, _) => Guidance {
            exposure: concat!(
                "Capture a 30-60 s clean-air baseline first; record the exposure, then a recovery window. ",
                "Weak signal expected — maximize headspace (increase surface area, slightly warm the sample) and use a longer exposure window (60-120 s)."
            ),
            dilution: "Avoid dilution — you need the maximum headspace concentration.",
        },
        _ => Guidance {
            exposure: "Capture a 30-60 s clean-air baseline first; record the exposure, then a recovery window. No usable signal is expected under normal conditions.",
            dilution: "N/A — not expected to be detectable.",
        },
    }
}

/// Look up a bundled reference compound by id.
pub fn reference_by_id(id: &str) -> Option<Chemical> {
    reference_dataset().remove(id)
}

/// Run the feasibility chain for a chemical.
///
/// `in_catalogue` should be true for bundled reference compounds (their
/// identity step says "resolved from the compound dictionary") and false for a
/// caller-constructed `Chemical` (reconstructed from structure).
pub fn run_chemical_verdict(chemical: &Chemical, opts: &ChainOptions, in_catalogue: bool) -> FeasibilityVerdict {
    let sensor_count = opts.sensor_count.unwrap_or(DEFAULT_SENSOR_COUNT);
    let c = run_constituent_chain(chemical, in_catalogue);
    let library = opts.library_substances.clone().unwrap_or_default();
    let cross_check = build_cross_check(sensor_count, &library, &chemical.name, &chemical.synonyms);
    let confidence = confidence_of(&[c.clone()]);
    let g = guidance(c.signal_strength, c.response_speed);
    FeasibilityVerdict {
        entity_id: chemical.id.clone(),
        entity_name: chemical.name.clone(),
        kind: ResolvedEntityKind::Chemical,
        verdict: c.verdict,
        confidence,
        signal_strength: c.signal_strength,
        response_speed: c.response_speed,
        constituents: vec![c.clone()],
        steps: c.steps,
        exposure_guidance: g.exposure.to_string(),
        dilution_guidance: g.dilution.to_string(),
        computed_at: chrono::Utc::now().to_rfc3339(),
        sensor_count,
        notes: vec![],
        cross_check: Some(cross_check),
    }
}

/// Resolve and run the chain against the bundled reference dataset.
pub fn resolve_and_run(entity_id: &str, kind: &str, opts: &ChainOptions) -> Option<FeasibilityVerdict> {
    if kind == "chemical" {
        let c = reference_by_id(entity_id)?;
        return Some(run_chemical_verdict(&c, opts, true));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict_for(id: &str) -> FeasibilityVerdict {
        let c = reference_by_id(id).expect(id);
        run_chemical_verdict(&c, &ChainOptions::default(), true)
    }

    #[test]
    fn reference_parity_with_python() {
        // Values cross-checked against opensmell (python) run_chemical_verdict.
        let cases = [
            ("ethanol", Verdict::Green, VerdictConfidence::High, SignalStrength::Strong, ResponseSpeed::Fast, 1.0),
            ("acetone", Verdict::Green, VerdictConfidence::High, SignalStrength::Strong, ResponseSpeed::Fast, 1.0),
            ("water", Verdict::Yellow, VerdictConfidence::High, SignalStrength::Strong, ResponseSpeed::Fast, 1.0),
            ("limonene", Verdict::Green, VerdictConfidence::Medium, SignalStrength::Strong, ResponseSpeed::Medium, 1.0),
            ("methane", Verdict::Green, VerdictConfidence::High, SignalStrength::Strong, ResponseSpeed::Fast, 1.0),
        ];
        for (id, verdict, conf, sig, speed, score) in cases {
            let v = verdict_for(id);
            assert_eq!(v.verdict, verdict, "{id} verdict");
            assert_eq!(v.confidence, conf, "{id} confidence");
            assert_eq!(v.signal_strength, sig, "{id} signal");
            assert_eq!(v.response_speed, speed, "{id} speed");
            assert_eq!(v.constituents[0].signal_score, score, "{id} score");
        }
    }

    #[test]
    fn chain_has_four_steps_in_order() {
        let v = verdict_for("ethanol");
        let ids: Vec<&str> = v.steps.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["identity", "volatility", "signal", "reactivity"]);
    }

    #[test]
    fn signal_ratio_ethanol_reference_is_one() {
        let c = reference_by_id("ethanol").unwrap();
        let refc = reference_compound();
        let (ratio, source) = signal_ratio(&c, &refc);
        assert!((ratio - 1.0).abs() < 1e-9);
        assert_eq!(source, DataSource::Measured);
    }

    #[test]
    fn water_is_not_redox_active() {
        let v = verdict_for("water");
        assert_eq!(v.verdict, Verdict::Yellow);
    }

    #[test]
    fn resolve_and_run_missing_returns_none() {
        assert!(resolve_and_run("not-a-real-chemical", "chemical", &ChainOptions::default()).is_none());
    }
}
