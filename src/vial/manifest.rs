//! Vial manifest: parse and validate the single source of truth (`vial.toml`).
//!
//! The rules enforced here mirror the ecosystem spec in
//! `private/SDK_AND_VIAL_AUTHORING.md`:
//!   * required `[vial]` fields are present and well-formed;
//!   * `claims.evidence` is MANDATORY for any Vial that provides a risk/decision
//!     output — we never ship a claim without a source;
//!   * paid Vials must be *Sealed* (no obfuscated DRM);
//!   * capability namespaces are closed (unknown ones are rejected), and network
//!     capabilities must be exact hostnames.
//!
//! Errors block shipping; warnings are advisory. `noses lint`/`validate`/`publish`
//! fail on any error.

use semver::Version;
use std::path::Path;
use toml::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Error,
    Warn,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: Level,
    pub message: String,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Self { level: Level::Error, message: message.into() }
    }
    pub fn warn(message: impl Into<String>) -> Self {
        Self { level: Level::Warn, message: message.into() }
    }
}

pub const KNOWN_TIERS: [&str; 3] = ["recipe", "wasm", "sidecar"];
pub const KNOWN_CONTRIBUTES: [&str; 4] = ["sources", "transforms", "views", "models"];
pub const KNOWN_CAPABILITIES: [&str; 3] = ["network", "sensors", "storage"];
pub const KNOWN_PRICING: [&str; 3] = ["free", "paid-one-time", "paid-subscription"];

/// A parsed `vial.toml`. We keep the raw TOML value so unknown/extra fields
/// survive round-trips and future schema growth doesn't break older CLIs.
pub struct VialManifest {
    pub value: Value,
}

impl VialManifest {
    pub fn parse(text: &str) -> Result<Self, String> {
        let value: Value = toml::from_str(text).map_err(|e| format!("invalid TOML: {e}"))?;
        Ok(Self { value })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::parse(&text)
    }

    fn table(&self, key: &str) -> Option<&toml::Table> {
        self.value.get(key).and_then(Value::as_table)
    }

    /// The declared tier (`recipe` | `wasm` | `sidecar`), if valid.
    pub fn tier(&self) -> Option<&str> {
        self.table("vial")?.get("tier")?.as_str()
    }

    /// The Vial id, if present.
    pub fn id(&self) -> Option<&str> {
        self.table("vial")?.get("id")?.as_str()
    }

    /// All validation diagnostics. Empty result = the manifest is shippable.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut d = Vec::new();

        let vial = match self.table("vial") {
            Some(v) => v,
            None => {
                d.push(Diagnostic::error("missing required [vial] table"));
                return d;
            }
        };

        // --- [vial] required fields ---
        for key in ["id", "name", "version", "tier", "license", "min_runtime"] {
            if vial.get(key).and_then(Value::as_str).map_or(true, |s| s.trim().is_empty()) {
                d.push(Diagnostic::error(format!("vial.{key} is required")));
            }
        }

        if let Some(id) = vial.get("id").and_then(Value::as_str) {
            if !valid_id(id) {
                d.push(Diagnostic::error(format!(
                    "vial.id `{id}` must be reverse-DNS (lowercase segments split by dots), e.g. org.example.my-vial"
                )));
            }
        }
        if let Some(v) = vial.get("version").and_then(Value::as_str) {
            if Version::parse(v).is_err() {
                d.push(Diagnostic::error(format!("vial.version `{v}` is not semver (MAJOR.MINOR.PATCH)")));
            }
        }
        if let Some(t) = vial.get("tier").and_then(Value::as_str) {
            if !KNOWN_TIERS.contains(&t) {
                d.push(Diagnostic::error(format!("vial.tier `{t}` must be one of {KNOWN_TIERS:?}")));
            }
        }
        if let Some(m) = vial.get("min_runtime").and_then(Value::as_str) {
            if Version::parse(m).is_err() {
                d.push(Diagnostic::warn(format!("vial.min_runtime `{m}` is not semver")));
            }
        }

        match vial.get("noses") {
            Some(Value::Array(a)) => {
                if a.is_empty() {
                    d.push(Diagnostic::warn("vial.noses is empty — credit the author (Nose)"));
                }
                for n in a {
                    match n.as_str() {
                        Some(s) if s.starts_with('@') => {}
                        Some(s) => d.push(Diagnostic::warn(format!(
                            "vial.noses entry `{s}` should start with '@' (a Nose handle)"
                        ))),
                        None => d.push(Diagnostic::error("vial.noses entries must be strings")),
                    }
                }
            }
            Some(_) => d.push(Diagnostic::error("vial.noses must be an array of handles")),
            None => d.push(Diagnostic::warn("vial.noses is missing — credit the author (Nose)")),
        }

        // --- [contributes] ---
        match self.table("contributes") {
            Some(c) => {
                for (k, v) in c {
                    if !KNOWN_CONTRIBUTES.contains(&k.as_str()) {
                        d.push(Diagnostic::error(format!(
                            "contributes.{k} is not a known contribution namespace (allowed: {KNOWN_CONTRIBUTES:?})"
                        )));
                    }
                    if !v.is_array() {
                        d.push(Diagnostic::error(format!("contributes.{k} must be an array")));
                    }
                }
            }
            None => d.push(Diagnostic::warn("no [contributes] table — the Vial adds nothing to the app")),
        }

        // --- [capabilities] ---
        if let Some(c) = self.table("capabilities") {
            for (k, v) in c {
                if !KNOWN_CAPABILITIES.contains(&k.as_str()) {
                    d.push(Diagnostic::error(format!(
                        "capabilities.{k} is not a known capability (allowed: {KNOWN_CAPABILITIES:?})"
                    )));
                }
                if k == "network" {
                    if let Some(arr) = v.as_array() {
                        for h in arr {
                            match h.as_str() {
                                Some("") => d.push(Diagnostic::error("capabilities.network contains an empty hostname")),
                                Some(s) if s.contains('*') => d.push(Diagnostic::error(format!(
                                    "capabilities.network `{s}` must be an exact hostname — no wildcards"
                                ))),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // --- [claims] : the honesty gate ---
        let provides = self
            .table("claims")
            .and_then(|c| c.get("provides"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !provides.is_empty() {
            let has_evidence = self
                .table("claims")
                .and_then(|c| c.get("evidence"))
                .and_then(Value::as_array)
                .map_or(false, |e| !e.is_empty());
            if !has_evidence {
                d.push(Diagnostic::error(
                    "claims.evidence is MANDATORY when claims.provides is non-empty — cite a DOI or a public benchmark",
                ));
            }
            let has_limits = self
                .table("claims")
                .and_then(|c| c.get("limits"))
                .and_then(Value::as_array)
                .map_or(false, |l| !l.is_empty());
            if !has_limits {
                d.push(Diagnostic::warn("claims.limits is empty — state the envelope this claim holds in"));
            }
        }

        // --- [pricing] ---
        match self.table("pricing") {
            Some(p) => match p.get("model").and_then(Value::as_str) {
                Some(m) if KNOWN_PRICING.contains(&m) => {
                    if m != "free" {
                        let sealed = p.get("sealed").and_then(Value::as_bool).unwrap_or(false);
                        if !sealed {
                            d.push(Diagnostic::error(
                                "paid Vials must be Sealed (pricing.sealed = true) — no obfuscated DRM",
                            ));
                        }
                    }
                }
                Some(m) => d.push(Diagnostic::error(format!(
                    "pricing.model `{m}` must be one of {KNOWN_PRICING:?}"
                ))),
                None => d.push(Diagnostic::warn("pricing.model is missing — assumed free")),
            },
            None => d.push(Diagnostic::warn("no [pricing] table — the Vial is assumed free")),
        }

        d
    }
}

/// Reverse-DNS id: two or more dot-separated lowercase segments of `[a-z0-9-]`,
/// with no leading/trailing hyphen in any segment.
fn valid_id(id: &str) -> bool {
    let parts: Vec<&str> = id.split('.').collect();
    if parts.len() < 2 {
        return false;
    }
    parts.iter().all(|p| {
        !p.is_empty()
            && p.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
            && !p.starts_with('-')
            && !p.ends_with('-')
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
[vial]
id = "org.example.fermentation"
name = "Fermentation Monitor"
version = "1.2.0"
tier = "recipe"
noses = ["@jane"]
license = "MIT"
min_runtime = "0.2.0"

[contributes]
views = ["olfactogram"]

[capabilities]
network = ["api.example.com"]

[claims]
provides = ["spoilage-risk"]
evidence = ["doi:10.1021/cr068121q"]
limits = ["4+ MOX channels"]

[pricing]
model = "free"
"#;

    #[test]
    fn good_manifest_has_no_errors() {
        let m = VialManifest::parse(GOOD).unwrap();
        let errs: Vec<_> = m.validate().into_iter().filter(|d| d.level == Level::Error).collect();
        assert!(errs.is_empty(), "unexpected errors: {:?}", errs);
    }

    #[test]
    fn claim_without_evidence_is_an_error() {
        let bad = GOOD.replace(r#"evidence = ["doi:10.1021/cr068121q"]"#, "evidence = []");
        let m = VialManifest::parse(&bad).unwrap();
        assert!(m.validate().iter().any(|d| d.level == Level::Error && d.message.contains("claims.evidence")));
    }

    #[test]
    fn paid_must_be_sealed() {
        let bad = GOOD.replace(r#"model = "free""#, r#"model = "paid-one-time""#);
        let m = VialManifest::parse(&bad).unwrap();
        assert!(m.validate().iter().any(|d| d.level == Level::Error && d.message.contains("Sealed")));
    }

    #[test]
    fn wildcard_network_is_rejected() {
        let bad = GOOD.replace(r#""api.example.com""#, r#""*.example.com""#);
        let m = VialManifest::parse(&bad).unwrap();
        assert!(m.validate().iter().any(|d| d.level == Level::Error && d.message.contains("wildcards")));
    }

    #[test]
    fn bad_id_is_rejected() {
        assert!(!valid_id("Fermentation"));
        assert!(!valid_id("single"));
        assert!(valid_id("org.example.fermentation-monitor"));
    }
}
