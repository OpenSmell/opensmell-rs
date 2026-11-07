//! Domain Vials: the shipped manifest format, its policy gate, and the
//! authoring metadata `noses` needs.
//!
//! The desktop app is the authority. `src/domainPacks.ts` in the Osmograph
//! desktop repository runs the same gate before a manifest can be installed,
//! and the sandbox is the execution boundary. This module mirrors those rules
//! so an author gets the answer locally, before pushing — it never replaces the
//! host gate. When the two disagree, the host wins.
//!
//! One format: `vial.json` with schema `osmograph.domain-pack/2` — the
//! programmable profile, the UI contract, and the evidence — plus optional
//! sandboxed HTML fragments (`ui.html`, `home.html`) beside it. `/1` is accepted
//! as a legacy manifest but cannot be listed in the registry. There is no
//! privileged tier.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The largest UI fragment the host will mount.
pub const UI_MAX_BYTES: usize = 512 * 1024;

/// Bridge methods the sandbox exposes. A Vial may call these and nothing else.
pub const BRIDGE_METHODS: &[&str] = &[
    "version", "onData", "action", "ready", "toast",
    "setDetection", "setTransforms", "setTrajectory",
];

#[derive(Debug, Clone)]
pub struct Check {
    pub id: String,
    pub label: String,
    pub passed: bool,
    pub critical: bool,
    pub detail: String,
}

#[derive(Debug, Default)]
pub struct Validation {
    pub checks: Vec<Check>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl Validation {
    pub fn passed(&self) -> bool {
        self.errors.is_empty()
    }
    fn add(&mut self, id: &str, label: &str, passed: bool, critical: bool, detail: impl Into<String>) {
        let detail = detail.into();
        if !passed {
            if critical { self.errors.push(detail.clone()); } else { self.warnings.push(detail.clone()); }
        }
        self.checks.push(Check {
            id: id.to_string(),
            label: label.to_string(),
            passed,
            critical,
            detail,
        });
    }
}

pub struct Vial {
    pub path: PathBuf,
    pub raw: String,
    pub json: Value,
}

impl Vial {
    pub fn load(path: &Path) -> Result<Vial, String> {
        let path = if path.is_dir() { path.join("vial.json") } else { path.to_path_buf() };
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let json: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;
        Ok(Vial { path, raw, json })
    }

    fn str(&self, key: &str) -> &str {
        self.json.get(key).and_then(Value::as_str).unwrap_or("")
    }
    pub fn id(&self) -> &str { self.str("id") }
    pub fn name(&self) -> &str { self.str("name") }
    pub fn version(&self) -> &str { self.str("version") }
    pub fn author(&self) -> &str { self.str("author") }
    pub fn license(&self) -> &str { self.str("license") }
    pub fn domain(&self) -> &str { self.str("domain") }
    pub fn tagline(&self) -> &str { self.str("tagline") }
    pub fn summary(&self) -> &str { self.str("summary") }
    pub fn has_ui(&self) -> bool {
        self.json.get("ui").map(|v| !v.is_null()).unwrap_or(false)
            || self.json.get("homeUi").map(|v| !v.is_null()).unwrap_or(false)
    }
    pub fn is_v2(&self) -> bool {
        self.str("schema") == "osmograph.domain-pack/2"
    }
    /// Positive scope. v2 uses `scope`; v1 used `whatItIs`.
    pub fn scope(&self) -> &str {
        let scope = self.str("scope");
        if scope.is_empty() { self.str("whatItIs") } else { scope }
    }
    pub fn limits(&self) -> Vec<String> {
        let declared: Vec<String> = arr(&self.json, "limits").iter()
            .filter_map(Value::as_str).map(str::to_string).collect();
        if declared.is_empty() {
            let legacy = self.str("whatItIsNot").trim();
            if legacy.is_empty() { Vec::new() } else { vec![legacy.to_string()] }
        } else {
            declared
        }
    }
    pub fn anomaly(&self) -> Option<&Value> {
        self.json.get("anomaly").filter(|v| v.is_object())
    }
    pub fn evidence(&self) -> Option<&Value> {
        self.json.get("evidence").filter(|v| v.is_object())
    }
    pub fn dir(&self) -> &Path {
        self.path.parent().unwrap_or_else(|| Path::new("."))
    }

    /// sha256 of the manifest bytes, exactly as the registry pins it.
    pub fn sha256_hex(&self) -> String {
        let digest = Sha256::digest(self.raw.as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// All the text the policy gate reads (mirrors `allPackText` in the host).
    fn policy_text(&self) -> String {
        let mut chunks: Vec<String> = vec![
            self.id().into(), self.name().into(), self.author().into(), self.license().into(),
            self.domain().into(), self.tagline().into(), self.summary().into(),
            self.scope().into(),
        ];
        for limit in self.limits() { chunks.push(limit); }
        if let Some(ev) = self.evidence() {
            chunks.push(text(ev, "dataset"));
            chunks.push(text(ev, "reportUrl"));
        }
        for c in arr(&self.json, "citations") {
            if let Some(s) = c.as_str() { chunks.push(s.to_string()); }
        }
        for d in arr(&self.json, "docs") {
            chunks.push(text(d, "title"));
            for b in arr(d, "body") { if let Some(s) = b.as_str() { chunks.push(s.to_string()); } }
        }
        for w in arr(&self.json, "widgets") {
            chunks.push(w.to_string());
        }
        if let Some(h) = self.json.get("home") {
            chunks.push(text(h, "headline"));
            chunks.push(text(h, "detail"));
            for l in arr(h, "limits") { if let Some(s) = l.as_str() { chunks.push(s.to_string()); } }
            for s in arr(h, "sources") { if let Some(s) = s.as_str() { chunks.push(s.to_string()); } }
        }
        chunks.join("\n")
    }

    pub fn validate(&self) -> Validation {
        let mut v = Validation::default();
        let j = &self.json;

        let is_v2 = self.is_v2();
        v.add("schema", "Manifest schema",
              self.str("schema") == "osmograph.domain-pack/1" || is_v2, true,
              if is_v2 {
                  "osmograph.domain-pack/2: profile, UI, and evidence."
              } else {
                  "Legacy osmograph.domain-pack/1. Installable; re-author as v2 to be listed."
              });

        if let Some(min) = j.get("minHost").and_then(Value::as_str) {
            let ok = semver::Version::parse(min).is_ok()
                && semver::Version::parse(env!("CARGO_PKG_VERSION"))
                    .map(|host| semver::Version::parse(min).map(|m| host >= m).unwrap_or(false))
                    .unwrap_or(false);
            v.add("min-host", "Host version", ok, true, if ok {
                format!("Requires Osmograph >= {min}.")
            } else {
                format!("Requires Osmograph >= {min}; this tool is {}.", env!("CARGO_PKG_VERSION"))
            });
        }

        let id_re = Regex::new(r"^[a-z0-9][a-z0-9._-]{2,80}$").unwrap();
        v.add("id", "Stable id", id_re.is_match(self.id()), true,
              "Use a lowercase, globally unique id (3–80 chars), for example publisher.slug.");
        v.add("name", "Name", in_range(self.name(), 2, 80), true,
              "A pack needs a short display name.");
        v.add("version", "Semantic version", semver::Version::parse(self.version()).is_ok(), true,
              "Use semver, for example 0.1.0.");
        v.add("author", "Author identity", in_range(self.author(), 2, 80), true,
              "Credit the author or handle.");
        v.add("license", "Licence", in_range(self.license(), 2, 80), true,
              "State a licence (SPDX id or a licence URL).");
        v.add("risk", "Risk level", ["low", "medium", "high", "critical"].contains(&self.str("risk")), true,
              "Declare low, medium, high, or critical risk.");
        v.add("domain", "Domain", self.domain().trim().len() >= 2, true,
              "Name the domain, for example fermentation or plants.");
        v.add("tagline", "Tagline", in_range(self.tagline(), 4, 160), true,
              "One honest line about the workspace.");
        v.add("summary", "Summary", in_range(self.summary(), 30, 600), true,
              "Explain the workspace and its scope.");

        v.add("scope", "Scope", self.scope().trim().len() >= 30, true,
              "State the decision this Vial supports.");
        if is_v2 {
            let limits = self.limits();
            v.add("limits", "Operating envelope",
                  !limits.is_empty() && limits.iter().all(|l| l.trim().len() >= 10), true,
                  "Name at least one condition where the Vial refuses or stops.");
            validate_anomaly(j, &mut v);
            validate_evidence(j, &mut v);
        } else {
            let what_it_is_not = self.str("whatItIsNot");
            let refuses = Regex::new(r"(?i)(?:not|cannot|no\s|never|without|refuse)").unwrap();
            v.add("what-it-is-not", "Operating envelope",
                  what_it_is_not.trim().len() >= 30 && refuses.is_match(what_it_is_not), true,
                  "Name the conditions where the check refuses.");
            v.add("anomaly", "Anomaly profile", false, false,
                  "Legacy manifest has no programmable profile. It installs as Ungated; re-author as v2 to be listed.");
        }

        let citations = arr(j, "citations");
        v.add("citations", "Citations",
              !citations.is_empty() && citations.iter().all(|c| c.as_str().map(|s| s.trim().len() >= 10).unwrap_or(false)),
              true, "At least one real reference, not a vibe.");

        let docs = arr(j, "docs");
        v.add("docs", "Field notes",
              docs.len() >= 2 && docs.iter().all(|d| !text(d, "title").trim().is_empty() && !arr(d, "body").is_empty()),
              true, "At least two documented sections with body text.");

        let types: BTreeSet<&str> = arr(j, "widgets").iter()
            .filter_map(|w| w.get("type").and_then(Value::as_str)).collect();
        v.add("hero", "Overview panel", types.contains("hero"), true, "A domain Vial needs an overview.");
        v.add("risk-panel", "Risk panel", types.contains("risk"), true, "A domain Vial must show its risk framing.");
        v.add("docs-panel", "Docs panel", types.contains("docs"), true, "A domain Vial must expose its field notes.");
        v.add("rig-panel", "Rig handoff", types.contains("rig"), true, "Show how the workspace connects to Check.");
        v.add("workspace", "A workspace surface",
              types.contains("tracking") || types.contains("diagram") || types.contains("rules"), true,
              "Add a log, a diagram, or decision rules.");

        if let Some(home) = j.get("home") {
            let actions = arr(home, "actions");
            let kinds = ["open-workspace", "open-check", "open-datasets", "open-gallery", "external"];
            v.add("home-contract", "Home contribution",
                  !text(home, "headline").trim().is_empty() && !text(home, "detail").trim().is_empty(), true,
                  "A Home contribution needs a headline and a detail line.");
            v.add("home-actions", "Home actions",
                  (1..=3).contains(&actions.len()) && actions.iter().all(|a| {
                      let kind = text(a, "kind");
                      !text(a, "id").trim().is_empty() && !text(a, "label").trim().is_empty()
                          && kinds.contains(&kind.as_str())
                          && (kind != "external" || !text(a, "href").trim().is_empty())
                  }), true,
                  "Provide one to three valid actions; external actions need an href.");
            let metrics = arr(home, "metrics");
            v.add("home-metrics", "Home metrics",
                  metrics.len() <= 4 && metrics.iter().all(|m| !text(m, "label").trim().is_empty() && !text(m, "value").trim().is_empty()),
                  true, "At most four metrics, each with a label and a value.");
            v.add("home-limits", "Home limits",
                  !arr(home, "limits").is_empty() && arr(home, "limits").iter().all(|l| l.as_str().map(|s| s.trim().len() >= 10).unwrap_or(false)),
                  true, "State at least one explicit limit in the Home contribution.");
            v.add("home-sources", "Home sources",
                  !arr(home, "sources").is_empty() && arr(home, "sources").iter().all(|s| s.as_str().map(|s| s.trim().len() >= 10).unwrap_or(false)),
                  true, "Cite at least one source for the Home contribution.");
        }

        let risk_widgets: Vec<&Value> = arr(j, "widgets").iter()
            .filter(|w| w.get("type").and_then(Value::as_str) == Some("risk")).collect();
        v.add("risk-lists", "Always / never limits",
              !risk_widgets.is_empty() && risk_widgets.iter().all(|w| !arr(w, "always").is_empty() && !arr(w, "never").is_empty()),
              true, "The risk panel needs explicit always and never lists.");
        let consistent = risk_widgets.iter().all(|w| text(w, "level") == self.str("risk"));
        v.add("risk-match", "Risk level consistency", consistent, false,
              "The risk panel and pack should declare the same level.");

        let risk = self.str("risk");
        if risk == "high" || risk == "critical" {
            let stop_rules = arr(j, "widgets").iter()
                .filter(|w| w.get("type").and_then(Value::as_str) == Some("rules"))
                .flat_map(|w| arr(w, "rules").to_vec())
                .filter(|r| text(r, "severity") == "stop")
                .count();
            v.add("high-risk-stop", "High-risk stop rules", stop_rules >= 1, true,
                  "High and critical risk packs need at least one explicit stop rule.");
            v.add("high-risk-docs", "High-risk documentation", docs.len() >= 3, true,
                  "High and critical risk packs need at least three field-note sections.");
        }

        for widget in arr(j, "widgets") {
            match widget.get("type").and_then(Value::as_str).unwrap_or("") {
                "tracking" => v.add("tracking-fields", "Tracking fields",
                    arr(widget, "fields").len() >= 2 && arr(widget, "logFields").len() >= 2, true,
                    "A log needs at least two context fields and two reading fields."),
                "diagram" => v.add("diagram-hotspots", "Diagram hotspots",
                    !arr(widget, "hotspots").is_empty() && arr(widget, "hotspots").iter().all(|h| {
                        let x = h.get("x").and_then(Value::as_f64).unwrap_or(-1.0);
                        let y = h.get("y").and_then(Value::as_f64).unwrap_or(-1.0);
                        (0.0..=100.0).contains(&x) && (0.0..=100.0).contains(&y)
                            && !text(h, "label").trim().is_empty() && !text(h, "body").trim().is_empty()
                    }), true, "Diagram points need labels, notes, and 0–100 coordinates."),
                "detector" => {
                    let levels = arr(widget, "levels");
                    v.add("detector-levels", "Detector states",
                        levels.len() >= 3 && levels.iter().all(|l| l.get("level").and_then(Value::as_f64).is_some()
                            && !text(l, "label").trim().is_empty() && !text(l, "meaning").trim().is_empty()
                            && !text(l, "action").trim().is_empty()),
                        true, "Document at least three detector states with a meaning and an action.");
                    v.add("detector-limits", "Detector limits",
                        arr(widget, "limits").len() >= 2 && arr(widget, "limits").iter().all(|l| l.as_str().map(|s| s.trim().len() >= 10).unwrap_or(false)),
                        true, "A detector panel needs at least two explicit limits.");
                    v.add("detector-evidence", "Detector evidence",
                        text(widget, "evidence").trim().len() >= 20, true,
                        "State the evidence behind the detector mapping.");
                }
                "rules" => v.add("rule-text", "Rule text",
                    arr(widget, "rules").iter().all(|r| !text(r, "when").trim().is_empty() && !text(r, "then").trim().is_empty()),
                    true, "Every rule needs a when and a then."),
                "checklist" => v.add("checklist-items", "Checklist items",
                    !arr(widget, "items").is_empty() && arr(widget, "items").iter().all(|i| !text(i, "id").trim().is_empty() && !text(i, "label").trim().is_empty()),
                    true, "Checklist items need stable ids and labels."),
                _ => {}
            }
        }

        // The manifest itself must stay declarative.
        let ui_specs = ui_specs(j);
        let policy_text = self.policy_text();
        let executable = first_executable(&policy_text);
        match executable {
            None => v.add("no-code", "Manifest stays declarative", true, true, "Manifest text only."),
            Some(id) => v.add("no-code", "Manifest stays declarative", false, true,
                format!("Rejected {id}: executable UI belongs in the separate sandboxed asset.")),
        }

        // The programmable surface is a property of the pack: a workspace UI and
        // a Home UI can split the work, so warn only when neither declares it.
        let programmable_any = ui_specs.iter().any(|(_, _, spec)| {
            std::fs::read_to_string(self.dir().join(text(spec, "entry")))
                .map(|b| b.contains("setTransforms") || b.contains("setTrajectory"))
                .unwrap_or(false)
        });
        let entry_re = Regex::new(r"(?i)^[a-z0-9][a-z0-9._-]*\.html$").unwrap();
        let v2 = j.get("anomaly").is_some();
        for (key, label, spec) in &ui_specs {
            v.add(&format!("{key}-runtime"), &format!("{label} runtime"),
                  spec.get("runtime").and_then(Value::as_str) == Some("sandbox/1"), true,
                  "Only sandbox/1 is supported. The UI runs in an iframe with no network, storage, file, or Tauri access.");
            let entry = text(spec, "entry");
            let entry_ok = entry_re.is_match(&entry) && !entry.contains("..");
            v.add(&format!("{key}-entry"), &format!("{label} entry"), entry_ok, true,
                  "Use a relative .html filename beside the manifest; no paths, schemes, or query strings.");
            if entry_ok {
                v.add_asset(&format!("{key}-asset"), &format!("{label} asset"), self.dir().join(&entry), programmable_any, v2);
            }
        }

        if j.get("home").is_some() && j.get("homeUi").is_none() {
            v.add("home-ui", "Home UI", true, false,
                  "No homeUi declared; the host renders the Home contribution from the manifest.");
        }
        if j.get("homeUi").is_some() && j.get("home").is_none() {
            v.add("home-ui", "Home UI", false, true,
                  "homeUi without a home contribution never mounts: add a home block or drop homeUi.");
        }

        // Placeholders are refused: a scaffold must not be publishable as-is.
        let placeholder = Regex::new(r"(?i)REPLACE ME|TODO|TBD|FIXME|yourname|your-handle|example\.(?:com|org|net)").unwrap();
        match placeholder.find(&policy_text) {
            Some(m) => v.add("placeholder", "No placeholder text", false, true,
                format!("Replace the placeholder text (\"{}\") before validating.", m.as_str())),
            None => v.add("placeholder", "No placeholder text", true, true, "No placeholder markers."),
        }

        let overclaims = forbidden_claims(&policy_text);
        v.add("no-overclaim", "No overclaim language", overclaims.is_empty(), true, if overclaims.is_empty() {
            "No unnegated forbidden claims found.".to_string()
        } else {
            format!("Remove or negate: {}.", overclaims.join(", "))
        });

        // UI text is user-facing: scan it too, but only warn (the host reads the manifest).
        for (key, label, spec) in &ui_specs {
            let asset = self.dir().join(text(spec, "entry"));
            if let Ok(body) = std::fs::read_to_string(&asset) {
                let claims = forbidden_claims(&body);
                if !claims.is_empty() {
                    v.add(&format!("{key}-claims"), &format!("{label} claims"), false, false,
                          format!("The UI text contains {}; the host gate only scans the manifest, so fix it here.", claims.join(", ")));
                }
            }
        }

        v
    }

    /// sha256 of every UI asset the manifest declares, keyed by filename.
    /// The host verifies each fetched asset against the registry, so a
    /// manifest hash alone cannot cover a swapped ui.html.
    pub fn asset_hashes(&self) -> Value {
        let mut assets = serde_json::Map::new();
        for key in ["ui", "homeUi"] {
            let Some(entry) = self.json.get(key).and_then(|v| v.get("entry")).and_then(Value::as_str) else { continue; };
            let path = self.dir().join(entry);
            if let Ok(bytes) = std::fs::read(&path) {
                let digest = Sha256::digest(&bytes);
                let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
                assets.insert(entry.to_string(), Value::String(hex));
            }
        }
        Value::Object(assets)
    }

    /// sha256 of every declared fixture. The host does not execute fixtures at
    /// runtime; these pins record the evidence claim a reviewer checked.
    pub fn fixture_hashes(&self) -> Value {
        let mut fixtures = serde_json::Map::new();
        let Some(entries) = self.evidence().and_then(|e| e.get("fixtures")).and_then(Value::as_array) else {
            return Value::Object(fixtures);
        };
        for fixture in entries {
            let Some(file) = fixture.get("file").and_then(Value::as_str) else { continue; };
            if !file.starts_with("fixtures/") || file.contains("..") { continue; }
            if let Ok(bytes) = std::fs::read(self.dir().join(file)) {
                let digest = Sha256::digest(&bytes);
                let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
                fixtures.insert(file.to_string(), Value::String(hex));
            }
        }
        Value::Object(fixtures)
    }

    /// The registry entry an author pastes into `opensmell-web/public/vials/registry.json`.
    pub fn registry_entry(&self, repo: &str, branch: &str, evidence: Option<&str>) -> Result<Value, String> {
        let (owner, name) = parse_github_repo(repo)
            .ok_or("--repo must be a GitHub repository URL, for example https://github.com/owner/repo")?;
        let manifest_url = format!("https://raw.githubusercontent.com/{owner}/{name}/{branch}/vial.json");
        let source_url = format!("https://github.com/{owner}/{name}");
        let evidence = evidence
            .map(str::to_string)
            .or_else(|| self.evidence().and_then(|e| e.get("grade")).and_then(Value::as_str).map(str::to_string));
        let assets = self.asset_hashes();
        Ok(serde_json::json!({
            "schema": self.str("schema"),
            "id": self.id(),
            "name": self.name(),
            "version": self.version(),
            "author": self.author(),
            "license": self.license(),
            "domain": self.domain(),
            "tagline": self.tagline(),
            "summary": self.summary(),
            "hasUi": self.has_ui(),
            "evidence": evidence,
            "manifestUrl": manifest_url,
            "sourceUrl": source_url,
            "hash": self.sha256_hex(),
            "assets": assets,
            "fixtures": self.fixture_hashes(),
            "install": "Install from source above, or paste the manifest URL in Osmograph → Vials → Aromatory."
        }))
    }
}

impl Validation {
    fn add_asset(&mut self, id: &str, label: &str, path: PathBuf, programmable_any: bool, v2: bool) {
        let body = match std::fs::read_to_string(&path) {
            Ok(b) => b,
            Err(e) => {
                self.add(id, label, false, true, format!("cannot read {}: {e}", path.display()));
                return;
            }
        };
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if body.trim().is_empty() {
            self.add(id, label, false, true, format!("{name} is empty."));
            return;
        }
        if body.len() > UI_MAX_BYTES {
            self.add(id, label, false, true, format!("{name} is larger than 512 KB."));
            return;
        }
        if let Some(reason) = fragment_problem(&body) {
            self.add(id, label, false, true, format!("{name}: {reason}"));
            return;
        }
        self.add(id, label, true, true, format!("{name} is a sandboxable fragment ({} bytes).", body.len()));

        let mut unknown: Vec<String> = Vec::new();
        let re = Regex::new(r"osmograph\s*\.\s*([A-Za-z_$][A-Za-z0-9_$]*)").unwrap();
        for cap in re.captures_iter(&body) {
            let method = cap[1].to_string();
            if !BRIDGE_METHODS.contains(&method.as_str()) && !unknown.contains(&method) {
                unknown.push(method);
            }
        }
        self.add(&format!("{id}-bridge"), &format!("{label} bridge"),
              unknown.is_empty(), true, if unknown.is_empty() {
                  "Only host bridge calls are used.".to_string()
              } else {
                  format!("Unknown bridge method(s): {}.", unknown.join(", "))
              });
        let live = body.contains("onData");
        self.add(&format!("{id}-live"), &format!("{label} live data"), live, false,
              if live { "Listens to host data." } else { "The UI never calls osmograph.onData; it is a static surface." });
        if v2 {
            let legacy: Vec<&str> = ["setDetection", "setTransforms"].into_iter()
                .filter(|method| body.contains(method))
                .collect();
            self.add(&format!("{id}-legacy"), &format!("{label} no legacy overrides"), legacy.is_empty(), true,
                  if legacy.is_empty() {
                      "The v2 profile is compiled from the manifest; the UI makes no runtime override.".to_string()
                  } else {
                      format!("A v2 UI must not call {}; the profile is compiled from the manifest.", legacy.join(" or "))
                  });
        }
        let programmable = v2 || body.contains("setTransforms") || body.contains("setTrajectory");
        let ok = programmable || programmable_any;
        self.add(&format!("{id}-program"), &format!("{label} programmable"), ok, false,
              if v2 {
                  "The anomaly profile in the manifest is the program."
              } else if body.contains("setTransforms") || body.contains("setTrajectory") {
                  "Declares host transforms or a trajectory."
              } else if programmable_any {
                  "No declarations here; the pack declares them in another surface."
              } else {
                  "Declares neither host transforms nor a trajectory; the host cannot compute or persist anything for this pack."
              });
    }
}

fn number(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn validate_anomaly(j: &Value, v: &mut Validation) {
    let Some(a) = j.get("anomaly").filter(|x| x.is_object()) else {
        v.add("anomaly", "Anomaly profile", false, true,
              "A v2 Vial must declare the anomaly block: baseline, transforms, detection, escalation.");
        return;
    };
    let baseline = a.get("baseline").unwrap_or(&Value::Null);
    let detection = a.get("detection").unwrap_or(&Value::Null);
    let escalation = detection.get("escalation").unwrap_or(&Value::Null);
    let admissibility = a.get("admissibility").unwrap_or(&Value::Null);
    let mut errors: Vec<String> = Vec::new();

    let warmup = number(baseline, "warmupS").unwrap_or(f64::NAN);
    if !(0.5..=86400.0).contains(&warmup) { errors.push("baseline.warmupS must be between 0.5 and 86400 seconds.".into()); }
    let r0 = number(baseline, "r0WindowS").unwrap_or(f64::NAN);
    if !(0.5..=3600.0).contains(&r0) { errors.push("baseline.r0WindowS must be between 0.5 and 3600 seconds.".into()); }
    let sensitivity = number(detection, "sensitivity").unwrap_or(f64::NAN);
    if !(0.1..=10.0).contains(&sensitivity) { errors.push("detection.sensitivity must be between 0.1 and 10.".into()); }
    let min_channels = number(detection, "minChannels").unwrap_or(f64::NAN);
    if !(min_channels >= 1.0) || min_channels.fract() != 0.0 { errors.push("detection.minChannels must be an integer of at least 1.".into()); }
    let level_budget = number(detection, "levelBudget").unwrap_or(0.0);
    if !(0.0..=12.0).contains(&level_budget) { errors.push("detection.levelBudget must be between 0 and 12 sigma.".into()); }
    let level_ref = number(detection, "levelRefS").unwrap_or(0.0);
    if level_ref < 0.0 { errors.push("detection.levelRefS cannot be negative.".into()); }
    if level_ref > 0.0 && level_budget == 0.0 { errors.push("detection.levelRefS needs a non-zero levelBudget.".into()); }
    let threshold = number(detection, "alertThreshold").unwrap_or(f64::NAN);
    if !(0.0..=1.0).contains(&threshold) { errors.push("detection.alertThreshold must be between 0 and 1.".into()); }
    let warning = number(escalation, "warningS").unwrap_or(f64::NAN);
    let critical = number(escalation, "criticalS").unwrap_or(f64::NAN);
    let emergency = number(escalation, "emergencyS").unwrap_or(f64::NAN);
    let reset = number(escalation, "resetNormalS").unwrap_or(f64::NAN);
    if !(warning > 0.0) { errors.push("detection.escalation.warningS must be positive.".into()); }
    if !(critical >= warning) { errors.push("detection.escalation.criticalS must be at least warningS.".into()); }
    if !(emergency >= critical) { errors.push("detection.escalation.emergencyS must be at least criticalS.".into()); }
    if !(reset > 0.0) { errors.push("detection.escalation.resetNormalS must be positive.".into()); }
    let hysteresis = number(detection, "hysteresis").unwrap_or(0.0);
    if !(0.0..=0.95).contains(&hysteresis) { errors.push("detection.hysteresis must be between 0 and 0.95.".into()); }
    if let Some(far) = number(detection, "farTargetPerMonth") {
        if far < 0.0 { errors.push("detection.farTargetPerMonth cannot be negative.".into()); }
    }
    if let Some(limits) = detection.get("perChannelLimits").and_then(Value::as_object) {
        let key_re = Regex::new(r"^[a-z0-9][a-z0-9_-]{0,47}$").unwrap();
        for (key, raw) in limits {
            if !key_re.is_match(key) { errors.push(format!("detection.perChannelLimits has an invalid channel id: {key}.")); }
            if !raw.as_f64().map(|n| n > 0.0).unwrap_or(false) {
                errors.push(format!("detection.perChannelLimits.{key} must be greater than zero."));
            }
        }
    }
    let coverage = number(admissibility, "minCoverage").unwrap_or(0.8);
    if !(0.0..=1.0).contains(&coverage) { errors.push("admissibility.minCoverage must be between 0 and 1.".into()); }

    v.add("anomaly", "Anomaly profile", true, true, "Baseline, transforms, detection, and escalation are declared.");
    v.add("anomaly-values", "Profile values", errors.is_empty(), true,
          errors.first().cloned().unwrap_or_else(|| "Profile values are in range.".into()));
}

fn validate_evidence(j: &Value, v: &mut Validation) {
    let Some(e) = j.get("evidence").filter(|x| x.is_object()) else {
        // The host installs a v2 without evidence as Ungated. Lint and test
        // must agree; publish (the listing path) requires the block instead.
        v.add("evidence", "Held-out evidence", false, false,
              "No evidence block declared; this Vial installs as Ungated and cannot be listed.");
        return;
    };
    let report = text(e, "reportUrl");
    let dataset = text(e, "dataset");
    let metrics = e.get("metrics").unwrap_or(&Value::Null);
    let has_number = ["farPerMonth", "latencyS", "mde", "nSamples", "nAlarms"]
        .iter().any(|k| number(metrics, k).map(|n| n.is_finite()).unwrap_or(false));
    v.add("evidence", "Held-out evidence",
          report.starts_with("https://") && dataset.trim().len() >= 10 && has_number, true,
          "A v2 Vial needs an https evidence report, the dataset it describes, and at least one measured number.");
    let measured = text(e, "measuredAt");
    v.add("evidence-date", "Evidence date",
          Regex::new(r"^\d{4}-\d{2}-\d{2}").unwrap().is_match(&measured), false,
          if measured.is_empty() { "Add the measurement date.".into() } else { format!("Measured {measured}.") });
    let fixtures = arr(e, "fixtures");
    let file_re = Regex::new(r"^fixtures/[a-z0-9][a-z0-9._-]*\.json$").unwrap();
    let bad = fixtures.iter().any(|f| !file_re.is_match(&text(f, "file")));
    v.add("evidence-fixtures", "Declarative fixtures",
          !fixtures.is_empty() && !bad, false,
          if fixtures.is_empty() {
              "No fixtures declared; `noses test` can only check the static surface.".into()
          } else if bad {
              "Fixture path must be fixtures/<name>.json.".into()
          } else {
              format!("{} fixture(s).", fixtures.len())
          });
}

fn in_range(s: &str, min: usize, max: usize) -> bool {
    let n = s.trim().chars().count();
    n >= min && n <= max
}

fn arr<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

fn text(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

fn ui_specs(j: &Value) -> Vec<(String, String, Value)> {
    let mut out = Vec::new();
    if let Some(spec) = j.get("ui") { out.push(("ui".to_string(), "UI".to_string(), spec.clone())); }
    if let Some(spec) = j.get("homeUi") { out.push(("home-ui".to_string(), "Home UI".to_string(), spec.clone())); }
    out
}

/// Mirror of `assertSandboxableUi` in the desktop host.
fn fragment_problem(html: &str) -> Option<&'static str> {
    let re = |p: &str| Regex::new(p).unwrap();
    if re(r"(?i)<!doctype|<html[\s>]|<head[\s>]|<body[\s>]|<base[\s>]").is_match(html) {
        return Some("a Vial UI must be a body fragment, not a full HTML document");
    }
    if re(r"(?i)<(?:script|iframe|object|embed|link)[^>]+(?:src|href)\s*=").is_match(html) {
        return Some("a Vial UI cannot load external scripts, frames, objects, or stylesheets");
    }
    if let Ok(media) = Regex::new(r#"(?i)<(?:img|source|video|audio)[^>]+src\s*=\s*["']([^"']*)"#) {
        for cap in media.captures_iter(html) {
            let src = &cap[1];
            if !(src.starts_with('#') || src.starts_with("data:")) {
                return Some("a Vial UI cannot load remote images or media");
            }
        }
    }
    if re(r"(?i)javascript\s*:").is_match(html) {
        return Some("a Vial UI cannot use javascript: URLs");
    }
    // The sandbox has no network and no storage; these calls would only fail.
    if re(r"(?i)\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(").is_match(html) {
        return Some("a Vial UI cannot use the network");
    }
    if re(r"(?i)\b(?:localStorage|sessionStorage|indexedDB)\b").is_match(html) {
        return Some("a Vial UI cannot use browser storage");
    }
    if re(r"(?i)\beval\s*\(|new\s+Function\s*\(").is_match(html) {
        return Some("a Vial UI cannot evaluate strings as code");
    }
    None
}

fn first_executable(text: &str) -> Option<&'static str> {
    let patterns: [(&str, &str); 6] = [
        ("script-tag", r"(?i)<script\b"),
        ("iframe", r"(?i)<iframe\b"),
        ("object", r"(?i)<object\b|<embed\b|<img\b|<link\b"),
        ("javascript-url", r"(?i)javascript:"),
        ("inline-handler", r"(?i)\bon[a-z]+\s*="),
        ("data-html", r"(?i)data:text/html"),
    ];
    for (id, pat) in patterns {
        if Regex::new(pat).unwrap().is_match(text) { return Some(id); }
    }
    None
}

fn forbidden_claims(text: &str) -> Vec<&'static str> {
    let patterns: [(&str, &str); 11] = [
        ("a “safe to eat” claim", r"(?i)\bsafe to eat\b"),
        ("a pathogen test claim", r"(?i)\bpathogen(?: test| detection| identification)\b"),
        ("a diagnosis claim", r"(?i)\bdiagnos(?:e|es|is|tic)\b"),
        ("an ethylene detection claim", r"(?i)\bdetect(?:s|ing)? ethylene\b"),
        ("a universal smell ID claim", r"(?i)\buniversal smell (?:id|identifier)\b"),
        ("an absolute concentration claim", r"(?i)\b(?:absolute concentration|parts per million|ppm)\b"),
        ("a cross-rig claim", r"(?i)\bcross[- ]rig\b"),
        ("a works-on-any-rig claim", r"(?i)\bworks? on any (?:rig|array|device)\b"),
        ("a guarantee", r"(?i)\bguarantee(?:d|s)?\b"),
        ("a medical claim", r"(?i)\bmedical (?:diagnosis|device|advice)\b"),
        ("a safety-device claim", r"(?i)\b(?:gas leak detector|certified safety device|life[- ]safety device|safety instrument)\b"),
    ];
    let mut out = Vec::new();
    for (label, pat) in patterns {
        let re = Regex::new(pat).unwrap();
        if re.find_iter(text).any(|m| !negated_before(text, m.start())) {
            out.push(label);
        }
    }
    out
}

/// The sentence before a match decides whether it is negated, matching the
/// host's rule (`.` `!` `?` and newlines end a sentence).
fn negated_before(text: &str, index: usize) -> bool {
    let before = &text[..index];
    let cut = before.rfind(['.', '!', '?', '\n']).map(|i| i + 1).unwrap_or(0);
    let sentence = before[cut..].to_lowercase();
    Regex::new(r"\b(?:not|no|cannot|can't|never|without|refuse|refuses|refused|doesn't|does not|isn't|is not|zero)\b")
        .unwrap()
        .is_match(&sentence)
}

fn parse_github_repo(url: &str) -> Option<(String, String)> {
    let url = url.trim().trim_end_matches('/').trim_end_matches(".git");
    let rest = url.strip_prefix("https://github.com/").or_else(|| url.strip_prefix("http://github.com/"))?;
    let mut parts = rest.split('/');
    let owner = parts.next()?;
    let name = parts.next()?;
    if owner.is_empty() || name.is_empty() { return None; }
    Some((owner.to_string(), name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("vial/examples/minimal")
    }

    #[test]
    fn the_bundled_example_passes_the_gate() {
        let vial = Vial::load(&example()).expect("example loads");
        let validation = vial.validate();
        assert!(validation.passed(), "example fails: {:?}", validation.errors);
        assert!(validation.errors.is_empty());
    }

    #[test]
    fn a_missing_what_it_is_not_fails() {
        let mut vial = Vial::load(&example()).unwrap();
        vial.json["whatItIsNot"] = Value::String("It tells you everything.".into());
        let v = vial.validate();
        assert!(!v.passed());
        assert!(v.errors.iter().any(|e| e.contains("refuses")));
    }

    #[test]
    fn an_unnegated_overclaim_fails() {
        let mut vial = Vial::load(&example()).unwrap();
        vial.json["summary"] = Value::String(
            "A field kit that guarantees pathogen detection on any rig, with absolute concentration.".into());
        let v = vial.validate();
        assert!(!v.passed());
        assert!(v.errors.iter().any(|e| e.contains("Remove or negate")));
    }

    #[test]
    fn a_negated_overclaim_is_allowed() {
        let mut vial = Vial::load(&example()).unwrap();
        vial.json["summary"] = Value::String(
            "A change log for one array. It cannot diagnose a plant and does not provide absolute concentration.".into());
        let v = vial.validate();
        assert!(v.passed(), "{:?}", v.errors);
    }

    #[test]
    fn placeholder_text_is_refused() {
        let mut vial = Vial::load(&example()).unwrap();
        vial.json["summary"] = Value::String("REPLACE ME with a real summary of this workspace.".into());
        let v = vial.validate();
        assert!(!v.passed());
        assert!(v.errors.iter().any(|e| e.contains("placeholder")));
    }

    #[test]
    fn the_hash_is_the_manifest_bytes() {
        let vial = Vial::load(&example()).unwrap();
        // Same bytes → same hash; the registry pins this value.
        assert_eq!(vial.sha256_hex(), Vial::load(&example()).unwrap().sha256_hex());
        assert_eq!(vial.sha256_hex().len(), 64);
    }

    #[test]
    fn a_github_repo_resolves() {
        assert_eq!(
            parse_github_repo("https://github.com/praisejamesx/plant-respiration.git"),
            Some(("praisejamesx".into(), "plant-respiration".into()))
        );
        assert_eq!(parse_github_repo("https://example.com/x"), None);
    }
}
