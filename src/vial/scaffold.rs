//! `noses new` — scaffold a real, programmable Vial.
//!
//! The scaffold is deliberately incomplete in exactly two places: it has no
//! citations and it carries `REPLACE ME` markers. Both are errors in the gate,
//! so `noses lint` refuses until the author writes real text. A scaffold that
//! already passed would be a publishable placeholder.

use std::path::{Path, PathBuf};

use serde_json::json;

pub fn kebab(name: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Create a pack directory. Returns the directory it wrote.
pub fn create(name: &str, publisher: &str, domain: &str, parent: &Path) -> Result<PathBuf, String> {
    let slug = kebab(name);
    if slug.is_empty() {
        return Err("a Vial needs a name with at least one letter or digit".into());
    }
    let publisher = publisher.trim().trim_start_matches('@').to_lowercase();
    if !publisher.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') || publisher.is_empty() {
        return Err("--publisher must be a handle: letters, digits, dashes, underscores".into());
    }
    let dir = parent.join(&slug);
    if dir.exists() {
        return Err(format!("{} already exists", dir.display()));
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;

    let id = format!("{publisher}.{slug}");
    let manifest = json!({
        "schema": "osmograph.domain-pack/1",
        "id": id,
        "name": name,
        "version": "0.1.0",
        "author": format!("@{publisher}"),
        "license": "MIT",
        "risk": "medium",
        "domain": domain,
        "tagline": "REPLACE ME: one honest line about this workspace.",
        "summary": "REPLACE ME: what this workspace logs, what it measures, and what it refuses to claim.",
        "whatItIs": "REPLACE ME: the protocol or useful claim this Vial carries.",
        "whatItIsNot": "It is not a laboratory instrument. It does not identify compounds, quantify concentration, or compare readings from a different array.",
        "citations": [],
        "ui": { "runtime": "sandbox/1", "entry": "ui.html", "height": 520 },
        "widgets": [
            {
                "type": "hero",
                "art": "generic",
                "title": name,
                "subtitle": "One array, one workspace, one honest limit.",
                "note": "Sandboxed UI · host-owned data"
            },
            {
                "type": "risk",
                "level": "medium",
                "headline": "A change on this array is not a verdict.",
                "body": "The array responds to a mixture of volatiles. A reading is evidence that the sample changed on this rig, not what changed or why.",
                "always": [
                    "Record the sample, the moment, and the conditions with every reading.",
                    "Compare only against a baseline taken on the same array and the same preparation."
                ],
                "never": [
                    "Never treat a reading as an identification or a safety verdict.",
                    "Never compare readings from a different array."
                ]
            },
            {
                "type": "tracking",
                "title": "Sample log",
                "description": "The log is the record. The reading is one column in it.",
                "fields": [
                    { "id": "sample", "label": "Sample", "type": "text", "placeholder": "what is in the jar" },
                    { "id": "context", "label": "Context", "type": "text", "placeholder": "preparation, room, time" }
                ],
                "logFields": [
                    { "id": "reading", "label": "Reading", "type": "number", "placeholder": "0–10" },
                    { "id": "note", "label": "Note", "type": "text", "placeholder": "what you observed" }
                ],
                "logLabel": "Add reading"
            },
            {
                "type": "rules",
                "title": "Stop rules",
                "rules": [
                    {
                        "when": "A reading moves and nothing you changed explains it",
                        "then": "Re-baseline on the same array and repeat the measurement before acting.",
                        "severity": "warn"
                    },
                    {
                        "when": "The array has drifted, saturated, or lost a channel",
                        "then": "Stop. Repair or recalibrate before recording again.",
                        "severity": "stop"
                    }
                ]
            },
            { "type": "docs", "title": "Field notes" },
            {
                "type": "rig",
                "title": "Use this with your rig",
                "body": "Record a clean reference and a sample on the same array. Compare them in Check. If the arrays differ, the app refuses the comparison.",
                "cta": "Open Check"
            }
        ],
        "docs": [
            {
                "id": "how",
                "title": "How to use this workspace",
                "body": [
                    "Start one record. Fill the sample and context fields, then add a reading whenever you measure or observe the sample.",
                    "The Home UI shows the array's own noise per channel and lets you arm a single-unit watch. The limit you arm is in transformed units (ΔR/R₀), not concentration.",
                    "Use Check to compare a sample against a clean reference on the same array. The comparison is per-array and per-session."
                ]
            },
            {
                "id": "sensor",
                "title": "What the sensor is doing",
                "body": [
                    "A MOX array responds to a mixture of volatile compounds. The response depends on the sensor, the temperature, the humidity, the distance, and the sample preparation.",
                    "The useful signal is a change relative to a reference on the same array. The absolute number does not transfer to another rig.",
                    "A channel that is dead, stuck, or saturated is excluded by the host. A reading from a dead channel is not a measurement."
                ]
            },
            {
                "id": "limits",
                "title": "Limits",
                "body": [
                    "This scaffold does not yet name a scientific claim. Add the real references behind your protocol to citations; the gate refuses a Vial without them.",
                    "This is not a diagnosis, a safety verdict, or a calibrated concentration."
                ]
            }
        ]
    });

    let manifest_path = dir.join("vial.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap() + "\n")
        .map_err(|e| format!("cannot write {}: {e}", manifest_path.display()))?;

    let ui_path = dir.join("ui.html");
    std::fs::write(&ui_path, UI_HTML.replace("__PACK_NAME__", name))
        .map_err(|e| format!("cannot write {}: {e}", ui_path.display()))?;

    let readme_path = dir.join("README.md");
    std::fs::write(&readme_path, readme(name, &id))
        .map_err(|e| format!("cannot write {}: {e}", readme_path.display()))?;

    Ok(dir)
}

fn readme(name: &str, id: &str) -> String {
    format!(
        r#"# {name}

`{id}` — a programmable Osmograph Vial.

## What is here

| File | Role |
|---|---|
| `vial.json` | The manifest (`osmograph.domain-pack/1`). The host re-runs this gate at install. |
| `ui.html` | A sandboxed body fragment. No network, no storage, no Tauri, no external files. |
| `README.md` | This file. |

## The loop

```
noses lint      # schema, claims, UI sandbox and bridge, placeholders
noses test      # lint + what the UI actually declares
noses publish   # sha256 + the exact registry entry to open a PR with
```

`noses new` leaves **two deliberate gaps**: `citations` is empty and the text
carries `REPLACE ME` markers. Both are errors in the gate, so `lint` refuses
until you write real text. That is the point: a scaffold must never be
publishable by accident.

## What the host owns

- Sampling, baseline, anomaly detection, and the hardware-sufficiency gate.
- The transforms (declared through `osmograph.setTransforms`) and the
  trajectory persisted every interval (`osmograph.setTrajectory`).
- The single-unit limits (`channelLimits`: measured `cv`, `min`, `max`, `r0`)
  and whether a channel is dead. A Vial may tighten a limit, never bypass a
  refusal.

Your UI receives a typed snapshot (`payload.snapshot`) and a bounded series
window (`payload.series`). It never gets raw buffers and never owns the
detector. The suggested watch limit in the scaffold is `4 × measured cv`; it is
a starting point derived from the rig's own noise, not a claim about the sample.

## Ship it

1. Push this directory to a public GitHub repository.
2. `noses publish --repo https://github.com/<you>/<repo>` prints the hash and
   the registry entry.
3. Open a PR against `OpenSmell/OpenSmell-web` that adds the entry to
   `public/vials/registry.json`.

A listing is discovery, not a review. The host runs the same gate on every
install, and every UI runs in the same sandbox.

## Two live examples

- `praisejamesx/fermentation-lab` — high risk, stop rules, batch log.
- `praisejamesx/plant-respiration` — medium risk, weekly trend, explicit blind spot.
"#
    )
}

const UI_HTML: &str = r##"<style>
.np { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(240px, .8fr); gap: 12px; padding: 12px; height: 100%; min-height: 480px; box-sizing: border-box; background: var(--bg); color: var(--ink); }
.np * { box-sizing: border-box; }
.np-panel { border: 1px solid var(--line); background: var(--panel); padding: 11px 12px; min-width: 0; }
.np-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
.np-eyebrow { font-family: var(--mono); font-size: 8px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted-2); }
.np h1 { margin: 2px 0 8px; font-size: 18px; line-height: 1.1; letter-spacing: -.02em; font-weight: 650; }
.np-pipe { font-family: var(--mono); font-size: 8px; color: var(--muted); margin-bottom: 10px; }
.np-table { width: 100%; border-collapse: collapse; }
.np-table th { text-align: left; font-family: var(--mono); font-size: 7px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted-2); font-weight: 600; padding: 0 4px 4px; border-bottom: 1px solid var(--line); }
.np-table td { padding: 5px 4px; border-bottom: 1px solid var(--line); font-size: 11px; font-variant-numeric: tabular-nums; }
.np-table td.num { font-family: var(--mono); text-align: right; }
.np-table tr.dead td { color: var(--muted-2); }
.np-input { width: 74px; border: 1px solid var(--line); background: var(--bg); color: var(--ink); padding: 3px 5px; font-family: var(--mono); font-size: 11px; text-align: right; }
.np-input:disabled { opacity: .4; }
.np-actions { display: flex; gap: 6px; margin-top: 9px; flex-wrap: wrap; }
.np button { border: 1px solid var(--line-strong); background: var(--panel); color: var(--ink); padding: 5px 10px; font-family: var(--mono); font-size: 9px; letter-spacing: .06em; text-transform: uppercase; cursor: pointer; }
.np button.primary { background: var(--ink); border-color: var(--ink); color: var(--bg); }
.np button:disabled { opacity: .4; cursor: not-allowed; }
.np-status { margin-top: 8px; font-size: 10px; line-height: 1.45; color: var(--muted); }
.np-status b { color: var(--ink); font-weight: 600; }
.np-right { display: grid; gap: 12px; align-content: start; }
.np-kv { display: flex; justify-content: space-between; gap: 8px; font-size: 10px; padding: 3px 0; border-bottom: 1px solid var(--line); }
.np-kv span:first-child { color: var(--muted); }
.np-kv span:last-child { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.np-alert { border-left: 2px solid var(--red); padding-left: 8px; font-size: 10px; color: var(--ink); }
.np-ok { border-left: 2px solid var(--line-strong); padding-left: 8px; font-size: 10px; color: var(--muted); }
.np-trail { font-family: var(--mono); font-size: 9px; color: var(--muted); max-height: 86px; overflow: hidden; }
</style>

<div class="np">
  <section class="np-panel">
    <div class="np-head">
      <span class="np-eyebrow">Live array · sandbox</span>
      <span class="np-eyebrow" id="np-source">waiting</span>
    </div>
    <h1>__PACK_NAME__</h1>
    <div class="np-pipe" id="np-pipe">transforms: not declared yet</div>
    <table class="np-table">
      <thead><tr><th>Channel</th><th class="num">ΔR/R₀</th><th class="num">cv</th><th class="num">R₀</th><th class="num">Watch limit</th></tr></thead>
      <tbody id="np-rows"><tr><td colspan="5" style="color:var(--muted)">Waiting for host data. Connect a rig or open a recording; this panel reads only what the host sends.</td></tr></tbody>
    </table>
    <div class="np-actions">
      <button class="primary" id="np-arm">Arm watch</button>
      <button id="np-disarm">Disable</button>
    </div>
    <div class="np-status" id="np-status">No watch armed. The limit is in ΔR/R₀, not concentration.</div>
  </section>
  <aside class="np-right">
    <section class="np-panel">
      <div class="np-head"><span class="np-eyebrow">Host verdict</span></div>
      <div id="np-verdict" class="np-ok">No config in force.</div>
      <div class="np-kv"><span>Baseline</span><span id="np-base">—</span></div>
      <div class="np-kv"><span>Anomaly</span><span id="np-anom">—</span></div>
      <div class="np-kv"><span>Channels</span><span id="np-chans">—</span></div>
    </section>
    <section class="np-panel">
      <div class="np-head"><span class="np-eyebrow">Tracked trajectory</span><span class="np-eyebrow" id="np-tr-count">0 pts</span></div>
      <div class="np-trail" id="np-trail">Nothing tracked yet.</div>
    </section>
  </aside>
</div>
<script>
(function () {
  "use strict";
  var LIM = {};
  var ARMED = false;
  var LATEST = null;

  function fmt(v, digits) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    return Number(v).toFixed(digits);
  }
  function limits() {
    var s = LATEST && LATEST.snapshot;
    return (s && s.channelLimits) || {};
  }
  function rowsFromSnapshot() {
    var s = LATEST && LATEST.snapshot;
    var series = LATEST && LATEST.series;
    var resp = series && series.response && series.response.values;
    var body = document.getElementById("np-rows");
    if (!s || !s.channels || !s.channels.length) {
      body.innerHTML = '<tr><td colspan="5" style="color:var(--muted)">Waiting for host data. Connect a rig or open a recording; this panel reads only what the host sends.</td></tr>';
      return;
    }
    var html = "";
    s.channels.forEach(function (ch, i) {
      var lim = (s.channelLimits && s.channelLimits[ch.id]) || null;
      var dead = !lim || lim.dead;
      var last = resp && resp[i] ? resp[i][resp[i].length - 1] : null;
      var key = ch.id;
      var prev = LIM[key];
      LIM[key] = prev;
      html += '<tr class="' + (dead ? "dead" : "") + '">'
        + '<td>' + ch.id + (dead ? " · no limit" : "") + '</td>'
        + '<td class="num">' + fmt(last, 3) + '</td>'
        + '<td class="num">' + (lim ? fmt(lim.cv, 4) : "—") + '</td>'
        + '<td class="num">' + (lim && lim.r0 !== null ? Math.round(lim.r0) : "—") + '</td>'
        + '<td class="num"><input class="np-input" data-ch="' + key + '" type="number" step="0.001" min="0"'
        + (dead ? " disabled" : "") + ' value="' + (prev !== undefined ? prev : (lim && !dead ? (lim.cv * 4).toFixed(3) : "")) + '"></td>'
        + '</tr>';
    });
    body.innerHTML = html;
  }
  function renderVerdict() {
    var s = LATEST && LATEST.snapshot;
    var verdict = document.getElementById("np-verdict");
    var det = s && s.detection;
    var cfg = s && s.detectionConfig;
    if (cfg && det) {
      if (det.triggeredChannels && det.triggeredChannels.length) {
        verdict.className = "np-alert";
        verdict.textContent = "Watch triggered on " + det.triggeredChannels.join(", ") + ". Look at the sample; the array cannot tell you why.";
      } else {
        verdict.className = "np-ok";
        verdict.textContent = "Armed on " + Object.keys(cfg.perChannelLimits || {}).length + " channel(s). No channel over its limit.";
      }
    } else if (cfg) {
      verdict.className = "np-ok";
      verdict.textContent = "Config declared; the host has not evaluated a live window yet.";
    } else {
      verdict.className = "np-ok";
      verdict.textContent = "No config in force.";
    }
    var base = s && s.baseline ? s.baseline.state + " · " + Math.round((s.baseline.progress || 0) * 100) + "%" : "—";
    document.getElementById("np-base").textContent = base;
    var a = s && s.anomaly;
    document.getElementById("np-anom").textContent = a ? (a.state + (a.score !== null ? " · " + fmt(a.score, 2) : "")) : "—";
    var chans = s && s.channels ? s.channels.length : 0;
    var dead = 0;
    var lims = limits();
    for (var k in lims) { if (lims[k] && lims[k].dead) dead++; }
    document.getElementById("np-chans").textContent = chans ? chans + " · " + dead + " dead" : "—";
    var src = s && s.source ? s.source : "waiting";
    document.getElementById("np-source").textContent = src;
    var t = s && s.transformsApplied;
    document.getElementById("np-pipe").textContent = t && t.length
      ? "host transforms: " + t.join(" · ")
      : "transforms: not declared yet";
    var trail = s && s.trajectory ? s.trajectory : [];
    document.getElementById("np-tr-count").textContent = trail.length + " pts";
    if (trail.length) {
      var p = trail[trail.length - 1];
      var parts = [];
      for (var f in p.values) parts.push(f + " " + fmt(p.values[f], 3));
      document.getElementById("np-trail").textContent = new Date(p.at).toLocaleTimeString() + " · " + parts.join(" · ");
    }
  }
  function arm() {
    var inputs = document.querySelectorAll("#np-rows input[data-ch]");
    var cfg = {};
    for (var i = 0; i < inputs.length; i++) {
      var el = inputs[i];
      if (el.disabled) continue;
      LIM[el.getAttribute("data-ch")] = el.value;
      var v = parseFloat(el.value);
      if (isFinite(v) && v > 0) cfg[el.getAttribute("data-ch")] = v;
    }
    if (!Object.keys(cfg).length) {
      document.getElementById("np-status").innerHTML = "<b>No limit to arm.</b> Enter a positive ΔR/R₀ limit on at least one live channel.";
      return;
    }
    ARMED = true;
    window.osmograph.setDetection({ enabled: true, alertThreshold: 0.5, perChannelLimits: cfg });
    document.getElementById("np-status").innerHTML = "<b>Armed on " + Object.keys(cfg).length + " channel(s).</b> Limits are in ΔR/R₀ on this array. The host refuses dead channels and non-live sources.";
  }
  function disarm() {
    ARMED = false;
    window.osmograph.setDetection({ enabled: false, alertThreshold: 0.5, perChannelLimits: {} });
    document.getElementById("np-status").innerHTML = "<b>Watch disabled.</b> No config is in force.";
  }

  document.getElementById("np-arm").addEventListener("click", arm);
  document.getElementById("np-disarm").addEventListener("click", disarm);

  window.osmograph.setTransforms({ baseline: "first-n", response: "delta-over-r0", conditioning: ["median3"] });
  window.osmograph.setTrajectory({
    fields: ["anomaly.score", "anomaly.confidence", "baseline.progress"],
    intervalMs: 60000
  });

  window.osmograph.onData(function (payload) {
    LATEST = payload;
    rowsFromSnapshot();
    renderVerdict();
  });
  window.osmograph.ready();
})();
</script>"##;
