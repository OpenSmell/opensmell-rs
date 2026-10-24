# Build your first Vial

Time to a working Vial: **under a minute.** No hardware, no accounts, no network.

The Vial system has one format: `vial.json` (`osmograph.domain-pack/1`) plus optional
sandboxed HTML fragments beside it. There is no second manifest format and no privileged
tier.

## 1. Scaffold

```
noses new "My First Vial" --publisher yourhandle --domain sensors
cd my-first-vial
```

You get three files: `vial.json`, `ui.html`, `README.md`.

The scaffold is **deliberately incomplete** in two places: `citations` is empty and the text
carries `REPLACE ME` markers. Both are errors in the gate. `noses lint` refuses until you
write real text — a scaffold must never be publishable by accident.

## 2. Write the manifest

Open `vial.json`. The required parts, in the gate's order:

| Field | Rule |
|---|---|
| `id` | lowercase `publisher.slug`, 3–80 chars |
| `version` | semver |
| `risk` | `low` · `medium` · `high` · `critical` |
| `whatItIs` | 30+ chars: the protocol or useful claim |
| `whatItIsNot` | 30+ chars **and** an explicit negation: the claim you refuse to make |
| `citations` | at least one real reference (10+ chars) |
| `docs` | at least two sections with body text |
| `widgets` | `hero`, `risk`, `docs`, `rig`, and at least one of `tracking`, `diagram`, `rules` |

High and critical risk packs additionally need a `rules` widget with at least one
`severity: "stop"` rule, and three `docs` sections.

The `risk` widget needs both `always` and `never` lists. The gate also scans all manifest
text for unnegated overclaims — “safe to eat”, pathogen detection, diagnosis, ethylene
detection, absolute concentration/ppm, cross-rig, works-on-any-rig, guarantees, medical
claims, safety-device claims. Negation is read per sentence: “It cannot diagnose a plant”
passes; “It diagnoses a plant” does not.

## 3. Write the UI (optional)

`ui.html` is a **body fragment**: no `<!doctype>`, no `<html>`, no `<head>`, no `<body>`.
The host wraps it in a sandboxed iframe with its own base CSS and the bridge, so start with
your markup and a `<style>` block.

The bridge is the only way in and out:

```js
window.osmograph.onData(function (payload) { /* payload.snapshot, payload.series */ });
window.osmograph.setTransforms({ baseline: "first-n", response: "delta-over-r0", conditioning: ["median3"] });
window.osmograph.setDetection({ enabled: true, alertThreshold: 0.5, perChannelLimits: { ch0: 0.024 } });
window.osmograph.setTrajectory({ fields: ["anomaly.score", "baseline.progress"], intervalMs: 60000 });
window.osmograph.ready();
```

What the host sends:

- `payload.snapshot.channels` — `{ id, label, kind, unit, health }`
- `payload.snapshot.channelLimits` — per channel `{ dead, cv, min, max, r0 }`, measured from
  the live stream. A dead channel has no measurable variation; the host refuses it.
- `payload.snapshot.detectionConfig` / `.detection` — the config in force and the host's
  verdict `{ active, triggeredChannels }`
- `payload.snapshot.baseline` — `{ state, progress, ageSec }`
- `payload.snapshot.trajectory` — the points the host persisted for you
- `payload.series` — the bounded window; `series.response.values` is your declared
  transform, already computed by the host

Supported transforms: baseline `first-n` · `session-mean`; response `delta-over-r0` ·
`relative` · `delta`; conditioning `median3`. Anything else is refused.

The single-unit rule: a limit is in transformed units on **this array**. The suggested
starting point in the scaffold is `4 × measured cv` — the rig's own noise floor — not a
claim about the sample. Never hard-code a limit that came from a different rig.

What the sandbox refuses at load: external scripts/styles/frames, remote images, `fetch`,
`XMLHttpRequest`, `WebSocket`, `localStorage`, `eval`, `javascript:` URLs, and any
`osmograph.*` method that is not on the bridge. `noses lint` checks all of it before the
app does.

## 4. Validate

```
noses lint      # every check, with the reason
noses test      # lint + what your UI actually declares (transforms, trajectory, watch)
```

`noses test` is static: it proves the contract, not the science. The live proof is the app:
install the Vial, connect a rig, arm the watch, and read the host's own report of the
transforms it applied and the trajectory it persisted.

## 5. Ship it

1. Push the directory to a public GitHub repository. `vial.json` must be at the repo root,
   beside `ui.html` / `home.html`.
2. ```
   noses publish --repo https://github.com/you/your-vial [--branch main] [--evidence B]
   ```
   It prints `sha256(vial.json)` and the exact registry entry.
3. Open a PR against `OpenSmell/OpenSmell-web` adding that entry to
   `public/vials/registry.json`.
4. Install it in Osmograph → Vials → Aromatory by pasting the manifest URL, or use the
   registry listing.

A listing is discovery, not a review. Every install re-runs the gate, and every UI runs in
the same sandbox.

## Using an AI assistant

Point it at [`llms.txt`](../llms.txt) and `src/vial/pack.rs`, and have it run `noses lint`
in a loop until clean. The gate's error messages are written to be the instructions.
