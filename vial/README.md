# The Vial toolchain — `noses`

One format. Build an Osmograph **Vial** without hardware, validate it against the same
gate the app runs, and print the exact registry entry to ship it.

`noses` is a binary of the core `opensmell` crate (`src/bin/noses.rs`). The validator in
`src/vial/pack.rs` mirrors the host gate in the desktop app (`src/domainPacks.ts`); the
desktop is the authority and re-runs that gate on every install. When the two disagree,
the host wins.

```
noses new <Name> [--publisher <handle>] [--domain <domain>]  scaffold a real Vial
noses lint [path]                                            the policy gate, locally
noses test [path]                                            lint + what the UI declares
noses publish [path] --repo <github-url> [--branch main]     hash + registry entry
```

## What a Vial is

| File | Role |
|---|---|
| `vial.json` | The manifest: `osmograph.domain-pack/1`. Text, widgets, claims, risk framing. |
| `ui.html` | Optional sandboxed body fragment: the workspace surface. |
| `home.html` | Optional sandboxed body fragment: the compact Home takeover. |

The UI runs inside an iframe with **no network, no storage, no files, no Tauri**. It
receives a typed snapshot (`payload.snapshot`) and a bounded series window
(`payload.series`) and can only call the host bridge: `onData`, `setTransforms`,
`setDetection`, `setTrajectory`, `toast`, `action`, `ready`, `version`.

The host owns sampling, the baseline, anomaly detection, the hardware-sufficiency gate,
the single-unit channel limits (`cv`, `min`, `max`, `r0`, `dead`), and the trajectory it
persists. A Vial **declares** what it wants; the host **enforces** what the rig supports.

## What the gate refuses

- A pack without `whatItIsNot` naming the claim it refuses to make.
- A pack without at least one real citation.
- Unnegated overclaims: “safe to eat”, pathogen detection, diagnosis, ethylene detection,
  absolute concentration/ppm, cross-rig, works-on-any-rig, guarantees, medical claims,
  safety-device claims.
- Executable content in the manifest (the manifest stays declarative).
- A UI that is a full document, loads remote assets, uses the network or storage, or
  calls a bridge method that does not exist.
- High and critical risk packs without an explicit stop rule and three field-note sections.
- Placeholder text (`REPLACE ME`) — a scaffold must never be publishable by accident.

## Status

`new`, `lint`, `test`, and `publish` are implemented and tested. `publish` does **not**
sign, upload, or open a PR: it prints the sha256 and the exact registry entry for a manual
PR. There is no hosted registry service; discovery is a JSON file in
`OpenSmell/OpenSmell-web`. `noses test` is static — it reads the manifest and the UI; the
live proof is the app, where the host reports the transforms it applied and the watch it
armed.

## Run it (no separate install)

From the crate root (`opensmell-rs/`):
```
cargo run --bin noses -- new "My First Vial" --publisher yourhandle --domain sensors
cargo run --bin noses -- lint  path/to/vial
cargo test --lib vial::    # the gate's own tests
```

## Example

[`examples/minimal`](./examples/minimal) is a complete, passing Vial: a sample log, a live
ΔR/R₀ panel that declares host transforms and a tracked trajectory, and an explicit limit
on what the array can claim. It is validated by a unit test so it cannot rot.

The two live packs are author-hosted, not bundled here:
`praisejamesx/fermentation-lab` and `praisejamesx/plant-respiration`.

## Docs

- [Build your first Vial](./docs/BUILD_YOUR_FIRST_VIAL.md)
- [Example gallery](./examples/README.md)
- [llms.txt](./llms.txt)
