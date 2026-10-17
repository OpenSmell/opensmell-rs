# The Vial toolchain — `noses`

One manifest, three tiers. Build an Osmograph **Vial** in under a minute, test it
without hardware, validate it, and publish it.

`noses` is not a separate SDK — it is a **binary of the core `opensmell` crate**
(`src/bin/noses.rs`), so the manifest rules, recipe engine and templates live in
exactly one place, shared with the engine.

```
noses create <name> [--template recipe|wasm]   # scaffold a working Vial
noses test  [path]                             # run fixtures — no hardware needed
noses lint  [path]                             # schema + permission + licence checks
noses dev   [path]                             # load + hot-reload in Osmograph
noses publish [path]                           # validate, then OIDC trusted publish
```

## Tiers
| Tier | For | Tech | Isolation |
|---|---|---|---|
| **recipe** (default) | domain experts, AI-assisted coders | declarative `recipe.json` | hardened host sandbox |
| **wasm** | expert coders needing real power/egress | Rust/TS via one WIT contract | memory-isolated, capability-based |
| **sidecar** | scientists with existing Python | Tauri sidecar | process-isolated |

## The manifest is the source of truth
A Vial is a single `vial.toml`, validated strictly by `noses lint`:
- `claims.evidence` is **mandatory** whenever `claims.provides` is non-empty — we never
  ship a claim without a source.
- Paid Vials must be **Sealed** (`pricing.sealed = true`); no obfuscated DRM.
- Capability namespaces are **closed**; `capabilities.network` allows exact hostnames only.

Schema: [`schema/vial.schema.json`](./schema/vial.schema.json).

## Run it (no separate install)
From the crate root (`opensmell-rs/`):
```
cargo run --bin noses -- create "My Vial"
cargo run --bin noses -- lint  path/to/vial
cargo test --lib vial::    # 10 Vial unit tests: validator + recipe engine + scaffold + gallery
```

## Examples (the gallery)
Three Official Vials under [`examples/`](./examples) — beer-fermentation, cold-chain,
leak-alarm. Each is validated by a unit test so the gallery can't rot.

## Docs
- [Build your first Vial](./docs/BUILD_YOUR_FIRST_VIAL.md)
- [Example gallery](./examples/README.md)
- [llms.txt](./llms.txt)
