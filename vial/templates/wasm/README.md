# __NAME__

A tier-2 (**WASM component**) Vial for Osmograph. Use this when a Recipe isn't
enough and you need real code — it runs memory-isolated with capability-based access.

## Develop
1. Implement the exports in `src/lib.rs` against `wit/vial.wit`.
2. `noses dev` to load and hot-reload in Osmograph Desktop.
3. `noses lint`, then `noses publish`.

## TODO
- [ ] Set your `@handle` and Vial name in `vial.toml`.
- [ ] Keep `[capabilities]` minimal; network must be exact hostnames.
- [ ] Cite evidence for anything under `[claims].provides`.
