# Vial examples

One example lives here, and it is validated by a unit test so it cannot rot:

- [`minimal/`](./minimal) — the smallest complete Vial. A sample log, a live ΔR/R₀ panel
  that declares host transforms and a tracked trajectory, and an explicit statement of what
  the array cannot claim.

The two live packs are author-hosted repositories, not bundled examples:

- [`praisejamesx/fermentation-lab`](https://github.com/praisejamesx/fermentation-lab) —
  high risk, stop rules, batch log, single-unit watch.
- [`praisejamesx/plant-respiration`](https://github.com/praisejamesx/plant-respiration) —
  medium risk, weekly trend, explicit blind spot (no ethylene channel).

Run the gate against any of them:

```
cargo run --bin noses -- lint   vial/examples/minimal
cargo run --bin noses -- test   vial/examples/minimal
cargo run --bin noses -- lint   ~/code/osmograph-vials/plant-respiration
```
