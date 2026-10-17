# __NAME__

A tier-1 (**Recipe**) Vial for Osmograph — declarative, sandboxed, no code.

## Develop
- Edit `recipe.json` to change the pipeline.
- `noses test` evaluates the recipe against `fixtures/sample.csv` and compares the
  result with `fixtures/expected.json`. **No hardware needed.**
- `noses lint` before `noses publish`.

## Recipe ops
`mean` · `min` · `max` · `range` · `first` · `last` · `std` · `delta(a - b)`

## TODO
- [ ] Rename the Vial and set your `@handle` in `vial.toml`.
- [ ] Describe what decision it makes under `[claims]`, and cite real evidence.
- [ ] Replace `fixtures/sample.csv` with a recording of your own process.
