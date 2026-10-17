# Build your first Vial

Time to a working Vial: **under a minute.** No hardware, no accounts, no network.

## 1. Scaffold
```
noses create "Beer Ferment Monitor" --template recipe
cd beer-ferment-monitor
```
You now have a **valid** Vial: `vial.toml`, `recipe.json`, a fixture recording, and a
`README.md` full of TODOs.

## 2. Change one thing and see it
Open `recipe.json` and add a step, e.g. a channel's standard deviation:
```json
{ "op": "std", "channel": "ch0", "as": "ch0_std" }
```
Run the loop:
```
noses test
```
It evaluates the recipe against `fixtures/sample.csv` and compares to
`fixtures/expected.json`. No hardware needed — this is the whole point.

## 3. Make it yours
- Set your `@handle` and the Vial name in `vial.toml`.
- Under `[claims]`, say what decision the Vial makes, then **cite evidence** (a DOI or a
  public benchmark). `noses lint` will refuse to pass without it — by design.
- Replace `fixtures/sample.csv` with a recording of your own process.

## 4. Validate
```
noses lint
```
Rules enforced: reverse-DNS id, semver, known tiers/namespaces, closed capabilities,
mandatory evidence for claims, Sealed-paid rule.

## 5. Load it
```
noses dev
```
Then open Osmograph Desktop → **System → Plugins**. Edit files; it hot-reloads.

## 6. Publish
```
noses publish
```
Validation first, then OIDC trusted publishing (no tokens in the repo) opens a store PR
with a signed artifact and provenance.

## Using an AI assistant
Point it at `llms.txt` and the Vial schema, then have it call `noses lint` in a loop until
clean. Declarative recipes are what LLMs emit most reliably — that's why recipe is the
default tier.
