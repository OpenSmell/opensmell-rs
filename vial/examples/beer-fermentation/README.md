# Beer Fermentation Monitor

Watches the headspace of a fermentation vessel and reports **relative chemical change**
against the start of the run.

- **Tier:** recipe (declarative, sandboxed).
- **Claim:** `fermentation-change` — a change vs baseline, with evidence.
- **Limits:** MOX arrays of ≥2 channels; reports *relative* change, never absolute CO2/VOC.

## Why it's honest
Fermentation is slow-changing, and MOX sensors drift — so this Vial reports a *change*
signal (mean level and span per channel), not a concentration. That is the Tier-A claim
the physics actually supports.

## Dev
```
noses test        # run against fixtures/sample.csv (no hardware)
noses lint        # validate
```
