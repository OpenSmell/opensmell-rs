# Example Vials (the gallery)

Official, validated Vials. Each is a real, honest example and a live test — the
`example_vials_are_valid_and_fixtures_match` unit test runs every one so the gallery
can't rot.

| Vial | Claim | Why it's honest |
|---|---|---|
| [`beer-fermentation`](./beer-fermentation) | fermentation-change | reports *relative* change, never absolute CO2/VOC |
| [`cold-chain`](./cold-chain) | spoilage-risk | sealed storage only; change vs baseline, not species ID |
| [`leak-alarm`](./leak-alarm) | leak-alarm | one dedicated target; not a certified toxic-gas detector |

Run any of them without hardware:

```
noses test examples/beer-fermentation
noses lint examples/beer-fermentation
```

Add your own: `noses create "My Vial"` then submit a PR — the same test will validate it.
