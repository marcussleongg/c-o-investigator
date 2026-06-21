# Dataset Hop Distribution

Hop count = `(len(path) - 1) // 2` (1-hop = one shared company; neg = no connection).
Generated 2026-06-21.

| Task set | File | Total | neg | 1-hop | 2-hop | 3-hop |
| -------- | ---- | ----- | --- | ----- | ----- | ----- |
| Train | `cases_train.json` | 140 | 45 | 80 | 15 | 0 |
| Test (held-out) | `cases_test.json` | 115 | 40 | 35 | 25 | 15 |
| Test — full dist (archived) | `cases_test_full.json` | 184 | 95 | 8 | 14 | 67 |
| Calib | `cases_calib.json` | 35 | 10 | 5 | 10 | 10 |

Regenerate the test set with `python build_test_set.py` (seeded, leakage-checked
against train + calib). The COI strength of a path decays with distance: 1-hop is a
direct interlocking directorate (a real COI), 2-hop is an indirect tie via one
shared associate, 3-hop is distant network proximity (not really a COI).

## Notes

- **Train** is the deadline mix: weighted to the fast + learnable tiers (negatives
  abstain in ~25s with no enrich; 1-hop is the gradient tier). 3-hop dropped — slow
  (minute-scale sixtyfour calls) and 0% on the base model, so it only yields
  errored / zero-gradient rollouts. Also conceptually justified: 3-hop paths aren't
  real conflicts of interest, so training on them would reinforce chasing distant links.
- **Test** is rebuilt toward the headline tiers (1-/2-hop = real/indirect COI) plus
  negatives, with a small 3-hop probe kept for the generalization story (does a
  policy trained on 1-/2-hop transfer to deeper traversal?). The graph supplies
  ~22.5k disjoint 1-hop and thousands of 2-hop pairs, so the mix is supply-unconstrained.
  **Always report eval results stratified by hop** — a single blended number is
  dominated by the tier where neither base nor trained model performs.
- **Test (full dist, archived)** is the original natural-distribution set (~75% 3-hop
  positives), kept in `cases_test_full.json` for reference / a generalization-heavy eval.
- **Calib** is the stratified sanity set (~even across tiers).
