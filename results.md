# COI Investigator — Eval Results

## Configuration

| Field | Value |
| ----- | ----- |
| Base model | **Qwen3 8B (Tinker)** (base id `22b93b24-...`). Training fork: **`coi-clean`** (id `abd3da86-6f7c-4644-b95a-448c364790e7`), pristine from base. (Earlier `coi-investigator` fork drifted 3 micro optim steps from smoke/check/trial runs — abandoned for clean before/after.) |
| Training set | `cases_train.json` — 140 cases (45 neg, 80 1-hop, 15 2-hop, 0 3-hop) — deadline mix: fast + learnable tiers; 3-hop dropped (slow + 0% base) |
| Test set | `cases_test.json` — 184 held-out cases |
| Calib set | `cases_calib.json` — 35 stratified cases |
| Grader | deterministic per-edge (edge_correctness 0.5, endpoints 0.2, citation_coverage 0.3) |
| Eval config | max_steps 50, group_size 5, max_concurrent 3 |

### Key config changes (2026-06-20)
- Rebalanced `cases_train.json` from 201 → 180, weighted toward 1-hop (curriculum foundation) while keeping 2-/3-hop for multi-hop competence.
- `enrich_person` / `enrich_company` structs now request `list[str]` for multi-entity fields (board_seats, prior_companies, board_members, key_executives, founders, sources) so the agent gets parsed entity lists instead of prose.
- Prompt now pushes submitting a **best partial path** over an empty one (empty earns zero; correct edges earn per-edge credit).

## Calibration runs (tasks_calib.py, 35 tasks × 5 rollouts)

| Date       | Rollouts completed | Successful | Failed | Success % | Notes                                           |
| ---------- | ------------------ | ---------- | ------ | --------- | ----------------------------------------------- |
| 2026-06-20 | 51 / 175           | 15         | 36     | 29%       | Run stopped early; all failures called submit() |

### Pre-training behavior probes (base Qwen3-7B, n=1 each)
Multi-hop is at flat 0.0 — the base model gives up rather than submitting partial paths:

| Task | Tier | Behavior | Reward |
| ---- | ---- | -------- | ------ |
| Marcela Kirberger ↔ Mark Blinn | 3-hop | explored, then `submit []` ×9 | 0.0 |
| Randall J. Lewis ↔ Ann C. Hoff (run a) | 2-hop | bailed after 2 steps, raw `<think>` dump, no submit | 0.0 |
| Randall J. Lewis ↔ Ann C. Hoff (run b) | 2-hop | enriched 2 endpoints only, then `submit []` ×4 | 0.0 |

Takeaway: base model doesn't explore deep enough to find a partial trail, so the prompt fix has nothing to rescue. This is the gap training should close — watch whether 1-hop reward climbs first and multi-hop follows.

## Training runs (tasks_train.py)

| Date | Rollouts | Successful | Failed | Success % | Notes |
| ---- | -------- | ---------- | ------ | --------- | ----- |
| 2026-06-20 | 8 (4 tasks × group 2) | 4 | 4 (1 tinker error) | 37.5% | Local smoke test of `train.py` — loop wiring confirmed. reward 0.375, loss +0.0000 (group=2 too small for within-group variance; expected). 713s rollout local → go remote for real run. |

## Test runs (tasks_test.py, 184 held-out cases)

| Date | Rollouts | Successful | Failed | Success % | Notes |
| ---- | -------- | ---------- | ------ | --------- | ----- |
