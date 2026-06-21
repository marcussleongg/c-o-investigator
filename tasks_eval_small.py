"""Small stratified held-out eval — 20 cases (6 neg, 8 1-hop, 4 2-hop, 2 3-hop).

A fast before/after probe sampled from cases_test.json (held out from training).
Run the SAME set against base and trained model, report stratified by hop:

    # before (base Qwen3 8B)
    hud eval tasks_eval_small.py 22b93b24-e8e6-4864-8083-0a2a8b987c88 \\
        --gateway --group-size 1 --max-concurrent 20 -y
    # after (trained)
    hud eval tasks_eval_small.py coi-clean2 --gateway --group-size 1 --max-concurrent 20 -y
"""

import json
import logging
from pathlib import Path

from env import conflict_of_interest, env  # noqa: F401

logger = logging.getLogger("coi-investigator")

with open(Path(__file__).parent / "cases_eval_small.json") as f:
    cases = json.load(f)

tasks = []
_seen: dict[str, int] = {}
for _case in cases:
    _t = conflict_of_interest(
        person_a=_case["person_a"],
        person_b=_case["person_b"],
        label=_case["label"],
        ground_truth_path=_case.get("path") or [],
        ground_truth_citations=_case.get("citations") or [],
    )

    def _slugify(name: str) -> str:
        return name.lower().replace(" ", "-").replace(".", "").replace(",", "")

    _base = f"evalsm-{_slugify(_case['person_a'])}-{_slugify(_case['person_b'])}"
    if _base in _seen:
        _seen[_base] += 1
        _t.slug = f"{_base}-{_seen[_base]}"
    else:
        _seen[_base] = 0
        _t.slug = _base
    tasks.append(_t)

logger.info("Small eval tasks registered: %d", len(tasks))
