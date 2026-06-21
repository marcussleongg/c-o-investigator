"""Held-out test set — 184 cases never seen during training.

Run this against the base model BEFORE training to get the baseline,
then against the trained model AFTER training to measure improvement.

    # Baseline (run before training)
    hud eval tasks_test.py coi-investigator -y

    # Post-training (run after training completes)
    hud eval tasks_test.py coi-investigator -y
"""

import json
import logging
from pathlib import Path

from env import conflict_of_interest, env  # noqa: F401

logger = logging.getLogger("coi-investigator")

with open(Path(__file__).parent / "cases_test.json") as f:
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

    _base = f"test-{_slugify(_case['person_a'])}-{_slugify(_case['person_b'])}"
    if _base in _seen:
        _seen[_base] += 1
        _t.slug = f"{_base}-{_seen[_base]}"
    else:
        _seen[_base] = 0
        _t.slug = _base
    tasks.append(_t)

logger.info("Test tasks registered: %d", len(tasks))
