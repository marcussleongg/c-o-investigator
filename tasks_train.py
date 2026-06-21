"""Training task list — 180 cases (40 neg, 60 1-hop, 45 2-hop, 35 3-hop).

Curriculum-weighted toward 1-hop (the foundational skill the agent must learn first:
find a connection and submit a non-empty path) while keeping enough 2-/3-hop for
multi-hop competence. Used by the training loop. Never use this for evaluation — see
tasks_test.py.
"""

import json
import logging
from pathlib import Path

from env import conflict_of_interest, env  # noqa: F401

logger = logging.getLogger("coi-investigator")

with open(Path(__file__).parent / "cases_train.json") as f:
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

    _base = f"train-{_slugify(_case['person_a'])}-{_slugify(_case['person_b'])}"
    if _base in _seen:
        _seen[_base] += 1
        _t.slug = f"{_base}-{_seen[_base]}"
    else:
        _seen[_base] = 0
        _t.slug = _base
    tasks.append(_t)

logger.info("Training tasks registered: %d", len(tasks))
