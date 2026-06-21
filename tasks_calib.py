"""Calibration task list — 35 stratified cases (10 neg, 5 1-hop, 10 2-hop, 10 3-hop).

Run against the forked model before training to verify reward spread is 20-50%.

    hud eval tasks_calib.py coi-investigator -y --runtime local
"""

import json
import logging
from pathlib import Path

from env import conflict_of_interest, env  # noqa: F401

logger = logging.getLogger("coi-investigator")

_CALIB_CASES = Path(__file__).parent / "cases_calib.json"

with open(_CALIB_CASES) as f:
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

    _base = f"calib-{_slugify(_case['person_a'])}-{_slugify(_case['person_b'])}"
    if _base in _seen:
        _seen[_base] += 1
        _t.slug = f"{_base}-{_seen[_base]}"
    else:
        _seen[_base] = 0
        _t.slug = _base
    tasks.append(_t)

logger.info("Calibration tasks registered: %d", len(tasks))
