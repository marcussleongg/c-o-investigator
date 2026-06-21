"""COI Investigator task list.

Loads cases from data/cases.json (the real graph output from build_graph.py).
Falls back to cases_stub.json for smoke-testing while the graph is still building.

    hud eval tasks.py claude --task-ids coi-jane-smith-robert-chen -y --runtime local
"""

import json
import logging
from pathlib import Path

from env import conflict_of_interest, env  # noqa: F401  (re-exports env for `hud eval tasks.py`)

logger = logging.getLogger("coi-investigator")

_REAL_CASES = Path(__file__).parent / "data" / "cases.json"
_STUB_CASES = Path(__file__).parent / "cases_stub.json"


def _load_cases() -> list[dict]:
    """
    Load COI test cases from disk, preferring the real graph output over the stub.

    Returns:
        list[dict] - Each case has person_a, person_b, label, path, citations.

    Examples:
        cases = _load_cases()
        len(cases)
        # 400  (or 3 if falling back to stub)
    """
    if _REAL_CASES.exists():
        with open(_REAL_CASES) as f:
            cases = json.load(f)
        logger.info("Loaded %d cases from %s", len(cases), _REAL_CASES)
        return cases

    logger.warning(
        "Real cases not found at %s — falling back to cases_stub.json for smoke-testing", _REAL_CASES
    )
    with open(_STUB_CASES) as f:
        cases = json.load(f)
    logger.info("Loaded %d stub cases", len(cases))
    return cases


def _make_slug(person_a: str, person_b: str) -> str:
    """
    Generate a URL-safe task slug from two person names.

    Parameters:
        person_a : str - First person's name.
        person_b : str - Second person's name.

    Returns:
        str - Slug like 'coi-jane-smith-robert-chen'.

    Examples:
        _make_slug("Jane Smith", "Robert Chen")
        # 'coi-jane-smith-robert-chen'
    """
    def _slugify(name: str) -> str:
        return name.lower().replace(" ", "-").replace(".", "").replace(",", "")

    return f"coi-{_slugify(person_a)}-{_slugify(person_b)}"


cases = _load_cases()

tasks = []
_seen_slugs: dict[str, int] = {}
for _case in cases:
    _t = conflict_of_interest(
        person_a=_case["person_a"],
        person_b=_case["person_b"],
        label=_case["label"],
        ground_truth_path=_case.get("path") or [],
        ground_truth_citations=_case.get("citations") or [],
    )
    _base_slug = _make_slug(_case["person_a"], _case["person_b"])
    if _base_slug in _seen_slugs:
        _seen_slugs[_base_slug] += 1
        _t.slug = f"{_base_slug}-{_seen_slugs[_base_slug]}"
    else:
        _seen_slugs[_base_slug] = 0
        _t.slug = _base_slug
    tasks.append(_t)

logger.info("Registered %d tasks", len(tasks))
