"""DEMO tasks — generalized COI, including people who may not be in SEC filings.

This is for *inference / demos* (watch the trained policy traverse), NOT a measured
eval — it uses the coi_general template, which has no fixed ground truth and grades
only the SEC-verifiable portion of whatever the agent submits. For real before/after
numbers use tasks_eval_small.py with conflict_of_interest.

Run (hud eval is just how HUD executes the agent — ignore the score, watch the trace):

    uv run hud eval tasks_demo.py coi-clean2 --gateway -y          # first task only (default)
    uv run hud eval tasks_demo.py coi-clean2 --gateway --full -y   # all demo tasks
    uv run hud eval tasks_demo.py coi-clean2 --gateway --task-ids demo-0 -y   # a specific one

Swap the entries below for real names + their companies. company_a/company_b are
optional but help: they seed the search and disambiguate common names.
"""

import logging

from env import coi_general, env  # noqa: F401  (re-exports env for `hud eval`)

logger = logging.getLogger("coi-investigator")

# (person_a, company_a, person_b, company_b) — edit these. Leave a company "" if unknown.
# Idea: a private/startup person (not in SEC graph) -> someone who IS in SEC filings,
# so the agent has to traverse out of the private world into public-company records.
_DEMOS: list[tuple[str, str, str, str]] = [
    ("Christopher Price", "Sixtyfour AI", "Elon Musk", ""),
    ("Jay Ram", "HUD (YCW25)", "Michael Truell", "Cursor"),
    ("Elon Musk", "", "Joe Gebbia", ""),
]

tasks = []
for _i, (_pa, _ca, _pb, _cb) in enumerate(_DEMOS):
    _t = coi_general(person_a=_pa, person_b=_pb, company_a=_ca, company_b=_cb)
    _t.slug = f"demo-{_i}"
    tasks.append(_t)

logger.info("Demo tasks registered: %d", len(tasks))
