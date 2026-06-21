"""Minimal tinker health check: one fast rollout batch + one train round-trip.

Runs a single negative COI task (the agent abstains quickly, ~25s) at group 2,
then does one forward_backward + optim_step. Exercises the same tinker-backed
paths as real training — inference for rollouts and the train round-trip — but in
~1-2 min instead of the full smoke test's ~12 min. Use it to check whether the
intermittent tinker error is still occurring.

    uv run check_tinker.py

Reports rollout failures and whether the train step raised, then PASS / FAIL.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from hud import TrainingClient
from hud.agents import create_agent
from hud.eval import Job, LocalRuntime, Taskset

from env import conflict_of_interest

MODEL = "coi-investigator"


async def main() -> None:
    # Pick the first negative case — abstention rollouts are the fastest tier.
    cases = json.loads((Path(__file__).parent / "cases_train.json").read_text())
    neg = next(c for c in cases if not c["label"])
    task = conflict_of_interest(
        person_a=neg["person_a"],
        person_b=neg["person_b"],
        label=False,
        ground_truth_path=[],
        ground_truth_citations=[],
    )

    taskset = Taskset("tinker-check", [task])
    agent = create_agent(
        MODEL,
        max_steps=8,
        completion_kwargs={"max_tokens": 1024, "extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(MODEL)

    session = await Job.start("tinker-check", group=2)
    await taskset.run(
        agent,
        runtime=LocalRuntime("env.py"),
        job=session,
        max_concurrent=2,
        rollout_timeout=300.0,
    )

    batch = session.runs
    failed = [r for r in batch if r.trace.status == "error"]
    print(f"\nrollouts: {len(batch)} | failed: {len(failed)}")
    for r in failed:
        err = getattr(r.trace, "error", None) or getattr(r, "error", None) or r.trace.status
        print(f"  ROLLOUT ERROR: {err}")

    try:
        fb = await trainer.forward_backward(batch, loss_fn="importance_sampling", group_size=2)
        result = await trainer.optim_step(learning_rate=1e-5)
        print(f"train round-trip OK: optim step {result.step}, datums {fb.num_datums}")
        if failed:
            print("\nTINKER CHECK: rollout error(s) present — see above")
        else:
            print("\nTINKER CHECK: PASS — no tinker errors")
    except Exception as e:
        print(f"  TRAIN ERROR: {type(e).__name__}: {e}")
        print("\nTINKER CHECK: FAIL — train round-trip raised (likely the tinker error)")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
