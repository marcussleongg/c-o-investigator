"""On-policy GRPO RL for COI Investigator: roll out the taskset, train, repeat.

Adapted from hud-python cookbooks/rl-training/simple_train.py. The loop is the
same five lines: run the taskset under one long-lived job, take the batch of fresh
runs, and hand them to the trainer. ``forward_backward`` does one server-side loss
(importance sampling — GRPO) followed by ``optim_step``, which checkpoints and
promotes the new weights behind the *same* model string, so the next rollout
samples the updated policy.

COI-specific vs the cookbook:
  - MODEL is the Qwen3-7B fork ``coi-investigator`` (set with `hud models`).
  - Each rollout is a multi-step agentic investigation (enrich / search / sec_search
    then submit), so we set ``max_steps`` and a generous ``max_tokens`` for the
    reasoning + tool calls, plus a per-rollout timeout so a slow rollout can't
    stall the batch.

Smoke-test the wiring locally first (cheap), then flip to remote for the real run:
    uv run train.py --steps 1 --group 2 --max-concurrent 2   # local (TASKSET="")
    # then set TASKSET in common.py to your deployed taskset and re-run:
    uv run train.py --steps 20 --group 5 --max-concurrent 8
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

from dotenv import load_dotenv

from common import load_taskset_and_runtime
from hud import TrainingClient
from hud.agents import create_agent
from hud.agents.types import AgentStep
from hud.eval import Job, Taskset

# The trainable gateway model to sample from and train, in place.
# coi-clean2: fresh Qwen3 8B (Tinker) fork — pristine base, no drift. (coi-clean
# was degraded by a run at lr=1e-5 that eroded reward 0.66 -> 0.46; abandoned.
# Restarting here with a lower learning rate.)
MODEL = "coi-clean2"


def _output_tokens(runs: list) -> int:
    """
    Total generated tokens across a batch of runs (a throughput numerator).

    Parameters:
        runs : list - A batch of Run objects from the training job.

    Returns:
        int - Sum of output token ids across every agent-step Sample in the batch.

    Examples:
        _output_tokens(session.runs[batch_start:])
        # 18342
    """
    return sum(
        len(sample.output_token_ids)
        for run in runs
        for sample in run.trace.collect(
            lambda s: s.sample if isinstance(s, AgentStep) and s.sample else None
        )
    )


async def main(
    *,
    steps: int,
    group: int,
    max_steps: int,
    max_tasks: int | None,
    learning_rate: float,
    max_concurrent: int,
    rollout_timeout: float,
) -> None:
    """
    Run the GRPO training loop: rollout the COI taskset, train on the batch, repeat.

    Parameters:
        steps           : int   - Number of training iterations (rollout + optim).
        group           : int   - Rollouts per task forming one GRPO group.
        max_steps       : int   - Max agent tool-call turns per rollout.
        learning_rate   : float - Optimizer step size.
        max_concurrent  : int   - Cap on simultaneous rollouts.
        rollout_timeout : float - Per-rollout wall-clock cap in seconds.

    Returns:
        None - Prints per-step reward, solve rate, throughput, and loss.

    Examples:
        asyncio.run(main(steps=20, group=5, max_steps=30, learning_rate=1e-5,
                         max_concurrent=8, rollout_timeout=600.0))
    """
    model = MODEL

    # return_token_ids tells the gateway this is a training rollout: each turn's
    # response carries token ids + per-token logprobs, recorded on the trace Sample
    # — the token-level data TrainingClient trains on. max_steps caps the agent's
    # tool loop; max_tokens leaves room for the reasoning model's CoT + a tool call
    # each turn.
    agent = create_agent(
        model,
        max_steps=max_steps,
        completion_kwargs={"max_tokens": 4096, "extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(model)
    # A deployed taskset on remote HUD boxes (TASKSET in common.py), or a local slice.
    taskset, runtime = load_taskset_and_runtime()
    all_tasks = list(taskset.tasks.values())  # .tasks is a dict keyed by slug
    if max_tasks is not None:
        n = min(max_tasks, len(all_tasks))
        print(f"sampling {n} of {len(all_tasks)} tasks/step (~{n * group} rollouts/step)", flush=True)

    # One job spans the whole session; each iteration appends its batch of runs.
    session = await Job.start("coi-rl", group=group)
    for step in range(steps):
        # Fresh random batch each step so training covers the whole taskset over
        # many steps, instead of the same fixed slice every time.
        if max_tasks is not None and max_tasks < len(all_tasks):
            step_tasks = random.sample(all_tasks, max_tasks)
        else:
            step_tasks = all_tasks
        step_taskset = Taskset(taskset.name, step_tasks)

        batch_start = len(session.runs)

        # --- rollout phase (sampling throughput) ---
        t0 = time.perf_counter()
        await step_taskset.run(
            agent,
            runtime=runtime,
            job=session,
            max_concurrent=max_concurrent,
            rollout_timeout=rollout_timeout,  # a slow rollout can't stall the batch
        )
        rollout_s = time.perf_counter() - t0
        batch = session.runs[batch_start:]
        tokens = _output_tokens(batch)
        failed = sum(1 for run in batch if run.trace.status == "error")

        # Drop any GRPO group containing an errored run: that run has reward 0 and
        # ~no tokens, which skews its group's advantage baseline. Groups are
        # contiguous chunks of `group`; dropping whole chunks keeps divisibility.
        clean: list = []
        dropped = 0
        for i in range(0, len(batch), group):
            grp = batch[i : i + group]
            if any(r.trace.status == "error" for r in grp):
                dropped += 1
                continue
            clean.extend(grp)

        if not clean:
            print(
                f"step {step:2d} | all {len(batch) // group} groups errored — skipping train",
                flush=True,
            )
            continue

        # --- train phase (forward_backward + optim_step) ---
        t1 = time.perf_counter()
        fb = await trainer.forward_backward(
            clean,
            loss_fn="importance_sampling",
            group_size=group,  # each task's `group` repeats form one GRPO group
        )
        result = await trainer.optim_step(learning_rate=learning_rate)
        train_s = time.perf_counter() - t1

        mean_reward = sum(run.reward for run in clean) / len(clean)
        solved = sum(1 for run in clean if run.reward > 0)
        tok_per_s = tokens / rollout_s if rollout_s > 0 else 0.0
        loss = fb.metrics.get("loss:sum", float("nan"))
        print(
            f"step {step:2d} | reward {mean_reward:.3f} ({solved}/{len(clean)} clean) "
            f"| rollout {rollout_s:5.1f}s {tokens:6d}tok {tok_per_s:4.0f}tok/s "
            f"| train {train_s:5.1f}s loss {loss:+.4f} "
            f"| optim {result.step} datums {fb.num_datums} "
            f"| failed {failed}/{len(batch)} dropped {dropped}g",
            flush=True,
        )


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--group", type=int, default=5, help="rollouts per task (GRPO group)")
    parser.add_argument("--max-steps", type=int, default=30, help="max agent turns per rollout")
    parser.add_argument(
        "--max-tasks", type=int, default=None,
        help="cap tasks per step for fast feedback (default: all tasks in the taskset)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-concurrent", type=int, default=8, help="cap on simultaneous rollouts")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-rollout wall-clock cap (s)")
    args = parser.parse_args()
    asyncio.run(
        main(
            steps=args.steps,
            group=args.group,
            max_steps=args.max_steps,
            max_tasks=args.max_tasks,
            learning_rate=args.learning_rate,
            max_concurrent=args.max_concurrent,
            rollout_timeout=args.timeout,
        )
    )
