"""Shared scaffolding for COI RL training (adapted from hud-python cookbooks/rl-training).

The training loop is agnostic to where rollouts come from — it only consumes
``job.runs`` (each carrying a trajectory + reward). So the real flow and the local
smoke-test differ only in *which taskset* and *which runtime* we hand to
``Taskset.run``; the training code in ``train.py`` never changes.

``load_taskset_and_runtime()`` picks between them from the ``TASKSET`` constant:

- ``TASKSET`` set — the real flow: load the taskset you deployed
  (``hud deploy`` + ``hud sync``) from the platform with ``Taskset.from_api`` and
  run every rollout on a leased HUD box with ``HUDRuntime`` (agent runs remotely,
  next to the env — parallel and fast). Nothing runs locally.
- empty — a cheap local smoke-test: a small slice of the COI training tasks driven
  against the bundled ``env.py`` locally, to verify the loop wiring before
  spending on the full remote run.
"""

from __future__ import annotations

from hud.eval import HUDRuntime, LocalRuntime, Provider, Taskset

# Importing tasks_train builds the 180-case COI training taskset from
# cases_train.json (40 neg, 60 1-hop, 45 2-hop, 35 3-hop).
from tasks_train import tasks as _train_tasks

# Deployed taskset to train on (its name or id, from `hud deploy` + `hud sync`).
# Leave empty for the local smoke-test against env.py.
TASKSET = "coi-train"

# How many tasks the local smoke-test uses (keep small — it's only a wiring check).
LOCAL_SAMPLE = 4


def load_taskset_and_runtime() -> tuple[Taskset, Provider | HUDRuntime]:
    """
    Resolve the rollout source for training from the ``TASKSET`` constant.

    When ``TASKSET`` is set, loads the deployed taskset and runs rollouts remotely
    on HUD infra (fast, parallel). When empty, returns a small local slice of the
    COI training tasks against the bundled ``env.py`` for a cheap wiring check.

    Parameters:
        None — configured via the module-level ``TASKSET`` / ``LOCAL_SAMPLE`` constants.

    Returns:
        tuple[Taskset, Provider | HUDRuntime] - The taskset to roll out and the
        runtime that executes each rollout.

    Examples:
        taskset, runtime = load_taskset_and_runtime()
        await taskset.run(agent, runtime=runtime, job=session)
    """
    if TASKSET:
        return Taskset.from_api(TASKSET), HUDRuntime()

    local = Taskset("coi-train-local", _train_tasks[:LOCAL_SAMPLE])
    return local, LocalRuntime("env.py")
