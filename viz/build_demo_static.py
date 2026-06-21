"""
Build static, Vercel-deployable pages from REAL trained-model investigations.

Unlike build_static.py (which *synthesizes* trajectories from the SEC graph and so
can only render people who are in graph.json), this runs each demo pair through the
trained model in-process and renders the agent's ACTUAL trajectory — so it works for
non-SEC people (startup founders, private execs). If the agent finds no connection,
or the rollout fails, the page shows "No connection found".

Cheap to re-run because the sixtyfour results are disk-cached (sixtyfour_cache.json):
entities enriched in a prior run are served from cache, not the live API.

Demo pairs come from tasks_demo._DEMOS (single source of truth). Writes
viz/dist/<slug>.html per pair + viz/dist/index.html, then prints the deploy command.

Usage:
  python viz/build_demo_static.py                 # trained model (coi-clean2)
  python viz/build_demo_static.py --model claude  # cleaner trajectories for a polished demo
"""

from __future__ import annotations

import argparse
import asyncio
import html as _html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viz"))

import build_viz  # noqa: E402  (trace_to_events, build_html, load_graph, build_no_connection_events)
import build_static  # noqa: E402  (reuse its landing page: gallery + live-search form)

DIST = ROOT / "viz" / "dist"
GRAPH = ROOT / "data" / "graph.json"


def _slug(a: str, b: str) -> str:
    """Filesystem-safe slug for a pair, e.g. ('Jane Smith','Bob Lee') -> 'jane-smith__bob-lee'."""
    def s(x: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", x.lower()).strip("-")
    return f"{s(a)}__{s(b)}"


_LABEL = {0: "no connection", 1: "1-hop · direct interlock",
          2: "2-hop · indirect tie", 3: "3-hop · distant link"}


def _card_from_html(path: Path) -> dict | None:
    """Recover a gallery card (slug, names, hop label) from a rendered page's embedded META."""
    m = re.search(r"const META = (\{.*?\});", path.read_text())
    if not m:
        return None
    try:
        meta = json.loads(m.group(1))
    except Exception:
        return None
    hops = int(meta.get("hops", 0) or 0)
    return {"slug": path.stem, "a": meta.get("person_a", ""), "b": meta.get("person_b", ""),
            "label": _LABEL.get(hops, f"{hops}-hop"), "hops": hops}


async def _investigate(pa: str, pb: str, ca: str, cb: str, model: str, max_steps: int, timeout: float):
    """Run one real rollout for a pair via coi_general (with company hints) and return its trace."""
    from dotenv import load_dotenv
    load_dotenv()
    from hud import Job
    from hud.agents import create_agent
    from hud.eval import LocalRuntime, Taskset

    from env import coi_general

    task = coi_general(person_a=pa, person_b=pb, company_a=ca, company_b=cb)
    agent = create_agent(model, max_steps=max_steps, completion_kwargs={"max_tokens": 4096})
    session = await Job.start("coi-demo-static", group=1)
    await Taskset("coi-demo-static", [task]).run(
        agent, runtime=LocalRuntime(str(ROOT / "env.py")), job=session,
        max_concurrent=1, rollout_timeout=timeout,
    )
    return session.runs[0].trace


def _noconn_events(pa: str, pb: str, adj, node_type) -> list[dict]:
    """Fallback events when a rollout fails entirely — a clean 'no connection' view."""
    try:
        return build_viz.build_no_connection_events(pa, pb, adj, node_type)
    except Exception:
        return [{
            "i": 0, "tool": "submit", "arg": "", "caption": "no connection found",
            "nodes": [{"id": "__noconn__", "type": "noconn", "label": "✕  No connection", "x": 0, "y": 0}],
            "edges": [], "path": [], "citations": [],
        }]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static demo pages from real trained-model investigations.")
    parser.add_argument("--model", default="coi-clean2", help="Model to run (e.g. coi-clean2, claude)")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--index-only", action="store_true",
                        help="Rebuild index.html from existing dist pages; run no rollouts")
    parser.add_argument("--backend", default="http://localhost:8000",
                        help="Local backend URL for the live-search form on the landing page")
    args = parser.parse_args()

    DIST.mkdir(parents=True, exist_ok=True)

    if not args.index_only:
        from tasks_demo import _DEMOS  # single source of truth for the demo pairs

        node_type, adj = build_viz.load_graph(GRAPH)
        for (pa, ca, pb, cb) in _DEMOS:
            print(f"investigating {pa} ↔ {pb} ...", flush=True)
            try:
                trace = asyncio.run(_investigate(pa, pb, ca, cb, args.model, args.max_steps, args.timeout))
                events, path = build_viz.trace_to_events(trace, node_type)
                if not events:
                    raise ValueError("empty trace")
            except Exception as e:
                print(f"  ! failed ({type(e).__name__}: {e}); rendering as no-connection", flush=True)
                events, path = _noconn_events(pa, pb, adj, node_type), []

            case = {"person_a": pa, "person_b": pb, "path": path}
            slug = _slug(pa, pb)
            back = '<a href="index.html" style="color:#8b949e;text-decoration:none">← All examples</a>'
            (DIST / f"{slug}.html").write_text(build_viz.build_html(events, case, form_html=back, ready=False))
            label = f"{(len(path) - 1) // 2}-hop" if path else "no connection"
            print(f"  -> {slug}.html ({label})", flush=True)

    # Rebuild the index over EVERY page in dist/ — the new real demos AND the older
    # synthesized SEC examples — by recovering each page's metadata. Connections
    # (higher hop count) first, no-connection last, then alphabetical.
    cards = [c for p in DIST.glob("*.html") if p.name != "index.html" and (c := _card_from_html(p))]
    cards.sort(key=lambda c: (-c["hops"], c["a"].lower()))
    # build_static._index_html renders the full landing page: gallery + live-search form.
    (DIST / "index.html").write_text(build_static._index_html(cards, args.backend))
    print(f"\nindex links {len(cards)} pages (+ live search) -> {DIST}/index.html")
    print("deploy: cd viz/dist && vercel deploy --prod   (or drag viz/dist into vercel.com)")


if __name__ == "__main__":
    main()
