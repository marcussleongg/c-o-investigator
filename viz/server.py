"""
Local demo server: type two names, run an investigation, watch it animate.

Routes:
  GET /                         -> the name-input form (empty canvas)
  GET /investigate?a=..&b=..    -> the animated investigation for that pair

Trajectory source (per pair):
  1. cache hit                  -> instant (viz/cache/<slug>.json)
  2. --real                     -> run the trained agent live and animate its real
                                   tool calls (slow; uses model + SixtyFour/Exa quota)
  3. default (synth)            -> shortest path in the SEC graph (instant, no quota);
                                   any two of the ~3,700 real people in the graph

Every computed result is cached, so pairs you demo load instantly next time. Pre-run
your demo pairs once to warm the cache.

    python viz/server.py                 # http://localhost:8000  (synth, safe during training)
    python viz/server.py --real          # live agent inference (post-training)
    python viz/server.py --port 8080
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import networkx as nx
from networkx.readwrite import json_graph
from rapidfuzz import process
from rapidfuzz.utils import default_process

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "viz"))
import build_viz  # noqa: E402  (build_events, build_html, trace_to_events, run_investigation)

CACHE_DIR = ROOT / "viz" / "cache"


def _load_graph() -> tuple[nx.Graph, dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Load the SEC graph plus node-type and adjacency indexes (once, at startup)."""
    G = json_graph.node_link_graph(json.loads((ROOT / "data" / "graph.json").read_text()), edges="edges")
    node_type = {n: d.get("type", "person") for n, d in G.nodes(data=True)}
    adj: dict[str, list[tuple[str, str]]] = {}
    for u, v, d in G.edges(data=True):
        adj.setdefault(u, []).append((v, d.get("role", "")))
        adj.setdefault(v, []).append((u, d.get("role", "")))
    return G, node_type, adj


def _slug(a: str, b: str) -> str:
    """Order-independent cache key for a name pair."""
    norm = "__".join(sorted(re.sub(r"[^a-z0-9]+", "-", x.lower()).strip("-") for x in (a, b)))
    return norm or "pair"


def _resolve(name: str, people: list[str]) -> str | None:
    """Map a typed name to the closest graph person (exact, then fuzzy ≥ 82), or None."""
    if name in people:
        return name
    m = process.extractOne(name, people, processor=default_process, score_cutoff=82)
    return m[0] if m else None


class Investigator:
    """Holds the graph and produces (events, case) for a pair, with on-disk caching."""

    def __init__(self, real: bool):
        self.real = real
        self.G, self.node_type, self.adj = _load_graph()
        self.people = [n for n, t in self.node_type.items() if t == "person"]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"loaded graph: {len(self.people)} people | mode: {'REAL agent' if real else 'synth'}", flush=True)

    def investigate(self, a: str, b: str) -> tuple[list[dict], dict]:
        """Return (events, case) for the pair — from cache, real inference, or synth."""
        cache_file = CACHE_DIR / f"{_slug(a, b)}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            print(f"cache hit: {a} ↔ {b}", flush=True)
            return data["events"], data["case"]

        if self.real:
            events, case = self._real(a, b)
        else:
            events, case = self._synth(a, b)

        cache_file.write_text(json.dumps({"events": events, "case": case}))
        return events, case

    def _synth(self, a: str, b: str) -> tuple[list[dict], dict]:
        ra, rb = _resolve(a, self.people), _resolve(b, self.people)
        if ra and rb and ra != rb and nx.has_path(self.G, ra, rb):
            path = nx.shortest_path(self.G, ra, rb)
            citations = [self.G[path[i]][path[i + 1]].get("citation", "") for i in range(len(path) - 1)]
            case = {"person_a": ra, "person_b": rb, "label": True, "path": path, "citations": citations}
            events = build_viz.build_events(case, self.node_type, self.adj)
            print(f"synth path ({(len(path) - 1)//2}-hop): {' → '.join(path)}", flush=True)
            return events, case
        # No path (or unknown names): expand both people's board networks as two
        # separate clusters that never meet, ending in a central "no connection" badge.
        na, nb = ra or a, rb or b
        events = build_viz.build_no_connection_events(na, nb, self.adj, self.node_type)
        return events, {"person_a": na, "person_b": nb, "label": False, "path": []}

    def _real(self, a: str, b: str) -> tuple[list[dict], dict]:
        print(f"running live agent: {a} ↔ {b} (this can take minutes) ...", flush=True)
        trace = build_viz.run_investigation(a, b)
        events, path = build_viz.trace_to_events(trace, self.node_type)
        case = {"person_a": a, "person_b": b, "label": bool(path), "path": path, "citations": []}
        return events, case


def _form(a: str = "", b: str = "") -> str:
    return (
        '<form class="formbar" action="/investigate" method="get">'
        f'<input name="a" placeholder="Person A" value="{html.escape(a)}" required>'
        '<span class="arrow">↔</span>'
        f'<input name="b" placeholder="Person B" value="{html.escape(b)}" required>'
        '<button type="submit">Investigate</button>'
        "</form>"
    )


def make_handler(inv: Investigator):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quieter console
            pass

        def _send(self, body: str, status: int = 200):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(build_viz.build_html([], {"person_a": "", "person_b": ""}, form_html=_form()))
                return
            if parsed.path == "/investigate":
                q = parse_qs(parsed.query)
                a = (q.get("a", [""])[0]).strip()
                b = (q.get("b", [""])[0]).strip()
                if not a or not b:
                    self._send(build_viz.build_html([], {"person_a": "", "person_b": ""}, form_html=_form(a, b)))
                    return
                try:
                    events, case = inv.investigate(a, b)
                except Exception as e:  # never crash the demo
                    self._send(build_viz.build_html(
                        [], {"person_a": a, "person_b": b}, form_html=_form(a, b)
                    ).replace("</header>", f'<div style="color:#f85149;padding:6px 20px">Error: {html.escape(str(e))}</div></header>'))
                    return
                self._send(build_viz.build_html(events, case, form_html=_form(case.get("person_a", a), case.get("person_b", b)), ready=True))
                return
            self._send("not found", 404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="COI investigation demo server.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--real", action="store_true", help="Run the live trained agent (slow; uses quota)")
    args = parser.parse_args()

    inv = Investigator(real=args.real)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(inv))
    print(f"\n  COI demo → http://localhost:{args.port}\n  Ctrl-C to stop\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
