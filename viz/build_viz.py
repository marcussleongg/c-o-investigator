"""
Build a self-contained, watchable visualization of a COI investigation.

Synthesizes a plausible agent trajectory from a labeled case's ground-truth path
(real names, roles, and citations pulled from the SEC graph), then renders a single
HTML file that animates the investigation: each simulated tool call adds nodes and
edges to the graph, and the final submit() lights up the discovered connection path
between the two people.

No server, no build step, and no live rollouts — it reads only local JSON, so it
never touches the SixtyFour/Exa quota the running training depends on. Open the
output HTML in any browser (needs internet for the vis-network CDN script).

Usage:
  python viz/build_viz.py                          # first 2-hop positive test case
  python viz/build_viz.py --case-index 3
  python viz/build_viz.py --person-a "X" --person-b "Y"
  python viz/build_viz.py --cases cases_test.json --hop 2 --out viz/coi_viz.html
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "graph.json"


def load_graph(graph_path: Path) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """
    Load node types and a neighbor index from the serialized COI graph.

    Parameters:
        graph_path : Path - Path to data/graph.json (networkx node_link format).

    Returns:
        tuple[dict, dict] - (node_type, adjacency) where node_type maps name -> 'person'
        | 'company', and adjacency maps name -> list of (neighbor, role).

    Examples:
        types, adj = load_graph(Path("data/graph.json"))
        types["EOG RESOURCES INC"]            # 'company'
        adj["EOG RESOURCES INC"][0]           # ('Janet Clark', 'Director')
    """
    data = json.loads(graph_path.read_text())
    node_type = {n["id"]: n.get("type", "person") for n in data["nodes"]}
    adj: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    # networkx 3.x serializes edges under "edges"; older builds use "links".
    for e in data.get("edges", data.get("links", [])):
        role = e.get("role", "")
        adj[e["source"]].append((e["target"], role))
        adj[e["target"]].append((e["source"], role))
    return node_type, adj


def select_case(
    cases: list[dict], *, case_index: int | None, person_a: str | None,
    person_b: str | None, hop: int | None,
) -> dict:
    """
    Pick one case to visualize from the loaded case list.

    Selection precedence: explicit index, then a person_a/person_b name match, then
    the first positive case matching the requested hop count (default: first 2-hop
    positive, falling back to any positive).

    Parameters:
        cases      : list[dict] - Cases loaded from cases_*.json.
        case_index : int | None - Exact index into the list, if given.
        person_a   : str | None - Substring match on person_a, if given.
        person_b   : str | None - Substring match on person_b, if given.
        hop        : int | None - Desired hop count ((len(path)-1)//2); default 2.

    Returns:
        dict - The selected case (person_a, person_b, label, path, citations).

    Examples:
        select_case(cases, case_index=None, person_a=None, person_b=None, hop=2)
        # {'person_a': 'Robert M. Muraro', ..., 'path': [...]}
    """
    if case_index is not None:
        return cases[case_index]

    if person_a or person_b:
        for c in cases:
            if (not person_a or person_a.lower() in c["person_a"].lower()) and (
                not person_b or person_b.lower() in c["person_b"].lower()
            ):
                return c
        raise SystemExit(f"No case matching person_a={person_a!r} person_b={person_b!r}")

    want_hop = 2 if hop is None else hop
    positives = [c for c in cases if c.get("label") and c.get("path")]
    for c in positives:
        if (len(c["path"]) - 1) // 2 == want_hop:
            return c
    if positives:
        return positives[0]
    raise SystemExit("No positive (connected) case found to visualize.")


def accession_label(citation: str) -> str:
    """
    Turn a raw filing citation into a short, human-readable SEC label.

    Parameters:
        citation : str - A citation string (local filing path or URL) containing an
                         EDGAR accession number like 0001193125-26-059296.

    Returns:
        str - "SEC 0001193125-26-059296" if an accession is found, else the input
        trimmed to its last path segment.

    Examples:
        accession_label("/Volumes/.../TRGP/10-K/0001193125-26-059296/full-submission.txt")
        # 'SEC 0001193125-26-059296'
    """
    m = re.search(r"\d{10}-\d{2}-\d{6}", citation or "")
    if m:
        return f"SEC {m.group(0)}"
    return (citation or "").rstrip("/").split("/")[-1] or "source"


def build_events(
    case: dict, node_type: dict[str, str], adj: dict[str, list[tuple[str, str]]],
    max_distractors: int = 2,
) -> list[dict]:
    """
    Turn a case's ground-truth path into an ordered list of investigation events.

    Walks the path from person_a to person_b, emitting one event per simulated tool
    call: sec_search on the start, then enrich_person / enrich_company at each hop.
    Enriching a company also surfaces a few real (non-path) board members as
    distractors, so the demo shows genuine exploration rather than a clean line. A
    final submit() event carries the full path for highlighting.

    Parameters:
        case            : dict - The selected case with path + citations.
        node_type       : dict - name -> 'person' | 'company' from the graph.
        adj             : dict - name -> [(neighbor, role)] from the graph.
        max_distractors : int  - Max real off-path board members revealed per company.

    Returns:
        list[dict] - Events, each: {i, tool, arg, caption, nodes:[{id,type}],
        edges:[{from,to,role,citation,onpath}]} with a trailing submit event that
        also carries {path, citations}.

    Examples:
        evs = build_events(case, node_type, adj)
        evs[0]["tool"]      # 'sec_search'
        evs[-1]["tool"]     # 'submit'
    """
    path: list[str] = case["path"]
    cits: list[str] = case.get("citations") or []

    def typ(name: str, idx: int) -> str:
        # Prefer the graph's label; fall back to bipartite alternation (even=person).
        return node_type.get(name) or ("person" if idx % 2 == 0 else "company")

    on_path = set(path)
    added_nodes: set[str] = set()
    added_edges: set[frozenset[str]] = set()
    events: list[dict] = []

    def node_payload(name: str, idx: int) -> list[dict]:
        if name in added_nodes:
            return []
        added_nodes.add(name)
        return [{"id": name, "type": typ(name, idx)}]

    def edge_payload(a: str, b: str, role: str, citation: str, onpath: bool) -> list[dict]:
        key = frozenset((a, b))
        if key in added_edges:
            return []
        added_edges.add(key)
        return [{"from": a, "to": b, "role": role, "citation": citation, "onpath": onpath}]

    # Opening move: locate the start person in SEC filings.
    start = path[0]
    events.append({
        "i": 0, "tool": "sec_search", "arg": start,
        "caption": f"sec_search('{start}') — locate primary-source filings",
        "nodes": node_payload(start, 0), "edges": [],
    })

    # Walk each node; enriching it reveals the next hop (+ distractors for companies).
    for idx, name in enumerate(path):
        kind = typ(name, idx)
        tool = "enrich_company" if kind == "company" else "enrich_person"
        nodes = node_payload(name, idx)
        edges: list[dict] = []

        if kind == "company":
            shown = 0
            for neighbor, role in adj.get(name, []):
                if shown >= max_distractors:
                    break
                if neighbor in on_path or neighbor in added_nodes:
                    continue
                nodes += node_payload(neighbor, idx + 1)
                edges += edge_payload(name, neighbor, role or "Director", "", False)
                shown += 1

        if idx < len(path) - 1:
            nxt = path[idx + 1]
            nodes += node_payload(nxt, idx + 1)
            role = ""
            for neighbor, r in adj.get(name, []):
                if neighbor == nxt:
                    role = r
                    break
            citation = cits[idx] if idx < len(cits) else ""
            edges += edge_payload(name, nxt, role or "Director", citation, True)

        verb = "board members & executives" if kind == "company" else "board seats & affiliations"
        events.append({
            "i": idx + 1, "tool": tool, "arg": name,
            "caption": f"{tool}('{name}') — {verb}",
            "nodes": nodes, "edges": edges,
        })

    events.append({
        "i": len(events), "tool": "submit", "arg": "",
        "caption": f"submit() — connection found across {len(path) - 1} edges",
        "nodes": [], "edges": [],
        "path": path,
        "citations": [accession_label(c) for c in cits],
    })

    _assign_positions(events, path)
    return events


def _assign_positions(events: list[dict], path: list[str], x_gap: int = 260, y_gap: int = 180) -> None:
    """
    Annotate every node in the events with a fixed (x, y) layout position.

    The main connection path is laid out left-to-right on a single horizontal line;
    each company's off-path distractors fan out directly above and below it. Fixed
    positions let the renderer disable physics entirely, so nodes appear in place
    with no bouncing or re-stabilization.

    Parameters:
        events : list[dict] - Events from build_events (mutated in place).
        path   : list[str] - The ground-truth path (defines the horizontal spine).
        x_gap  : int - Horizontal spacing between consecutive path nodes.
        y_gap  : int - Vertical spacing between fanned distractors.

    Returns:
        None - Adds "x"/"y" keys to each node dict in events.

    Examples:
        _assign_positions(events, ["A", "Co", "B"])
        # events[1]["nodes"][0] now has {"x": 260, "y": 0, ...}
    """
    pos: dict[str, tuple[int, int]] = {n: (i * x_gap, 0) for i, n in enumerate(path)}

    # Group each company's off-path distractors, then fan them up/down around it.
    distractors: dict[str, list[str]] = {}
    for ev in events:
        for e in ev.get("edges", []):
            if not e["onpath"]:
                distractors.setdefault(e["from"], []).append(e["to"])
    for company, kids in distractors.items():
        cx, _ = pos.get(company, (0, 0))
        for j, kid in enumerate(kids):
            level = j // 2 + 1
            sign = -1 if j % 2 == 0 else 1
            pos[kid] = (cx, sign * level * y_gap)

    for ev in events:
        for n in ev.get("nodes", []):
            x, y = pos.get(n["id"], (0, 0))
            n["x"], n["y"] = x, y


def compute_soft_ties(path: list[str], overlay: dict | None) -> list[dict]:
    """
    Derive unverified person↔person ghost edges from the SixtyFour overlay.

    Compares the people on the path pairwise and emits a tie wherever they share a
    non-corporate signal: an alma mater (co-alumni), a named associate, a non-public
    board (co-director), or a direct family/associate naming. Multiple signals for
    one pair are merged into a single labelled edge. These ties are illustrative and
    never scored.

    Parameters:
        path    : list[str] - The connection path (people sit at even indices).
        overlay : dict | None - name -> enrichment entry from enrich_overlay.py.

    Returns:
        list[dict] - Each: {"from", "to", "label"} for a dashed ghost edge.

    Examples:
        compute_soft_ties(["A","Co","B"], {"A":{"education":["MIT"]},"B":{"education":["MIT"]}})
        # [{'from': 'A', 'to': 'B', 'label': 'Co-alumni: MIT'}]
    """
    if not overlay:
        return []

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

    def _clean(s: str) -> str:
        return re.sub(r"\(demo\)", "", str(s)).strip()

    def _inst(s: str) -> str:
        head = re.split(r"[—,(]| - ", _clean(s))[0]
        return _norm(head) or _norm(s)

    people = [n for i, n in enumerate(path) if i % 2 == 0 and n in overlay]
    ties: list[dict] = []
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            a, b = people[i], people[j]
            ea, eb = overlay[a], overlay[b]
            labels: list[str] = []

            shared_edu = ({_inst(x) for x in ea.get("education", [])} & {_inst(x) for x in eb.get("education", [])}) - {""}
            for raw in ea.get("education", []):
                if _inst(raw) in shared_edu:
                    labels.append(f"Co-alumni: {re.split(r'[—,(]| - ', _clean(raw))[0].strip()}")
                    break

            shared_assoc = ({_norm(x) for x in ea.get("associates", [])} & {_norm(x) for x in eb.get("associates", [])}) - {""}
            for raw in ea.get("associates", []):
                if _norm(raw) in shared_assoc:
                    labels.append(f"Shared associate: {_clean(raw).split('(')[0].strip()}")
                    break

            shared_board = ({_inst(x) for x in ea.get("other_boards", [])} & {_inst(x) for x in eb.get("other_boards", [])}) - {""}
            for raw in ea.get("other_boards", []):
                if _inst(raw) in shared_board:
                    labels.append(f"Co-director: {re.split(r'[—,(]| - ', _clean(raw))[0].strip()}")
                    break

            # Direct family/associate naming: one entry mentions the other person.
            for kind in ("family", "associates"):
                if any(_norm(b) and _norm(b) in _norm(x) for x in ea.get(kind, [])) or \
                   any(_norm(a) and _norm(a) in _norm(x) for x in eb.get(kind, [])):
                    labels.append("Family" if kind == "family" else "Associate")

            if labels:
                ties.append({"from": a, "to": b, "label": " · ".join(dict.fromkeys(labels))})
    return ties


def build_no_connection_events(person_a: str, person_b: str,
                               adj: dict[str, list[tuple[str, str]]], node_type: dict[str, str],
                               max_companies: int = 3, max_people: int = 2, gap: int = 220) -> list[dict]:
    """
    Build events for a negative case: two separate board clusters that never meet.

    Expands each person's local network (their board seats, then a couple of fellow
    directors) as a cluster on its own side of the canvas, then drops a central
    "✕ No connection" badge in the gap between them. This shows a real two-sided
    investigation that fails to connect, rather than two lonely dots.

    Parameters:
        person_a, person_b : str - The two people (already resolved to graph names if possible).
        adj           : dict - name -> [(neighbor, role)] from the graph.
        node_type     : dict - name -> 'person'|'company'.
        max_companies : int - Board seats to show per person.
        max_people    : int - Fellow directors to show under the first board.
        gap           : int - Layout spacing.

    Returns:
        list[dict] - Events (each enrich expands a side; final submit drops the badge).

    Examples:
        build_no_connection_events("A", "B", adj, node_type)[-1]["tool"]  # 'submit'
    """
    events: list[dict] = []
    added: set[str] = set()
    right0 = 6 * gap
    center_x = 3 * gap

    def companies_of(name: str) -> list[str]:
        return [n for n, _ in adj.get(name, []) if node_type.get(n) == "company"][:max_companies]

    def people_of(comp: str, exclude: str) -> list[str]:
        return [n for n, _ in adj.get(comp, []) if node_type.get(n) == "person" and n != exclude][:max_people]

    def role(x: str, y: str) -> str:
        return next((r for n, r in adj.get(x, []) if n == y), "") or "Director"

    def node(nid, ntype, x, y, label=None):
        if nid in added:
            return None
        added.add(nid)
        n = {"id": nid, "type": ntype, "x": x, "y": y}
        if label:
            n["label"] = label
        return n

    def emit(tool, arg, nodes, edges, caption):
        events.append({"i": len(events), "tool": tool, "arg": arg,
                       "caption": caption, "nodes": [n for n in nodes if n], "edges": edges})

    def side(person, base_x, direction):
        # direction: +1 expands rightward (left cluster), -1 leftward (right cluster)
        emit("sec_search", person, [node(person, "person", base_x, 0)], [], f"sec_search('{person}')")
        comps = companies_of(person)
        nodes, edges = [], []
        for k, c in enumerate(comps):
            cy = (k - (len(comps) - 1) / 2) * gap
            nodes.append(node(c, "company", base_x + direction * gap, cy))
            edges.append({"from": person, "to": c, "role": role(person, c), "citation": "", "onpath": False})
        emit("enrich_person", person, nodes, edges, f"enrich_person('{person}') — board seats")
        if comps:
            c0 = comps[0]
            ppl = people_of(c0, person)
            nodes, edges = [], []
            for k, p in enumerate(ppl):
                py = (k - (len(ppl) - 1) / 2) * gap * 0.7
                nodes.append(node(p, "person", base_x + direction * 2 * gap, py))
                edges.append({"from": c0, "to": p, "role": role(c0, p), "citation": "", "onpath": False})
            if any(nodes):
                emit("enrich_company", c0, nodes, edges, f"enrich_company('{c0}')")

    side(person_a, 0, +1)
    side(person_b, right0, -1)

    badge = {"id": "__noconn__", "type": "noconn", "label": "✕  No connection", "x": center_x, "y": 0}
    events.append({"i": len(events), "tool": "submit", "arg": "",
                   "caption": "submit([]) — no connection found",
                   "nodes": [badge], "edges": [], "path": [], "citations": []})
    return events


def render_html(events: list[dict], case: dict, out_path: Path, overlay: dict | None = None,
                soft_ties: list[dict] | None = None) -> None:
    """
    Write a single self-contained HTML file that animates the investigation events.

    Parameters:
        events   : list[dict] - Output of build_events.
        case     : dict - The selected case (for the title/question banner).
        out_path : Path - Destination .html file.

    Returns:
        None - Writes the file to disk.

    Examples:
        render_html(events, case, Path("viz/coi_viz.html"))
    """
    html = build_html(events, case, overlay=overlay, soft_ties=soft_ties)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)


def build_html(events: list[dict], case: dict, overlay: dict | None = None,
               soft_ties: list[dict] | None = None, form_html: str = "",
               ready: bool = False) -> str:
    """
    Build the standalone viz HTML as a string (shared by render_html and the server).

    Parameters:
        events    : list[dict] - Investigation events.
        case      : dict - Case with person_a/person_b and optional path.
        overlay   : dict | None - SixtyFour overlay (panel + heuristic ties source).
        soft_ties : list[dict] | None - Precomputed ties; falls back to heuristic if None.
        form_html : str - Optional name-input form bar injected into the header (server mode).
        ready     : bool - Job finished — show the "investigation complete" start popup.

    Returns:
        str - The full HTML document.

    Examples:
        build_html(events, case, form_html="<form>…</form>", autoplay=True)
    """
    p = case.get("path") or []
    meta = {
        "person_a": case.get("person_a", ""),
        "person_b": case.get("person_b", ""),
        "hops": (len(p) - 1) // 2 if len(p) >= 1 else 0,
        "ready": ready,
    }
    if soft_ties is None:
        soft_ties = compute_soft_ties(p, overlay)
    return (
        _HTML_TEMPLATE.replace("__EVENTS__", json.dumps(events))
        .replace("__META__", json.dumps(meta))
        .replace("__OVERLAY__", json.dumps(overlay or {}))
        .replace("__SOFTTIES__", json.dumps(soft_ties))
        .replace("__FORMBAR__", form_html)
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>COI Investigator — live trace</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3; --muted:#8b949e;
          --person:#2dd4bf; --company:#d29922; --path:#2ea043; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:var(--bg); color:var(--text); height:100vh; display:flex; flex-direction:column; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:15px; font-weight:600; letter-spacing:.2px; }
  header .q { margin-top:4px; font-size:13px; color:var(--muted); }
  header .q b { color:var(--text); font-weight:600; }
  main { flex:1; display:flex; min-height:0; }
  #net { flex:1; min-width:0; }
  aside { width:340px; border-left:1px solid var(--line); background:var(--panel);
          display:flex; flex-direction:column; }
  .controls { padding:12px; border-bottom:1px solid var(--line); display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  button { background:#21262d; color:var(--text); border:1px solid var(--line); border-radius:6px;
           padding:6px 12px; font-size:13px; cursor:pointer; }
  button:hover { background:#30363d; }
  button.primary { background:#238636; border-color:#2ea043; }
  .controls label { font-size:12px; color:var(--muted); display:flex; align-items:center; gap:6px; }
  .log { flex:1; overflow-y:auto; padding:8px 12px; }
  .log .row { padding:7px 9px; margin:5px 0; border-radius:6px; background:#0d1117; border:1px solid var(--line);
              font-size:12px; opacity:0; transform:translateY(4px); transition:all .25s; }
  .log .row.show { opacity:1; transform:none; }
  .log .row .tool { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--person); }
  .log .row.submit .tool { color:var(--path); }
  .verdict { margin:6px 12px 12px; padding:10px 12px; border-radius:8px; background:#0d2818;
             border:1px solid var(--path); font-size:13px; display:none; }
  .verdict.show { display:block; }
  .verdict .pathline { margin-top:6px; color:var(--muted); font-size:12px; line-height:1.5; }
  .legend { display:flex; gap:14px; font-size:12px; color:var(--muted); }
  .legend span::before { content:""; display:inline-block; width:9px; height:9px; border-radius:50%;
                          margin-right:5px; vertical-align:middle; }
  .legend .p::before { background:var(--person); }
  .legend .c::before { background:var(--company); border-radius:2px; }
  .intel { margin:0 12px 12px; padding:10px 12px; border-radius:8px; background:#1c1408;
           border:1px dashed var(--company); font-size:12px; display:none; }
  .intel.show { display:block; }
  .intel .hd { font-weight:600; color:#f0d999; display:flex; justify-content:space-between; align-items:center; }
  .intel .tag { font-weight:500; font-size:10px; color:#0d1117; background:var(--company);
                padding:2px 6px; border-radius:10px; }
  .intel .who { margin-top:6px; color:var(--text); font-weight:600; }
  .intel .field { margin-top:5px; color:var(--muted); line-height:1.45; }
  .intel .field b { color:#cbd5d1; font-weight:600; }
  .intel .hint { color:var(--muted); font-style:italic; }
  .formbar { margin-top:10px; display:flex; gap:8px; align-items:center; }
  .formbar input { background:#0d1117; border:1px solid var(--line); color:var(--text);
                   border-radius:6px; padding:6px 10px; font-size:13px; min-width:180px; }
  .formbar button { background:#238636; border:1px solid #2ea043; color:#fff;
                    border-radius:6px; padding:6px 14px; font-size:13px; cursor:pointer; }
  .formbar .arrow { color:var(--muted); }
  .modal { position:fixed; inset:0; background:rgba(13,17,23,.72); backdrop-filter:blur(2px);
           display:none; align-items:center; justify-content:center; z-index:50; }
  .modal.show { display:flex; }
  .modal .card { background:var(--panel); border:1px solid var(--path); border-radius:12px;
                 padding:26px 32px; text-align:center; max-width:440px;
                 box-shadow:0 12px 48px rgba(0,0,0,.55); animation:pop .2s ease; }
  @keyframes pop { from { transform:scale(.94); opacity:0; } to { transform:scale(1); opacity:1; } }
  .modal .check { width:46px; height:46px; line-height:44px; border-radius:50%; margin:0 auto 12px;
                  background:#0d2818; border:1px solid var(--path); color:var(--path); font-size:22px; }
  .modal .title { font-size:17px; font-weight:600; }
  .modal .sub { color:var(--muted); font-size:13px; margin-top:6px; }
  .modal .btns { margin-top:20px; display:flex; gap:10px; justify-content:center; }
  .modal .btns button { background:#21262d; color:var(--text); border:1px solid var(--line);
                        border-radius:6px; padding:9px 16px; font-size:13px; cursor:pointer; }
  .modal .btns .primary { background:#238636; border-color:#2ea043; color:#fff; }
  .loading { position:fixed; inset:0; background:rgba(13,17,23,.82); backdrop-filter:blur(2px);
             display:none; align-items:center; justify-content:center; z-index:60; }
  .loading.show { display:flex; }
  .loading .card { text-align:center; max-width:420px; padding:8px 24px; }
  .loading .spinner { width:36px; height:36px; border-radius:50%; margin:0 auto 16px;
                      border:3px solid #30363d; border-top-color:var(--person); animation:spin .8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .loading .title { font-size:16px; font-weight:600; }
  .loading .sub { color:var(--text); font-size:14px; margin-top:6px; }
  .loading .note { color:var(--muted); font-size:12px; margin-top:10px; line-height:1.4; }
  .vis-tooltip { background:#161b22 !important; color:#e6edf3 !important; border:1px solid #30363d !important;
                 border-radius:6px !important; font-family:inherit !important; font-size:12px !important;
                 padding:6px 9px !important; max-width:280px !important; white-space:normal !important; }
</style>
</head>
<body>
<header>
  <h1>COI Investigator — agent trace</h1>
  <div class="q">Do <b id="pa"></b> and <b id="pb"></b> share a conflict of interest?
    <span class="legend"> &nbsp; <span class="p">person</span><span class="c">company</span></span>
  </div>
  __FORMBAR__
</header>
<main>
  <div id="net"></div>
  <aside>
    <div class="controls">
      <button class="primary" id="play">▶ Play</button>
      <button id="step">Step</button>
      <button id="reset">Reset</button>
      <button id="ties" style="display:none">⌁ Soft ties</button>
      <label>speed
        <input id="speed" type="range" min="300" max="2000" value="1100" step="100">
      </label>
    </div>
    <div class="verdict" id="verdict"></div>
    <div class="intel" id="intel">
      <div class="hd">Supplementary intel · SixtyFour <span class="tag">unverified · not scored</span></div>
      <div id="intelBody" class="hint" style="margin-top:6px">Click a person node to view enrichment.</div>
    </div>
    <div class="log" id="log"></div>
  </aside>
</main>
<div class="modal" id="modal">
  <div class="card">
    <div class="check">✓</div>
    <div class="title">Investigation complete</div>
    <div class="sub" id="modalSub"></div>
    <div class="btns">
      <button class="primary" id="mPlay">▶ Play investigation</button>
      <button id="mStep">Step through</button>
    </div>
  </div>
</div>
<div class="loading" id="loading">
  <div class="card">
    <div class="spinner"></div>
    <div class="title">Investigating…</div>
    <div class="sub" id="loadingSub"></div>
    <div class="note">Tracing connections through SEC filings — this can take a moment.</div>
  </div>
</div>
<script>
const EVENTS = __EVENTS__;
const META = __META__;
const OVERLAY = __OVERLAY__;   // post-hoc SixtyFour enrichment (unverified, not scored)
const SOFTTIES = __SOFTTIES__; // derived person↔person ghost edges (unverified)
document.getElementById('pa').textContent = META.person_a;
document.getElementById('pb').textContent = META.person_b;

const nodes = new vis.DataSet();
const edges = new vis.DataSet();
const network = new vis.Network(document.getElementById('net'), {nodes, edges}, {
  nodes: { font:{color:'#e6edf3', size:14}, borderWidth:2 },
  edges: { color:{color:'#30363d', highlight:'#8b949e'}, width:1.5,
           font:{ color:'#e6edf3', size:12, background:'#0d1117', strokeWidth:0, vadjust:0 },
           smooth:false, arrows:{to:{enabled:false}} },
  physics: false,            // fixed positions — no bouncing / re-stabilization
  layout: { improvedLayout:false },
  interaction:{ hover:true, tooltipDelay:120, dragView:true, zoomView:true },
});

// Smoothly glide the camera to frame whatever is on screen, capped so a lone
// first node isn't blown up to full zoom.
function frame() {
  network.fit({ animation:{ duration:650, easingFunction:'easeInOutQuad' } });
  setTimeout(() => { if (network.getScale() > 1.3) network.moveTo({ scale:1.3,
    animation:{ duration:300, easingFunction:'easeInOutQuad' } }); }, 670);
}

// --- reveal engine: nodes fade + extend out from their parent, edges stretch in ---
const REVEAL_MS = 520;
let _anims = [], _running = false;
function _ease(t) { return t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2) / 2; }
function _tick(now) {
  for (let i = _anims.length - 1; i >= 0; i--) {
    const a = _anims[i];
    let t = (now - a.start) / a.dur; if (t > 1) t = 1;
    const e = _ease(t);
    if (a.kind === 'node') {
      nodes.update({ id:a.id, x:a.fx + (a.tx - a.fx)*e, y:a.fy + (a.ty - a.fy)*e, opacity:e });
    } else {
      edges.update({ id:a.id, color:{ color:'rgba(' + a.rgb + ',' + e + ')' } });
    }
    if (t >= 1) _anims.splice(i, 1);
  }
  if (_anims.length) requestAnimationFrame(_tick); else _running = false;
}
function _animate(a) { _anims.push(a); if (!_running) { _running = true; requestAnimationFrame(_tick); } }

function styleNode(t) {
  if (t === 'company') return { shape:'box', color:{background:'#3b2f12', border:'#d29922'}, font:{color:'#f0d999'} };
  if (t === 'noconn') return { shape:'box', color:{background:'#2d1416', border:'#f85149'},
    font:{color:'#ff7b72', size:15, bold:true }, borderWidth:2, margin:10 };
  return { shape:'dot', size:18, color:{background:'#0d3b3b', border:'#2dd4bf'} };
}

function truncate(s, n) { s = s || ''; return s.length > n ? s.slice(0, n - 1) + '…' : s; }

const logEl = document.getElementById('log');
function logRow(ev) {
  const r = document.createElement('div');
  r.className = 'row' + (ev.tool === 'submit' ? ' submit' : '');
  const call = ev.arg ? `${ev.tool}('${ev.arg}')` : `${ev.tool}()`;
  r.innerHTML = `<span class="tool">${call}</span>`;
  logEl.appendChild(r);
  requestAnimationFrame(() => r.classList.add('show'));
  logEl.scrollTop = logEl.scrollHeight;
}

function applyEvent(ev) {
  const existing = new Set(nodes.getIds());
  let added = false;
  (ev.nodes || []).forEach(n => {
    if (nodes.get(n.id)) return;
    // Start at the position of the already-placed node this one connects from,
    // so it visibly extends out toward its final spot.
    let fx = n.x, fy = n.y;
    (ev.edges || []).forEach(e => {
      const parent = (e.from === n.id && existing.has(e.to)) ? e.to
                   : (e.to === n.id && existing.has(e.from)) ? e.from : null;
      if (parent) { const p = nodes.get(parent); if (p) { fx = p.x; fy = p.y; } }
    });
    nodes.add(Object.assign({ id:n.id, label:n.label || n.id, x:fx, y:fy, opacity:0, _type:n.type }, styleNode(n.type)));
    _animate({ kind:'node', id:n.id, fx, fy, tx:n.x, ty:n.y, start:performance.now(), dur:REVEAL_MS });
    added = true;
  });
  (ev.edges || []).forEach(e => {
    const id = e.from + '||' + e.to;
    if (edges.get(id)) return;
    const role = e.role || '';
    edges.add({ id, from:e.from, to:e.to, label: truncate(role, 22),
      title: [role, e.citation].filter(Boolean).join('  ·  ') || (e.onpath ? 'connection edge' : 'explored'),
      color:{ color:'rgba(48,54,61,0)' } });
    _animate({ kind:'edge', id, rgb:'48,54,61', start:performance.now(), dur:REVEAL_MS });
  });
  logRow(ev);
  if (added) setTimeout(frame, REVEAL_MS);   // reframe once nodes have settled
  if (ev.tool === 'submit') highlightPath(ev.path, ev.citations);
}

function highlightPath(path, citations) {
  const v0 = document.getElementById('verdict');
  if (!path || !path.length) {   // agent reported no connection
    v0.style.background = '#161b22'; v0.style.borderColor = '#8b949e';
    v0.innerHTML = '<b>No connection found</b><div class="pathline">No COI path reported between the two people.</div>';
    v0.classList.add('show');
    return;
  }
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i], b = path[i + 1];
    const id = edges.get(a + '||' + b) ? a + '||' + b : b + '||' + a;
    if (edges.get(id)) edges.update({ id, color:{color:'#2ea043'}, width:4 });
  }
  nodes.update({ id:path[0], color:{background:'#0d3b3b', border:'#f0d061'}, borderWidth:4 });
  nodes.update({ id:path[path.length-1], color:{background:'#0d3b3b', border:'#f0d061'}, borderWidth:4 });
  const v = document.getElementById('verdict');
  v.innerHTML = `<b>✓ Conflict of interest found</b> — ${META.hops} hop(s)`
    + `<div class="pathline">${path.join('  →  ')}</div>`
    + `<div class="pathline">citations: ${(citations || []).join(' · ')}</div>`;
  v.classList.add('show');
  if (SOFTTIES.length) setTimeout(drawSoftTies, 450);   // reveal soft ties after the spine lands
}

let idx = 0, timer = null;
const speed = document.getElementById('speed');
const playBtn = document.getElementById('play');

function stepOnce() {
  if (idx >= EVENTS.length) { stop(); return; }
  applyEvent(EVENTS[idx++]);
}
function play() {
  if (idx >= EVENTS.length) reset();
  playBtn.textContent = '❚❚ Pause';
  timer = setInterval(stepOnce, +speed.value);
}
function stop() {
  clearInterval(timer); timer = null; playBtn.textContent = '▶ Play';
}
function reset() {
  stop(); idx = 0; _anims = []; _running = false; tiesShown = false; nodes.clear(); edges.clear();
  logEl.innerHTML = ''; document.getElementById('verdict').classList.remove('show');
}

playBtn.onclick = () => timer ? stop() : play();
document.getElementById('step').onclick = () => { stop(); stepOnce(); };
document.getElementById('reset').onclick = reset;
speed.oninput = () => { if (timer) { stop(); play(); } };

// --- supplementary intel panel (only if an overlay was supplied) ---
const intelEl = document.getElementById('intel');
const intelBody = document.getElementById('intelBody');
function fieldRow(label, val) {
  const items = Array.isArray(val) ? val.filter(Boolean) : (val ? [val] : []);
  if (!items.length) return '';
  return `<div class="field"><b>${label}:</b> ${items.join('; ')}</div>`;
}
function showIntel(name) {
  const e = OVERLAY[name];
  if (!e) { intelBody.className = 'hint'; intelBody.innerHTML = 'Click a person node to view enrichment.'; return; }
  intelBody.className = '';
  intelBody.innerHTML = `<div class="who">${name}${e.demo ? ' (demo)' : ''}</div>`
    + (e.summary ? `<div class="field">${e.summary}</div>` : '')
    + fieldRow('Education', e.education)
    + fieldRow('Family', e.family)
    + fieldRow('Known associates', e.associates)
    + fieldRow('Other boards', e.other_boards);
}
if (Object.keys(OVERLAY).length) {
  intelEl.classList.add('show');
  network.on('click', p => { if (p.nodes && p.nodes.length) showIntel(p.nodes[0]); });
}

// --- soft ties: dashed amber person↔person ghost edges (unverified) ---
const tiesBtn = document.getElementById('ties');
let tiesShown = false;
function drawSoftTies() {
  SOFTTIES.forEach((t, k) => {
    const id = 'soft||' + t.from + '||' + t.to;
    if (edges.get(id) || !nodes.get(t.from) || !nodes.get(t.to)) return;
    edges.add({ id, from:t.from, to:t.to, label: truncate(t.label, 24),
      title: [t.label, t.title].filter(Boolean).join('  ·  ') || 'unverified soft tie',
      dashes:[6,5], width:t.width || 2, color:{ color:'rgba(210,153,34,0)' },
      font:{ color:'#f0d999', size:11, background:'#0d1117', strokeWidth:0 },
      smooth:{ enabled:true, type: (k % 2) ? 'curvedCW' : 'curvedCCW', roundness:0.55 } });
    _animate({ kind:'edge', id, rgb:'210,153,34', start:performance.now(), dur:REVEAL_MS });
  });
  tiesShown = true;
}
function hideSoftTies() {
  SOFTTIES.forEach(t => edges.remove('soft||' + t.from + '||' + t.to));
  tiesShown = false;
}
if (SOFTTIES.length) {
  tiesBtn.style.display = '';
  tiesBtn.onclick = () => { tiesShown ? hideSoftTies() : drawSoftTies(); };
}

// --- completion popup: appears when a job's result has loaded; click to begin ---
const modal = document.getElementById('modal');
function hideModal() { modal.classList.remove('show'); }
document.getElementById('mPlay').onclick = () => { hideModal(); reset(); play(); };
document.getElementById('mStep').onclick = () => { hideModal(); reset(); stepOnce(); };
if (META.ready && EVENTS.length) {
  document.getElementById('modalSub').textContent = META.person_a + '  ↔  ' + META.person_b;
  modal.classList.add('show');
}

// --- investigating overlay: shown on submit, stays up until the result page loads ---
const _form = document.querySelector('.formbar');
if (_form) {
  _form.addEventListener('submit', () => {
    const a = (_form.querySelector('input[name=a]') || {}).value || '';
    const b = (_form.querySelector('input[name=b]') || {}).value || '';
    document.getElementById('loadingSub').textContent = a + '  ↔  ' + b;
    document.getElementById('loading').classList.add('show');
    // no preventDefault — the browser keeps this page (with the overlay) visible
    // until the /investigate response arrives, then swaps in the result.
  });
}
</script>
</body>
</html>
"""


def trace_to_events(trace, node_type: dict[str, str]) -> tuple[list[dict], list[str]]:
    """
    Convert a real HUD agent trace into viz events — the agent's ACTUAL tool calls.

    Walks the trace's tool round-trips in order: each enrich/sec_search adds the
    queried entity as a node (off-path exploration, incl. dead ends), and the final
    submit() draws the claimed path and lights it up. Unlike the synthesized view,
    this shows what the agent really did, backtracks included.

    Parameters:
        trace     : Trace - A HUD run trace (run.trace).
        node_type : dict - name -> 'person'|'company' from the graph (for typing/fallback).

    Returns:
        tuple[list[dict], list[str]] - (events, submitted_path).

    Examples:
        events, path = trace_to_events(run.trace, node_type)
    """
    from hud.agents.types import ToolStep  # lazy: only needed for real runs

    steps = trace.collect(lambda s: s if isinstance(s, ToolStep) else None) or []
    events: list[dict] = []
    added: set[str] = set()
    submit_path: list[str] = []
    submit_cits: list[str] = []

    def _typ(name: str, fallback: str) -> str:
        return node_type.get(name) or fallback

    i = 0
    for ts in steps:
        call = getattr(ts, "call", None)
        if not call or not getattr(call, "name", None):
            continue
        i += 1
        name = call.name
        args = call.arguments or {}

        if name == "submit":
            submit_path = list(args.get("path") or [])
            submit_cits = list(args.get("citations") or [])
            nodes, edges = [], []
            for idx, nd in enumerate(submit_path):
                if nd not in added:
                    nodes.append({"id": nd, "type": _typ(nd, "person" if idx % 2 == 0 else "company")})
                    added.add(nd)
                if idx < len(submit_path) - 1:
                    edges.append({"from": nd, "to": submit_path[idx + 1], "role": "",
                                  "citation": submit_cits[idx] if idx < len(submit_cits) else "", "onpath": True})
            events.append({"i": i, "tool": "submit", "arg": "",
                           "caption": f"submit() — {len(submit_path)} nodes",
                           "nodes": nodes, "edges": edges, "path": submit_path,
                           "citations": [accession_label(c) for c in submit_cits]})
        else:
            q = args.get("name") or args.get("query") or args.get("company") or ""
            nodes = []
            if q and q not in added:
                t = "company" if name == "enrich_company" else _typ(q, "person")
                nodes.append({"id": q, "type": t})
                added.add(q)
            events.append({"i": i, "tool": name, "arg": q,
                           "caption": f"{name}('{q}')", "nodes": nodes, "edges": []})

    # Layout: submitted path on the spine; everything else (exploration / dead ends) fans out.
    gap = 260
    pos: dict[str, tuple[int, int]] = {nd: (k * gap, 0) for k, nd in enumerate(submit_path)}
    others = [n for n in added if n not in pos]
    for k, n in enumerate(others):
        pos[n] = ((k // 2) * gap, (-2 if k % 2 == 0 else 2) * gap)
    for ev in events:
        for n in ev["nodes"]:
            n["x"], n["y"] = pos.get(n["id"], (0, 0))

    return events, submit_path


def run_investigation(person_a: str, person_b: str, model: str = "coi-clean",
                      max_steps: int = 30, timeout: float = 600.0):
    """
    Run a real agent rollout for two names and return its trace (the live inference job).

    Uses the conflict_of_interest template with a placeholder label — the grade is
    ignored; only the agent's trajectory (its real tool calls) is consumed. Slow
    (minutes) and uses the model + SixtyFour/Exa quota; intended for post-training.

    Parameters:
        person_a, person_b : str - The two people to investigate.
        model     : str - Trainable/served model id (e.g. "coi-clean", or "claude").
        max_steps : int - Max agent turns.
        timeout   : float - Per-rollout wall-clock cap (s).

    Returns:
        Trace - The agent run trace (pass to trace_to_events).

    Examples:
        trace = run_investigation("Tim Cook", "Jane Smith")
    """
    import asyncio
    import sys

    from dotenv import load_dotenv
    load_dotenv()
    sys.path.insert(0, str(ROOT))
    from hud import Job
    from hud.agents import create_agent
    from hud.eval import LocalRuntime, Taskset
    from env import conflict_of_interest

    task = conflict_of_interest(person_a=person_a, person_b=person_b,
                                label=False, ground_truth_path=[], ground_truth_citations=[])

    async def _run():
        agent = create_agent(model, max_steps=max_steps, completion_kwargs={"max_tokens": 4096})
        session = await Job.start("coi-demo", group=1)
        await Taskset("coi-demo", [task]).run(
            agent, runtime=LocalRuntime(str(ROOT / "env.py")), job=session,
            max_concurrent=1, rollout_timeout=timeout,
        )
        return session.runs[0].trace

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a COI investigation trace as a standalone HTML file.")
    parser.add_argument("--cases", default="cases_test.json", help="Cases JSON to draw from")
    parser.add_argument("--graph", default=str(GRAPH_PATH), help="Path to graph.json")
    parser.add_argument("--case-index", type=int, default=None, help="Exact case index to visualize")
    parser.add_argument("--person-a", default=None, help="Match a case by person_a substring")
    parser.add_argument("--person-b", default=None, help="Match a case by person_b substring")
    parser.add_argument("--hop", type=int, default=None, help="Preferred hop count (default 2)")
    parser.add_argument("--distractors", type=int, default=2, help="Off-path board members shown per company")
    parser.add_argument("--overlay", default=None, help="Optional SixtyFour overlay JSON (from enrich_overlay.py)")
    parser.add_argument("--ties", default=None, help="Optional LLM-reasoned ties JSON (overrides heuristic)")
    parser.add_argument("--out", default="viz/coi_viz.html", help="Output HTML path")
    args = parser.parse_args()

    cases = json.loads((ROOT / args.cases).read_text()) if not Path(args.cases).is_absolute() \
        else json.loads(Path(args.cases).read_text())
    node_type, adj = load_graph(Path(args.graph))
    case = select_case(
        cases, case_index=args.case_index, person_a=args.person_a,
        person_b=args.person_b, hop=args.hop,
    )
    overlay = None
    if args.overlay:
        ov_path = ROOT / args.overlay if not Path(args.overlay).is_absolute() else Path(args.overlay)
        overlay = json.loads(ov_path.read_text())

    soft_ties = None
    if args.ties:
        ties_path = ROOT / args.ties if not Path(args.ties).is_absolute() else Path(args.ties)
        raw_ties = json.loads(ties_path.read_text())
        width_by_strength = {"strong": 3.0, "moderate": 2.0, "weak": 1.5}
        soft_ties = [{
            "from": t["from"], "to": t["to"],
            "label": t.get("label", ""),
            "title": (t.get("basis", "") + f" [{t.get('strength', 'unverified')}]").strip(),
            "width": width_by_strength.get(t.get("strength", ""), 2.0),
        } for t in raw_ties]

    events = build_events(case, node_type, adj, max_distractors=args.distractors)
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    render_html(events, case, out_path, overlay=overlay, soft_ties=soft_ties)

    hops = (len(case["path"]) - 1) // 2
    print(f"Visualized: {case['person_a']} ↔ {case['person_b']} ({hops}-hop, {len(events)} events)")
    print(f"Path: {' → '.join(case['path'])}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
