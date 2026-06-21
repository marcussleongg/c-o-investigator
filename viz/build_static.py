"""
Build a static, Vercel-deployable version of the COI demo.

Generates viz/dist/:
  index.html      landing page — pre-rendered example gallery + a live-search form
  <slug>.html     one fully self-contained animated investigation per demo pair

The static site CANNOT run agent inference (Vercel serverless can't host a multi-minute
rollout + MCP tool server). So the example pages are pre-rendered and fully static,
while the live-search form navigates to a LOCAL backend you run yourself
(`python viz/server.py`, default http://localhost:8000). The landing page states this
explicitly. Top-level navigation to localhost is not blocked by HTTPS mixed-content
rules, so no CORS setup is needed — the local server just has to be running.

Usage:
  python viz/build_static.py                                   # examples from cases_test.json
  python viz/build_static.py --pair "Tim Cook" "Jane Smith"    # add custom pairs
  python viz/build_static.py --backend http://localhost:8000   # set the live backend URL
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "viz"))
import build_viz  # noqa: E402
import server as srv  # noqa: E402  (reuse the tested synth Investigator + _slug)

DIST = ROOT / "viz" / "dist"


def default_pairs() -> list[tuple[str, str]]:
    """
    Pick one representative pair per hop tier from the held-out test set.

    Returns:
        list[tuple[str, str]] - (person_a, person_b) pairs covering 1-hop, 2-hop,
        3-hop, and a negative, in that order (skipping any tier not present).

    Examples:
        default_pairs()
        # [('John A. Bryant', 'Mandy Glew'), ('Angela Hwang', 'Ricardo Fernandez'), ...]
    """
    cases = json.loads((ROOT / "cases_test.json").read_text())
    pick: dict = {}
    for c in cases:
        tier = "neg" if not c.get("path") else (len(c["path"]) - 1) // 2
        pick.setdefault(tier, (c["person_a"], c["person_b"]))
    return [pick[k] for k in (1, 2, 3, "neg") if k in pick]


_HOP_LABEL = {0: "no connection", 1: "1-hop · direct interlock",
              2: "2-hop · indirect tie", 3: "3-hop · distant link"}


def _index_html(cards: list[dict], backend: str) -> str:
    """Render the landing page: example gallery + live-search form."""
    card_html = "\n".join(
        f'<a class="card" href="{c["slug"]}.html">'
        f'<div class="names">{html.escape(c["a"])} <span class="x">↔</span> {html.escape(c["b"])}</div>'
        f'<div class="tag">{c["label"]}</div></a>'
        for c in cards
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>COI Investigator — demo</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3; --muted:#8b949e;
          --person:#2dd4bf; --company:#d29922; --path:#2ea043; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:var(--bg); color:var(--text); padding:40px 24px; max-width:860px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .lede {{ color:var(--muted); font-size:14px; margin-bottom:28px; }}
  h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted);
        border-bottom:1px solid var(--line); padding-bottom:8px; margin:32px 0 16px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  .card {{ display:block; background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:16px; text-decoration:none; color:var(--text); transition:border-color .15s; }}
  .card:hover {{ border-color:var(--person); }}
  .card .names {{ font-weight:600; font-size:14px; }}
  .card .x {{ color:var(--muted); margin:0 4px; }}
  .card .tag {{ color:var(--muted); font-size:12px; margin-top:6px; }}
  .live {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px; }}
  .live .row {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
  .live input {{ background:#0d1117; border:1px solid var(--line); color:var(--text);
                border-radius:6px; padding:8px 12px; font-size:14px; min-width:200px; }}
  .live button {{ background:#238636; border:1px solid #2ea043; color:#fff; border-radius:6px;
                 padding:8px 16px; font-size:14px; cursor:pointer; }}
  .note {{ color:var(--muted); font-size:12.5px; line-height:1.5; margin-top:12px; }}
  .note code {{ background:#0d1117; border:1px solid var(--line); border-radius:4px; padding:1px 6px; color:var(--person); }}
  .backrow {{ display:flex; gap:8px; align-items:center; margin-top:10px; }}
  .backrow input {{ flex:1; background:#0d1117; border:1px solid var(--line); color:var(--muted);
                   border-radius:6px; padding:6px 10px; font-size:12px; }}
</style></head>
<body>
  <h1>COI Investigator</h1>
  <div class="lede">Does a chain of board seats and executive roles connect two people?
    A verifiable RL environment for hidden-graph investigation, traced against SEC filings.</div>

  <h2>Live search <span style="text-transform:none;letter-spacing:0">— requires the local backend</span></h2>
  <div class="live">
    <form class="row" onsubmit="return goLive(event)">
      <input id="la" placeholder="Person A" required>
      <span style="color:var(--muted)">↔</span>
      <input id="lb" placeholder="Person B" required>
      <button type="submit">Investigate (live)</button>
    </form>
    <div class="note">
      Live investigation runs on a <b>backend you host locally</b> — the static site can't
      run agent inference. Start it with <code>python viz/server.py</code> (or
      <code>--real</code> for the trained agent), then submit above. If the backend isn't
      running, the link won't resolve; the pre-rendered examples below always work.
      <div class="backrow"><span>backend:</span><input id="be" value="{html.escape(backend)}"></div>
    </div>
  </div>

  <h2>Pre-rendered examples</h2>
  <div class="grid">{card_html}</div>

<script>
  const beField = document.getElementById('be');
  beField.value = localStorage.getItem('coi_backend') || beField.value;
  beField.addEventListener('change', () => localStorage.setItem('coi_backend', beField.value.trim()));
  function goLive(e) {{
    e.preventDefault();
    const a = document.getElementById('la').value.trim();
    const b = document.getElementById('lb').value.trim();
    if (!a || !b) return false;
    const backend = (beField.value.trim() || '{html.escape(backend)}').replace(/\\/$/, '');
    // top-level navigation to the local backend (not a fetch — avoids mixed-content/CORS)
    window.location.href = backend + '/investigate?a=' + encodeURIComponent(a) + '&b=' + encodeURIComponent(b);
    return false;
  }}
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static, Vercel-deployable demo site.")
    parser.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                        help="Add a custom example pair (repeatable)")
    parser.add_argument("--backend", default="http://localhost:8000", help="Local backend URL for live search")
    args = parser.parse_args()

    pairs = args.pair or default_pairs()
    inv = srv.Investigator(real=False)
    DIST.mkdir(parents=True, exist_ok=True)

    cards: list[dict] = []
    for a, b in pairs:
        events, case = inv.investigate(a, b)
        slug = srv._slug(case.get("person_a", a), case.get("person_b", b))
        back = '<a class="back" href="index.html" style="color:#8b949e;text-decoration:none">← All examples</a>'
        html_doc = build_viz.build_html(events, case, form_html=back, ready=False)
        (DIST / f"{slug}.html").write_text(html_doc)
        p = case.get("path") or []
        hop = (len(p) - 1) // 2 if p else 0
        cards.append({"slug": slug, "a": case.get("person_a", a), "b": case.get("person_b", b),
                      "label": _HOP_LABEL.get(hop, f"{hop}-hop")})
        print(f"  {a} ↔ {b}  ->  {slug}.html ({_HOP_LABEL.get(hop, hop)})")

    (DIST / "index.html").write_text(_index_html(cards, args.backend))
    print(f"\nwrote {len(cards)} examples + index → {DIST}")
    print("deploy: `cd viz/dist && vercel deploy --prod`  (or drag the folder into vercel.com)")


if __name__ == "__main__":
    main()
