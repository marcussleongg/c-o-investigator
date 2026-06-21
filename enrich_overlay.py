"""
Post-hoc SixtyFour enrichment overlay for the COI visualization.

This is an ADD-ON that runs *after* the scored investigation, never inside it. The
trained model, the grader, the deployed env, and the before/after eval are all left
untouched — this script only takes the people on a discovered path and asks SixtyFour
for richer, COI-relevant context (education, family, known associates, other boards)
that SEC filings don't capture. The output is supplementary and explicitly
*unverified*: it is rendered as side context in the viz, never scored.

Because it calls the SixtyFour API (which shares quota with live training), results
are cached to disk, and a --demo mode produces clearly-labelled placeholder data so
the viz can be wired up and tested without spending quota. Run it for real only after
training finishes.

Usage:
  python enrich_overlay.py --demo                       # offline placeholder overlay
  python enrich_overlay.py --hop 2                      # real: enrich a 2-hop case's people
  python enrich_overlay.py --person-a "Hwang" --out viz/overlay.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "viz" / "overlay_cache.json"
SIXTYFOUR_BASE = "https://api.sixtyfour.ai"

# Fields SEC filings don't give you — the "soft" relationships that make a COI real.
_PERSON_STRUCT = {
    "summary": "One-sentence description of who this person is",
    "education": {"description": "Universities/colleges attended and degrees", "type": "list[str]"},
    "family_relationships": {
        "description": "Known family members and any notable companies/roles they hold",
        "type": "list[str]",
    },
    "known_associates": {
        "description": "Notable personal/professional associates — co-founders, longtime "
        "colleagues, mentors, college or club connections",
        "type": "list[str]",
    },
    "other_boards": {
        "description": "Nonprofit, private, or advisory boards beyond public-company directorships",
        "type": "list[str]",
    },
    "sources": {"description": "Source URLs the research is based on", "type": "list[str]"},
}


def _load_cache() -> dict[str, dict]:
    """Load the on-disk enrichment cache (name -> overlay entry), or {} if absent."""
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    """Persist the enrichment cache to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _extract(result: dict, key: str):
    """
    Best-effort pull of a struct field from a SixtyFour response.

    SixtyFour may return structured fields at the top level or nested under
    "structured_data"; this checks both.

    Parameters:
        result : dict - The raw SixtyFour JSON response.
        key    : str - The struct field name to extract.

    Returns:
        The field value, or None if not present.

    Examples:
        _extract({"structured_data": {"education": ["MIT"]}}, "education")
        # ['MIT']
    """
    if key in result:
        return result[key]
    sd = result.get("structured_data")
    if isinstance(sd, dict) and key in sd:
        return sd[key]
    return None


async def _sixtyfour_enrich(name: str, company: str = "", timeout: float = 900.0) -> dict:
    """
    Call SixtyFour /enrich-lead for one person with the COI-relevant struct.

    Parameters:
        name    : str - Person's full name.
        company : str - An associated company (from the path) to disambiguate common names.
        timeout : float - HTTP timeout in seconds (SixtyFour deep research is slow).

    Returns:
        dict - An overlay entry: {summary, education, family, associates, other_boards,
        sources, unverified:True}, or {error:...} if the API is unavailable.

    Examples:
        await _sixtyfour_enrich("Angela Hwang", company="United Parcel Service Inc")
        # {'summary': '...', 'education': [...], 'unverified': True, ...}
    """
    key = os.getenv("SIXTYFOUR_API_KEY")
    if not key:
        return {"error": "SIXTYFOUR_API_KEY not set"}

    lead_info = {"name": name}
    if company:
        lead_info["company"] = company
    payload = {"lead_info": lead_info, "struct": _PERSON_STRUCT, "tier": "micro"}
    headers = {"x-api-key": key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{SIXTYFOUR_BASE}/enrich-lead", headers=headers, json=payload)
        if r.status_code >= 500:
            r = await client.post(f"{SIXTYFOUR_BASE}/enrich-lead", headers=headers, json=payload)
        if r.status_code >= 400:
            return {"error": f"SixtyFour returned {r.status_code}"}
        result = r.json()

    return {
        "summary": _extract(result, "summary") or "",
        "education": _extract(result, "education") or [],
        "family": _extract(result, "family_relationships") or [],
        "associates": _extract(result, "known_associates") or [],
        "other_boards": _extract(result, "other_boards") or [],
        "sources": _extract(result, "sources") or [],
        "unverified": True,
    }


def reason_ties(overlay: dict[str, dict], model: str = "claude-sonnet-4-6") -> list[dict]:
    """
    Use an LLM to reason about non-corporate ties across the enriched people.

    Replaces brittle string-intersection with judgment: the model resolves entity
    variants, weighs significance, synthesizes across fields, and explains itself.
    It is grounded — instructed to use ONLY the supplied facts and to allow "no tie".
    This runs post-hoc over the (unverified) SixtyFour overlay and is never scored; it
    does not touch the trained policy or the deterministic grader.

    Parameters:
        overlay : dict - name -> enrichment entry (from SixtyFour or --demo).
        model   : str - Anthropic model id for the reasoning step.

    Returns:
        list[dict] - Each: {from, to, label, basis, strength}. Empty if no API key
        or nothing supported.

    Examples:
        reason_ties({"A": {"education": ["MIT"]}, "B": {"education": ["MIT"]}})
        # [{'from': 'A', 'to': 'B', 'label': 'Co-alumni (MIT)', 'basis': '...', 'strength': 'moderate'}]
    """
    import re as _re

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("  ANTHROPIC_API_KEY not set — skipping LLM tie reasoning")
        return []

    import anthropic

    people = list(overlay)
    facts = {
        p: {k: overlay[p].get(k) for k in ("summary", "education", "family", "associates", "other_boards")}
        for p in people
    }
    prompt = (
        "You are analyzing possible NON-CORPORATE connections between people for a "
        "conflict-of-interest review. Use ONLY the enrichment facts provided; do not use "
        "outside knowledge and do not invent facts.\n\n"
        f"People: {people}\n\n"
        f"Enrichment facts (per person, from a research API — possibly noisy):\n"
        f"{json.dumps(facts, indent=2)}\n\n"
        "For every PAIR of these people, decide whether the facts support a genuine "
        "personal/non-corporate tie (shared school, family relationship, close associate, "
        "shared private/nonprofit board, etc.). Assert a tie ONLY when the facts for BOTH "
        "people support it. Judge significance: drop coincidental overlaps (e.g. both "
        "attended a very large school with no other link); prefer specific, meaningful links.\n\n"
        'Return ONLY a JSON array. Each item: {"from": name, "to": name, "label": short phrase, '
        '"basis": which facts justify it, "strength": "weak"|"moderate"|"strong"}. '
        "Omit pairs with no supported tie. No prose, no markdown."
    )
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}])
        raw = resp.content[0].text.strip()
        raw = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=_re.DOTALL).strip()
        ties = json.loads(raw)
        return [t for t in ties if isinstance(t, dict) and t.get("from") and t.get("to")] if isinstance(ties, list) else []
    except Exception as e:
        print(f"  tie reasoning failed: {e}")
        return []


def _demo_overlay(names: list[str]) -> dict[str, dict]:
    """
    Build a clearly-labelled placeholder overlay with deliberate, sparse overlaps.

    Crafts demo enrichment so a few believable ghost edges emerge instead of a
    clique: the two endpoints share a school and a nonprofit (a tie that visibly
    bridges them), and the first two people share an associate. Everything is
    tagged demo/unverified.

    Parameters:
        names : list[str] - People to enrich (path order; first/last = endpoints).

    Returns:
        dict[str, dict] - name -> overlay entry.

    Examples:
        _demo_overlay(["A", "M", "B"])["A"]["education"]   # ['Stanford University — BA']
    """
    schools = ["Stanford University — BA", "Harvard Business School — MBA",
               "Yale University — JD", "MIT — BS", "University of Chicago — MBA"]
    nonprofits = ["Coastal Education Fund", "Civic Arts Foundation", "Global Health Trust"]
    ov: dict[str, dict] = {}
    for i, p in enumerate(names):
        ov[p] = {
            "summary": f"(demo) Illustrative SixtyFour enrichment for {p}.",
            "education": [schools[i % len(schools)]],
            "family": [],
            "associates": [],
            "other_boards": [f"(demo) {nonprofits[i % len(nonprofits)]}"],
            "sources": ["(demo) https://example.com/profile"],
            "unverified": True,
            "demo": True,
        }
    if len(names) >= 2:
        # Endpoints: shared alma mater + shared nonprofit → a tie that bridges them.
        ov[names[-1]]["education"] = list(ov[names[0]]["education"])
        ov[names[-1]]["other_boards"] = list(ov[names[0]]["other_boards"])
        # First pair: a shared personal associate.
        ov[names[0]]["associates"].append("Pat Lee (former colleague)")
        ov[names[1]]["associates"].append("Pat Lee (former colleague)")
    return ov


def select_people(args) -> list[tuple[str, str]]:
    """
    Resolve which people to enrich, as (name, disambiguating_company) pairs.

    Uses an explicit --names list if given; otherwise selects a case from the cases
    file (by index / name / hop, same logic as build_viz) and returns ONLY the two
    principals under investigation — the path's endpoints — each paired with its
    adjacent company for disambiguation. Intermediaries on the chain are not enriched.

    Parameters:
        args : argparse.Namespace - Parsed CLI args.

    Returns:
        list[tuple[str, str]] - (person_name, company) pairs to enrich (the two endpoints).

    Examples:
        select_people(args)
        # [('Angela Hwang', 'UNITED PARCEL SERVICE INC'), ('Ricardo Fernandez', 'GENERAL MILLS INC')]
    """
    if args.names:
        return [(n.strip(), "") for n in args.names.split(",") if n.strip()]

    import sys
    sys.path.insert(0, str(ROOT / "viz"))
    from build_viz import select_case  # reuse identical case selection

    cases = json.loads((ROOT / args.cases).read_text())
    case = select_case(
        cases, case_index=args.case_index, person_a=args.person_a,
        person_b=args.person_b, hop=args.hop,
    )
    path = case.get("path") or [case["person_a"], case["person_b"]]
    a, b = path[0], path[-1]
    a_co = path[1] if len(path) > 1 else ""
    b_co = path[-2] if len(path) > 2 else a_co  # both endpoints share the company on a 1-hop path
    return [(a, a_co), (b, b_co)]


async def main_async(args) -> None:
    people = select_people(args)
    names = [n for n, _ in people]
    print("enriching:", ", ".join(names))

    if args.demo:
        overlay = _demo_overlay(names)
    else:
        cache = _load_cache()
        overlay = {}
        for name, company in people:
            ck = name.lower()
            if ck in cache and not args.refresh:
                overlay[name] = cache[ck]
                print(f"  cache hit: {name}")
                continue
            print(f"  SixtyFour: {name} ...", flush=True)
            entry = await _sixtyfour_enrich(name, company)
            if "error" in entry:
                print(f"    skipped ({entry['error']})")
                continue
            overlay[name] = entry
            cache[ck] = entry
            _save_cache(cache)

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(overlay, indent=2))
    print(f"wrote {len(overlay)} entries → {out_path}")

    if args.reason:
        ties = reason_ties(overlay, model=args.model)
        ties_path = ROOT / args.ties_out if not Path(args.ties_out).is_absolute() else Path(args.ties_out)
        ties_path.write_text(json.dumps(ties, indent=2))
        print(f"wrote {len(ties)} LLM-reasoned ties → {ties_path}")
        for t in ties:
            print(f"  {t['from']} ⌁ {t['to']} : {t.get('label','')} [{t.get('strength','')}]")
        print(f"render with: python viz/build_viz.py --hop {args.hop or 2} --overlay {args.out} --ties {args.ties_out}")
    else:
        print(f"render with: python viz/build_viz.py --hop {args.hop or 2} --overlay {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc SixtyFour enrichment overlay (not scored).")
    parser.add_argument("--cases", default="cases_test.json", help="Cases JSON to draw people from")
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--person-a", default=None)
    parser.add_argument("--person-b", default=None)
    parser.add_argument("--hop", type=int, default=None, help="Preferred hop count (default 2)")
    parser.add_argument("--names", default=None, help="Comma-separated names to enrich (overrides case selection)")
    parser.add_argument("--demo", action="store_true", help="Write labelled placeholder data (no SixtyFour calls)")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and re-fetch")
    parser.add_argument("--reason", action="store_true", help="Use an LLM to reason ties across the overlay")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic model for tie reasoning")
    parser.add_argument("--ties-out", dest="ties_out", default="viz/ties.json", help="Output for reasoned ties")
    parser.add_argument("--out", default="viz/overlay.json", help="Output overlay JSON")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    load_dotenv()
    main()
