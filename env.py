"""COI Investigator HUD environment.

Tools: search/fetch (Exa), enrich_person/enrich_company (SixtyFour), sec_search (EDGAR),
       submit (structured answer).
Template: conflict_of_interest — agent finds the connection path between two people and
          submits it with one citation per edge. Graded deterministically against graph.json.
"""

# NOTE: do NOT add `from __future__ import annotations` here. Under it, a
# `@env.template` param annotated with Literal/alias/model crashes the
# sync/deploy manifest path (TypeAdapter on a string forward-ref). Keep
# annotations as real objects.
import asyncio
import contextlib
import logging
import os
import socket
import sys
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from dotenv import load_dotenv

from hud import Environment
from hud.capabilities import Capability
from hud.graders import EvaluationResult, SubScore

load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[%(levelname)s] %(name)s | %(message)s")
for noisy in ("httpx", "httpcore", "FastMCP", "mcp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("coi-investigator")

env = Environment(name="coi-investigator")

_MCP_PORT: int | None = None
_MCP_SERVER_TASK: asyncio.Task[None] | None = None

# Holds the agent's submitted answer for the current episode
_submitted: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Graph (loaded once at module import; used by grader)
# ---------------------------------------------------------------------------

def _load_graph():
    """
    Load the COI networkx graph from disk, with a stub fallback for testing.

    Returns:
        nx.Graph - The loaded or stub graph.

    Examples:
        G = _load_graph()
        G.number_of_nodes()
        # 1024
    """
    import networkx as nx
    from networkx.readwrite import json_graph
    import json
    from pathlib import Path

    graph_path = Path(__file__).parent / "data" / "graph.json"
    if graph_path.exists():
        with open(graph_path) as f:
            data = json.load(f)
        G = json_graph.node_link_graph(data)
        logger.info("Loaded graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    # Stub graph built from cases_stub.json for smoke-testing before real graph is ready
    stub_path = Path(__file__).parent / "cases_stub.json"
    G = nx.Graph()
    if stub_path.exists():
        with open(stub_path) as f:
            cases = json.load(f)
        for case in cases:
            if case.get("path"):
                path = case["path"]
                citations = case.get("citations") or ["stub"] * (len(path) - 1)
                for i in range(len(path) - 1):
                    node_a, node_b = path[i], path[i + 1]
                    node_type_a = "company" if i % 2 == 1 else "person"
                    node_type_b = "company" if i % 2 == 0 else "person"
                    G.add_node(node_a, type=node_type_a)
                    G.add_node(node_b, type=node_type_b)
                    G.add_edge(node_a, node_b, role="Director", citation=citations[i])
        logger.info("Loaded stub graph from cases_stub.json: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    else:
        logger.warning("No graph.json or cases_stub.json found — grader will use an empty graph")
    return G


G = _load_graph()


# ---------------------------------------------------------------------------
# Exa: web search + fetch
# ---------------------------------------------------------------------------

async def _exa_search(query: str, k: int = 5) -> list[dict[str, str]]:
    key = os.getenv("EXA_API_KEY")
    if not key:
        return [{"message": "Live web search is not configured. Set EXA_API_KEY.", "query": query}]
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"query": query, "numResults": k, "contents": {"text": {"maxCharacters": 800}}},
        )
        r.raise_for_status()
        data = r.json()
    out = [
        {"title": it.get("title", ""), "url": it.get("url", ""), "snippet": (it.get("text") or "")[:200]}
        for it in data.get("results", [])
        if it.get("url")
    ]
    return out or [{"message": "No results found", "query": query}]


async def _exa_fetch(url: str, max_chars: int = 2500) -> str:
    key = os.getenv("EXA_API_KEY")
    if not key:
        return "Live fetch is not configured. Set EXA_API_KEY."
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://api.exa.ai/contents",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"urls": [url], "text": {"maxCharacters": max_chars}},
        )
        r.raise_for_status()
        data = r.json()
    results = data.get("results", [])
    if not results:
        return "No content available for this URL"
    return (results[0].get("text") or "")[:max_chars] or "No content available for this URL"


async def search(query: str) -> list[dict[str, str]]:
    """Search the web for a query. Returns a list of {title, url, snippet}."""
    return await _exa_search(query)


async def fetch(url: str) -> str:
    """Fetch the full text of a web page by its URL (from a prior search result)."""
    return await _exa_fetch(url)


# ---------------------------------------------------------------------------
# SixtyFour: deep person + company enrichment
# ---------------------------------------------------------------------------

_SIXTYFOUR_BASE = "https://api.sixtyfour.ai"
_sixtyfour_cache: dict[str, dict[str, Any]] = {}


async def _sixtyfour_post(path: str, payload: dict[str, Any], timeout: float = 900.0) -> dict[str, Any]:
    key = os.getenv("SIXTYFOUR_API_KEY")
    if not key:
        return {"error": "SixtyFour is not configured. Set SIXTYFOUR_API_KEY."}
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{_SIXTYFOUR_BASE}{path}", headers=headers, json=payload)
        if r.status_code >= 500:
            r = await client.post(f"{_SIXTYFOUR_BASE}{path}", headers=headers, json=payload)
        logger.info("sixtyfour %s -> %s in %.0fs", path, r.status_code, time.monotonic() - t0)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text[:300]
            return {"error": f"SixtyFour returned {r.status_code}", "detail": detail}
        return r.json()


async def enrich_person(name: str, company: str = "", linkedin: str = "") -> dict[str, Any]:
    """Deep-research a person: current role, company, prior roles, board seats, sources.

    Pass company and/or linkedin to disambiguate common names.
    """
    cache_key = f"person:{name.lower()}:{company.lower()}"
    if cache_key in _sixtyfour_cache:
        logger.info("sixtyfour cache hit: %s", cache_key)
        return _sixtyfour_cache[cache_key]

    lead_info: dict[str, str] = {"name": name}
    if company:
        lead_info["company"] = company
    if linkedin:
        lead_info["linkedin"] = linkedin
    struct = {
        "current_role": "Current job title and company",
        "board_seats": {"description": "Other public company boards this person sits on", "type": "list[str]"},
        "prior_companies": {"description": "Notable prior roles or companies", "type": "list[str]"},
        "sources": {"description": "Source URLs the research is based on", "type": "list[str]"},
    }
    result = await _sixtyfour_post("/enrich-lead", {"lead_info": lead_info, "struct": struct, "tier": "micro"})
    if "error" not in result:
        _sixtyfour_cache[cache_key] = result
    return result


async def enrich_company(company: str, website: str = "") -> dict[str, Any]:
    """Deep-research a company: what it does, founders, board members, key executives, sources."""
    cache_key = f"company:{company.lower()}:{website.lower()}"
    if cache_key in _sixtyfour_cache:
        logger.info("sixtyfour cache hit: %s", cache_key)
        return _sixtyfour_cache[cache_key]

    target = f"{company} ({website})" if website else company
    struct = {
        "what_they_do": "One-sentence description of the company",
        "board_members": {"description": "Names of current board members or directors", "type": "list[str]"},
        "key_executives": {"description": "Names and titles of key executive officers", "type": "list[str]"},
        "founders": {"description": "Names of the founders", "type": "list[str]"},
        "sources": {"description": "Source URLs the research is based on", "type": "list[str]"},
    }
    result = await _sixtyfour_post(
        "/company-intelligence", {"target_company": target, "struct": struct, "tier": "micro"}
    )
    if "error" not in result:
        _sixtyfour_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# SEC EDGAR: primary source search
# ---------------------------------------------------------------------------

async def sec_search(name: str) -> dict[str, Any]:
    """Search SEC EDGAR for DEF 14A and 10-K filings mentioning a person or company name.

    Returns up to 5 unique filings with company name, form type, date, accession number,
    and a direct EDGAR URL usable as a citation.

    Parameters:
        name : str - A person name or company name to look up.
    """
    headers = {"User-Agent": "marcusleongjx@gmail.com"}
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        try:
            r = await client.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": f'"{name}"', "forms": "DEF 14A,10-K"},
            )
            r.raise_for_status()
            hits = r.json().get("hits", {}).get("hits", [])

            # Deduplicate by accession number — one filing can have many documents
            seen_adsh: set[str] = set()
            filings = []
            for h in hits:
                src = h.get("_source", {})
                adsh = src.get("adsh", "")
                if not adsh or adsh in seen_adsh:
                    continue
                seen_adsh.add(adsh)

                cik = (src.get("ciks") or [""])[0].lstrip("0")
                display = (src.get("display_names") or [""])[0]
                company = display.split("(")[0].strip() if display else ""
                form = (src.get("root_forms") or src.get("form") or [""])[0]
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/" if cik else ""

                filings.append({
                    "company": company,
                    "form": form,
                    "date": src.get("file_date", ""),
                    "accession_no": adsh,
                    "url": url,
                })
                if len(filings) >= 5:
                    break

            return {"name": name, "filings": filings, "total_hits": len(hits)}

        except Exception as e:
            logger.warning("EDGAR search failed for %r: %s", name, e)
            return {"name": name, "error": str(e), "filings": []}


# ---------------------------------------------------------------------------
# Submit: structured final answer
# ---------------------------------------------------------------------------

async def submit(path: list[str], citations: list[str]) -> str:
    """REQUIRED: Call this tool to finish the task. This is the ONLY valid way to end.

    You MUST call submit() — writing your conclusion as text does not count and scores zero.
    Call it as soon as you have enough evidence, or to report no connection found.

    Parameters:
        path      : Ordered list of alternating person/company names forming the
                    connection path, e.g. ["Jane Smith", "Acme Corp", "Robert Chen"].
                    Pass an empty list [] if no connection exists.
        citations : One citation URL or EDGAR accession number per edge in path
                    (len must equal len(path) - 1). Pass [] for no-connection cases.
    """
    _submitted["path"] = path
    _submitted["citations"] = citations
    logger.info("Agent submitted: path=%s", path)
    return "Submitted. Task complete — stop here and do not call any more tools."


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade_submission(
    submitted: dict[str, Any],
    label: bool,
    ground_truth_path: list[str],
    ground_truth_citations: list[str],
    graph,
) -> EvaluationResult:
    """
    Score the agent's submitted answer against the ground-truth graph.

    Rewards correct abstentions, penalizes fabrications, gives per-edge partial
    credit for correct edges, and adds a bonus for citation coverage.

    Parameters:
        submitted            : dict - The _submitted dict populated by the submit tool.
        label                : bool - True if a real connection exists.
        ground_truth_path    : list[str] - The correct path from cases.json.
        ground_truth_citations : list[str] - Ground-truth citations (unused in scoring but
                                              available for debugging).
        graph                : nx.Graph - The COI graph from graph.json.

    Returns:
        EvaluationResult - HUD result with reward and subscores.

    Examples:
        grade_submission({"path": [], "citations": []}, False, [], [], G)
        # EvaluationResult(reward=1.0)  — correct abstention
    """
    path = submitted.get("path", [])
    citations = submitted.get("citations", [])
    subscores: list[SubScore] = []

    # ── negative cases ────────────────────────────────────────────────────
    if label is False:
        if not path:
            return EvaluationResult(
                reward=1.0,
                content="Correct: agent abstained on a no-connection case.",
                subscores=[SubScore(name="abstention", value=1.0, weight=1.0)],
            )
        else:
            return EvaluationResult(
                reward=0.0,
                content="Incorrect: agent fabricated a connection on a no-connection case.",
                subscores=[SubScore(name="fabrication_penalty", value=0.0, weight=1.0)],
            )

    # ── positive cases — agent submitted nothing ──────────────────────────
    if label is True and not path:
        return EvaluationResult(
            reward=0.0,
            content="Missed: agent submitted no path for a case where a connection exists.",
            subscores=[SubScore(name="missed_connection", value=0.0, weight=1.0)],
        )

    # ── positive cases — score the path ──────────────────────────────────
    n_edges = len(path) - 1

    # Per-edge correctness against graph — fuzzy node matching handles name variants
    # (e.g. "L3Harris Technologies" vs "L3Harris Technologies, Inc.")
    def _best_match(name: str) -> str:
        from rapidfuzz import process
        from rapidfuzz.utils import default_process
        match = process.extractOne(name, list(graph.nodes()), processor=default_process, score_cutoff=80)
        return match[0] if match else name

    edge_scores: list[float] = []
    for i in range(n_edges):
        a = _best_match(path[i])
        b = _best_match(path[i + 1])
        edge_scores.append(1.0 if graph.has_edge(a, b) else 0.0)

    edge_reward = sum(edge_scores) / max(n_edges, 1)
    subscores.append(SubScore(name="edge_correctness", value=edge_reward, weight=0.5))

    # Endpoint bonus: path connects the right two people (fuzzy to handle name variants)
    from rapidfuzz import fuzz
    endpoints_correct = (
        len(path) >= 2
        and len(ground_truth_path) >= 2
        and fuzz.token_sort_ratio(path[0], ground_truth_path[0]) >= 85
        and fuzz.token_sort_ratio(path[-1], ground_truth_path[-1]) >= 85
    )
    subscores.append(SubScore(name="endpoints", value=1.0 if endpoints_correct else 0.0, weight=0.2))

    # Citation coverage: at least one citation per edge
    citation_coverage = min(len(citations), n_edges) / max(n_edges, 1)
    subscores.append(SubScore(name="citation_coverage", value=citation_coverage, weight=0.3))

    reward = sum(s.value * s.weight for s in subscores)
    reward = max(0.0, min(1.0, reward))

    return EvaluationResult(
        reward=reward,
        content=f"Path: {' → '.join(path)} | edge_score={edge_reward:.2f} endpoints={endpoints_correct}",
        subscores=subscores,
    )


# ---------------------------------------------------------------------------
# MCP lifecycle
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _listening(host: str, port: int, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"MCP server never came up on {host}:{port}")


@env.initialize
async def _up() -> None:
    from fastmcp import FastMCP

    global _MCP_PORT, _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is None:
        server = FastMCP(name="coi-tools")
        server.tool(search)
        server.tool(fetch)
        server.tool(enrich_person)
        server.tool(enrich_company)
        server.tool(sec_search)
        server.tool(submit)
        _MCP_PORT = _free_port()
        _MCP_SERVER_TASK = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=_MCP_PORT, show_banner=False)
        )
        await _listening("127.0.0.1", _MCP_PORT)
    env.add_capability(Capability.mcp(name="coi-tools", url=f"http://127.0.0.1:{_MCP_PORT}/mcp"))
    if not os.getenv("EXA_API_KEY"):
        logger.warning("EXA_API_KEY not set; search/fetch tools will return error messages.")
    if not os.getenv("SIXTYFOUR_API_KEY"):
        logger.warning("SIXTYFOUR_API_KEY not set; enrich tools will return error messages.")


@env.shutdown
async def _down() -> None:
    global _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is not None:
        _MCP_SERVER_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _MCP_SERVER_TASK
        _MCP_SERVER_TASK = None


# ---------------------------------------------------------------------------
# Task template
# ---------------------------------------------------------------------------

@env.template()
async def conflict_of_interest(
    person_a: str,
    person_b: str,
    label: bool,
    ground_truth_path: list[str],
    ground_truth_citations: list[str],
) -> AsyncGenerator[Any, Any]:
    """Investigate whether two people share a COI via board/executive connections.

    The agent uses the research tools to trace a connection path, then calls submit()
    with the path and one citation per edge. Graded deterministically against graph.json.
    """
    _submitted.clear()

    yield (
        f"Determine whether {person_a} and {person_b} are connected through shared board "
        "memberships or executive roles at public companies — a potential conflict of interest.\n\n"
        f"They may be directly connected through one shared company, connected through a chain "
        "of intermediaries, or have no connection at all. All three outcomes are equally valid — "
        "do not assume a connection exists. To find indirect connections, search the companies "
        "and people you discover along the way — a chain may require several hops.\n\n"
        "Tools available:\n"
        "  - sec_search(name) — search SEC EDGAR filings for a person or company name. "
        "Returns filing URLs suitable as citations.\n"
        "  - search(query) / fetch(url) — web search and page retrieval for additional context.\n"
        "  - enrich_person(name) / enrich_company(name) — deep research on a person or company; "
        "returns board seats, roles, and affiliations.\n\n"
        "When you have reached a conclusion, call submit() — this is the only way to record your answer:\n"
        "  submit(path=['Person A', 'Company X', 'Person B', ...], citations=['one_url_per_edge', ...]) "
        "if a connection exists.\n"
        "  submit(path=[], citations=[]) ONLY if you are confident no connection exists.\n\n"
        "IMPORTANT — do not give up on a partial trail. If you have found part of a connection "
        "but cannot complete the full chain, submit your best partial path rather than an empty one. "
        "Every correct edge you submit earns credit; an empty path earns nothing when a connection "
        "actually exists. Submit an empty path only when you genuinely believe the two people are "
        "unconnected — never as a way to give up on a hard chain.\n\n"
        "Writing your conclusion as text without calling submit() scores zero."
    )

    result = grade_submission(_submitted, label, ground_truth_path, ground_truth_citations, G)
    logger.info(
        "conflict_of_interest reward=%.3f | %s ↔ %s | label=%s",
        result.reward, person_a, person_b, label,
    )
    yield result
