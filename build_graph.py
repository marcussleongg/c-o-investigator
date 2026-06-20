"""
Build the ground-truth COI graph from SEC filings (DEF 14A + 10-K).

Runs five steps in sequence:
  1. Fetch real S&P 500 tickers from Wikipedia, cross-reference with SEC CIK map
  2. Download the most recent DEF 14A and 10-K for each company
  3. Parse each filing type with a dedicated parser:
       DEF 14A → director nominees + their outside affiliations
       10-K    → executive officers + their role at the filing company
  4. Build a bipartite networkx graph (person ↔ company)
  5. Generate labeled positive/negative COI test cases

Outputs:
  data/graph.json  — serialized networkx graph (the answer key)
  data/cases.json  — labeled COI test cases consumed by tasks.py

Usage:
  python build_graph.py [--companies N] [--pos N] [--neg N]
                        [--download-only] [--skip-download]
"""

import argparse
import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path

import anthropic
import html2text
import networkx as nx
import requests
from bs4 import BeautifulSoup
from networkx.readwrite import json_graph
from rapidfuzz import fuzz
from sec_edgar_downloader import Downloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("/Volumes/T7 Shield/c-o-investigator/data")
FILINGS_DIR = DATA_DIR / "filings"
GRAPH_PATH = DATA_DIR / "graph.json"
CASES_PATH = DATA_DIR / "cases.json"

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {"User-Agent": "marcusleongjx@gmail.com"}

FUZZY_THRESHOLD = 88
LLM_MODEL = "claude-haiku-4-5-20251001"
MAX_SECTION_CHARS = 12_000  # ~3k tokens; enough for any director/officer section

_anthropic_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _anthropic_client


# ---------------------------------------------------------------------------
# Step 1 — company list
# ---------------------------------------------------------------------------

def fetch_sp500_tickers() -> list[str]:
    """
    Scrape the current S&P 500 constituent tickers from Wikipedia.

    Parameters:
        None

    Returns:
        list[str] - Ticker symbols for all ~503 S&P 500 constituents.

    Examples:
        fetch_sp500_tickers()[:3]
        # ['MMM', 'AOS', 'ABT']
    """
    log.info("Fetching S&P 500 tickers from Wikipedia...")
    resp = requests.get(WIKIPEDIA_SP500_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # The constituents table has id="constituents"
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError("Could not find S&P 500 constituents table on Wikipedia")

    tickers = []
    for row in table.find_all("tr")[1:]:  # skip header
        cells = row.find_all("td")
        if cells:
            ticker = cells[0].get_text(strip=True).replace(".", "-")  # BRK.B → BRK-B
            tickers.append(ticker)

    log.info(f"Found {len(tickers)} S&P 500 tickers")
    return tickers


def fetch_sp500_ciks(limit: int) -> list[tuple[str, str, str]]:
    """
    Return S&P 500 companies as (ticker, company_name, cik) tuples.

    Fetches the real S&P 500 ticker list from Wikipedia, then cross-references
    with the SEC CIK map. Companies not found in the SEC map are skipped.

    Parameters:
        limit : int - Cap on how many companies to return (useful for testing).
                      Pass 500 (or any large number) to get the full index.

    Returns:
        list[tuple[str, str, str]] - List of (ticker, company_name, cik_padded).

    Examples:
        fetch_sp500_ciks(5)
        # [('AAPL', 'Apple Inc.', '0000320193'), ...]
    """
    sp500_tickers = set(fetch_sp500_tickers())

    log.info("Fetching CIK map from SEC...")
    resp = requests.get(SEC_CIK_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    # Build ticker→(name, cik) lookup; SEC tickers are uppercase
    cik_map: dict[str, tuple[str, str]] = {
        entry["ticker"].upper(): (entry["title"], str(entry["cik_str"]).zfill(10))
        for entry in resp.json().values()
    }

    companies = []
    missing = []
    for ticker in sp500_tickers:
        if ticker.upper() in cik_map:
            name, cik = cik_map[ticker.upper()]
            companies.append((ticker, name, cik))
        else:
            missing.append(ticker)

    if missing:
        log.warning(f"No CIK found for {len(missing)} tickers: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    companies = companies[:limit]
    log.info(f"Matched {len(companies)} S&P 500 companies to CIKs")
    return companies


# ---------------------------------------------------------------------------
# Step 2 — download filings
# ---------------------------------------------------------------------------

def download_filings(companies: list[tuple[str, str, str]], download_dir: Path) -> None:
    """
    Download the most recent DEF 14A and 10-K for each company from EDGAR.

    Parameters:
        companies : list[tuple[str, str, str]] - Output of fetch_sp500_ciks.
        download_dir : Path - Root directory where filings will be saved.

    Returns:
        None

    Examples:
        download_filings([('AAPL', 'Apple Inc.', '0000320193')], Path('data/filings'))
    """
    dl = Downloader("COIInvestigator", "marcusleongjx@gmail.com", str(download_dir))
    for ticker, name, _ in companies:
        for form in ("DEF 14A", "10-K"):
            try:
                dl.get(form, ticker, limit=1)
                log.info(f"  {ticker} {form} OK")
            except Exception as e:
                log.warning(f"  {ticker} {form} failed: {e}")
        time.sleep(0.15)  # SEC rate limit: ~6–8 req/s


# ---------------------------------------------------------------------------
# Step 3 — parsers (LLM-based)
# ---------------------------------------------------------------------------

# Prompts instruct the model to return only JSON — no prose, no markdown fences.
_DEF14A_PROMPT = """You are extracting structured data from a section of an SEC DEF 14A proxy statement filed by {company}.

Extract every director or director nominee mentioned. For each person return a JSON object with:
- "name": their full name
- "role_at_filing_company": their role at {company} (e.g. "Director", "Chair", "Lead Independent Director")
- "other_affiliations": array of OTHER companies they are currently affiliated with, each as {{"company": "...", "role": "..."}}. Include occupations listed under their name, bio descriptions, and any "Other Public Company Directorships" sections.

Return ONLY a valid JSON array with no explanation or markdown."""

_TENK_PROMPT = """You are extracting structured data from the Executive Officers section of an SEC 10-K annual report filed by {company}.

Extract every executive officer listed. For each person return a JSON object with:
- "name": their full name
- "role": their exact title or position at {company}

Return ONLY a valid JSON array with no explanation or markdown."""

# Stop extracting a section when we hit these headers — they signal a new unrelated section
_SECTION_STOP_RE = re.compile(
    r"PROPOSAL\s*[2-9]|COMPENSATION DISCUSSION|AUDIT COMMITTEE REPORT|"
    r"SECURITY OWNERSHIP|EXECUTIVE COMPENSATION|RELATED.PERSON TRANSACTIONS",
    re.IGNORECASE,
)


def _extract_primary_html(submission_path: Path, form_type: str) -> str | None:
    """
    Pull the primary document HTML out of an EDGAR full-submission.txt SGML file.

    EDGAR bundles all filing documents (HTML, XBRL, exhibits) into one SGML
    archive. The primary filing document is the first <DOCUMENT> block whose
    <TYPE> matches the form type we want.

    Parameters:
        submission_path : Path - Path to full-submission.txt.
        form_type : str - The SEC form type to extract, e.g. "10-K" or "DEF 14A".

    Returns:
        str | None - The HTML content of the primary document, or None if not found.

    Examples:
        html = _extract_primary_html(Path('data/.../full-submission.txt'), '10-K')
    """
    try:
        content = submission_path.read_text(errors="ignore")
    except Exception:
        return None

    # Match the first DOCUMENT block of the right type and pull its TEXT content
    pattern = re.compile(
        r"<DOCUMENT>\s*<TYPE>" + re.escape(form_type) + r"[\s\S]*?<TEXT>([\s\S]*?)</TEXT>",
        re.IGNORECASE,
    )
    match = pattern.search(content)
    if match:
        return match.group(1).strip()

    # Fallback: just grab the first TEXT block in the file
    fallback = re.search(r"<TEXT>([\s\S]*?)</TEXT>", content, re.IGNORECASE)
    return fallback.group(1).strip() if fallback else None


def _extract_section_text(html: str, anchor_re: re.Pattern) -> str | None:
    """
    Find a named section in an HTML document and return it as clean plain text.

    Searches the raw HTML string for all anchor matches, takes a fixed-size
    window of HTML after each match, converts to plain text, then scores each
    window by the density of person-name-like patterns. The highest-scoring
    window is returned. This approach is robust to iXBRL nesting and TOC entries,
    which score low because they contain mostly page numbers and links.

    Parameters:
        html : str - Full HTML document content.
        anchor_re : re.Pattern - Regex matching the section header text.

    Returns:
        str | None - Plain-text content of the best-matching section window.

    Examples:
        text = _extract_section_text(html, re.compile(r"PROPOSAL 1", re.I))
    """
    positions = [m.start() for m in re.finditer(anchor_re, html)]
    if not positions:
        return None

    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0

    # HTML is ~6× more verbose than plain text, so take a 6× window
    window_html_chars = MAX_SECTION_CHARS * 6

    best_text: str | None = None
    best_score: int = -1

    for pos in positions:
        window = html[pos: pos + window_html_chars]
        text = converter.handle(window)[:MAX_SECTION_CHARS]

        # Score = proper-name density minus page-number noise near the start
        proper_names = len(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text))
        page_numbers = len(re.findall(r"\b\d{1,3}\b", text[:300]))
        score = proper_names - page_numbers

        if score > best_score:
            best_score = score
            best_text = text

    return best_text


def _llm_extract(section_text: str, prompt: str, cache_key: str) -> list[dict]:
    """
    Call Claude Haiku to extract structured data from a filing section.

    Results are cached to data/llm_cache/ by cache_key so re-runs don't
    re-call the API for already-processed sections.

    Parameters:
        section_text : str - Plain-text content of the filing section.
        prompt : str - Fully formatted extraction prompt.
        cache_key : str - Unique key for caching (derived from file path + form type).

    Returns:
        list[dict] - Parsed JSON array from the LLM, or [] on failure.

    Examples:
        records = _llm_extract(section_text, prompt, "AAPL-DEF14A")
    """
    cache_dir = DATA_DIR / "llm_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{hashlib.md5(cache_key.encode()).hexdigest()}.json"

    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    try:
        resp = _get_client().messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt + "\n\n" + section_text}],
        )
        raw = resp.content[0].text.strip()

        # Strip markdown code fences if the model adds them despite the prompt
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

        result = json.loads(raw)
        if isinstance(result, list):
            cache_file.write_text(json.dumps(result))
            return result
    except Exception as e:
        log.warning(f"LLM extraction failed ({cache_key}): {e}")

    return []


def parse_def14a(submission_path: Path, filing_company: str) -> list[tuple[str, str, str, str]]:
    """
    Extract person→company edges from a DEF 14A full-submission.txt file.

    Pulls the primary HTML from the SGML bundle, locates the director nominees
    section, and uses Claude Haiku to extract names, roles, and outside
    affiliations. Each person yields one edge to the filing company plus one
    edge per outside affiliation.

    Parameters:
        submission_path : Path - Path to the full-submission.txt SGML file.
        filing_company : str - The company that filed this proxy.

    Returns:
        list[tuple[str, str, str, str]] - (person, company, role, citation) records.

    Examples:
        parse_def14a(Path('.../full-submission.txt'), 'Microsoft Corp')
        # [('Reid G. Hoffman', 'Microsoft Corp', 'Director', '...'),
        #  ('Reid G. Hoffman', 'Greylock Partners', 'Partner', '...')]
    """
    html = _extract_primary_html(submission_path, "DEF 14A")
    if not html:
        return []

    anchor_re = re.compile(
        r"PROPOSAL\s*1|ELECTION OF DIRECTORS|DIRECTOR NOMINEES|"
        r"NOMINEES FOR DIRECTOR|OUR DIRECTOR NOMINEES",
        re.IGNORECASE,
    )
    section_text = _extract_section_text(html, anchor_re)
    if not section_text:
        log.warning(f"  No director nominees section found in {submission_path.parent.name}")
        return []

    prompt = _DEF14A_PROMPT.format(company=filing_company)
    cache_key = f"{submission_path}-DEF14A"
    extracted = _llm_extract(section_text, prompt, cache_key)

    citation = str(submission_path)
    records: list[tuple[str, str, str, str]] = []

    for person in extracted:
        name = str(person.get("name", "")).strip()
        if len(name) < 4:
            continue
        role = str(person.get("role_at_filing_company", "Director")).strip()
        records.append((name, filing_company, role, citation))

        for aff in person.get("other_affiliations", []):
            aff_co = str(aff.get("company", "")).strip()
            aff_role = str(aff.get("role", "")).strip()
            if aff_co and aff_role and fuzz.token_sort_ratio(aff_co, filing_company) < 85:
                records.append((name, aff_co, aff_role, citation))

    return records


def parse_10k(submission_path: Path, filing_company: str) -> list[tuple[str, str, str, str]]:
    """
    Extract executive officer→company edges from a 10-K full-submission.txt file.

    Pulls the primary HTML from the SGML bundle, locates the Executive Officers
    section, and uses Claude Haiku to extract names and titles.

    Parameters:
        submission_path : Path - Path to the full-submission.txt SGML file.
        filing_company : str - The company that filed this 10-K.

    Returns:
        list[tuple[str, str, str, str]] - (person, company, role, citation) records.

    Examples:
        parse_10k(Path('.../full-submission.txt'), 'Apple Inc.')
        # [('Tim Cook', 'Apple Inc.', 'Chief Executive Officer', '...')]
    """
    html = _extract_primary_html(submission_path, "10-K")
    if not html:
        return []

    anchor_re = re.compile(
        r"EXECUTIVE OFFICERS OF THE REGISTRANT|"
        r"INFORMATION ABOUT.{0,30}EXECUTIVE OFFICERS|"
        r"EXECUTIVE OFFICERS AND DIRECTORS|"
        r"EXECUTIVE OFFICERS",
        re.IGNORECASE,
    )
    section_text = _extract_section_text(html, anchor_re)
    if not section_text:
        log.warning(f"  No executive officers section found in {submission_path.parent.name}")
        return []

    prompt = _TENK_PROMPT.format(company=filing_company)
    cache_key = f"{submission_path}-10K"
    extracted = _llm_extract(section_text, prompt, cache_key)

    citation = str(submission_path)
    records: list[tuple[str, str, str, str]] = []

    for person in extracted:
        name = str(person.get("name", "")).strip()
        role = str(person.get("role", "")).strip()
        if len(name) >= 4 and role:
            records.append((name, filing_company, role, citation))

    return records


def collect_all_people(
    companies: list[tuple[str, str, str]], download_dir: Path
) -> list[tuple[str, str, str, str]]:
    """
    Walk downloaded filings and extract all (person, company, role, citation) records.

    Looks for full-submission.txt files (EDGAR SGML format) under each ticker's
    DEF 14A and 10-K directories, then calls the appropriate parser.

    Parameters:
        companies : list[tuple[str, str, str]] - Output of fetch_sp500_ciks.
        download_dir : Path - Root directory of downloaded filings.

    Returns:
        list[tuple[str, str, str, str]] - Combined records from both filing types.

    Examples:
        records = collect_all_people([('MSFT', 'Microsoft Corp', '...')], Path('data/filings'))
    """
    edgar_root = download_dir / "sec-edgar-filings"
    records: list[tuple[str, str, str, str]] = []

    for ticker, company_name, _ in companies:
        ticker_dir = edgar_root / ticker

        for form, parser in (("DEF 14A", parse_def14a), ("10-K", parse_10k)):
            form_dir = ticker_dir / form
            if not form_dir.exists():
                continue

            # Take the most recent filing directory; skip macOS metadata (._*) entries
            filing_dirs = sorted(
                [d for d in form_dir.iterdir() if d.is_dir() and not d.name.startswith("._")],
                reverse=True,
            )
            if not filing_dirs:
                continue

            submission = filing_dirs[0] / "full-submission.txt"
            if not submission.exists():
                log.warning(f"  {ticker} {form}: no full-submission.txt found")
                continue

            new_records = parser(submission, company_name)
            if new_records:
                records.extend(new_records)
                log.info(f"  {ticker} {form}: {len(new_records)} records")
            else:
                log.warning(f"  {ticker} {form}: 0 records extracted")

    log.info(f"Total raw records: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Step 4 — entity resolution + build graph
# ---------------------------------------------------------------------------

def resolve_name(name: str, known_names: list[str]) -> str:
    """
    Return a matching canonical name from known_names, or name itself if no match.

    Parameters:
        name : str - The name to resolve.
        known_names : list[str] - Already-seen canonical names.

    Returns:
        str - The canonical name.

    Examples:
        resolve_name('Jane A. Smith', ['Jane Smith'])
        # 'Jane Smith'
    """
    for known in known_names:
        if fuzz.token_sort_ratio(name, known) >= FUZZY_THRESHOLD:
            return known
    return name


def build_graph(records: list[tuple[str, str, str, str]]) -> nx.Graph:
    """
    Build a bipartite networkx graph from (person, company, role, citation) records.

    Nodes carry a `type` attribute ('person' or 'company').
    Edges carry `role` and `citation` attributes.
    Duplicate person-company edges accumulate roles; first citation is kept.

    Parameters:
        records : list[tuple[str, str, str, str]] - Output of collect_all_people.

    Returns:
        nx.Graph - Bipartite graph.

    Examples:
        G = build_graph(records)
        G.nodes['Tim Cook']         # {'type': 'person'}
        G['Tim Cook']['Apple Inc.'] # {'role': 'CEO', 'citation': '...'}
    """
    G = nx.Graph()
    canonical_names: list[str] = []

    for raw_name, company, role, citation in records:
        canonical = resolve_name(raw_name, canonical_names)
        if canonical == raw_name and raw_name not in canonical_names:
            canonical_names.append(raw_name)

        G.add_node(canonical, type="person")
        G.add_node(company, type="company")

        if G.has_edge(canonical, company):
            existing_role = G[canonical][company]["role"]
            if role not in existing_role:
                G[canonical][company]["role"] = existing_role + "; " + role
        else:
            G.add_edge(canonical, company, role=role, citation=citation)

    n_people = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "person")
    n_companies = G.number_of_nodes() - n_people
    log.info(
        f"Graph: {n_people} people, {n_companies} companies, {G.number_of_edges()} edges"
    )
    return G


# ---------------------------------------------------------------------------
# Step 5 — generate labeled test cases
# ---------------------------------------------------------------------------

def generate_cases(
    G: nx.Graph,
    n_positive: int = 200,
    n_negative: int = 200,
    min_path: int = 3,
    max_path: int = 7,
    seed: int = 42,
) -> list[dict]:
    """
    Generate labeled COI test cases by sampling person pairs from the graph.

    Positive cases: shortest path exists with length in [min_path, max_path].
    Negative cases: no path exists between the two people.
    Path length is in nodes (3 = one company hop, 5 = two company hops).

    Parameters:
        G : nx.Graph - The COI graph from build_graph.
        n_positive : int - Target number of positive cases.
        n_negative : int - Target number of negative cases.
        min_path : int - Minimum path length in nodes.
        max_path : int - Maximum path length in nodes.
        seed : int - Random seed for reproducibility.

    Returns:
        list[dict] - Each dict: person_a, person_b, label, path, citations.

    Examples:
        cases = generate_cases(G, n_positive=5, n_negative=5)
        cases[0]
        # {'person_a': 'Jane Smith', 'person_b': 'Robert Chen',
        #  'label': True, 'path': ['Jane Smith', 'Blackrock', 'Robert Chen'],
        #  'citations': ['filing1.htm', 'filing2.htm']}
    """
    random.seed(seed)
    people = [n for n, d in G.nodes(data=True) if d.get("type") == "person"]
    log.info(f"Sampling cases from {len(people)} people...")

    positive: list[dict] = []
    negative: list[dict] = []
    seen: set[frozenset] = set()
    max_attempts = (n_positive + n_negative) * 100

    for _ in range(max_attempts):
        if len(positive) >= n_positive and len(negative) >= n_negative:
            break
        a, b = random.sample(people, 2)
        pair = frozenset([a, b])
        if pair in seen:
            continue
        seen.add(pair)

        try:
            path = nx.shortest_path(G, a, b)
        except nx.NetworkXNoPath:
            if len(negative) < n_negative:
                negative.append({"person_a": a, "person_b": b, "label": False, "path": None, "citations": None})
            continue

        if len(positive) < n_positive and min_path <= len(path) <= max_path:
            citations = [G[path[i]][path[i + 1]].get("citation", "") for i in range(len(path) - 1)]
            positive.append({"person_a": a, "person_b": b, "label": True, "path": path, "citations": citations})

    log.info(f"Generated {len(positive)} positive, {len(negative)} negative cases")
    cases = positive + negative
    random.shuffle(cases)
    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build COI ground-truth graph from SEC filings")
    parser.add_argument("--companies", type=int, default=500, help="Number of S&P 500 companies to process (default: all)")
    parser.add_argument("--pos", type=int, default=200, help="Target positive cases")
    parser.add_argument("--neg", type=int, default=200, help="Target negative cases")
    parser.add_argument("--download-only", action="store_true", help="Download filings then exit (no parsing)")
    parser.add_argument("--skip-download", action="store_true", help="Skip filing download, go straight to parsing")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    FILINGS_DIR.mkdir(exist_ok=True)

    companies = fetch_sp500_ciks(args.companies)

    if not args.skip_download:
        log.info(f"Downloading DEF 14A + 10-K for {len(companies)} companies...")
        download_filings(companies, FILINGS_DIR)

    if args.download_only:
        log.info("--download-only: done.")
        return

    records = collect_all_people(companies, FILINGS_DIR)
    if not records:
        log.error("No records parsed — check filing downloads and HTML structure.")
        return

    G = build_graph(records)

    with open(GRAPH_PATH, "w") as f:
        json.dump(json_graph.node_link_data(G), f, indent=2)
    log.info(f"Graph saved → {GRAPH_PATH}")

    cases = generate_cases(G, n_positive=args.pos, n_negative=args.neg)
    with open(CASES_PATH, "w") as f:
        json.dump(cases, f, indent=2)
    log.info(f"Cases saved → {CASES_PATH} ({len(cases)} total)")


if __name__ == "__main__":
    main()
