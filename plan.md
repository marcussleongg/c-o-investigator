# RSI Hackathon — COI Discovery RL Environment

## What you're building (the big picture)

An RL environment where an AI agent plays investigator. It gets a task — "find if Person A and Person B have a conflict of interest" — and uses research tools to discover a chain of connections between them. You hold a secret answer key derived from public SEC records. The agent never sees it. You score the agent's work against it.

The graph is your answer key. The agent never sees the graph. Everything below either builds the answer key, builds the harness around it, or runs the loop and proves it learned.

---

## The dependency chain

```
Stage 0 (ground-truth graph)
    └── Stage 1 (tool layer)      ← can build in parallel with Stage 2
    └── Stage 2 (verifier)        ← can build in parallel with Stage 1
            └── Stage 3 (close the loop)
                    └── Stage 4 (train — optional, attempt if Stage 3 is solid by hour ~14)

Stage 5 (visualization) runs in parallel throughout
```

**Team split (2–3 people):**
- Person 1: Stage 0 → graph
- Person 2: Stages 1–2 → harness
- Person 3: Stage 5 → demo
- Everyone converges on Stage 3

---

## Stage 0 — Build the answer key

### What you're doing

SEC filings are public documents where companies must disclose who sits on their boards and who their executives are. You pull this data and build a graph:

- **Nodes:** people and companies
- **Edges:** relationships (`Alice —[director of]→ Acme Corp`, `Bob —[CFO of]→ Acme Corp`)

From this graph you generate labeled test cases. You find two people, check whether there's a multi-hop path between them in the graph, and package that as an example:

```
person_A:               "Jane Smith"
person_B:               "Robert Chen"
label:                  True
ground_truth_path:      Jane Smith → Blackrock → Palantir → Robert Chen
ground_truth_citations: [SEC filing 1, SEC filing 2]
```

You generate both **positive cases** (real path exists) and **negative cases** (no path, or a path so long it's implausible).

### The principle

This is your reward signal in latent form. The networkx graph you build here is what Stage 2's verifier queries. Without it, nothing else can be verified or tested.

### Scope-cut lever

If SEC parsing is slow or fiddly, fall back to **OpenAlex co-authorship** (clean graph, person↔person via shared papers) to prove the mechanism and note SEC as the "real" target. A working loop on a simpler graph beats a half-built graph with no loop.

---

### Stage 0 in detail — how to actually build the dataset

There are three distinct sub-problems: getting the raw data, parsing it into structured edges, and generating test cases from the graph. Budget **3–4 hours** for this stage. If you hit hour 3 and SEC parsing is still broken, switch to OpenAlex immediately.

#### Step 1 — Get a company list

Start with the S&P 500. Small enough to be tractable, large enough to produce a dense graph with interesting multi-hop paths. What you need from it is each company's **CIK number** — EDGAR's internal identifier.

SEC publishes a full ticker→CIK mapping as a JSON file you can download once and use as a lookup table:

```
https://www.sec.gov/files/company_tickers.json
```

#### Step 2 — Download the filings

The filing type you want is **DEF 14A** (proxy statements) — filed annually, contains every director and executive officer by name. You only need the most recent filing per company.

Install one of these:

```bash
pip install edgartools           # newer, nicer API
pip install sec-edgar-downloader # simpler, battle-tested
```

With `edgartools`:

```python
from edgar import Company

company = Company("Apple Inc", "0000320193")
filing = company.get_filings(form="DEF 14A").latest(1)
doc = filing.obj()
```

With `sec-edgar-downloader`:

```python
from sec_edgar_downloader import Downloader

dl = Downloader("YourName", "your@email.com", "./data")
dl.get("DEF 14A", "AAPL", limit=1)
```

Aim for 100–200 companies to start — that gives a working graph. You don't need all 500.

#### Step 3 — Parse names and roles out of the filings

Proxy statements are HTML. The section you want is usually titled "DIRECTORS AND EXECUTIVE OFFICERS" or "PROPOSAL 1: ELECTION OF DIRECTORS". Most contain a table like:

```
Name          | Age | Position              | Since
Jane Smith    | 58  | Director              | 2018
Robert Chen   | 61  | Director, CEO         | 2015
```

Parse with BeautifulSoup:

```python
from bs4 import BeautifulSoup
import re

def extract_directors(html_text):
    soup = BeautifulSoup(html_text, "lxml")
    headers = soup.find_all(string=re.compile(
        r"DIRECTORS AND EXECUTIVE OFFICERS|ELECTION OF DIRECTORS",
        re.IGNORECASE
    ))
    # Find the nearest table after the header and extract name + role columns
```

The parsing will break on some filings — that's fine. Aim for 70–80% coverage. If a filing is malformed, skip and move on.

**Entity resolution:** The same person appears as "Jane A. Smith" in one filing and "Jane Smith" in another. Use fuzzy matching to deduplicate:

```bash
pip install rapidfuzz
```

```python
from rapidfuzz import fuzz

def same_person(name_a, name_b):
    return fuzz.token_sort_ratio(name_a, name_b) > 88
```

#### Step 4 — Build the graph

```python
import networkx as nx

G = nx.Graph()

def add_edge(person_name, company_name, role, filing_accession_number):
    G.add_node(person_name, type="person")
    G.add_node(company_name, type="company")
    G.add_edge(person_name, company_name, role=role, citation=filing_accession_number)
```

This is a **bipartite graph** — people connect to companies, companies connect to people. A conflict-of-interest path looks like:

```
Jane Smith → [Blackrock] → Robert Chen → [Palantir] → Alice Wong
  person        company       person         company      person
```

That's a path of length 4 (4 edges). A 2-hop COI (one shared company) is path length 2. Store each edge's EDGAR accession number — that becomes the ground-truth citation.

Serialize the graph to disk once built so you don't re-parse:

```python
from networkx.readwrite import json_graph
import json

with open("graph.json", "w") as f:
    json.dump(json_graph.node_link_data(G), f)
```

#### Step 5 — Generate labeled test cases

```python
import random

def generate_cases(G, n_positive=200, n_negative=200):
    people = [n for n, d in G.nodes(data=True) if d["type"] == "person"]
    cases = []

    for _ in range(n_positive * 5):  # oversample, filter to target length
        a, b = random.sample(people, 2)
        try:
            path = nx.shortest_path(G, a, b)
            if 2 <= len(path) <= 6:  # 1–3 company hops; 4 is the sweet spot
                citations = [G[path[i]][path[i+1]]["citation"]
                             for i in range(len(path) - 1)]
                cases.append({
                    "person_a": a, "person_b": b,
                    "label": True, "path": path, "citations": citations
                })
        except nx.NetworkXNoPath:
            continue
        if len([c for c in cases if c["label"]]) >= n_positive:
            break

    for _ in range(n_negative * 5):
        a, b = random.sample(people, 2)
        if not nx.has_path(G, a, b):
            cases.append({
                "person_a": a, "person_b": b,
                "label": False, "path": None, "citations": None
            })
        if len([c for c in cases if not c["label"]]) >= n_negative:
            break

    return cases
```

**Difficulty tuning:** path length 2 (one shared company) is trivial. Path length 4 (two companies, one intermediary person) is the sweet spot — hard enough to require multi-hop reasoning, tractable enough for the agent to succeed sometimes. Generate mostly length-4 cases. Path length 6+ is probably too hard for a hackathon demo.

Save cases to `cases.json` — this is consumed directly by `tasks.py` in Stage 3.

---

## Stage 1 — Give the agent hands

### What you're doing

You wrap SixtyFour and Exa as callable functions the agent can invoke:

**SixtyFour tool — `enrich(entity_name)`**
Agent gives it a name, it returns related entities and sources. This is how the agent traverses the graph it can't see — it discovers edges one hop at a time.

**Exa tool — `corroborate(claim)`**
Agent gives it a specific claim like "Jane Smith sits on Blackrock's board" and it finds primary source documents that confirm or refute it. This is how the agent gets citable evidence.

**SEC EDGAR tool — `sec_search(name)`**
Takes either a company name or a person name. Always goes through EDGAR directly — no pre-built maps.

For a **company name**, the flow is:
1. Query EDGAR entity search to resolve name → CIK: `https://efts.sec.gov/LATEST/search-index?q=&entity={company_name}&forms=DEF+14A,10-K`
2. Use the CIK to fetch the full filing history: `https://data.sec.gov/submissions/CIK{cik}.json`
3. Return the most recent DEF 14A and 10-K accession numbers as citable sources

For a **person name**, query EDGAR full-text search to find which filings mention them: `https://efts.sec.gov/LATEST/search-index?q="Person+Name"&forms=DEF+14A,10-K`

Returns filing accession numbers the agent can cite. This is the primary source layer — EDGAR citations are the strongest possible evidence since they're the same source the Stage 0 ground truth was built from. This flow works for any company, not just S&P 500 — making it valid in production for private or international companies where the agent has to discover structure from scratch.

**Submit action — `submit(path, citations)`**
The agent says "I'm done, here's my answer" — a claimed connection path plus the sources backing each edge.

After each tool call, the agent gets back an **observation**: what new entities were found, what sources are now available, what's now reachable from where it started.

### The principle

This is the action space. The agent's job is deciding which entity to enrich next, when it has enough evidence to submit, and when to say "no connection found." These decisions are what the RL is trying to optimize. Keep the tools thin — they're API wrappers, not products.

---

## Stage 2 — Build the judge

### What you're doing

When the agent submits, you run its answer through a scoring function that checks it against the Stage 0 graph:

| What happened | Score |
|---|---|
| Claimed edge exists in ground-truth graph | +reward |
| Claimed edge doesn't exist (hallucinated) | -penalty |
| Citation actually supports its claimed edge | +reward |
| Citation doesn't support its claimed edge | heavy penalty |
| Path connects A to B end-to-end | bonus |
| Correctly says "no connection" on a negative case | +reward |
| Fabricates a connection on a negative case | heavy penalty |
| Each tool call made | small -cost |

### The principle

Two things make this design defensible:

1. **Per-edge partial credit** — a 3-hop path that gets 2 hops right still gets signal, instead of scoring 0 for being incomplete. This makes learning tractable.
2. **Citation-gating** — the agent can't just guess the right path and claim it; each edge must be backed by a real source. This forces genuine reasoning rather than pattern-matching on the answer key.

This is the defensible core of the whole project. The per-edge decomposition is what a sharp judge will probe — it's where your rigor needs to show.

---

## Stage 3 — Close the loop

### What you're doing

HUD is the RL framework. You register your environment into it:

- Tools from Stage 1 → action space
- Scoring function from Stage 2 → evaluation logic
- Cases from Stage 0 → tasks

You wire Claude in as the agent via the model gateway and run full rollouts:

```
task given → agent calls tools → agent submits → verifier scores → reward returned → repeat
```

You're checking:
- Do trajectories complete?
- Do tools fire correctly?
- Are rewards sensible — not all 0, not all 1?

If rewards are all 0, your cases are too hard. If all 1, too easy. Re-tune case difficulty here.

### The principle

This is the first moment the machine runs end-to-end. Before this you had components. Here you have a closed loop. **This is a legitimate stopping point** — a working environment + verifier + sampled successful/failed trajectories is a submittable RFT-ready project.

---

## Stage 4 — Train a model on it (optional)

### What you're doing

You take the rollout trajectories from Stage 3 and use them to train an open model (Qwen or Llama class, hosted on Fireworks) with **GRPO** — a reinforcement learning algorithm that adjusts the model's weights based on which outputs got high rewards.

The model runs Stage 3's loop repeatedly. Good behavior (correct paths, cited evidence, knowing when to stop) gets reinforced. Bad behavior (hallucinated edges, unsupported citations, excessive tool calls) gets suppressed.

You produce a **before/after comparison**: task success rate on held-out cases before training vs. after.

### The principle

This is where it becomes RFT (reinforcement fine-tuning) rather than just an eval. The environment you built in Stages 0–3 is the training signal. The model is learning a policy — not memorizing answers, but learning how to investigate. Even a modest bump on 50–100 cases is the whole point.

**Be ruthless here under time pressure.** A non-converged training run demos worse than a clean environment with no training.

---

## Stage 5 — Make it watchable (parallel track)

### What you're doing

A real-time visualization that renders the agent's tool calls as a graph building up on screen:

- Each `enrich()` call adds nodes
- Each confirmed edge gets drawn with a tooltip showing the citation
- When the agent finds the path, it lights up
- When it fails or gives up, you see the dead ends

### The principle

COI discovery is uniquely good for this because the reasoning is spatial — paths between nodes. A viewer who has never heard of RL understands in 10 seconds what the agent is doing and why it matters. This is how you win the room. Build it in parallel — don't let it slip to the end.

---

## Using the hud-deepresearch template

The repo at `github.com/hud-evals/hud-deepresearch` is your starting point for Stages 1–3. It already ships working Exa and SixtyFour tool wrappers, the MCP server lifecycle, and the HUD environment registration pattern. You're forking it and swapping the grader and task template, not building from scratch.

### What you inherit verbatim

- **Exa tools** (`search`, `fetch`) in `env.py` — your corroboration step, copy unchanged
- **SixtyFour tools** (`enrich_person`, `enrich_company`) — your entity enrichment step, copy unchanged; API call structure, timeout/retry, and tier selection are already handled
- **MCP server lifecycle** (`_up`, `_down`, FastMCP, `_listening`) — boilerplate you don't write
- **`@env.template()` pattern** — how you define your task: yield a prompt, receive the agent's answer, run your grader, yield the result

### What you replace

**Add a `submit` tool.** The template has no explicit submit — the agent sends its final answer as prose. For COI you need structured output (a path list + citations list), so add this to the MCP server:

```python
_submitted: dict = {}

async def submit(path: list[str], citations: list[str]) -> str:
    """Submit the connection path and one citation URL per edge."""
    _submitted["path"] = path
    _submitted["citations"] = citations
    return "Submitted."

# in _up(), add: server.tool(submit)
```

**Replace the task template.** Swap `research_person` for a `conflict_of_interest` template:

```python
@env.template()
async def conflict_of_interest(
    person_a: str, person_b: str,
    label: bool, ground_truth_path: list[str], ground_truth_citations: list[str],
) -> AsyncGenerator[Any, Any]:
    _submitted.clear()
    answer = yield (
        f"Investigate whether {person_a} and {person_b} have a conflict of interest — "
        "a shared connection through board memberships or executive roles. "
        "Use enrich_person and search to discover the chain. "
        "When confident, call submit() with the path and one citation URL per edge. "
        "If there is no connection, call submit(path=[], citations=[])."
    )
    result = grade_submission(_submitted, label, ground_truth_path, ground_truth_citations, G)
    yield result
```

**Replace the grader.** The template uses `LLMJudgeGrader` because it grades prose quality. You have a deterministic graph, so use a direct function. This is strictly better for RL — no LLM latency in the reward loop:

```python
def grade_submission(submitted, label, ground_truth_path, ground_truth_citations, graph):
    path = submitted.get("path", [])
    reward = 0.0

    if label is False and not path:
        return GradeResult(reward=1.0)   # correct abstention
    if label is False and path:
        return GradeResult(reward=-0.5)  # fabricated connection
    if label is True and not path:
        return GradeResult(reward=0.0)   # missed connection

    # Check each claimed edge against the graph
    edge_rewards = []
    for i in range(len(path) - 1):
        edge_rewards.append(0.2 if graph.has_edge(path[i], path[i+1]) else -0.1)

    # Bonus for connecting the right endpoints
    if path[0] == ground_truth_path[0] and path[-1] == ground_truth_path[-1]:
        reward += 0.2

    # Citation coverage
    citations = submitted.get("citations", [])
    citation_score = min(len(citations), len(path) - 1) / max(len(path) - 1, 1)
    reward += sum(edge_rewards) + (citation_score * 0.3)

    return GradeResult(reward=max(0.0, min(1.0, reward)))
```

**Replace `tasks.py`.** Instead of two hardcoded tasks, generate from `cases.json`:

```python
from env import env, conflict_of_interest  # noqa: F401
import json

with open("cases.json") as f:
    cases = json.load(f)

tasks = []
for case in cases:
    t = conflict_of_interest(
        person_a=case["person_a"], person_b=case["person_b"],
        label=case["label"],
        ground_truth_path=case["path"] or [],
        ground_truth_citations=case["citations"] or [],
    )
    t.slug = f"coi-{case['person_a'].lower().replace(' ', '-')}-{case['person_b'].lower().replace(' ', '-')}"
    tasks.append(t)
```

### Concrete starting point

```bash
git clone https://github.com/hud-evals/hud-deepresearch c-o-investigator
cd c-o-investigator
uv sync
# env.py:   keep Exa + SixtyFour tools, add submit tool, replace template, add grader
# tasks.py: generate from cases.json
# graph.py: load Stage 0 networkx graph
hud eval tasks.py claude --task-ids coi-jane-smith-robert-chen -y --runtime local
```

---

## Key risks and mitigations

| Risk | Mitigation |
|---|---|
| SEC parsing eats the day | Fall back to OpenAlex co-authorship at hour 3 |
| Rewards are all 0 or all 1 | Re-tune case difficulty in Stage 3 before training |
| GRPO doesn't converge in time | Stop at Stage 3, demo the environment + trajectories instead |
| Tools are flaky / rate-limited | Build thin fallback mocks for SixtyFour / Exa during dev |

---

## The question judges will ask

> "Does this actually require RL, or is it a prompting problem?"

Your answer: zero-shot Claude consistently halluccinates edges, fabricates citations, and doesn't know when to stop. The trained policy doesn't. Show the curve, or show qualitative trajectory comparisons. The reward decomposition (per-edge, citation-gated) is the mechanism that makes this learnable — lead with that.

---

## Framing for the pitch

Don't pitch this as a compliance tool. Pitch it as: **a blueprint for any task where an agent must traverse a hidden graph using noisy external evidence**. COI discovery is the instantiation. The mechanism generalizes to competitive intelligence, scientific literature mapping, regulatory investigation — any domain where the answer is a path through a network you can't see directly.

The 2040 framing: agents that can autonomously investigate any network of hidden relationships, verify their findings against primary sources, and know when they've found enough.
