# C-o-Investigator

Much of this project was built on top of HUD's [deep research environment](https://github.com/hud-evals/hud-deepresearch), where the retrieve → inspect → answer loop was utilized in the scenario of identifying conflicts-of-interest between 2 corporate directors/executives.

Use cases include:

- Due diligence for audits
- Information for journalist
- Findings warm connections

1. Improved Qwen3-8B model at determining whether conflicts-of-interest exists, potentially multi-hop ones. While frontier models with the same tools might be able to do as well or even better, the trained model can run at a **fraction of the cost**.

2. A verifiable environment and training framework reusable with larger models (Qwen3-8B was chosen for hackathon scope). The same reward signal could train or benchmark stronger base models.

**The contribution is the verifiable environment, not so much the trained Qwen3-8B model.**

What was built is a reward signal:

- a ground-truth graph (answer key) from SEC primary sources,
- a deterministic, per-edge, **citation-gated** grader,
- a stratified task distribution.

This can generalize to other verifiable graph-traversal-esque tasks: any domain where the answer is a path through a relationship network with checkable intermediate steps (supply chains, org charts, citation graphs).

[Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://aclanthology.org/2024.acl-long.550) (Yang et al., 2024) finds that LLMs show strong evidence of multi-hop reasoning on the first hop but only moderate evidence on the second, degrading further beyond that. This project directly targets that weakness: RL training on a task requiring 1–3 correct hops to earn reward gives the model an explicit signal to improve at the chaining behavior the paper identifies as hard. The paper ran experiments with larger models (up to 70B), and so it follows that smaller models would have the same struggles.

## Environment

A single episode of conflict-of-interest investigation. It:

- Presents the agent with two people's names and asks whether they share board/executive connections
- Exposes 5 MCP tools: search (Exa), fetch (Exa), enrich_person (sixtyfour), enrich_company (sixtyfour), sec_search (SEC Edgar), submit
- Terminates when the agent calls submit() (or hits max_steps)
- Scores the submission against graph.json and returns a reward in [0, 1]

## Agent

The agent is a ReAct-style tool-calling loop built by HUD's create_agent(). Each turn it:

1. Reads the current conversation history
2. Makes a single call to the model for a response (reasoning + tool call)
3. Executes the tool, appends the result to history
4. Repeats until submit() is called or max_steps is exhausted

## Policy

π(a | s) where s is the full conversation history and a is the next complete response (reasoning + tool call).

A stochastic policy determined by the LLM.

## Reward signal

| Subscore            | Weight | What it measures                                                                    |
| ------------------- | ------ | ----------------------------------------------------------------------------------- |
| `edge_correctness`  | 0.5    | Fraction of submitted path edges that exist in graph.json (fuzzy match, score ≥ 80) |
| `endpoints`         | 0.2    | Binary — did the path connect the right two people? (fuzzy token_sort_ratio ≥ 85)   |
| `citation_coverage` | 0.3    | min(citations, edges) / edges — at least one URL per edge                           |

## Data pipeline (`build_graph.py`)

**Step 1 — Company list.** Scrapes the current S&P 500 constituent tickers from Wikipedia, then cross-references against the SEC's `company_tickers.json` CIK map to get the CIK needed for EDGAR downloads. Companies not in the SEC map are dropped.

**Step 2 — Filing download.** Downloads the most recent DEF 14A (proxy statement) and 10-K (annual report) for each company via `sec-edgar-downloader`. Filings arrive as EDGAR full-submission SGML bundles (`full-submission.txt`), which contain all documents for a single filing.

**Step 3 — LLM (Claude Haiku) parsing.** Each SGML bundle is split into its primary HTML document, then the relevant section is extracted:

- DEF 14A → "Proposal 1 / Election of Directors" section → director nominees + their outside board seats and affiliations
- 10-K → "Executive Officers" section → officer names and titles

Section text is scored by proper-name density to avoid matching TOC entries or iXBRL noise. Claude Haiku then extracts structured JSON (name, role, affiliations) from the plain text. Results are cached to disk so re-runs don't re-call the API.

**Step 4 — Graph construction.** Each extracted record becomes a `(person, company, role, citation)` tuple. These are assembled into a bipartite `networkx.Graph` where person and company are node types and edges carry `role` and `citation` (the filing path used as a verifiable source). Fuzzy name matching (rapidfuzz token_sort_ratio ≥ 88) deduplicates name variants like "Jane A. Smith" → "Jane Smith" before adding nodes.

**Step 5 — Case generation.** Person pairs are sampled from the graph and labeled by shortest-path length:

- **Positive cases**: shortest path exists with length in [3, 7] nodes (1 hop = one shared company, 2 hops = two companies, etc.). The ground-truth path and per-edge citations are stored.
- **Negative cases**: no path exists between the two people (different connected components).

The held-out test set (`build_test_set.py`) resamples from the graph with a weighted distribution — more 1-hop and 2-hop cases than the graph's natural ~75% 3-hop skew — and excludes all pairs already seen in training or calibration sets.

### Small eval set (`cases_eval_small.json`)

A 20-case stratified subset of the held-out test set, used for a fast before/after comparison (base Qwen3-8B vs. the trained model) on a fixed task set:

| Tier  | Cases |
| ----- | ----- |
| neg   | 6     |
| 1-hop | 8     |
| 2-hop | 4     |
| 3-hop | 2     |

Weighted toward the trained tier (1-hop) and the generalization tier (2-hop), with a small 3-hop probe. Run identically against both models (before and after training); using a _fixed_ set for both removes the task-draw variance that makes per-step training rewards noisy, so it's the clean comparison measure. Report results stratified by hop, not as a single blended number.

## Results

Mean reward in [0, 1] on the small eval set (20 cases, `--group-size 1 --full`), **stratified by hop**:

| Tier        | n   | Base Qwen3-8B | Trained Qwen3-8B | Claude    |
| ----------- | --- | ------------- | ---------------- | --------- |
| neg         | 6   | 1.000         | 0.833            | 1.00      |
| 1-hop       | 8   | 0.500         | 0.438            | 0.469     |
| 2-hop       | 4   | 0.000         | 0.000            | 0.188     |
| 3-hop       | 2   | 0.000         | 0.000            | 0.000     |
| **Overall** | 20  | **0.500**     | **0.425**        | **0.525** |

(Base: 0.500 ± 0.418, 35% full-success. Trained: 0.425 ± 0.426, 30%. Trained = the peak-reward checkpoint, step-000003, not the final dip checkpoint — a fair comparison.)

Reading the base model: it abstains correctly on every negative (1.000), earns **partial** credit on 1-hop (finds the right endpoints / some edges but rarely the complete cited path), and **fails every 2-hop and 3-hop case (0.000)** — the multi-hop reasoning gap this project targets.

**Training result (honest): no measurable improvement.** Trained ≈ base (marginally lower, well within noise). Case-by-case, the differences are stochastic, not systematic:

- The neg drop is a single case (1 of 6) where the trained model fabricated a connection on a no-connection pair instead of abstaining.
- The 1-hop drop is two cases flipping on the tool-call give-up lottery (one run fumbles a tool call and submits empty; see Challenges) — the trained model also _recovered_ one case the base gave up on. Net within noise.
- 2-hop / 3-hop unchanged at 0.000 — neither model can do deep traversal.

Possible reasons for the non-improvement:

- Too little optim steps at 4
- Base model of Qwen3-8B is too small

## Challenges

Due to the small base model, there are failures in following the JSON format for tool calls sometimes. As such, tool calls fail and the result is empty lists being submitted. These hard errors are dropped from training, so this behavior is not reinforced.

However, there are rarer completed-but-empty rollouts (agent gives up / never submits, but the rollout finishes cleanly). This results in scores of 1.0 on negatives, and these are included in training so GRPO reinforces this "giving up" behavior, likely biasing the policy toward abstention.

## Learning takeaways

Watch metrics like change in reward and loss magnitude. This tells us if training is improving the policy. The first run with LR 1e-5 saw reward go down and loss go up between steps. By halving the LR to 5e-6, the opposite was observed, suggesting policy improvement.
