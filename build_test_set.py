"""
Regenerate the held-out COI test set, weighted toward the tiers that matter.

The original cases_test.json followed the graph's natural distribution, which is
~75% 3-hop positives — the tier that is both the hardest and the least like a real
conflict of interest. This rebuilds the test set toward the headline tiers (1-hop =
direct interlock, 2-hop = indirect tie, plus negatives) while keeping a small 3-hop
probe to measure generalization (framing (a)).

Every sampled pair is checked against the training and calibration sets so the
held-out eval stays leakage-free. Output is shuffled and seeded for reproducibility.

Usage:
  python build_test_set.py                                  # default 35/25/15/40 mix
  python build_test_set.py --h1 30 --h2 20 --h3 12 --neg 30
  python build_test_set.py --seed 7 --out cases_test.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / "data" / "graph.json"
EXCLUDE_FILES = ("cases_train.json", "cases_calib.json")


def load_graph(graph_path: Path) -> nx.Graph:
    """
    Load the serialized COI graph from disk.

    Parameters:
        graph_path : Path - Path to data/graph.json (networkx node_link format).

    Returns:
        nx.Graph - The bipartite person/company graph.

    Examples:
        G = load_graph(Path("data/graph.json"))
        G.number_of_nodes()  # 7008
    """
    return json_graph.node_link_graph(json.loads(graph_path.read_text()), edges="edges")


def load_excluded_pairs(files: tuple[str, ...]) -> set[frozenset[str]]:
    """
    Collect person-pair keys already used by training/calibration sets.

    Parameters:
        files : tuple[str, ...] - JSON case files whose pairs must be excluded.

    Returns:
        set[frozenset[str]] - Each element is frozenset({person_a, person_b}).

    Examples:
        load_excluded_pairs(("cases_train.json",))
        # {frozenset({'Tim Cook', 'Apple Inc.'}), ...}
    """
    used: set[frozenset[str]] = set()
    for f in files:
        p = ROOT / f
        if p.exists():
            for c in json.loads(p.read_text()):
                used.add(frozenset((c["person_a"], c["person_b"])))
    return used


def _path_citations(G: nx.Graph, path: list[str]) -> list[str]:
    """Edge citations along a node path (one per edge), '' where missing."""
    return [G[path[i]][path[i + 1]].get("citation", "") for i in range(len(path) - 1)]


def sample_positives(
    G: nx.Graph, people: list[str], targets: dict[int, int],
    excluded: set[frozenset[str]], rng: random.Random,
) -> dict[int, list[dict]]:
    """
    Sample connected person pairs bucketed by exact hop distance.

    For each randomly ordered source person, runs one bounded BFS and harvests
    targets at the exact shortest-path distances for the requested hop tiers
    (1-hop = 2 edges, 2-hop = 4, 3-hop = 6). Pairs already in the excluded set or
    already chosen are skipped, keeping the test set leakage-free and de-duplicated.

    Parameters:
        G        : nx.Graph - The COI graph.
        people   : list[str] - All person-typed node names.
        targets  : dict[int, int] - hop -> desired count, e.g. {1:35, 2:25, 3:15}.
        excluded : set[frozenset[str]] - Pairs to avoid (train/calib).
        rng      : random.Random - Seeded RNG for reproducible sampling.

    Returns:
        dict[int, list[dict]] - hop -> list of case dicts (label True, with path +
        citations).

    Examples:
        sample_positives(G, people, {1:35,2:25,3:15}, excluded, random.Random(7))[1][0]
        # {'person_a': ..., 'person_b': ..., 'label': True, 'path': [...], 'citations': [...]}
    """
    hop_to_dist = {h: 2 * h for h in targets}
    dist_to_hop = {d: h for h, d in hop_to_dist.items()}
    max_dist = max(hop_to_dist.values())

    buckets: dict[int, list[dict]] = {h: [] for h in targets}
    chosen: set[frozenset[str]] = set()
    order = people[:]
    rng.shuffle(order)

    for src in order:
        if all(len(buckets[h]) >= targets[h] for h in targets):
            break
        dists = nx.single_source_shortest_path_length(G, src, cutoff=max_dist)
        for tgt, d in dists.items():
            hop = dist_to_hop.get(d)
            if hop is None or len(buckets[hop]) >= targets[hop]:
                continue
            if tgt == src or G.nodes[tgt].get("type") != "person":
                continue
            key = frozenset((src, tgt))
            if key in excluded or key in chosen:
                continue
            path = nx.shortest_path(G, src, tgt)
            buckets[hop].append({
                "person_a": src, "person_b": tgt, "label": True,
                "path": path, "citations": _path_citations(G, path),
            })
            chosen.add(key)
    return buckets


def sample_negatives(
    G: nx.Graph, people: list[str], n_neg: int,
    excluded: set[frozenset[str]], rng: random.Random,
) -> list[dict]:
    """
    Sample disconnected person pairs (no path) as negative cases.

    Uses connected-component membership: a pair in two different components has no
    path and is a clean negative. Skips excluded and duplicate pairs.

    Parameters:
        G        : nx.Graph - The COI graph.
        people   : list[str] - All person-typed node names.
        n_neg    : int - Desired number of negatives.
        excluded : set[frozenset[str]] - Pairs to avoid (train/calib).
        rng      : random.Random - Seeded RNG.

    Returns:
        list[dict] - Case dicts with label False and null path/citations.

    Examples:
        sample_negatives(G, people, 40, excluded, random.Random(7))[0]
        # {'person_a': ..., 'person_b': ..., 'label': False, 'path': None, 'citations': None}
    """
    comp_id: dict[str, int] = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            comp_id[n] = i

    negatives: list[dict] = []
    chosen: set[frozenset[str]] = set()
    attempts = 0
    max_attempts = n_neg * 200
    while len(negatives) < n_neg and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(people, 2)
        if comp_id.get(a) == comp_id.get(b):
            continue  # same component → a path exists, not a clean negative
        key = frozenset((a, b))
        if key in excluded or key in chosen:
            continue
        chosen.add(key)
        negatives.append({"person_a": a, "person_b": b, "label": False, "path": None, "citations": None})
    return negatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the held-out COI test set weighted toward 1-/2-hop + neg.")
    parser.add_argument("--h1", type=int, default=35, help="1-hop positives (direct interlock)")
    parser.add_argument("--h2", type=int, default=25, help="2-hop positives (indirect tie)")
    parser.add_argument("--h3", type=int, default=15, help="3-hop positives (generalization probe)")
    parser.add_argument("--neg", type=int, default=40, help="negative (disconnected) cases")
    parser.add_argument("--seed", type=int, default=7, help="random seed for reproducibility")
    parser.add_argument("--graph", default=str(GRAPH_PATH), help="path to graph.json")
    parser.add_argument("--out", default="cases_test.json", help="output cases file")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    G = load_graph(Path(args.graph))
    people = [n for n, d in G.nodes(data=True) if d.get("type") == "person"]
    excluded = load_excluded_pairs(EXCLUDE_FILES)
    print(f"graph: {len(people)} people | excluded train/calib pairs: {len(excluded)}")

    targets = {1: args.h1, 2: args.h2, 3: args.h3}
    pos_buckets = sample_positives(G, people, targets, excluded, rng)
    negatives = sample_negatives(G, people, args.neg, excluded, rng)

    cases: list[dict] = list(negatives)
    for h in sorted(pos_buckets):
        got = len(pos_buckets[h])
        if got < targets[h]:
            print(f"  WARNING: only found {got}/{targets[h]} for {h}-hop")
        cases.extend(pos_buckets[h])
    rng.shuffle(cases)

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.write_text(json.dumps(cases, indent=2))

    counts = {h: len(pos_buckets[h]) for h in sorted(pos_buckets)}
    print(f"wrote {len(cases)} cases → {out_path}")
    print(f"  neg:{len(negatives)} " + " ".join(f"{h}-hop:{counts[h]}" for h in counts))


if __name__ == "__main__":
    main()
