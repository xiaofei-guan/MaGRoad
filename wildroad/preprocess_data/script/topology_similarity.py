#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topology similarity utilities.

This module provides:
- Conversion from adjacency dict to networkx.Graph
- Weisfeiler-Lehman (WL) topology similarity using label histograms
- Jensen-Shannon distance without SciPy dependency
- Length density computation for a cropped road graph

Design goals:
- Deterministic and reproducible (no use of Python's built-in hash randomness)
- Minimal dependencies (only numpy and networkx)
- Readable and easy to audit
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any

import numpy as np
import networkx as nx


Node = Tuple[float, float]
AdjacencyDict = Dict[Node, List[Node]]


def build_networkx_graph(graph_dict: AdjacencyDict, rounding_decimals: int = 3) -> nx.Graph:
    """Convert adjacency dict to networkx undirected graph.

    - Nodes are 2D coordinates; we round to reduce floating duplicates.
    - Edges are undirected; duplicates are ignored.

    Args:
        graph_dict: mapping (r, c) -> List[(r, c)]
        rounding_decimals: decimals to round coordinates when creating nodes

    Returns:
        networkx.Graph
    """
    def round_node(n: Node) -> Node:
        return (round(float(n[0]), rounding_decimals), round(float(n[1]), rounding_decimals))

    G = nx.Graph()
    for node, neighbors in graph_dict.items():
        n0 = round_node(node)
        if n0 not in G:
            G.add_node(n0)
        for nb in neighbors:
            n1 = round_node(nb)
            if n1 not in G:
                G.add_node(n1)
            if n0 != n1:
                G.add_edge(n0, n1)
    return G


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon distance between two distributions.

    Both p and q must be non-negative and sum to 1 (up to numerical noise).
    Returns sqrt(JS divergence), consistent with scipy.spatial.distance.jensenshannon.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask]))))

    js_div = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    js_dist = math.sqrt(max(0.0, js_div))
    return js_dist


def compute_wl_histograms(G: nx.Graph, iterations: int = 3) -> List[Dict[str, int]]:
    """Compute WL label histograms for a graph for 0..iterations.

    Deterministic implementation using string labels:
    - iter 0: label = f"deg:{degree}"
    - iter t>0: label = f"{prev}|{','.join(sorted(neigh_labels))}"

    Args:
        G: networkx.Graph
        iterations: number of WL iterations (>=1 recommended)

    Returns:
        List of dicts (label -> count) for each iteration (length = iterations+1)
    """
    # Initialize labels by degree
    node_labels: Dict[Any, str] = {}
    for n in G.nodes():
        node_labels[n] = f"deg:{G.degree(n)}"

    histograms: List[Dict[str, int]] = []
    # Record histogram for iteration 0
    h0: Dict[str, int] = {}
    for lbl in node_labels.values():
        h0[lbl] = h0.get(lbl, 0) + 1
    histograms.append(h0)

    # Iteratively update labels
    for _ in range(iterations):
        new_labels: Dict[Any, str] = {}
        for n in G.nodes():
            neigh_labels = [node_labels[nb] for nb in G.neighbors(n)]
            neigh_labels.sort()
            new_labels[n] = f"{node_labels[n]}|{','.join(neigh_labels)}"
        node_labels = new_labels

        h: Dict[str, int] = {}
        for lbl in node_labels.values():
            h[lbl] = h.get(lbl, 0) + 1
        histograms.append(h)

    return histograms


def wl_similarity_from_histograms(
    h1: List[Dict[str, int]], h2: List[Dict[str, int]]
) -> float:
    """Compute WL similarity in [0,1] from two WL histogram sequences.

    - Each sequence is a list of label->count dicts for 0..T iterations
    - For each iteration, compute JS distance between distributions; sim = 1 - JSdist
    - Final score is weighted average with larger weight for later iterations
    """
    L = min(len(h1), len(h2))
    sims: List[float] = []
    for i in range(L):
        labels = set(h1[i].keys()) | set(h2[i].keys())
        if not labels:
            sims.append(1.0)
            continue
        labels = sorted(labels)
        v1 = np.array([h1[i].get(lbl, 0) for lbl in labels], dtype=np.float64)
        v2 = np.array([h2[i].get(lbl, 0) for lbl in labels], dtype=np.float64)
        jsd = _js_distance(v1, v2)
        sims.append(1.0 - jsd)

    # Later iterations receive higher weights
    if not sims:
        return 1.0
    base_weights = np.linspace(1, len(sims), num=len(sims), dtype=np.float64)
    weights = base_weights / base_weights.sum()
    return float(np.average(np.array(sims, dtype=np.float64), weights=weights))


def compute_length_density(graph_dict: AdjacencyDict, patch_size: int) -> float:
    """Compute length density = total unique edge length / patch_area.

    - Unique edges: undirected; duplicates removed by canonical ordering
    - Length in pixel units (Euclidean)
    - patch_area = patch_size^2 (assumes square patches)
    """
    seen_edges = set()
    total_length = 0.0
    for a, nbrs in graph_dict.items():
        ar, ac = float(a[0]), float(a[1])
        for b in nbrs:
            br, bc = float(b[0]), float(b[1])
            if ar == br and ac == bc:
                continue
            # Canonical undirected key
            key = (a, b) if a <= b else (b, a)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            total_length += math.hypot(bc - ac, br - ar)

    area = float(patch_size * patch_size)
    if area <= 0:
        return 0.0
    return total_length / area


class WLCache:
    """Cache WL histograms and pairwise similarities for speed.

    Typical usage:
        cache = WLCache(iterations=3)
        h = cache.get_hist(graph_id, G)
        sim = cache.get_similarity(id_a, id_b, G_a, G_b)
    """

    def __init__(self, iterations: int = 3) -> None:
        self.iterations = iterations
        self._hist_cache: Dict[str, List[Dict[str, int]]] = {}
        self._pair_cache: Dict[Tuple[str, str], float] = {}

    def get_hist(self, graph_id: str, G: nx.Graph) -> List[Dict[str, int]]:
        if graph_id in self._hist_cache:
            return self._hist_cache[graph_id]
        hist = compute_wl_histograms(G, iterations=self.iterations)
        self._hist_cache[graph_id] = hist
        return hist

    def get_similarity(
        self,
        id_a: str,
        id_b: str,
        G_a: nx.Graph,
        G_b: nx.Graph,
    ) -> float:
        key = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
        if key in self._pair_cache:
            return self._pair_cache[key]
        h_a = self.get_hist(id_a, G_a)
        h_b = self.get_hist(id_b, G_b)
        sim = wl_similarity_from_histograms(h_a, h_b)
        self._pair_cache[key] = sim
        return sim


