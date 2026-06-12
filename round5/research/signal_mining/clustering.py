from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering


def _cluster_id(graph_type: str, products: List[str]) -> str:
    raw = graph_type + "|" + "|".join(sorted(products))
    return "CLU_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def _edge_product_pair(row: Dict[str, Any]):
    if row.get("leader") and row.get("follower"):
        return row["leader"], row["follower"]
    if row.get("product_a") and row.get("product_b"):
        return row["product_a"], row["product_b"]
    return None


def build_graph_clusters(
    rows: List[Dict[str, Any]],
    graph_type: str,
    min_score: float = 0.01,
    allowed_tiers: Tuple[str, ...] = ("confirmed", "promising"),
    top_edges_per_node: int | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if top_edges_per_node is not None:
        node_edges: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            pair = _edge_product_pair(row)
            if pair is None:
                continue
            a, b = pair
            node_edges.setdefault(a, []).append(row)
            node_edges.setdefault(b, []).append(row)
        keep_ids = set()
        for edge_rows in node_edges.values():
            ranked = sorted(edge_rows, key=lambda r: abs(float(r.get("stability_score", 0.0) or 0.0)), reverse=True)
            for row in ranked[:top_edges_per_node]:
                keep_ids.add(id(row))
        rows = [row for row in rows if id(row) in keep_ids]

    graph = nx.Graph()
    for row in rows:
        if row.get("tier") not in allowed_tiers:
            continue
        pair = _edge_product_pair(row)
        if pair is None:
            continue
        score = abs(float(row.get("stability_score", 0.0) or 0.0))
        if score < min_score:
            continue
        a, b = pair
        if a == b:
            continue
        if graph.has_edge(a, b):
            graph[a][b]["weight"] += score
            graph[a][b]["evidence_count"] += 1
        else:
            graph.add_edge(a, b, weight=score, evidence_count=1)

    cluster_rows: List[Dict[str, Any]] = []
    clusters_json: Dict[str, Any] = {"graph_type": graph_type, "clusters": []}
    for component in nx.connected_components(graph):
        products = sorted(component)
        if len(products) < 2:
            continue
        sub = graph.subgraph(products)
        weights = [float(data.get("weight", 0.0)) for _, _, data in sub.edges(data=True)]
        evidence = sum(int(data.get("evidence_count", 0)) for _, _, data in sub.edges(data=True))
        score = float(np.mean(weights) * np.log1p(evidence) * np.sqrt(len(products))) if weights else 0.0
        cid = _cluster_id(graph_type, products)
        row = {
            "cluster_id": cid,
            "graph_type": graph_type,
            "size": len(products),
            "edge_count": sub.number_of_edges(),
            "evidence_count": evidence,
            "cluster_score": score,
            "products": "|".join(products),
        }
        cluster_rows.append(row)
        clusters_json["clusters"].append({**row, "products": products})
    cluster_rows.sort(key=lambda r: float(r["cluster_score"]), reverse=True)
    return cluster_rows, clusters_json


def build_all_clusters(
    pair_rows: List[Dict[str, Any]],
    spread_rows: List[Dict[str, Any]],
    mirror_rows: List[Dict[str, Any]],
    min_score: float = 0.01,
    top_edges_per_node: int | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    specs = [
        ("lead_lag", pair_rows),
        ("spread_mean_reversion", spread_rows),
        ("mirror_inversion", mirror_rows),
        ("microprice_lead", [r for r in pair_rows if "microprice" in str(r.get("feature", ""))]),
        ("liquidity_shock", [r for r in pair_rows if any(k in str(r.get("feature", "")) for k in ["depth", "imbalance", "volume"])]),
    ]
    all_rows: List[Dict[str, Any]] = []
    all_json: Dict[str, Any] = {"cluster_sets": {}}
    for graph_type, rows in specs:
        cluster_rows, cluster_json = build_graph_clusters(
            rows,
            graph_type,
            min_score=min_score,
            top_edges_per_node=top_edges_per_node,
        )
        all_rows.extend(cluster_rows)
        all_json["cluster_sets"][graph_type] = cluster_json["clusters"]
    all_rows.sort(key=lambda r: float(r["cluster_score"]), reverse=True)
    return all_rows, all_json

