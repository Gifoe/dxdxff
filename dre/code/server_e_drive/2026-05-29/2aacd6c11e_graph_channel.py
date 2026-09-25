from __future__ import annotations

from typing import Dict, Sequence

import networkx as nx
import numpy as np


GRAPH_FEATURE_NAMES: Sequence[str] = (
    "degree_norm",
    "strength_norm",
    "clustering_coeff",
    "eigenvector_centrality",
    "pagerank",
    "kcore_norm",
    "local_efficiency",
)

EPS = 1e-8


def compute_abs_pearson_connectivity(window_data: np.ndarray) -> np.ndarray:
    window_array = np.asarray(window_data, dtype=np.float32)
    if window_array.ndim != 2:
        raise ValueError("window_data must have shape [channels, time].")

    num_channels = int(window_array.shape[0])
    if num_channels == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if num_channels == 1:
        return np.zeros((1, 1), dtype=np.float32)

    corr = np.corrcoef(window_array)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr).astype(np.float32, copy=False)
    np.fill_diagonal(corr, 0.0)
    return corr


def _threshold_connectivity(
    corr_abs: np.ndarray,
    *,
    edge_quantile: float = 0.70,
    min_edge_weight: float = 0.10,
) -> np.ndarray:
    num_channels = int(corr_abs.shape[0])
    if num_channels <= 1:
        return np.zeros_like(corr_abs, dtype=np.float32)

    upper = corr_abs[np.triu_indices(num_channels, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return np.zeros_like(corr_abs, dtype=np.float32)

    quantile = float(np.clip(edge_quantile, 0.0, 1.0))
    threshold = max(float(min_edge_weight), float(np.quantile(positive, quantile)))
    edge_mask = np.triu(corr_abs >= threshold, k=1)

    if not np.any(edge_mask):
        max_idx = int(np.argmax(upper))
        tri_i, tri_j = np.triu_indices(num_channels, k=1)
        fallback_i = int(tri_i[max_idx])
        fallback_j = int(tri_j[max_idx])
        edge_mask = np.zeros_like(corr_abs, dtype=bool)
        edge_mask[fallback_i, fallback_j] = True

    weighted_adj = np.zeros_like(corr_abs, dtype=np.float32)
    weighted_adj[edge_mask] = corr_abs[edge_mask]
    weighted_adj = weighted_adj + weighted_adj.T
    return weighted_adj


def compute_thresholded_abs_pearson_adjacency(
    window_data: np.ndarray,
    *,
    edge_quantile: float = 0.70,
    min_edge_weight: float = 0.10,
) -> np.ndarray:
    """Return the weighted functional-connectivity adjacency used by the GNN path.

    This keeps the graph construction semantics shared between hand-crafted graph
    node features and neural message passing. The adjacency is based on the
    absolute Pearson correlation matrix, with the same quantile/minimum-weight
    thresholding and single-edge fallback as ``compute_graph_node_features``.
    """

    corr_abs = compute_abs_pearson_connectivity(window_data)
    return _threshold_connectivity(
        corr_abs,
        edge_quantile=edge_quantile,
        min_edge_weight=min_edge_weight,
    )


def _build_weighted_graph(weighted_adj: np.ndarray) -> nx.Graph:
    num_channels = int(weighted_adj.shape[0])
    graph = nx.Graph()
    graph.add_nodes_from(range(num_channels))

    rows, cols = np.where(np.triu(weighted_adj, k=1) > 0.0)
    for src, dst in zip(rows.tolist(), cols.tolist()):
        weight = float(weighted_adj[src, dst])
        graph.add_edge(src, dst, weight=weight, distance=1.0 / max(weight, EPS))
    return graph


def _global_efficiency_from_graph(graph: nx.Graph) -> float:
    num_nodes = graph.number_of_nodes()
    if num_nodes < 2 or graph.number_of_edges() == 0:
        return 0.0

    path_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="distance"))
    total_efficiency = 0.0
    total_pairs = 0
    for src in graph.nodes:
        src_lengths = path_lengths.get(src, {})
        for dst in graph.nodes:
            if src == dst:
                continue
            distance = src_lengths.get(dst)
            if distance is None or distance <= 0.0:
                continue
            total_efficiency += 1.0 / distance
            total_pairs += 1

    if total_pairs == 0:
        return 0.0
    return float(total_efficiency / total_pairs)


def _node_local_efficiency(graph: nx.Graph, node_idx: int) -> float:
    neighbors = list(graph.neighbors(node_idx))
    if len(neighbors) < 2:
        return 0.0
    subgraph = graph.subgraph(neighbors).copy()
    return _global_efficiency_from_graph(subgraph)


def compute_graph_node_features(
    window_data: np.ndarray,
    *,
    edge_quantile: float = 0.70,
    min_edge_weight: float = 0.10,
) -> np.ndarray:
    window_array = np.asarray(window_data, dtype=np.float32)
    if window_array.ndim != 2:
        raise ValueError("window_data must have shape [channels, time].")

    num_channels = int(window_array.shape[0])
    num_features = len(GRAPH_FEATURE_NAMES)
    if num_channels == 0:
        return np.zeros((0, num_features), dtype=np.float32)

    corr_abs = compute_abs_pearson_connectivity(window_array)
    weighted_adj = _threshold_connectivity(
        corr_abs,
        edge_quantile=edge_quantile,
        min_edge_weight=min_edge_weight,
    )
    graph = _build_weighted_graph(weighted_adj)

    degree_norm = np.asarray(
        [graph.degree(node_idx) / max(num_channels - 1, 1) for node_idx in range(num_channels)],
        dtype=np.float32,
    )
    strength_norm = weighted_adj.sum(axis=1) / max(num_channels - 1, 1)

    if graph.number_of_edges() == 0:
        clustering = np.zeros(num_channels, dtype=np.float32)
        eigenvector = np.zeros(num_channels, dtype=np.float32)
        pagerank = np.full(num_channels, 1.0 / max(num_channels, 1), dtype=np.float32)
        kcore = np.zeros(num_channels, dtype=np.float32)
    else:
        clustering_dict: Dict[int, float] = nx.clustering(graph, weight="weight")
        clustering = np.asarray([clustering_dict.get(node_idx, 0.0) for node_idx in range(num_channels)], dtype=np.float32)

        try:
            eigenvector_dict = nx.eigenvector_centrality_numpy(graph, weight="weight")
        except Exception:
            eigenvector_dict = {node_idx: 0.0 for node_idx in graph.nodes}
        eigenvector = np.asarray(
            [eigenvector_dict.get(node_idx, 0.0) for node_idx in range(num_channels)],
            dtype=np.float32,
        )

        try:
            pagerank_dict = nx.pagerank(graph, weight="weight")
        except Exception:
            pagerank_dict = {node_idx: 1.0 / max(num_channels, 1) for node_idx in graph.nodes}
        pagerank = np.asarray(
            [pagerank_dict.get(node_idx, 0.0) for node_idx in range(num_channels)],
            dtype=np.float32,
        )

        try:
            core_dict = nx.core_number(graph)
        except Exception:
            core_dict = {node_idx: 0 for node_idx in graph.nodes}
        max_core = max(core_dict.values()) if core_dict else 0
        kcore = np.asarray(
            [core_dict.get(node_idx, 0) / max(max_core, 1) for node_idx in range(num_channels)],
            dtype=np.float32,
        )

    local_efficiency = np.asarray(
        [_node_local_efficiency(graph, node_idx) for node_idx in range(num_channels)],
        dtype=np.float32,
    )

    return np.stack(
        [
            degree_norm,
            strength_norm.astype(np.float32, copy=False),
            clustering,
            eigenvector,
            pagerank,
            kcore,
            local_efficiency,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


__all__ = [
    "GRAPH_FEATURE_NAMES",
    "compute_abs_pearson_connectivity",
    "compute_thresholded_abs_pearson_adjacency",
    "compute_graph_node_features",
]

