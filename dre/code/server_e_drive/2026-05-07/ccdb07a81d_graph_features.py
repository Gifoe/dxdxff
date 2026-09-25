from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .peri_onset_windows import PeriOnsetWindowSet


GRAPH_FEATURE_NAMES = [
    "degree_norm",
    "strength_norm",
    "clustering_coeff",
    "eigenvector_centrality",
    "pagerank",
    "kcore_norm",
    "local_efficiency",
]


def compute_dynamic_adjacency(
    window_data: np.ndarray,
    sfreq: float,
    method: str = "high_gamma_envelope_corr",
    topk: int = 8,
    adaptive_topk: bool = False,
    edge_density: float = 0.20,
    topk_min: int = 2,
    topk_max: int = 12,
    channel_mask: np.ndarray | None = None,
) -> np.ndarray:
    data = np.asarray(window_data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("window_data must have shape [channels, time].")
    num_channels = data.shape[0]
    if num_channels == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if num_channels == 1:
        return np.zeros((1, 1), dtype=np.float32)

    method_l = str(method).lower()
    if method_l in {"pearson", "pearson_abs"}:
        source = data
    elif method_l in {"envelope_corr", "high_gamma_envelope_corr"}:
        source = _analytic_envelope(data, sfreq, low=80.0, high=150.0)
    elif method_l in {"low_gamma_envelope_corr"}:
        source = _analytic_envelope(data, sfreq, low=30.0, high=80.0)
    elif method_l in {"plv", "pli", "wpli"}:
        resolved_topk = _resolve_topk(num_channels, topk, adaptive_topk, edge_density, topk_min, topk_max, channel_mask)
        return _phase_connectivity(data, sfreq, method_l, resolved_topk, channel_mask=channel_mask)
    else:
        raise ValueError(f"Unsupported graph method: {method}")

    adjacency = np.abs(np.corrcoef(source))
    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    adjacency = _symmetrize_zero_diag(adjacency)
    resolved_topk = _resolve_topk(num_channels, topk, adaptive_topk, edge_density, topk_min, topk_max, channel_mask)
    return _topk_sparsify(adjacency, topk=resolved_topk, channel_mask=channel_mask)


def compute_graph_node_features_from_adjacency(
    adjacency: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    adj = np.asarray(adjacency, dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must have shape [channels, channels].")
    c = adj.shape[0]
    if c == 0:
        return np.zeros((0, len(GRAPH_FEATURE_NAMES)), dtype=np.float32), list(GRAPH_FEATURE_NAMES)

    adj = _symmetrize_zero_diag(adj)
    binary = adj > 0.0
    degree = binary.sum(axis=1).astype(np.float32)
    degree_norm = degree / max(c - 1, 1)
    strength = adj.sum(axis=1)
    strength_norm = strength / np.clip(strength.max(), 1e-8, None)
    clustering = _clustering_coeff(binary)
    eigenvector = _eigenvector_centrality(adj)
    pagerank = _pagerank(adj)
    kcore = _kcore_number(binary)
    kcore_norm = kcore / max(float(kcore.max()), 1.0)
    local_efficiency = _neighbor_density(binary)

    features = np.stack(
        [degree_norm, strength_norm, clustering, eigenvector, pagerank, kcore_norm, local_efficiency],
        axis=-1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False), list(GRAPH_FEATURE_NAMES)


def compute_dynamic_graph_features(
    window_set: PeriOnsetWindowSet,
    graph_method: str,
    topk: int = 8,
    adaptive_topk: bool = False,
    edge_density: float = 0.20,
    topk_min: int = 2,
    topk_max: int = 12,
    channel_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    graph_blocks = []
    adjacency_blocks = []
    for start, end in zip(window_set.window_starts, window_set.window_ends):
        window_data = window_set.signal_segment[:, int(start) : int(end)]
        adjacency = compute_dynamic_adjacency(
            window_data,
            window_set.sfreq,
            method=graph_method,
            topk=topk,
            adaptive_topk=adaptive_topk,
            edge_density=edge_density,
            topk_min=topk_min,
            topk_max=topk_max,
            channel_mask=channel_mask,
        )
        graph_features, names = compute_graph_node_features_from_adjacency(adjacency)
        graph_blocks.append(graph_features)
        adjacency_blocks.append(adjacency)
    return (
        np.stack(graph_blocks, axis=0).astype(np.float32, copy=False),
        np.stack(adjacency_blocks, axis=0).astype(np.float32, copy=False),
        names,
    )


def _analytic_envelope(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    nyquist = float(sfreq) / 2.0
    high = min(float(high), nyquist - 1.0)
    low = min(float(low), high - 1.0)
    if low <= 0.0 or high <= low or data.shape[-1] < 8:
        return np.abs(hilbert(data, axis=-1)).astype(np.float32, copy=False)
    try:
        sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, data, axis=-1)
    except ValueError:
        filtered = data
    return np.abs(hilbert(filtered, axis=-1)).astype(np.float32, copy=False)


def _phase_connectivity(
    data: np.ndarray,
    sfreq: float,
    method: str,
    topk: int,
    channel_mask: np.ndarray | None = None,
) -> np.ndarray:
    filtered = _bandpass_or_raw(data, sfreq, 4.0, 30.0)
    phase = np.angle(hilbert(filtered, axis=-1))
    c = phase.shape[0]
    adj = np.zeros((c, c), dtype=np.float32)
    for i in range(c):
        diff = phase[i][None, :] - phase
        if method == "plv":
            values = np.abs(np.mean(np.exp(1j * diff), axis=-1))
        elif method == "pli":
            values = np.abs(np.mean(np.sign(np.sin(diff)), axis=-1))
        else:
            imag = np.sin(diff)
            values = np.abs(np.mean(imag, axis=-1)) / np.clip(np.mean(np.abs(imag), axis=-1), 1e-8, None)
        adj[i] = values.astype(np.float32, copy=False)
    return _topk_sparsify(_symmetrize_zero_diag(adj), topk=topk, channel_mask=channel_mask)


def _bandpass_or_raw(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    nyquist = float(sfreq) / 2.0
    high = min(high, nyquist - 1.0)
    if high <= low:
        return data
    try:
        sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        return sosfiltfilt(sos, data, axis=-1)
    except ValueError:
        return data


def _symmetrize_zero_diag(adjacency: np.ndarray) -> np.ndarray:
    adj = (adjacency + adjacency.T) / 2.0
    np.fill_diagonal(adj, 0.0)
    return np.clip(adj, 0.0, None).astype(np.float32, copy=False)


def _resolve_topk(
    num_channels: int,
    topk: int,
    adaptive_topk: bool,
    edge_density: float,
    topk_min: int,
    topk_max: int,
    channel_mask: np.ndarray | None,
) -> int:
    valid_c = int(np.asarray(channel_mask, dtype=bool).sum()) if channel_mask is not None else int(num_channels)
    if not adaptive_topk:
        return int(topk)
    if valid_c <= 1:
        return 0
    k = int(round(float(edge_density) * float(valid_c)))
    k = max(int(topk_min), min(int(topk_max), k))
    return min(k, valid_c - 1)


def _topk_sparsify(adjacency: np.ndarray, topk: int, channel_mask: np.ndarray | None = None) -> np.ndarray:
    c = adjacency.shape[0]
    valid = np.ones(c, dtype=bool) if channel_mask is None else np.asarray(channel_mask, dtype=bool)
    if valid.shape[0] != c:
        raise ValueError("channel_mask must match adjacency channel dimension.")
    adjacency = adjacency.copy()
    adjacency[~valid, :] = 0.0
    adjacency[:, ~valid] = 0.0
    if topk <= 0 or topk >= c:
        return _symmetrize_zero_diag(adjacency)
    keep = np.zeros_like(adjacency, dtype=bool)
    for i in range(c):
        if not valid[i]:
            continue
        row = adjacency[i].copy()
        row[~valid] = -np.inf
        row[i] = -np.inf
        k_i = min(int(topk), max(int(valid.sum()) - 1, 0))
        if k_i <= 0:
            continue
        idx = np.argpartition(row, -k_i)[-k_i:]
        keep[i, idx] = row[idx] > 0.0
    keep = keep | keep.T
    return _symmetrize_zero_diag(np.where(keep, adjacency, 0.0))


def _clustering_coeff(binary: np.ndarray) -> np.ndarray:
    c = binary.shape[0]
    coeff = np.zeros(c, dtype=np.float32)
    for i in range(c):
        neighbors = np.flatnonzero(binary[i])
        degree = len(neighbors)
        if degree < 2:
            continue
        sub = binary[np.ix_(neighbors, neighbors)]
        edges = sub.sum() / 2.0
        coeff[i] = float((2.0 * edges) / (degree * (degree - 1)))
    return coeff


def _eigenvector_centrality(adjacency: np.ndarray, max_iter: int = 50) -> np.ndarray:
    c = adjacency.shape[0]
    vec = np.ones(c, dtype=np.float32) / max(c, 1)
    for _ in range(max_iter):
        next_vec = adjacency @ vec
        norm = np.linalg.norm(next_vec)
        if norm <= 1e-8:
            break
        next_vec = (next_vec / norm).astype(np.float32, copy=False)
        if np.linalg.norm(next_vec - vec) < 1e-5:
            vec = next_vec
            break
        vec = next_vec
    return vec / np.clip(vec.max(), 1e-8, None)


def _pagerank(adjacency: np.ndarray, damping: float = 0.85, max_iter: int = 50) -> np.ndarray:
    c = adjacency.shape[0]
    if c == 0:
        return np.zeros(0, dtype=np.float32)
    row_sum = adjacency.sum(axis=1, keepdims=True)
    transition = adjacency / np.clip(row_sum, 1e-8, None)
    transition[row_sum[:, 0] <= 1e-8] = 1.0 / c
    rank = np.ones(c, dtype=np.float32) / c
    teleport = (1.0 - damping) / c
    for _ in range(max_iter):
        next_rank = teleport + damping * (transition.T @ rank)
        if np.linalg.norm(next_rank - rank, ord=1) < 1e-6:
            rank = next_rank.astype(np.float32, copy=False)
            break
        rank = next_rank.astype(np.float32, copy=False)
    return rank / np.clip(rank.max(), 1e-8, None)


def _kcore_number(binary: np.ndarray) -> np.ndarray:
    c = binary.shape[0]
    remaining = np.ones(c, dtype=bool)
    core = np.zeros(c, dtype=np.float32)
    degrees = binary.sum(axis=1).astype(int)
    for k in range(1, c + 1):
        changed = True
        while changed:
            changed = False
            removable = remaining & (degrees < k)
            if np.any(removable):
                changed = True
                core[removable] = k - 1
                remaining[removable] = False
                degrees = (binary[:, remaining].sum(axis=1)).astype(int)
        if not np.any(remaining):
            break
    core[remaining] = degrees[remaining]
    return core.astype(np.float32, copy=False)


def _neighbor_density(binary: np.ndarray) -> np.ndarray:
    c = binary.shape[0]
    density = np.zeros(c, dtype=np.float32)
    for i in range(c):
        neighbors = np.flatnonzero(binary[i])
        n = len(neighbors)
        if n < 2:
            continue
        sub = binary[np.ix_(neighbors, neighbors)]
        density[i] = float(sub.sum() / max(n * (n - 1), 1))
    return density


__all__ = [
    "GRAPH_FEATURE_NAMES",
    "compute_dynamic_adjacency",
    "compute_dynamic_graph_features",
    "compute_graph_node_features_from_adjacency",
]
