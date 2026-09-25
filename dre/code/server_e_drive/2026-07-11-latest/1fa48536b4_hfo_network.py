from __future__ import annotations

from typing import Sequence

import numpy as np

from .robust_rank import percentile_rank_channels


def coincidence_adjacency(events: Sequence[np.ndarray], window_sec: float = .100) -> np.ndarray:
    n = len(events); matrix = np.zeros((n, n), dtype=float)
    for left in range(n):
        a = np.asarray(events[left], dtype=float)
        for right in range(left + 1, n):
            b = np.asarray(events[right], dtype=float)
            if not a.size or not b.size: continue
            count = sum(bool(np.any(np.abs(b - value) <= window_sec)) for value in a)
            matrix[left, right] = matrix[right, left] = float(count)
    return matrix


def weighted_eigenvector_centrality(adjacency: np.ndarray) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.any(matrix > 0):
        return np.full(matrix.shape[0] if matrix.ndim == 2 else 0, np.nan)
    try:
        values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0); vector = np.abs(vectors[:, np.argmax(values)])
    except np.linalg.LinAlgError:
        return np.full(matrix.shape[0], np.nan)
    if vector.max() <= 0: return np.full(matrix.shape[0], np.nan)
    return vector / vector.max()


def hfo_hub_rank(events: Sequence[np.ndarray], valid_mask: np.ndarray, window_sec: float = .100) -> np.ndarray:
    centrality = weighted_eigenvector_centrality(coincidence_adjacency(events, window_sec))
    return percentile_rank_channels(centrality, np.asarray(valid_mask, dtype=bool) & np.isfinite(centrality), high_is_abnormal=True)


__all__ = ["coincidence_adjacency", "hfo_hub_rank", "weighted_eigenvector_centrality"]
