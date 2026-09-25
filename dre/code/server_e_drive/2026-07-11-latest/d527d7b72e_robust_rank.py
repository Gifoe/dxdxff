from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def percentile_rank_channels(values: np.ndarray, valid_mask: np.ndarray, *, high_is_abnormal: bool) -> np.ndarray:
    data = np.asarray(values, dtype=float).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1) & np.isfinite(data)
    if data.shape != np.asarray(valid_mask).reshape(-1).shape:
        raise ValueError("values and valid_mask must have the same flattened shape")
    output = np.full(data.shape, np.nan, dtype=float)
    if int(valid.sum()) < 4:
        return output
    ranks = (rankdata(data[valid], method="average") - 1.0) / max(int(valid.sum()) - 1, 1)
    output[valid] = ranks if high_is_abnormal else 1.0 - ranks
    return np.clip(output, 0.0, 1.0)


__all__ = ["percentile_rank_channels"]
