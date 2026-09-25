from __future__ import annotations

import numpy as np


CONTROLS = ("none", "CHANNEL_PERMUTATION", "TARGET_PERMUTATION", "NEZ_PERMUTATION", "OUTCOME_PERMUTATION")


def permute_channel_map(values: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy(); indices = np.where(np.asarray(valid, dtype=bool))[0]; rng = np.random.default_rng(seed); output[indices] = output[rng.permutation(indices)]; return output


def permute_target(mask: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy(); indices = np.where(np.asarray(valid, dtype=bool))[0]; rng = np.random.default_rng(seed); output[indices] = output[rng.permutation(indices)]; return output


def permute_outcomes(y: np.ndarray, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(np.asarray(y, dtype=int))


__all__ = ["CONTROLS", "permute_channel_map", "permute_outcomes", "permute_target"]
