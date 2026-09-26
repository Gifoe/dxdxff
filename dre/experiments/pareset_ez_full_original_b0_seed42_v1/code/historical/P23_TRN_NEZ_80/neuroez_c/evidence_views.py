from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

BASE_SPECTRAL_FEATURE_NAMES = (
    "log_bp_delta",
    "log_bp_theta",
    "log_bp_alpha",
    "log_bp_beta",
    "log_bp_low_gamma",
    "log_bp_high_gamma",
    "log_total_power",
    "rms",
    "variance",
    "line_length_per_sec",
    "spectral_entropy",
    "hjorth_mobility",
    "hjorth_complexity",
)

PRUNED_SPECTRAL_FEATURE_NAMES = (
    "log_bp_delta",
    "log_bp_theta",
    "log_bp_beta",
    "log_bp_low_gamma",
    "log_bp_high_gamma",
    "rms",
    "variance",
    "line_length_per_sec",
    "spectral_entropy",
)

WINDOW_NODE_FEATURE_NAMES = BASE_SPECTRAL_FEATURE_NAMES + (
    "degree_norm",
    "strength_norm",
    "clustering_coeff",
    "eigenvector_centrality",
    "pagerank",
    "kcore_norm",
    "local_efficiency",
)

_PART_ALIASES = {
    "full": ("abs", "delta", "zdelta", "ratio"),
    "all": ("abs", "delta", "zdelta", "ratio"),
    "zdelta_ratio": ("zdelta", "ratio"),
    "zdelta-only": ("zdelta",),
    "zdelta_only": ("zdelta",),
    "ratio-only": ("ratio",),
    "ratio_only": ("ratio",),
    "abs-only": ("abs",),
    "abs_only": ("abs",),
    "delta-only": ("delta",),
    "delta_only": ("delta",),
}


def _selected_feature_parts(args: Any | None) -> tuple[str, ...]:
    raw = getattr(args, "b0_feature_parts", "abs,delta,zdelta,ratio") if args is not None else "abs,delta,zdelta,ratio"
    key = str(raw).strip().lower()
    if key in _PART_ALIASES:
        return _PART_ALIASES[key]
    parts = tuple(part.strip().lower() for part in key.split(",") if part.strip())
    invalid = [part for part in parts if part not in {"abs", "delta", "zdelta", "ratio"}]
    if invalid:
        raise ValueError(f"Unsupported b0_feature_parts item(s): {invalid}")
    return parts if parts else ("abs", "delta", "zdelta", "ratio")


def _selected_physics_parts(args: Any | None) -> tuple[str, ...]:
    raw = getattr(args, "physics_feature_parts", "zdelta,delta") if args is not None else "zdelta,delta"
    key = str(raw).strip().lower()
    if key in _PART_ALIASES:
        return _PART_ALIASES[key]
    parts = tuple(part.strip().lower() for part in key.split(",") if part.strip())
    invalid = [part for part in parts if part not in {"abs", "delta", "zdelta", "ratio"}]
    if invalid:
        raise ValueError(f"Unsupported physics_feature_parts item(s): {invalid}")
    return parts if parts else ("zdelta", "delta")


def _feature_group_indices(feature_dim: int, args: Any | None) -> tuple[np.ndarray, tuple[str, ...]]:
    group = str(getattr(args, "b0_feature_groups", "spectral_classical") if args is not None else "spectral_classical").strip().lower()
    names = _window_feature_names(feature_dim, args)
    if feature_dim < len(BASE_SPECTRAL_FEATURE_NAMES):
        raise ValueError(f"Expected at least {len(BASE_SPECTRAL_FEATURE_NAMES)} spectral features, got {feature_dim}.")

    if group in {"all", "spectral", "spectral_classical", "pruned", "pruned_spectral"}:
        selected = list(PRUNED_SPECTRAL_FEATURE_NAMES)
    elif group == "gamma_line_length":
        selected = ["log_bp_low_gamma", "log_bp_high_gamma", "line_length_per_sec"]
    else:
        raise ValueError(
            f"Unsupported b0_feature_groups={group!r}; expected spectral_classical or gamma_line_length. "
            "Graph-node feature groups were removed in B0-Pruned-EZBackbone."
        )

    invalid = [name for name in selected if name not in names or names.index(name) >= feature_dim]
    if invalid:
        raise ValueError(
            f"Unsupported b0_feature_groups={group!r}; missing feature(s): {invalid}. "
            f"Available features: {names[:feature_dim]}"
        )
    return np.asarray([names.index(name) for name in selected], dtype=np.int64), tuple(selected)


def _baseline_window_mask(window_centers: np.ndarray, num_windows: int) -> np.ndarray:
    centers = np.asarray(window_centers, dtype=np.float32)
    pre_mask = centers < 0.0
    if centers.shape[0] != num_windows or not np.any(pre_mask):
        pre_mask = np.ones((num_windows,), dtype=bool)
    return pre_mask


def b0_self_reference_features(window_features: np.ndarray, window_centers: np.ndarray, args: Any | None = None) -> np.ndarray:
    features = np.asarray(window_features, dtype=np.float32)
    indices, selected_names = _feature_group_indices(int(features.shape[-1]), args)
    b0_self_reference_features.selected_feature_names = selected_names
    features = features[:, :, indices]

    eps = float(getattr(args, "self_compare_eps", 1e-5) if args is not None else 1e-5)
    pre_mask = _baseline_window_mask(window_centers, features.shape[0])
    baseline = features[pre_mask]
    pre_mean = baseline.mean(axis=0, keepdims=True)
    pre_std = np.clip(baseline.std(axis=0, keepdims=True), eps, None)
    pre_mean_t = np.broadcast_to(pre_mean, features.shape)
    delta = features - pre_mean_t
    zdelta = delta / pre_std
    ratio = np.log((np.abs(features) + eps) / (np.abs(pre_mean_t) + eps))
    arrays = {
        "abs": features.astype(np.float32, copy=False),
        "delta": delta.astype(np.float32, copy=False),
        "zdelta": zdelta.astype(np.float32, copy=False),
        "ratio": ratio.astype(np.float32, copy=False),
    }
    parts = [arrays[name] for name in _selected_feature_parts(args)]
    return np.nan_to_num(np.concatenate(parts, axis=-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


b0_self_reference_features.selected_feature_names = PRUNED_SPECTRAL_FEATURE_NAMES


def _window_feature_names(feature_dim: int, args: Any | None) -> list[str]:
    raw_names = getattr(args, "window_feature_names", None) if args is not None else None
    if raw_names is not None:
        names = [str(name) for name in raw_names]
        if len(names) != feature_dim:
            raise ValueError(
                f"Cache window_feature_names length={len(names)} does not match window_features dim={feature_dim}."
            )
        return names
    names = list(WINDOW_NODE_FEATURE_NAMES)
    if feature_dim > len(names):
        names.extend(f"unnamed_window_feature_{idx}" for idx in range(len(names), feature_dim))
    return names


def _physics_feature_indices(feature_dim: int, args: Any | None) -> tuple[np.ndarray, tuple[str, ...]]:
    raw = (
        getattr(args, "physics_state_features", "log_bp_high_gamma,line_length_per_sec,rms,variance")
        if args is not None
        else "log_bp_high_gamma,line_length_per_sec,rms,variance"
    )
    requested = tuple(name.strip() for name in str(raw).split(",") if name.strip())
    names = _window_feature_names(feature_dim, args)
    if feature_dim < len(BASE_SPECTRAL_FEATURE_NAMES):
        raise ValueError(f"Expected at least {len(BASE_SPECTRAL_FEATURE_NAMES)} spectral features, got {feature_dim}.")
    invalid = [name for name in requested if name not in names or names.index(name) >= feature_dim]
    if invalid:
        raise ValueError(
            f"Unsupported physics_state_features item(s): {invalid}. "
            f"Available features: {names[:feature_dim]}"
        )
    if not requested:
        raise ValueError("Unsupported physics_state_features: at least one feature name is required.")
    return np.asarray([names.index(name) for name in requested], dtype=np.int64), requested


def physics_state_features(window_features: np.ndarray, window_centers: np.ndarray, args: Any | None = None) -> np.ndarray:
    features = np.asarray(window_features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(f"Expected window_features with shape [T,C,F], got ndim={features.ndim}.")
    indices, selected_names = _physics_feature_indices(int(features.shape[-1]), args)
    physics_state_features.selected_feature_names = selected_names
    features = features[:, :, indices]

    eps = float(getattr(args, "self_compare_eps", 1e-5) if args is not None else 1e-5)
    pre_mask = _baseline_window_mask(window_centers, features.shape[0])
    baseline = features[pre_mask]
    pre_mean = baseline.mean(axis=0, keepdims=True)
    pre_std = np.clip(baseline.std(axis=0, keepdims=True), eps, None)
    pre_mean_t = np.broadcast_to(pre_mean, features.shape)
    delta = features - pre_mean_t
    zdelta = delta / pre_std
    ratio = np.log((np.abs(features) + eps) / (np.abs(pre_mean_t) + eps))
    arrays = {
        "abs": features.astype(np.float32, copy=False),
        "delta": delta.astype(np.float32, copy=False),
        "zdelta": zdelta.astype(np.float32, copy=False),
        "ratio": ratio.astype(np.float32, copy=False),
    }
    parts = [arrays[name] for name in _selected_physics_parts(args)]
    return np.nan_to_num(np.concatenate(parts, axis=-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


physics_state_features.selected_feature_names = ("log_bp_high_gamma", "line_length_per_sec", "rms", "variance")


@dataclass
class EvidenceNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {arr.shape[-1]}.")
        return ((arr - self.mean) / self.std).astype(np.float32, copy=False)


@dataclass
class MultiViewEvidenceNormalizer:
    b0: EvidenceNormalizer
    physics: EvidenceNormalizer

    @property
    def feature_dim(self) -> int:
        return self.b0.feature_dim

    @property
    def mean(self) -> np.ndarray:
        return self.b0.mean

    @property
    def std(self) -> np.ndarray:
        return self.b0.std

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.transform_b0(values)

    def transform_b0(self, values: np.ndarray) -> np.ndarray:
        return self.b0.transform(values)

    def transform_physics(self, values: np.ndarray) -> np.ndarray:
        return self.physics.transform(values)


def fit_normalizer(arrays: Iterable[np.ndarray], fallback_dim: int = 1) -> EvidenceNormalizer:
    arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays if np.asarray(arr).ndim == 3 and np.asarray(arr).shape[-1] > 0]
    if not arrays:
        return EvidenceNormalizer(np.zeros((fallback_dim,), dtype=np.float32), np.ones((fallback_dim,), dtype=np.float32))
    dim = int(arrays[0].shape[-1])
    total = np.zeros((dim,), dtype=np.float64)
    total_sq = np.zeros((dim,), dtype=np.float64)
    count = 0
    for arr in arrays:
        if arr.shape[-1] != dim:
            continue
        flat = arr.reshape(-1, dim).astype(np.float64, copy=False)
        total += flat.sum(axis=0)
        total_sq += np.square(flat).sum(axis=0)
        count += int(flat.shape[0])
    if count <= 0:
        return EvidenceNormalizer(np.zeros((dim,), dtype=np.float32), np.ones((dim,), dtype=np.float32))
    mean = total / float(count)
    std = np.sqrt(np.clip(total_sq / float(count) - np.square(mean), 1e-8, None))
    return EvidenceNormalizer(mean.astype(np.float32), std.astype(np.float32))


__all__ = [
    "BASE_SPECTRAL_FEATURE_NAMES",
    "EvidenceNormalizer",
    "MultiViewEvidenceNormalizer",
    "PRUNED_SPECTRAL_FEATURE_NAMES",
    "WINDOW_NODE_FEATURE_NAMES",
    "b0_self_reference_features",
    "fit_normalizer",
    "physics_state_features",
]
