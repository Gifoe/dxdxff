from __future__ import annotations

from typing import Any

from .config import BNPDGSConfig


ABLATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "A0_static_like_baseline": {
        "temporal_encoder": "mean",
        "seizure_pooling": "mean",
        "use_adjacency_message_passing": False,
        "use_baseline_normalization": False,
    },
    "A1_spectral_temporal_only": {
        "use_graph_features": False,
        "use_adjacency_message_passing": False,
    },
    "A2_dynamic_graph_only": {
        "use_short_features": False,
        "use_lowfreq_features": False,
        "use_graph_features": True,
        "use_adjacency_message_passing": True,
    },
    "A3_no_baseline_normalization": {
        "use_baseline_normalization": False,
    },
    "A4_no_temporal_encoder": {
        "temporal_encoder": "mean",
    },
    "A5_no_positive_weighting": {
        "use_positive_weight": False,
    },
    "A6_full_bn_pdgs": {},
    "A7_mean_seizure_pooling": {
        "seizure_pooling": "mean",
    },
    "A8_250ms_only": {
        "use_lowfreq_features": False,
    },
    "A9_pearson_graph": {
        "graph_method": "pearson_abs",
    },
}


def make_ablation_config(base: BNPDGSConfig | Any | None, ablation_name: str) -> BNPDGSConfig:
    cfg = BNPDGSConfig.from_args(base)
    overrides = ABLATION_OVERRIDES.get(str(ablation_name))
    if overrides is None:
        raise KeyError(f"Unknown ablation: {ablation_name}")
    return cfg.replace(**overrides)


def list_ablations() -> list[str]:
    return list(ABLATION_OVERRIDES)


__all__ = ["ABLATION_OVERRIDES", "list_ablations", "make_ablation_config"]
