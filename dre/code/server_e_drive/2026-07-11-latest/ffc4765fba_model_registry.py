from __future__ import annotations

from typing import Any

from .baselines import SummaryMLModel
from .hifos_pact import HiFOSPACT
from outcome_hifos.baselines.hierarchical import SharedHierPoolV1, TokenHierarchicalOutcomeModel
from task1_baselines.models.raw_backbones import FrozenEmbeddingEncoder, build_raw_token_encoder


SUPPORTED_VARIANTS = (
    "H1_SUMMARY_ML",
    "H2_HIER_POOL",
    "H3_ATTENTION_MIL",
    "H4_MULTI_CORE",
    "H5_ANCHORED_CORE",
    "H6_ANCHORED_UOT_DESC",
    "H7_TRANSPORT_GRAPH",
    "H8_RECURRENCE",
    "H9_FM_RECURRENCE",
    "T2_EEGNET_HIERPOOL",
    "T2_SHALLOWFBCSPNET_HIERPOOL",
    "T2_DEEP4NET_HIERPOOL",
    "T2_EEGCONFORMER_HIERPOOL",
    "T2_BRAINBERT_FROZEN_HIERPOOL",
    "T2_CBRAMOD_FROZEN_HIERPOOL",
)


def build_model(variant: str, config: dict[str, Any], input_dim: int):
    name = str(variant).upper()
    if name not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported outcome variant {variant!r}; expected one of {SUPPORTED_VARIANTS}.")
    if name == "H1_SUMMARY_ML":
        return SummaryMLModel(random_seed=int(config.get("random_seed", 42)))
    raw_names = {
        "T2_EEGNET_HIERPOOL": "eegnet",
        "T2_SHALLOWFBCSPNET_HIERPOOL": "shallowfbcspnet",
        "T2_DEEP4NET_HIERPOOL": "deep4net",
        "T2_EEGCONFORMER_HIERPOOL": "eegconformer",
    }
    if name in raw_names:
        token_dim = int(config.get("token_embedding_dim", 128))
        encoder = build_raw_token_encoder(raw_names[name], n_times=int(input_dim), sfreq=float(config.get("raw_sfreq", 200.0)), embedding_dim=token_dim)
        return TokenHierarchicalOutcomeModel(
            encoder,
            SharedHierPoolV1(token_dim, hidden_dim=int(config.get("model_dim", 128)), dropout=float(config.get("dropout", 0.2))),
            frozen_backbone=False,
        )
    if name in {"T2_BRAINBERT_FROZEN_HIERPOOL", "T2_CBRAMOD_FROZEN_HIERPOOL"}:
        return TokenHierarchicalOutcomeModel(
            FrozenEmbeddingEncoder(int(input_dim)),
            SharedHierPoolV1(int(input_dim), hidden_dim=int(config.get("model_dim", 128)), dropout=float(config.get("dropout", 0.2))),
            frozen_backbone=True,
        )
    return HiFOSPACT(variant=name, input_dim=int(input_dim), config=config)


__all__ = ["SUPPORTED_VARIANTS", "build_model"]
