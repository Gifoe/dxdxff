from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class BNPDGSConfig:
    peri_pre_sec: float = 10.0
    peri_post_sec: float = 10.0
    short_window_sec: float = 0.250
    short_stride_sec: float = 0.150
    low_freq_window_sec: float = 1.0
    low_freq_stride_sec: float = 0.150
    baseline_start_sec: float = -10.0
    baseline_end_sec: float = -2.0

    target_sfreq: float = 512.0
    bandpass_low: float = 1.0
    bandpass_high: float = 150.0

    graph_method: str = "high_gamma_envelope_corr"
    graph_sparsity: str = "topk"
    graph_topk: int = 8
    graph_edge_quantile: float = 0.70
    use_adjacency_message_passing: bool = True
    use_short_features: bool = True
    use_lowfreq_features: bool = True
    use_graph_features: bool = True
    use_baseline_normalization: bool = True
    use_positive_weight: bool = True

    pos_weight_max: float = 20.0
    model_dim: int = 96
    temporal_encoder: str = "tcn"
    seizure_pooling: str = "attention"
    channel_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.25

    lambda_bce: float = 0.25
    lambda_pairwise: float = 1.0
    lambda_listwise: float = 1.0
    lambda_count: float = 0.20
    lambda_mass: float = 0.05
    lambda_temporal_smooth: float = 0.02

    ranking_margin: float = 1.0
    batch_size: int = 1
    feature_num_workers: int = 20
    epochs: int = 100
    patience: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_splits: int = 5
    seed: int = 42
    output_dir: Path = Path("outputs/bn_pdgs_ranker")

    datasets: Sequence[str] = ("lzu", "hup", "multicenter")
    success_only: bool = True

    @classmethod
    def from_args(cls, args: Any | None) -> "BNPDGSConfig":
        if args is None:
            return cls()
        if isinstance(args, cls):
            return args

        def get(name: str, default: Any) -> Any:
            if isinstance(args, Mapping):
                return args.get(name, default)
            return getattr(args, name, default)

        values = {}
        for field_name, field_def in cls.__dataclass_fields__.items():
            raw = get(field_name, field_def.default)
            if field_name == "output_dir":
                raw = Path(raw)
            values[field_name] = raw
        return cls(**values)

    def replace(self, **kwargs: Any) -> "BNPDGSConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        result["datasets"] = list(self.datasets)
        return result


# Backward-compatible constant aliases for scripts that import module globals.
PERI_PRE_SEC = BNPDGSConfig.peri_pre_sec
PERI_POST_SEC = BNPDGSConfig.peri_post_sec
SHORT_WINDOW_SEC = BNPDGSConfig.short_window_sec
SHORT_STRIDE_SEC = BNPDGSConfig.short_stride_sec
LOW_FREQ_WINDOW_SEC = BNPDGSConfig.low_freq_window_sec
LOW_FREQ_STRIDE_SEC = BNPDGSConfig.low_freq_stride_sec
BASELINE_START_SEC = BNPDGSConfig.baseline_start_sec
BASELINE_END_SEC = BNPDGSConfig.baseline_end_sec
TARGET_SFREQ = BNPDGSConfig.target_sfreq
BANDPASS_LOW = BNPDGSConfig.bandpass_low
BANDPASS_HIGH = BNPDGSConfig.bandpass_high
GRAPH_METHOD = BNPDGSConfig.graph_method
GRAPH_SPARSITY = BNPDGSConfig.graph_sparsity
GRAPH_TOPK = BNPDGSConfig.graph_topk
GRAPH_EDGE_QUANTILE = BNPDGSConfig.graph_edge_quantile
POS_WEIGHT_MAX = BNPDGSConfig.pos_weight_max
MODEL_DIM = BNPDGSConfig.model_dim
TEMPORAL_ENCODER = BNPDGSConfig.temporal_encoder
CHANNEL_LAYERS = BNPDGSConfig.channel_layers
NUM_HEADS = BNPDGSConfig.num_heads
DROPOUT = BNPDGSConfig.dropout
LAMBDA_BCE = BNPDGSConfig.lambda_bce
LAMBDA_PAIRWISE = BNPDGSConfig.lambda_pairwise
LAMBDA_LISTWISE = BNPDGSConfig.lambda_listwise
LAMBDA_COUNT = BNPDGSConfig.lambda_count
LAMBDA_MASS = BNPDGSConfig.lambda_mass
LAMBDA_TEMPORAL_SMOOTH = BNPDGSConfig.lambda_temporal_smooth


__all__ = ["BNPDGSConfig"]
