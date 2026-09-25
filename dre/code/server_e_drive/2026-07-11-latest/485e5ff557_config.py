from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class A12Config:
    """Resolved, serializable configuration for an A12 execution."""

    seeds: tuple[int, ...] = (42, 43, 44)
    inner_folds: int = 4
    max_swaps: int = 1
    strict: bool = True
    resume: bool = False
    device: str = "cpu"
    strict_device: bool = False
    num_workers: int = 0
    selected_tail_frac: float = 0.30
    selected_tail_min: int = 3
    selected_tail_max: int = 12
    boundary_width: int = 20
    add_pool_max: int = 30
    anchor_quantile: float = 0.50
    anchor_min_channels: int = 8
    minimum_anchor_channel_coverage: float = 0.0
    minimum_anchor_feature_finite_ratio: float = 0.0
    anchor_mode: str = "low_score_non_selected_nez"
    epsilon: float = 1e-12
    risk_lambda: float = 0.10
    uncertainty_lambda: float = 0.0
    minimum_action_coverage: float = 0.0
    patient_harm_limit: float = 0.15
    max_eject_candidates: int = 12
    max_add_candidates: int = 30
    max_pairs_per_patient: int = 360
    trajectory_mode: str = "trajectory_compact"
    anchor_parity_tolerance: float = 1e-6
    expected_old_v3_macro_f1: float | None = None
    old_v3_summary: str | None = None
    force_rebuild_pair_cache: bool = False
    input_fingerprint: str | None = None
    batch_size: int = 32
    max_epochs: int = 40
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    dropout: float = .2
    gradient_clip: float = 1.0
    catboost_depth: int = 3
    catboost_learning_rate: float = .05
    catboost_l2_leaf_reg: float = 20.
    catboost_max_iterations: int = 800
    catboost_early_stopping_rounds: int = 50
    cache_invalid_record_policy: str = "drop"
    smoke_mode: bool = False
    variants: tuple[str, ...] = field(default_factory=lambda: ("A12-V0", "A12-V1"))

    def __post_init__(self) -> None:
        if self.max_swaps < 0:
            raise ValueError("max_swaps must be nonnegative")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs, and patience must be positive")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        for name in ("minimum_anchor_channel_coverage", "minimum_anchor_feature_finite_ratio", "minimum_action_coverage", "patient_harm_limit"):
            value = float(getattr(self, name))
            if not 0. <= value <= 1.:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.cache_invalid_record_policy not in {"fail", "drop"}:
            raise ValueError("cache_invalid_record_policy must be fail or drop")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
