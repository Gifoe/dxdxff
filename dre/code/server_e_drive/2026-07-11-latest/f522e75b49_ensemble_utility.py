from __future__ import annotations

from pathlib import Path

import pandas as pd

from .catboost_utility import CatBoostUtilityModel
from .siamese_utility_mlp import SiameseUtilityMLP
from .trajectory_tcn import TrajectoryTCNUtilityModel


class EnsembleUtilityModel:
    """Mean of independently trained CatBoost, Siamese, and TCN utilities."""

    def __init__(self, *, random_seed: int = 42, cat_iterations: int = 80,
                 cat_depth: int = 3, cat_learning_rate: float = .05,
                 cat_l2_leaf_reg: float = 20., cat_early_stopping_rounds: int = 50,
                 neural_epochs: int = 30,
                 hidden_dim: int = 32, dropout: float = .2,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 32, num_workers: int = 0, patience: int = 8,
                 gradient_clip: float = 1., strict_device: bool = False,
                 output_dir=None, trajectory_store=None, device: str = "cpu",
                 risk_lambda: float = .10) -> None:
        root = None if output_dir is None else Path(output_dir)
        self.members = [
            CatBoostUtilityModel(iterations=cat_iterations, depth=cat_depth, learning_rate=cat_learning_rate, l2_leaf_reg=cat_l2_leaf_reg, early_stopping_rounds=cat_early_stopping_rounds, random_seed=random_seed, risk_lambda=risk_lambda),
            SiameseUtilityMLP(epochs=neural_epochs, hidden_dim=hidden_dim, dropout=dropout, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size, num_workers=num_workers, patience=patience, gradient_clip=gradient_clip, strict_device=strict_device, output_dir=None if root is None else root / "siamese", random_seed=random_seed, risk_lambda=risk_lambda, device=device),
            TrajectoryTCNUtilityModel(epochs=neural_epochs, hidden_dim=hidden_dim, dropout=dropout, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size, num_workers=num_workers, patience=patience, gradient_clip=gradient_clip, strict_device=strict_device, output_dir=None if root is None else root / "tcn", random_seed=random_seed, risk_lambda=risk_lambda, device=device, trajectory_store=trajectory_store),
        ]
        self.member_names = ("catboost", "siamese", "tcn")
        self.selected_member_names = self.member_names
        self.fit_subjects: set[str] = set()
        self.trajectory_store = trajectory_store
        self.tcn_member = self.members[2]

    def fit(self, pairs: pd.DataFrame, feature_columns: list[str]) -> "EnsembleUtilityModel":
        self.fit_subjects = set(pairs["subject_id"].astype(str))
        for member in self.members:
            member.fit(pairs, feature_columns)
        return self

    def predict(self, pairs: pd.DataFrame) -> pd.DataFrame:
        member_predictions = self.predict_members(pairs)
        predictions = [item.reset_index(drop=True) for name, item in member_predictions.items() if name in self.selected_member_names]
        output = sum(predictions) / len(predictions)
        output["member_variance"] = pd.concat([item["pred_delta"].reset_index(drop=True) for item in predictions], axis=1).var(axis=1, ddof=0)
        output.index = pairs.index
        return output

    def predict_members(self, pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {name: member.predict(pairs) for name, member in zip(self.member_names, self.members)}
