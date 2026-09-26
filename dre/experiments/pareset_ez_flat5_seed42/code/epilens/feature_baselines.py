"""Code-aligned feature baselines and patient-relative controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn


def build_classical_estimator(name: str, seed: int):
    """Return the fixed configurations reported in the appendix."""
    if name == "logistic_regression":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="liblinear",
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "rbf_svm":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=1.0,
                        gamma="scale",
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError("The LightGBM baseline requires the local lightgbm package") from error
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=300,
                        num_leaves=7,
                        max_depth=3,
                        learning_rate=0.05,
                        min_child_samples=10,
                        class_weight="balanced",
                        random_state=seed,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown feature baseline: {name}")


def patient_feature_zscore(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Label-free per-patient, per-feature channel z-scoring."""
    result = frame.copy()
    for _, indices in result.groupby("patient_id").groups.items():
        values = result.loc[indices, feature_columns].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        result.loc[indices, feature_columns] = (values - mean) / scale
    return result


def patient_score_rank(frame: pd.DataFrame, score_column: str) -> pd.Series:
    """Convert scores to average percentile ranks independently per patient."""
    output = pd.Series(index=frame.index, dtype=float)
    for _, indices in frame.groupby("patient_id").groups.items():
        values = frame.loc[indices, score_column].to_numpy(dtype=float)
        output.loc[indices] = rankdata(values, method="average") / (len(values) + 1.0)
    return output


class DeepSetsControl(nn.Module):
    """Permutation-invariant control using the 88-dimensional table."""

    def __init__(self, input_dimension: int = 88):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(192, 96), nn.GELU(), nn.Dropout(0.15), nn.Linear(96, 1)
        )

    def forward(self, channels: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(channels)
        mean = encoded.mean(dim=0, keepdim=True).expand_as(encoded)
        maximum = encoded.max(dim=0, keepdim=True).values.expand_as(encoded)
        return self.head(torch.cat((encoded, mean, maximum), dim=-1)).squeeze(-1)


class PatientMLPControl(nn.Module):
    """Channel-wise MLP used before patient-wise output ranking."""

    def __init__(self, input_dimension: int = 88):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, channels: torch.Tensor) -> torch.Tensor:
        return self.network(channels).squeeze(-1)


@dataclass(frozen=True)
class NeuralControlConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 6
    minimum_epochs: int = 6
    gradient_clip: float = 1.0
