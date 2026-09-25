from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SIMPLE_BASELINES = ("majority", "elasticnet", "rbf_svm", "random_forest", "lightgbm")


def parameter_grid(name: str, compact: bool = False) -> list[dict[str, Any]]:
    if name == "elasticnet":
        return [{"C": c, "l1_ratio": ratio} for c in ([1.0] if compact else [0.01, 0.1, 1.0, 10.0]) for ratio in ([0.5] if compact else [0, 0.25, 0.5, 0.75, 1])]
    if name == "rbf_svm":
        return [{"C": c, "gamma": gamma} for c in ([1.0] if compact else [0.1, 1, 10]) for gamma in (["scale"] if compact else ["scale", 0.01, 0.1])]
    if name == "random_forest":
        return [{"max_depth": depth, "min_samples_leaf": leaf} for depth in ([5] if compact else [3, 5, None]) for leaf in ([2] if compact else [2, 5, 10])]
    if name == "lightgbm":
        return [{"num_leaves": 7, "max_depth": 3, "learning_rate": 0.05, "min_child_samples": 10}] if compact else [{"num_leaves": leaves, "max_depth": depth, "learning_rate": rate, "min_child_samples": child} for leaves in [7, 15] for depth in [3, 5] for rate in [0.01, 0.05] for child in [10, 20]]
    if name == "majority":
        return [{}]
    raise ValueError(f"Unknown Task 2 baseline {name}")


def build_estimator(name: str, params: dict[str, Any], seed: int, compact: bool = False):
    if name == "elasticnet":
        estimator = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced", random_state=seed, max_iter=5000, **params)
        return Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("classifier", estimator)])
    if name == "rbf_svm":
        estimator = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed, **params)
        return Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("classifier", estimator)])
    if name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=50 if compact else 500, class_weight="balanced_subsample", random_state=seed, n_jobs=1, **params)
        return Pipeline([("imputer", SimpleImputer()), ("classifier", estimator)])
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM is unavailable; install lightgbm on the server.") from exc
        return Pipeline([
            ("imputer", SimpleImputer()),
            ("classifier", LGBMClassifier(
                objective="binary",
                n_estimators=100 if compact else 500,
                random_state=seed,
                n_jobs=1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
                **params,
            )),
        ])
    raise ValueError(f"No estimator for Task 2 baseline {name}")


__all__ = ["SIMPLE_BASELINES", "build_estimator", "parameter_grid"]
