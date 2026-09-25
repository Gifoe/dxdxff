from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def feature_parameter_grid(name: str, *, compact: bool = False) -> list[dict[str, Any]]:
    normalized = name.lower()
    if normalized == "logistic_regression":
        return [{"C": 1.0}]
    if normalized == "rbf_svm":
        return [{"C": c, "gamma": gamma} for c in ([1.0] if compact else [0.1, 1.0, 10.0]) for gamma in (["scale"] if compact else ["scale", 0.01, 0.1])]
    if normalized == "random_forest":
        return [
            {"max_depth": depth, "min_samples_leaf": leaf, "max_features": features}
            for depth in ([5] if compact else [3, 5, None])
            for leaf in ([2] if compact else [2, 5, 10])
            for features in (["sqrt"] if compact else ["sqrt", 0.5])
        ]
    if normalized == "lightgbm":
        return [
            {"num_leaves": leaves, "max_depth": depth, "learning_rate": rate, "min_child_samples": child}
            for leaves in ([7] if compact else [7, 15])
            for depth in ([3] if compact else [3, 5])
            for rate in ([0.05] if compact else [0.01, 0.05])
            for child in ([10] if compact else [10, 20])
        ]
    raise ValueError(f"Unknown Task 1 feature model: {name}")


def build_feature_estimator(name: str, params: dict[str, Any], *, seed: int):
    normalized = name.lower()
    if normalized == "logistic_regression":
        classifier = LogisticRegression(
            penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000,
            random_state=seed, **params,
        )
        return Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("classifier", classifier)])
    if normalized == "rbf_svm":
        classifier = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed, **params)
        return Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("classifier", classifier)])
    if normalized == "random_forest":
        classifier = RandomForestClassifier(n_estimators=50 if params.pop("_compact", False) else 300, class_weight="balanced_subsample", random_state=seed, n_jobs=-1, **params)
        return Pipeline([("imputer", SimpleImputer()), ("classifier", classifier)])
    if normalized == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM is unavailable; install lightgbm on the server.") from exc
        classifier = LGBMClassifier(objective="binary", n_estimators=300, class_weight="balanced", random_state=seed, verbosity=-1, **params)
        return Pipeline([("imputer", SimpleImputer()), ("classifier", classifier)])
    raise ValueError(f"Unknown Task 1 feature model: {name}")


__all__ = ["build_feature_estimator", "feature_parameter_grid"]
