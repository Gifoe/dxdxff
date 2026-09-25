from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .training import grouped_train_validation_split


@dataclass
class _Constant:
    value: float

    def predict(self, matrix):
        return np.full(len(matrix), self.value, dtype=float)

    def predict_proba(self, matrix):
        value = np.clip(self.value, 0.0, 1.0)
        return np.column_stack([np.full(len(matrix), 1.0 - value), np.full(len(matrix), value)])


class CatBoostUtilityModel:
    """Low-capacity dual-head CatBoost benefit/harm/delta utility model."""

    def __init__(self, *, iterations: int = 300, depth: int = 3, learning_rate: float = 0.05, l2_leaf_reg: float = 20.0, early_stopping_rounds: int = 50, risk_lambda: float = 0.10, random_seed: int = 42) -> None:
        self.iterations, self.depth, self.learning_rate, self.l2_leaf_reg, self.early_stopping_rounds, self.risk_lambda, self.random_seed = iterations, depth, learning_rate, l2_leaf_reg, early_stopping_rounds, risk_lambda, random_seed
        self.benefit_model = self.harm_model = self.delta_model = None
        self.feature_columns: list[str] = []
        self.fit_subjects: set[str] = set()
        self.validation_subjects: set[str] = set()
        self.best_iteration: int | None = None

    @staticmethod
    def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
        return np.nan_to_num(frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def _classification_model(self, matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray, *, eval_set=None):
        if len(np.unique(labels)) < 2:
            return _Constant(float(labels.mean()) if len(labels) else 0.0)
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("CatBoost is required for CatBoostUtilityModel") from exc
        model = CatBoostClassifier(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg, loss_function="Logloss", random_seed=self.random_seed, verbose=False, allow_writing_files=False, thread_count=1)
        kwargs = {"sample_weight": weights}
        if eval_set is not None and len(eval_set[0]):
            kwargs.update({"eval_set": eval_set, "use_best_model": True, "early_stopping_rounds": self.early_stopping_rounds})
        model.fit(matrix, labels, **kwargs)
        return model

    def fit(self, pairs: pd.DataFrame, feature_columns: list[str]) -> "CatBoostUtilityModel":
        required = {"subject_id", "beneficial_label", "harmful_label", "delta_patient_macro_f1"}
        missing = required - set(pairs.columns)
        if missing:
            raise ValueError(f"pair labels missing: {sorted(missing)}")
        self.feature_columns = list(feature_columns)
        source = pairs.reset_index(drop=True)
        train_index, validation_index = grouped_train_validation_split(source, seed=self.random_seed)
        train, validation = source.iloc[train_index], source.iloc[validation_index]
        self.fit_subjects = set(train["subject_id"].astype(str)); self.validation_subjects = set(validation["subject_id"].astype(str))
        matrix = self._matrix(train, self.feature_columns); validation_matrix = self._matrix(validation, self.feature_columns)
        weights = train.get("patient_pair_weight", pd.Series(1.0, index=train.index)).astype(float).to_numpy()
        benefit = train["beneficial_label"].astype(int).to_numpy(); benefit_validation = validation["beneficial_label"].astype(int).to_numpy()
        harm = train["harmful_label"].astype(int).to_numpy(); harm_validation = validation["harmful_label"].astype(int).to_numpy()
        delta = train["delta_patient_macro_f1"].astype(float).to_numpy(); delta_validation = validation["delta_patient_macro_f1"].astype(float).to_numpy()
        self.benefit_model = self._classification_model(matrix, benefit, weights, eval_set=(validation_matrix, benefit_validation))
        self.harm_model = self._classification_model(matrix, harm, weights, eval_set=(validation_matrix, harm_validation))
        if len(delta) < 2:
            self.delta_model = _Constant(float(delta.mean()) if len(delta) else 0.0)
        else:
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:
                raise RuntimeError("CatBoost is required for CatBoostUtilityModel") from exc
            self.delta_model = CatBoostRegressor(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg, loss_function="Huber:delta=0.05", random_seed=self.random_seed, verbose=False, allow_writing_files=False, thread_count=1)
            kwargs = {"sample_weight": weights}
            if len(validation): kwargs.update({"eval_set": (validation_matrix, delta_validation), "use_best_model": True, "early_stopping_rounds": self.early_stopping_rounds})
            self.delta_model.fit(matrix, delta, **kwargs)
        iterations = [getattr(model, "get_best_iteration", lambda: None)() for model in (self.benefit_model, self.harm_model, self.delta_model)]
        valid_iterations = [int(value) for value in iterations if value is not None and int(value) >= 0]
        self.best_iteration = max(valid_iterations) if valid_iterations else None
        return self

    def predict(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.benefit_model is None or self.harm_model is None or self.delta_model is None:
            raise RuntimeError("model must be fit before predict")
        matrix = self._matrix(pairs, self.feature_columns)
        benefit = self.benefit_model.predict_proba(matrix)[:, 1]
        harm = self.harm_model.predict_proba(matrix)[:, 1]
        delta = np.asarray(self.delta_model.predict(matrix), dtype=float)
        utility = benefit * np.maximum(delta, 0.0) - self.risk_lambda * harm
        return pd.DataFrame({"p_benefit": benefit, "p_harm": harm, "pred_delta": delta, "utility": utility}, index=pairs.index)
