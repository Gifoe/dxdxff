from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np


@dataclass
class FoldNormalizer:
    median: np.ndarray | None = None
    scale: np.ndarray | None = None
    fit_subject_ids: tuple[str, ...] = ()

    @property
    def feature_dim(self) -> int:
        return 0 if self.median is None else int(self.median.shape[0])

    @staticmethod
    def _mask_for(array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(valid_mask, dtype=bool)
        try:
            return np.broadcast_to(mask, array.shape[:-1])
        except ValueError as error:
            raise ValueError(f"Mask shape {mask.shape} cannot broadcast to token shape {array.shape[:-1]}.") from error

    def fit(
        self,
        arrays: Sequence[np.ndarray],
        *,
        valid_masks: Sequence[np.ndarray],
        subject_ids: Sequence[str],
    ) -> "FoldNormalizer":
        if len(arrays) != len(subject_ids) or len(arrays) != len(valid_masks):
            raise ValueError("Normalizer arrays, valid_masks, and subject_ids must have equal length.")
        flattened: list[np.ndarray] = []
        for array, valid_mask in zip(arrays, valid_masks):
            value = np.asarray(array, dtype=np.float32)
            if value.ndim < 2:
                raise ValueError(f"Normalizer expects arrays with a feature axis, got {value.shape}.")
            mask = self._mask_for(value, valid_mask)
            if mask.any():
                flattened.append(value[mask])
        if not flattened:
            raise ValueError("Normalizer cannot fit an empty training fold.")
        matrix = np.concatenate(flattened, axis=0)
        finite = np.where(np.isfinite(matrix), matrix, np.nan)
        self.median = np.nanmedian(finite, axis=0).astype(np.float32)
        q25 = np.nanpercentile(finite, 25.0, axis=0)
        q75 = np.nanpercentile(finite, 75.0, axis=0)
        scale = q75 - q25
        std = np.nanstd(finite, axis=0)
        self.scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0)).astype(np.float32)
        self.fit_subject_ids = tuple(dict.fromkeys(str(value) for value in subject_ids))
        return self

    def transform(self, array: np.ndarray, *, valid_mask: np.ndarray) -> np.ndarray:
        if self.median is None or self.scale is None:
            raise RuntimeError("FoldNormalizer must be fitted before transform.")
        value = np.asarray(array, dtype=np.float32)
        if value.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {value.shape[-1]}.")
        mask = self._mask_for(value, valid_mask)
        normalized = (value - self.median) / self.scale
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        normalized[~mask] = 0.0
        return normalized

    @staticmethod
    def _relative_statistics(arrays: Sequence[np.ndarray], valid_masks: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        selected = []
        for array, valid_mask in zip(arrays, valid_masks):
            value = np.asarray(array, dtype=np.float32)
            mask = FoldNormalizer._mask_for(value, valid_mask)
            if mask.any():
                selected.append(value[mask])
        if not selected:
            raise ValueError("Patient-relative normalization requires at least one valid token.")
        matrix = np.concatenate(selected, axis=0)
        finite = np.where(np.isfinite(matrix), matrix, np.nan)
        median = np.nanmedian(finite, axis=0)
        q25 = np.nanpercentile(finite, 25.0, axis=0)
        q75 = np.nanpercentile(finite, 75.0, axis=0)
        std = np.nanstd(finite, axis=0)
        scale = np.where((q75 - q25) > 1e-6, q75 - q25, np.where(std > 1e-6, std, 1.0))
        return median.astype(np.float32), scale.astype(np.float32)

    @staticmethod
    def patient_relative(array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        value = np.asarray(array, dtype=np.float32)
        mask = FoldNormalizer._mask_for(value, valid_mask)
        median, scale = FoldNormalizer._relative_statistics([value], [mask])
        relative = (value - median) / scale
        relative = np.nan_to_num(relative, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        relative[~mask] = 0.0
        return relative

    @staticmethod
    def patient_relative_many(arrays: Sequence[np.ndarray], valid_masks: Sequence[np.ndarray]) -> list[np.ndarray]:
        if len(arrays) != len(valid_masks):
            raise ValueError("Patient-relative arrays and valid_masks must have equal length.")
        median, scale = FoldNormalizer._relative_statistics(arrays, valid_masks)
        output = []
        for array, valid_mask in zip(arrays, valid_masks):
            value = np.asarray(array, dtype=np.float32)
            mask = FoldNormalizer._mask_for(value, valid_mask)
            relative = np.nan_to_num((value - median) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            relative[~mask] = 0.0
            output.append(relative)
        return output

    def transform_with_relative(self, array: np.ndarray, valid_mask: np.ndarray, relative: np.ndarray | None = None) -> np.ndarray:
        relative_view = self.patient_relative(array, valid_mask) if relative is None else np.asarray(relative, dtype=np.float32)
        return np.concatenate([self.transform(array, valid_mask=valid_mask), relative_view], axis=-1).astype(np.float32)

    def save(self, path: str | Path, audit_path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)
        Path(audit_path).write_text(
            json.dumps(
                {
                    "fit_subject_ids": list(self.fit_subject_ids),
                    "fit_subject_count": len(self.fit_subject_ids),
                    "feature_dim": self.feature_dim,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


__all__ = ["FoldNormalizer"]
