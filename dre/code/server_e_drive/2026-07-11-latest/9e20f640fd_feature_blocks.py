from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PairFeatureBlocks:
    """Explicit semantic feature partition for Siamese and temporal pair models."""

    eject_columns: tuple[str, ...]
    add_columns: tuple[str, ...]
    pair_columns: tuple[str, ...]
    patient_columns: tuple[str, ...]

    @classmethod
    def infer(cls, frame: pd.DataFrame, columns: list[str]) -> "PairFeatureBlocks":
        available = [name for name in columns if name in frame.columns]
        eject = tuple(name for name in available if name.startswith("eject_"))
        add = tuple(name for name in available if name.startswith("add_"))
        patient = tuple(name for name in available if name.startswith("patient_"))
        pair = tuple(name for name in available if name not in {*eject, *add, *patient})
        if not eject or not add:
            raise ValueError("explicit Siamese blocks require at least one eject_ and add_ feature")
        return cls(eject, add, pair, patient)

    def matrix(self, frame: pd.DataFrame, block: str) -> np.ndarray:
        columns = getattr(self, f"{block}_columns")
        if not columns:
            return np.zeros((len(frame), 0), dtype=np.float32)
        return np.nan_to_num(frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class BlockScalers:
    """Independent outer/inner-fit scalers whose fit subject IDs are auditable."""

    def __init__(self, blocks: PairFeatureBlocks) -> None:
        self.blocks = blocks
        self.scalers = {name: StandardScaler() for name in ("eject", "add", "pair", "patient")}
        self.fit_subjects: set[str] = set()

    def fit(self, frame: pd.DataFrame) -> "BlockScalers":
        self.fit_subjects = set(frame["subject_id"].astype(str))
        for name, scaler in self.scalers.items():
            matrix = self.blocks.matrix(frame, name)
            if matrix.shape[1]:
                scaler.fit(matrix)
        return self

    def transform(self, frame: pd.DataFrame, name: str) -> np.ndarray:
        matrix = self.blocks.matrix(frame, name)
        return self.scalers[name].transform(matrix) if matrix.shape[1] else matrix

