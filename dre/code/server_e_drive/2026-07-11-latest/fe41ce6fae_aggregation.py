from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def aggregate_window_embeddings_to_patient_channels(embeddings: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("embeddings must be [n_windows, embedding_dim].")
    if len(metadata) != emb.shape[0]:
        raise ValueError("metadata row count must match embeddings row count.")
    work = metadata.reset_index(drop=True).copy()
    record_vectors: List[dict] = []
    for keys, group in work.groupby(["fold_idx", "split_role", "subject_id", "center", "channel_name", "run_id"], sort=False):
        fold_idx, split_role, subject_id, center, channel_name, run_id = keys
        idx = group.index.to_numpy()
        vec = emb[idx]
        record_vec = np.concatenate([vec.mean(axis=0), vec.max(axis=0)]).astype(np.float32)
        record_vectors.append(
            {
                "fold_idx": int(fold_idx),
                "split_role": str(split_role),
                "subject_id": str(subject_id),
                "center": str(center),
                "channel_name": str(channel_name),
                "run_id": str(run_id),
                "label_ez": int(group["label_ez"].iloc[0]),
                "n_windows": int(len(group)),
                "vector": record_vec,
            }
        )
    rows: List[dict] = []
    for keys, group in pd.DataFrame(record_vectors).groupby(["fold_idx", "split_role", "subject_id", "center", "channel_name"], sort=False):
        fold_idx, split_role, subject_id, center, channel_name = keys
        vectors = np.vstack(group["vector"].to_list())
        final_vec = np.concatenate([vectors.mean(axis=0), vectors.max(axis=0)]).astype(np.float32)
        row = {
            "fold_idx": int(fold_idx),
            "split_role": str(split_role),
            "subject_id": str(subject_id),
            "center": str(center),
            "channel_name": str(channel_name),
            "label_ez": int(group["label_ez"].iloc[0]),
            "n_records": int(group["run_id"].nunique()),
            "n_windows": int(group["n_windows"].sum()),
        }
        for idx, value in enumerate(final_vec):
            row[f"emb_{idx:04d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)

