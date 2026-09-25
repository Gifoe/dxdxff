from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neuroez_c.clean_nez_utils import add_label_encoding_columns, apply_allowed_subject_filter, json_safe
from neuroez_c.raw_brainbert_data import normalize_channel_name, parse_contact_topology


FORBIDDEN_MODEL_FEATURES = {
    "center",
    "center_id",
    "subject_id",
    "patient_id",
    "fold_idx",
    "outcome_group",
    "surgery_success",
    "true_ez",
    "true_nez",
    "true_ez_count",
    "predicted_by_oracle_k",
    "predicted_by_kcal",
}

BASE_SCORE_DEFAULT_WEIGHTS = {
    "w_feature": 1.0,
    "w_raw_onset": 1.0,
    "w_raw_all": 0.5,
    "w_raw_preictal": 0.25,
}


def _patient_z(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    std = float(arr.std(ddof=0))
    if not np.isfinite(std) or std < 1e-8:
        return pd.Series(np.zeros(len(arr), dtype=float), index=values.index)
    return (arr - float(arr.mean())) / std


def _top_fraction_indices(group: pd.DataFrame, column: str, frac: float) -> set[int]:
    if column not in group.columns:
        return set()
    n = max(1, int(math.ceil(float(frac) * len(group))))
    values = pd.to_numeric(group[column], errors="coerce").fillna(0.0)
    ranked = group.assign(_candidate_score=values).sort_values("_candidate_score", ascending=False, kind="mergesort")
    return set(ranked.head(n).index.astype(int).tolist())


def _rule_parts(candidate_rule: str) -> tuple[float, bool, bool, bool, bool]:
    rule = str(candidate_rule).strip().lower()
    mapping: dict[str, tuple[float, bool, bool, bool, bool]] = {
        "union_top20_feature_raw_neighbors": (0.20, True, True, True, True),
        "union_top30_feature_raw_neighbors": (0.30, True, True, True, True),
        "union_top40_feature_raw_neighbors": (0.40, True, True, True, True),
        "feature_top20_only": (0.20, True, False, False, False),
        "feature_top30_only": (0.30, True, False, False, False),
        "feature_top40_only": (0.40, True, False, False, False),
        "raw_top30_only": (0.30, False, True, False, False),
        "feature_top30_neighbors": (0.30, True, False, True, False),
        "raw_top30_neighbors": (0.30, False, True, False, True),
    }
    if rule not in mapping:
        raise ValueError(f"Unknown candidate_rule={candidate_rule!r}")
    return mapping[rule]


def _parse_topology(group: pd.DataFrame) -> tuple[dict[int, tuple[str | None, int | None]], dict[str, int]]:
    parsed: dict[int, tuple[str | None, int | None]] = {}
    shaft_codes: dict[str, int] = {}
    for row_idx, row in group.iterrows():
        shaft, contact = parse_contact_topology(row.get("channel_name"))
        if shaft is None:
            norm = normalize_channel_name(row.get("channel_name"))
            shaft = "".join(ch for ch in norm if not ch.isdigit()) or None
        if shaft is not None and shaft not in shaft_codes:
            shaft_codes[shaft] = len(shaft_codes) + 1
        parsed[int(row_idx)] = (shaft, contact)
    return parsed, shaft_codes


def _neighbor_indices(group: pd.DataFrame, seed_indices: Sequence[int], parsed: dict[int, tuple[str | None, int | None]]) -> set[int]:
    selected: set[int] = set()
    for seed_idx in seed_indices:
        seed_shaft, seed_contact = parsed.get(int(seed_idx), (None, None))
        if seed_shaft is None or seed_contact is None:
            continue
        for row_idx in group.index:
            shaft, contact = parsed.get(int(row_idx), (None, None))
            if shaft == seed_shaft and contact is not None and abs(int(contact) - int(seed_contact)) <= 2:
                selected.add(int(row_idx))
    return selected


def _candidate_diagnostics(out: pd.DataFrame) -> tuple[float | None, float | None]:
    if "true_ez" not in out.columns:
        return None, None
    y = pd.to_numeric(out["true_ez"], errors="coerce").fillna(0).astype(int)
    cand = out["is_candidate"].astype(bool)
    n_true = int(y.eq(1).sum())
    n_cand = int(cand.sum())
    recall = float((cand & y.eq(1)).sum() / n_true) if n_true else None
    precision = float((cand & y.eq(1)).sum() / n_cand) if n_cand else None
    return recall, precision


def generate_clean_nez_candidates(
    rows: pd.DataFrame,
    *,
    candidate_rule: str = "union_top30_feature_raw_neighbors",
    top_neighbor_k: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    top_fraction, use_feature, use_raw, feature_neighbors, raw_neighbors = _rule_parts(candidate_rule)
    out = rows.copy()
    out["channel_norm"] = out["channel_name"].map(normalize_channel_name)
    out["is_candidate"] = False
    out["shaft_code"] = 0.0
    out["contact_number_raw"] = 0.0
    out["contact_number_norm"] = 0.0
    out["has_neighbor_pm1"] = 0.0
    out["has_neighbor_pm2"] = 0.0
    out["local_cluster_size"] = 0.0
    out["is_isolated_high_score"] = 0.0
    parse_failures = 0

    for _, group in out.groupby("subject_id", sort=False):
        candidate_idxs: set[int] = set()
        parsed, shaft_codes = _parse_topology(group)
        if use_feature:
            candidate_idxs |= _top_fraction_indices(group, "feature_suspicious_z", top_fraction)
        if use_raw:
            candidate_idxs |= _top_fraction_indices(group, "raw_dist_onset_z", top_fraction)
            candidate_idxs |= _top_fraction_indices(group, "raw_dist_all_z", top_fraction)
        if feature_neighbors:
            feature_seeds = list(
                group.sort_values("feature_suspicious_z", ascending=False, kind="mergesort")
                .head(int(top_neighbor_k))
                .index.astype(int)
            )
            candidate_idxs |= _neighbor_indices(group, feature_seeds, parsed)
        if raw_neighbors:
            raw_seed_scores = group.copy()
            raw_seed_scores["_raw_seed"] = pd.to_numeric(raw_seed_scores.get("raw_dist_onset_z", 0.0), errors="coerce").fillna(0.0)
            if "raw_dist_all_z" in raw_seed_scores.columns:
                raw_seed_scores["_raw_seed"] += pd.to_numeric(raw_seed_scores["raw_dist_all_z"], errors="coerce").fillna(0.0)
            raw_seeds = list(
                raw_seed_scores.sort_values("_raw_seed", ascending=False, kind="mergesort")
                .head(int(top_neighbor_k))
                .index.astype(int)
            )
            candidate_idxs |= _neighbor_indices(group, raw_seeds, parsed)
        out.loc[list(candidate_idxs), "is_candidate"] = True

        contacts_by_shaft: dict[str, list[int]] = defaultdict(list)
        for row_idx, (shaft, contact) in parsed.items():
            if shaft is None or contact is None:
                parse_failures += 1
                continue
            contacts_by_shaft[shaft].append(int(contact))
        for row_idx, (shaft, contact) in parsed.items():
            if shaft is None or contact is None:
                continue
            contacts = contacts_by_shaft.get(shaft, [])
            lo, hi = min(contacts), max(contacts)
            out.at[row_idx, "shaft_code"] = float(shaft_codes.get(shaft, 0))
            out.at[row_idx, "contact_number_raw"] = float(contact)
            out.at[row_idx, "contact_number_norm"] = 0.0 if hi == lo else (float(contact) - lo) / float(hi - lo)
            pm1 = [num for num in contacts if num != int(contact) and abs(num - int(contact)) <= 1]
            pm2 = [num for num in contacts if num != int(contact) and abs(num - int(contact)) <= 2]
            out.at[row_idx, "has_neighbor_pm1"] = float(bool(pm1))
            out.at[row_idx, "has_neighbor_pm2"] = float(bool(pm2))
            out.at[row_idx, "local_cluster_size"] = float(len(pm2) + 1)
            if bool(out.at[row_idx, "is_candidate"]) and not pm2:
                out.at[row_idx, "is_isolated_high_score"] = 1.0

    recall, precision = _candidate_diagnostics(out)
    counts = out.groupby("subject_id")["is_candidate"].sum().astype(int)
    audit = {
        "candidate_rule": str(candidate_rule),
        "top_fraction": float(top_fraction),
        "top_neighbor_k": int(top_neighbor_k),
        "row_count": int(len(out)),
        "candidate_count": int(out["is_candidate"].astype(bool).sum()),
        "candidate_fraction": float(out["is_candidate"].astype(bool).mean()) if len(out) else 0.0,
        "candidate_count_by_fold": {str(k): int(v) for k, v in out.groupby("fold_idx")["is_candidate"].sum().to_dict().items()} if "fold_idx" in out.columns else {},
        "candidate_count_by_patient_summary": {
            "min": int(counts.min()) if len(counts) else 0,
            "max": int(counts.max()) if len(counts) else 0,
            "mean": float(counts.mean()) if len(counts) else 0.0,
        },
        "topology_parse_failures": int(parse_failures),
        "candidate_recall_diagnostic": recall,
        "candidate_precision_diagnostic": precision,
        "diagnostic_uses_true_labels": True,
    }
    return out, audit


def compute_base_suspicious_logit(
    rows: pd.DataFrame,
    *,
    w_feature: float = 1.0,
    w_raw_onset: float = 1.0,
    w_raw_all: float = 0.5,
    w_raw_preictal: float = 0.25,
) -> tuple[pd.Series, list[str]]:
    work = rows.copy()
    missing: list[str] = []
    components = {
        "feature_non_nez_score": float(w_feature),
        "raw_dist_onset_z": float(w_raw_onset),
        "raw_dist_all_z": float(w_raw_all),
        "raw_dist_preictal_z": float(w_raw_preictal),
    }
    score = pd.Series(np.zeros(len(work), dtype=float), index=work.index)
    for col, weight in components.items():
        if col not in work.columns:
            missing.append(col)
            work[col] = 0.0
        score = score + weight * work.groupby("subject_id", sort=False)[col].transform(_patient_z)
    return pd.to_numeric(score, errors="coerce").fillna(0.0), missing


def _feature_columns(rows: pd.DataFrame) -> list[str]:
    candidates = [
        "p_clean_nez",
        "feature_non_nez_score",
        "feature_suspicious_z",
        "rank_feature_suspicious_norm",
        "base_suspicious_logit",
        "raw_dist_all_z",
        "raw_dist_onset_z",
        "raw_dist_preictal_z",
        "raw_dist_all_cosine",
        "raw_dist_onset_cosine",
        "n_records",
        "shaft_code",
        "contact_number_norm",
        "has_neighbor_pm1",
        "has_neighbor_pm2",
        "local_cluster_size",
        "is_isolated_high_score",
    ]
    cols = [col for col in candidates if col in rows.columns]
    forbidden = sorted(FORBIDDEN_MODEL_FEATURES.intersection(cols))
    if forbidden:
        raise ValueError(f"Forbidden SetTopo model features selected: {forbidden}")
    return cols


class CandidateSetDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, feature_columns: Sequence[str]) -> None:
        self.feature_columns = list(feature_columns)
        self.items: list[dict[str, Any]] = []
        candidates = rows[rows["is_candidate"].astype(bool)].copy()
        for subject_id, group in candidates.groupby("subject_id", sort=False):
            if group.empty:
                continue
            self.items.append(
                {
                    "subject_id": str(subject_id),
                    "fold_idx": int(pd.to_numeric(group["fold_idx"], errors="coerce").dropna().iloc[0]),
                    "x": group[self.feature_columns].fillna(0.0).to_numpy(dtype=np.float32),
                    "true_ez": pd.to_numeric(group.get("clinical_true_ez", group["true_ez"]), errors="coerce").fillna(0).to_numpy(dtype=np.float32),
                    "true_nez": pd.to_numeric(group.get("clinical_true_nez", group["true_nez"]), errors="coerce").fillna(0).to_numpy(dtype=np.float32),
                    "is_pseudo_clean_nez_anchor": pd.to_numeric(
                        group.get("is_pseudo_clean_nez_anchor", pd.Series([0] * len(group), index=group.index)),
                        errors="coerce",
                    ).fillna(0).to_numpy(dtype=np.float32),
                    "base_suspicious_logit": pd.to_numeric(group["base_suspicious_logit"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
                    "raw_dist_onset_z": pd.to_numeric(group.get("raw_dist_onset_z", pd.Series([0.0] * len(group), index=group.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
                    "shaft_code": pd.to_numeric(group.get("shaft_code", pd.Series([0.0] * len(group), index=group.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
                    "contact_number": pd.to_numeric(group.get("contact_number_raw", pd.Series([0.0] * len(group), index=group.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
                    "channel_indices": group.index.astype(int).to_numpy(dtype=np.int64),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def _collate_sets(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(item["x"].shape[0] for item in batch)
    feat_dim = batch[0]["x"].shape[1]
    out: dict[str, Any] = {
        "subject_id": [item["subject_id"] for item in batch],
        "fold_idx": [item["fold_idx"] for item in batch],
        "x": torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32),
        "mask": torch.zeros((len(batch), max_len), dtype=torch.bool),
        "true_ez": torch.zeros((len(batch), max_len), dtype=torch.float32),
        "true_nez": torch.zeros((len(batch), max_len), dtype=torch.float32),
        "is_pseudo_clean_nez_anchor": torch.zeros((len(batch), max_len), dtype=torch.float32),
        "base_suspicious_logit": torch.zeros((len(batch), max_len), dtype=torch.float32),
        "raw_dist_onset_z": torch.zeros((len(batch), max_len), dtype=torch.float32),
        "shaft_code": torch.zeros((len(batch), max_len), dtype=torch.float32),
        "contact_number": torch.zeros((len(batch), max_len), dtype=torch.float32),
        "channel_indices": [],
    }
    for row_idx, item in enumerate(batch):
        n = item["x"].shape[0]
        out["x"][row_idx, :n, :] = torch.as_tensor(item["x"], dtype=torch.float32)
        out["mask"][row_idx, :n] = True
        for key in ("true_ez", "true_nez", "is_pseudo_clean_nez_anchor", "base_suspicious_logit", "raw_dist_onset_z", "shaft_code", "contact_number"):
            out[key][row_idx, :n] = torch.as_tensor(item[key], dtype=torch.float32)
        out["channel_indices"].append(item["channel_indices"])
    return out


class SetTopoReranker(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.15) -> None:
        super().__init__()
        self.input_projection = nn.Linear(int(input_dim), int(d_model))
        self.candidate_type_embedding = nn.Parameter(torch.zeros(int(d_model)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=int(d_model) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(int(d_model))
        self.delta_head = nn.Linear(int(d_model), 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x) + self.candidate_type_embedding.view(1, 1, -1)
        all_invalid = ~mask.any(dim=1)
        key_padding_mask = ~mask
        if torch.any(all_invalid):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid] = False
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        delta = self.delta_head(self.norm(h)).squeeze(-1)
        return delta.masked_fill(~mask, 0.0)


def _settopo_losses(
    delta: torch.Tensor,
    batch: dict[str, Any],
    *,
    alpha: float,
    lambda_clean_nez: float,
    lambda_noisy_pos_rank: float,
    lambda_topology: float,
    lambda_residual: float,
    ranking_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"].to(delta.device)
    base = batch["base_suspicious_logit"].to(delta.device)
    final = base + float(alpha) * delta
    true_nez = batch["true_nez"].to(delta.device)
    true_ez = batch["true_ez"].to(delta.device)
    anchors = batch["is_pseudo_clean_nez_anchor"].to(delta.device)
    raw_onset = batch["raw_dist_onset_z"].to(delta.device)
    shaft = batch["shaft_code"].to(delta.device)
    contact = batch["contact_number"].to(delta.device)

    clean_mask = mask & true_nez.eq(1)
    clean_loss = F.binary_cross_entropy_with_logits(final[clean_mask], torch.zeros_like(final[clean_mask])) if clean_mask.any() else final.sum() * 0.0

    rank_losses = []
    topo_losses = []
    for row_idx in range(final.shape[0]):
        valid = mask[row_idx]
        pos_scores = final[row_idx][valid & true_ez[row_idx].eq(1)]
        anchor_scores = final[row_idx][valid & ((anchors[row_idx].eq(1)) | true_nez[row_idx].eq(1))]
        if pos_scores.numel() and anchor_scores.numel():
            diff = pos_scores[:, None] - anchor_scores[None, :]
            rank_losses.append(F.softplus(float(ranking_margin) - diff).mean())
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        for a_pos in range(valid_idx.numel()):
            i = valid_idx[a_pos]
            for j in valid_idx[a_pos + 1 :]:
                same_shaft = shaft[row_idx, i] > 0 and torch.isclose(shaft[row_idx, i], shaft[row_idx, j])
                if not bool(same_shaft):
                    continue
                if contact[row_idx, i] <= 0 or contact[row_idx, j] <= 0:
                    continue
                if abs(float(contact[row_idx, i] - contact[row_idx, j])) > 2.0:
                    continue
                raw_threshold = torch.quantile(raw_onset[row_idx][valid], 0.90)
                high_raw = raw_onset[row_idx, [i, j]] > raw_threshold
                if float(raw_threshold.detach().cpu()) > 0.0 and bool(high_raw.any()):
                    continue
                topo_losses.append((final[row_idx, i] - final[row_idx, j]).square())
    rank_loss = torch.stack(rank_losses).mean() if rank_losses else final.sum() * 0.0
    topo_loss = torch.stack(topo_losses).mean() if topo_losses else final.sum() * 0.0
    residual_loss = delta[mask].square().mean() if mask.any() else delta.sum() * 0.0
    loss = (
        float(lambda_clean_nez) * clean_loss
        + float(lambda_noisy_pos_rank) * rank_loss
        + float(lambda_topology) * topo_loss
        + float(lambda_residual) * residual_loss
    )
    return loss, {
        "clean_nez_loss": float(clean_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "topology_loss": float(topo_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
    }


def _device(name: str) -> torch.device:
    if str(name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(name))


def _parse_float_list(values: str | Sequence[float]) -> list[float]:
    if isinstance(values, str):
        return [float(item.strip()) for item in values.split(",") if item.strip()]
    return [float(value) for value in values]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def _split_train_val_subjects(subjects: Sequence[str], *, seed: int) -> tuple[set[str], set[str]]:
    subjects = list(subjects)
    if len(subjects) < 3:
        return set(subjects), set()
    rng = np.random.default_rng(int(seed))
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(0.2 * len(shuffled))))
    return set(shuffled[n_val:]), set(shuffled[:n_val])


def _fit_real_settopo_fold(
    rows: pd.DataFrame,
    *,
    fold_idx: int,
    feature_cols: Sequence[str],
    output_dir: Path,
    d_model: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    alpha: float,
    lambda_clean_nez: float,
    lambda_noisy_pos_rank: float,
    lambda_topology: float,
    lambda_residual: float,
    ranking_margin: float,
    device_name: str,
    amp: bool,
    patience: int,
    seed: int,
) -> tuple[pd.Series, list[dict[str, Any]], dict[str, Any]]:
    train_mask = ~rows["fold_idx"].astype(int).eq(int(fold_idx))
    test_mask = rows["fold_idx"].astype(int).eq(int(fold_idx))
    train_subjects_all = sorted(rows.loc[train_mask, "subject_id"].astype(str).unique())
    fit_subjects, val_subjects = _split_train_val_subjects(train_subjects_all, seed=int(seed) + int(fold_idx))
    fit_rows = rows[rows["subject_id"].astype(str).isin(fit_subjects)].copy()
    val_rows = rows[rows["subject_id"].astype(str).isin(val_subjects)].copy()
    test_rows = rows[test_mask].copy()

    scaler = StandardScaler()
    train_candidate = fit_rows[fit_rows["is_candidate"].astype(bool)]
    if len(train_candidate) == 0:
        train_candidate = rows[train_mask & rows["is_candidate"].astype(bool)]
    if len(train_candidate) == 0:
        return pd.Series(0.0, index=test_rows.index), [], {"fallback_zero_delta": True, "reason": "no_train_candidates"}
    scaler.fit(train_candidate[list(feature_cols)].fillna(0.0).to_numpy(dtype=np.float64))
    for frame in (fit_rows, val_rows, test_rows):
        if len(frame) > 0:
            frame.loc[:, list(feature_cols)] = scaler.transform(frame[list(feature_cols)].fillna(0.0).to_numpy(dtype=np.float64))

    fit_ds = CandidateSetDataset(fit_rows, feature_cols)
    val_ds = CandidateSetDataset(val_rows, feature_cols) if not val_rows.empty else CandidateSetDataset(fit_rows, feature_cols)
    test_ds = CandidateSetDataset(test_rows, feature_cols)
    if len(fit_ds) == 0 or len(test_ds) == 0:
        return pd.Series(0.0, index=test_rows.index), [], {"fallback_zero_delta": True, "reason": "empty_dataset"}
    device = _device(device_name)
    torch.manual_seed(int(seed) + int(fold_idx))
    model = SetTopoReranker(
        input_dim=len(feature_cols),
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    grad_scaler = torch.amp.GradScaler("cuda", enabled=bool(amp) and device.type == "cuda")
    loader = DataLoader(fit_ds, batch_size=max(1, int(batch_size)), shuffle=True, collate_fn=_collate_sets)
    val_loader = DataLoader(val_ds, batch_size=max(1, int(batch_size)), shuffle=False, collate_fn=_collate_sets)
    loss_rows: list[dict[str, Any]] = []
    best_val = None
    best_state = None
    bad_epochs = 0
    max_epochs = max(1, int(epochs))
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        train_parts: dict[str, list[float]] = defaultdict(list)
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=bool(amp) and device.type == "cuda"):
                delta = model(batch["x"], batch["mask"])
                loss, parts = _settopo_losses(
                    delta,
                    batch,
                    alpha=alpha,
                    lambda_clean_nez=lambda_clean_nez,
                    lambda_noisy_pos_rank=lambda_noisy_pos_rank,
                    lambda_topology=lambda_topology,
                    lambda_residual=lambda_residual,
                    ranking_margin=ranking_margin,
                )
            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt)
            grad_scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                train_parts[key].append(value)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                delta = model(batch["x"], batch["mask"])
                loss, _ = _settopo_losses(
                    delta,
                    batch,
                    alpha=alpha,
                    lambda_clean_nez=lambda_clean_nez,
                    lambda_noisy_pos_rank=lambda_noisy_pos_rank,
                    lambda_topology=lambda_topology,
                    lambda_residual=lambda_residual,
                    ranking_margin=ranking_margin,
                )
                val_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        loss_rows.append(
            {
                "epoch": int(epoch),
                "fold_idx": int(fold_idx),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "clean_nez_loss": float(np.mean(train_parts["clean_nez_loss"])) if train_parts["clean_nez_loss"] else 0.0,
                "rank_loss": float(np.mean(train_parts["rank_loss"])) if train_parts["rank_loss"] else 0.0,
                "topology_loss": float(np.mean(train_parts["topology_loss"])) if train_parts["topology_loss"] else 0.0,
                "residual_loss": float(np.mean(train_parts["residual_loss"])) if train_parts["residual_loss"] else 0.0,
            }
        )
        if best_val is None or val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(patience) > 0 and bad_epochs >= int(patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "feature_columns": list(feature_cols)}, output_dir / f"settopo_model_fold_{fold_idx}.pt")
    with (output_dir / f"settopo_scaler_fold_{fold_idx}.pkl").open("wb") as fout:
        pickle.dump(scaler, fout)
    pred = pd.Series(0.0, index=test_rows.index)
    if len(test_ds):
        test_loader = DataLoader(test_ds, batch_size=max(1, int(batch_size)), shuffle=False, collate_fn=_collate_sets)
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                delta = model(batch_on_device["x"], batch_on_device["mask"]).detach().cpu().numpy()
                for row_idx, indices in enumerate(batch["channel_indices"]):
                    n = len(indices)
                    pred.loc[indices] = delta[row_idx, :n]
    audit = {
        "train_subjects": int(len(train_subjects_all)),
        "fit_subjects": int(len(fit_subjects)),
        "val_subjects": int(len(val_subjects)),
        "test_subjects": int(test_rows["subject_id"].nunique()),
        "train_candidate_rows": int(rows.loc[train_mask, "is_candidate"].astype(bool).sum()),
        "test_candidate_rows": int(test_rows["is_candidate"].astype(bool).sum()),
        "fallback_zero_delta": False,
    }
    return pred, loss_rows, audit


def _fit_ridge_baseline_fold(rows: pd.DataFrame, *, fold_idx: int, feature_cols: Sequence[str]) -> tuple[pd.Series, dict[str, Any]]:
    train_mask = ~rows["fold_idx"].astype(int).eq(int(fold_idx))
    test_mask = rows["fold_idx"].astype(int).eq(int(fold_idx))
    candidate_train = train_mask & rows["is_candidate"].astype(bool)
    candidate_test = test_mask & rows["is_candidate"].astype(bool)
    pred = pd.Series(0.0, index=rows[test_mask].index)
    target = (1.0 - pd.to_numeric(rows.loc[candidate_train, "true_nez"], errors="coerce").fillna(1.0)).astype(float)
    if int(candidate_train.sum()) >= 2 and target.nunique() >= 2 and int(candidate_test.sum()) > 0:
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(rows.loc[candidate_train, feature_cols].fillna(0.0).to_numpy(dtype=np.float64), target.to_numpy(dtype=np.float64))
        pred.loc[rows.loc[candidate_test].index] = model.predict(rows.loc[candidate_test, feature_cols].fillna(0.0).to_numpy(dtype=np.float64))
        fallback = False
    else:
        fallback = True
    return pred, {"fallback_zero_delta": fallback, "train_candidate_rows": int(candidate_train.sum()), "test_candidate_rows": int(candidate_test.sum())}


def run_settopo_reranker(
    clean_nez_distance_ledger: str | Path,
    output_dir: str | Path,
    *,
    alpha_list: str | Sequence[float] = (0.10,),
    train_alpha: float = 0.10,
    candidate_rule: str = "union_top30_feature_raw_neighbors",
    model_type: str = "real_settopo",
    d_model: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.15,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    lambda_clean_nez: float = 1.0,
    lambda_noisy_pos_rank: float = 0.20,
    lambda_topology: float = 0.01,
    lambda_residual: float = 0.001,
    ranking_margin: float = 0.10,
    w_feature: float = 1.0,
    w_raw_onset: float = 1.0,
    w_raw_all: float = 0.5,
    w_raw_preictal: float = 0.25,
    device: str = "auto",
    amp: bool = False,
    patience: int = 8,
    seed: int = 42,
    allowed_subjects_ledger: str | Path | None = None,
    allowed_subjects_file: str | Path | None = None,
    require_n_patients: int | None = None,
    label_encoding_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if model_type not in {"real_settopo", "ridge_residual_baseline"}:
        raise ValueError("model_type must be real_settopo or ridge_residual_baseline")
    rows = pd.read_csv(clean_nez_distance_ledger)
    rows["subject_id"] = rows["subject_id"].astype(str)
    rows = add_label_encoding_columns(rows, label_encoding_mode=label_encoding_mode)
    rows, filter_audit = apply_allowed_subject_filter(
        rows,
        allowed_subjects_ledger=allowed_subjects_ledger,
        allowed_subjects_file=allowed_subjects_file,
        require_n_patients=require_n_patients,
    )
    rows, candidate_audit = generate_clean_nez_candidates(rows, candidate_rule=candidate_rule)
    if "rank_feature_suspicious" in rows.columns:
        rows["rank_feature_suspicious_norm"] = rows.groupby("subject_id")["rank_feature_suspicious"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").fillna(len(s)).astype(float) / max(len(s), 1)
        )
    base, missing_base = compute_base_suspicious_logit(
        rows,
        w_feature=w_feature,
        w_raw_onset=w_raw_onset,
        w_raw_all=w_raw_all,
        w_raw_preictal=w_raw_preictal,
    )
    rows["base_suspicious_logit"] = base
    rows["settopo_delta"] = 0.0
    feature_cols = _feature_columns(rows)
    folds = sorted(int(value) for value in pd.to_numeric(rows["fold_idx"], errors="coerce").dropna().unique())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_rows: list[dict[str, Any]] = []
    fold_audits: dict[str, Any] = {}
    alphas = _parse_float_list(alpha_list)
    train_alpha = float(train_alpha)
    if not any(abs(float(alpha) - train_alpha) <= 1e-12 for alpha in alphas):
        alphas = [train_alpha] + alphas
    for fold_idx in folds:
        if model_type == "real_settopo":
            pred, fold_losses, fold_audit = _fit_real_settopo_fold(
                rows,
                fold_idx=fold_idx,
                feature_cols=feature_cols,
                output_dir=output_dir,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                alpha=train_alpha,
                lambda_clean_nez=lambda_clean_nez,
                lambda_noisy_pos_rank=lambda_noisy_pos_rank,
                lambda_topology=lambda_topology,
                lambda_residual=lambda_residual,
                ranking_margin=ranking_margin,
                device_name=device,
                amp=amp,
                patience=patience,
                seed=seed,
            )
            loss_rows.extend(fold_losses)
        else:
            pred, fold_audit = _fit_ridge_baseline_fold(rows, fold_idx=fold_idx, feature_cols=feature_cols)
            loss_rows.append({"epoch": 1, "fold_idx": int(fold_idx), "train_loss": 0.0, "val_loss": 0.0, "clean_nez_loss": 0.0, "rank_loss": 0.0, "topology_loss": 0.0, "residual_loss": 0.0})
        rows.loc[pred.index, "settopo_delta"] = pred
        fold_audits[str(fold_idx)] = fold_audit

    main_output = rows.copy()
    file_prefix = "corrected_suspicious_ledger" if model_type == "real_settopo" else "ridge_residual_ledger"
    for alpha in alphas:
        out = rows.copy()
        out["alpha"] = float(alpha)
        out["main_alpha"] = float(train_alpha)
        out["train_alpha"] = float(train_alpha)
        out["diagnostic_only"] = bool(abs(float(alpha) - train_alpha) > 1e-12)
        out["alpha_role"] = "main" if abs(float(alpha) - train_alpha) <= 1e-12 else "diagnostic_only"
        out["final_suspicious_logit"] = out["base_suspicious_logit"].astype(float) + float(alpha) * out["settopo_delta"].astype(float)
        out["final_suspicious_score"] = _sigmoid(out["final_suspicious_logit"].to_numpy(dtype=np.float64))
        out["rank_final_suspicious"] = out.groupby("subject_id")["final_suspicious_score"].rank(ascending=False, method="first").astype(int)
        out["oracle_top_true_count_pred"] = 0
        out["predicted_by_oracle_k"] = 0
        for _, group in out.groupby("subject_id", sort=False):
            k = int(pd.to_numeric(group["true_ez"], errors="coerce").fillna(0).sum())
            if k > 0:
                idx = group.sort_values("final_suspicious_score", ascending=False, kind="mergesort").head(k).index
                out.loc[idx, ["oracle_top_true_count_pred", "predicted_by_oracle_k"]] = 1
        out.to_csv(output_dir / f"{file_prefix}_alpha{float(alpha):.2f}.csv", index=False)
        if abs(float(alpha) - train_alpha) <= 1e-12:
            main_output = out
    pd.DataFrame(loss_rows).to_csv(output_dir / "settopo_loss_curve.csv", index=False)
    pd.DataFrame([{"fold_idx": int(k), **v} for k, v in fold_audits.items()]).to_csv(output_dir / "settopo_fold_metrics.csv", index=False)
    audit = {
        **filter_audit,
        "model_type": model_type,
        "method_name": "CleanNEZ-RawBB-SetTopo" if model_type == "real_settopo" else "CleanNEZ-RawBB-RidgeResidual",
        "real_settopo_used": model_type == "real_settopo",
        "train_alpha": float(train_alpha),
        "main_alpha": float(train_alpha),
        "alpha_list": [float(alpha) for alpha in alphas],
        "alpha_protocol_no_test_selection": True,
        "diagnostic_alpha_outputs": any(abs(float(alpha) - train_alpha) > 1e-12 for alpha in alphas),
        "diagnostic_alphas": [float(alpha) for alpha in alphas if abs(float(alpha) - train_alpha) > 1e-12],
        "settopo_uses_clinical_true_ez_for_ranking": True,
        "settopo_uses_clinical_true_nez_for_clean_loss": True,
        "raw_binary_label_not_used_as_clinical_target": True,
        "architecture": {"d_model": int(d_model), "num_layers": int(num_layers), "num_heads": int(num_heads), "dropout": float(dropout)},
        "training": {"epochs": int(epochs), "learning_rate": float(learning_rate), "weight_decay": float(weight_decay), "batch_size": int(batch_size), "patience": int(patience), "seed": int(seed), "device": str(device), "amp": bool(amp)},
        "loss_weights": {"lambda_clean_nez": float(lambda_clean_nez), "lambda_noisy_pos_rank": float(lambda_noisy_pos_rank), "lambda_topology": float(lambda_topology), "lambda_residual": float(lambda_residual), "ranking_margin": float(ranking_margin)},
        "base_score_weights": {"w_feature": float(w_feature), "w_raw_onset": float(w_raw_onset), "w_raw_all": float(w_raw_all), "w_raw_preictal": float(w_raw_preictal)},
        "missing_base_score_columns": missing_base,
        "row_count": int(len(rows)),
        "feature_columns": feature_cols,
        "forbidden_feature_intersection": sorted(FORBIDDEN_MODEL_FEATURES.intersection(feature_cols)),
        "no_center_features": not {"center", "center_id"}.intersection(feature_cols),
        "no_outcome_features": not {"outcome_group", "surgery_success"}.intersection(feature_cols),
        "true_ez_count_not_used_as_feature": "true_ez_count" not in feature_cols,
        "candidate_audit": candidate_audit,
        "fold_audits": fold_audits,
    }
    if "SetTopo" in audit["method_name"] and model_type != "real_settopo":
        raise RuntimeError("SetTopo method name cannot be used when model_type is not real_settopo.")
    with (output_dir / "settopo_training_audit.json").open("w", encoding="utf-8") as fout:
        json.dump(json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    return main_output, audit


__all__ = [
    "BASE_SCORE_DEFAULT_WEIGHTS",
    "CandidateSetDataset",
    "FORBIDDEN_MODEL_FEATURES",
    "SetTopoReranker",
    "compute_base_suspicious_logit",
    "generate_clean_nez_candidates",
    "run_settopo_reranker",
]
