"""Frozen A1 temporal/recruitment/persistence matched probes (FIT/validation only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT, build_fold, install_interleaved_hlv_view, make_args, sha256  # noqa: E402
sys.path.insert(0, str(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import epoch_grid, finalize_fold  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402
from reproduce_source import A1_RUNTIME, LOCK_SHA256, RUNTIME, ensure_source  # noqa: E402

VARIANTS = {"A": ("A1", "A2"), "B": ("B1", "B2"), "C": ("C1", "C2")}
INPUT_DIMS = {"A": 32, "B": 20, "C": 30}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_source() -> None:
    ensure_source()
    path = EXPERIMENT / "SOURCE_REPRODUCTION.json"
    if not path.exists():
        raise RuntimeError("Run exact A1 source reproduction first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("pass") or abs(payload["observed_mean"] - 0.6259962097139906) > 1e-6:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")


class Head(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.tanh(self.net(x).squeeze(-1))


def extract(model, loader, device) -> list[dict]:
    captures: dict[str, torch.Tensor] = {}

    def capture_window(_module, _inputs, output):
        captures["w"] = output.detach()

    def capture_q(_module, _inputs, output):
        captures["q"] = output[0].detach()

    def capture_r(_module, inputs):
        captures["r"] = inputs[0].detach()

    handles = [model.b0_encoder.register_forward_hook(capture_window),
               model.temporal_encoder.register_forward_hook(capture_q),
               model.channel_classifier.classifier.register_forward_pre_hook(capture_r)]
    rows: list[dict] = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                captures.clear()
                out = model(core._move_tensors_to_device(batch, device))
                if set(captures) != {"w", "q", "r"}:
                    raise RuntimeError(f"Missing intermediate frozen outputs: {set(captures)}")
                reconstructed = model.channel_classifier.classifier(captures["r"]).squeeze(-1)
                valid_logits = batch["channel_mask"].to(device)
                if not torch.allclose(reconstructed[valid_logits], out["logits"][valid_logits], rtol=0, atol=1e-6):
                    raise RuntimeError("Pre-classifier embedding does not replay frozen A1 logits")
                for i, subject_id in enumerate(batch["subject_id"]):
                    channel_ids = torch.nonzero(batch["channel_mask"][i], as_tuple=False).flatten()
                    if channel_ids.numel() == 0:
                        raise RuntimeError("Patient has no valid channels")
                    seizures = []
                    for s in torch.nonzero(batch["seizure_mask"][i], as_tuple=False).flatten().tolist():
                        windows = torch.nonzero(batch["window_mask"][i, s], as_tuple=False).flatten()
                        if windows.numel() == 0:
                            raise RuntimeError("A1 seizure has no valid windows")
                        x = batch["b0_features"][i, s].index_select(0, windows).index_select(1, channel_ids).numpy()
                        w = captures["w"][i, s].index_select(0, windows.to(device)).index_select(1, channel_ids.to(device)).cpu().numpy()
                        if x.shape[-1] != 36 or w.shape[-1] != 32:
                            raise RuntimeError(f"Intermediate dimensions changed: {x.shape}, {w.shape}")
                        seizures.append({"x": x.astype(np.float32), "w": w.astype(np.float32),
                                         "q": captures["q"][i, s].index_select(0, channel_ids.to(device)).cpu().numpy().astype(np.float32),
                                         "centers": batch["window_centers"][i, s].index_select(0, windows).numpy().astype(np.float32),
                                         "channel_mask": batch["seizure_channel_mask"][i, s].index_select(0, channel_ids).numpy().astype(bool),
                                         "window_mask": np.ones(len(windows), dtype=bool)})
                    channel_ids_device = channel_ids.to(device)
                    h = out["patient_channel_embedding"][i].index_select(0, channel_ids_device).cpu().numpy().astype(np.float32)
                    r = captures["r"][i].index_select(0, channel_ids_device).cpu().numpy().astype(np.float32)
                    logits = out["logits"][i].index_select(0, channel_ids_device).cpu().numpy().astype(np.float32)
                    rows.append({"subject_id": str(subject_id), "seizures": seizures, "h": h, "r": r,
                                 "logits": logits, "score_nez": out["score_nez"][i].index_select(0, channel_ids_device).cpu().numpy().astype(np.float32),
                                 "score_ez": out["score_ez"][i].index_select(0, channel_ids_device).cpu().numpy().astype(np.float32),
                                 "y_nez": batch["labels_nez"][i].index_select(0, channel_ids).numpy().astype(np.float32),
                                 "y_ez": batch["labels_ez"][i].index_select(0, channel_ids).numpy().astype(np.float32)})
    finally:
        for handle in handles:
            handle.remove()
    if len(rows) != len({row["subject_id"] for row in rows}):
        raise RuntimeError("Duplicate frozen patient")
    return sorted(rows, key=lambda row: row["subject_id"])


def get_repr(exp, model, fit_loader, val_loader, folder: Path, source_hash: str) -> tuple[list[dict], list[dict]]:
    path = folder / "representation.pt"
    if path.exists():
        print(f"[TRACE] loading representation {folder.parent.name}/{folder.name}", flush=True)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["lock_sha256"] != LOCK_SHA256 or saved["source_checkpoint_sha256"] != source_hash:
            raise RuntimeError("Frozen representation resume provenance mismatch")
        return saved["fit"], saved["validation"]
    fit, val = extract(model, fit_loader, exp.device), extract(model, val_loader, exp.device)
    if len(val) != 13 or not fit:
        raise RuntimeError("Frozen FIT/validation roles changed")
    if shutil.disk_usage(RUNTIME).free < 2_000_000_000:
        raise RuntimeError("Insufficient private runtime disk space for intermediate representation")
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"fit": fit, "validation": val, "lock_sha256": LOCK_SHA256,
                "source_checkpoint_sha256": source_hash, "outer_test_accessed": False}, temporary)
    temporary.replace(path)
    return fit, val


def linear_slope(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n < 2:
        return np.zeros(values.shape[1:], dtype=np.float64)
    u = np.linspace(0, 1, n, dtype=np.float64)
    centered = u - u.mean()
    return np.tensordot(centered, values, axes=(0, 0)) / np.dot(centered, centered)


def trajectory_summary(w: np.ndarray) -> np.ndarray:
    n = len(w)
    k = max(1, math.ceil(0.2 * n))
    early, late = w[:k].mean(axis=0), w[-k:].mean(axis=0)
    first, last = w[0], w[-1]
    slope = linear_slope(w)
    variation = np.abs(np.diff(w, axis=0)).mean(axis=0) if n > 1 else np.zeros_like(first)
    return np.concatenate((early, late, late - early, first, last, last - first, slope, variation))


def trajectory(rows: list[dict], fold: int, epoch: int, shuffled: bool) -> list[np.ndarray]:
    features = []
    for row in rows:
        channels = len(row["logits"])
        per_channel: list[list[np.ndarray]] = [[] for _ in range(channels)]
        for seizure_index, seizure in enumerate(row["seizures"]):
            for channel in np.flatnonzero(seizure["channel_mask"]):
                w = seizure["w"][:, channel].astype(np.float64)
                if shuffled:
                    key = f"42|{fold}|{epoch}|{row['subject_id']}|{seizure_index}|{channel}".encode()
                    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
                    w = w[np.random.default_rng(seed).permutation(len(w))]
                per_channel[channel].append(trajectory_summary(w))
        features.append(np.stack([np.mean(items, axis=0) if items else np.zeros(256) for items in per_channel]).astype(np.float32))
    return features


def project_trajectory(fit: list[np.ndarray], val: list[np.ndarray], seed: int) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    matrix = np.concatenate(fit).astype(np.float64)
    if matrix.shape[1] != 256 or matrix.shape[0] < 32:
        raise RuntimeError("Unexpected FIT trajectory matrix")
    pca = PCA(n_components=32, svd_solver="randomized", random_state=seed)
    projected = pca.fit_transform(matrix)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k = min(32, int(np.searchsorted(cumulative, 0.95) + 1))
    def pad(a):
        out = np.zeros((len(a), 32), dtype=np.float32)
        out[:, :k] = a[:, :k].astype(np.float32)
        return out
    fit_out, start = [], 0
    for f in fit:
        fit_out.append(pad(projected[start:start + len(f)]))
        start += len(f)
    val_out = [pad(pca.transform(v.astype(np.float64))) for v in val]
    return fit_out, val_out, {"fit_rows": len(matrix), "selected_components": k,
                              "explained_variance_at_cap": float(cumulative[-1]),
                              "explained_variance_selected": float(cumulative[k - 1])}


def fit_change_scaler(fit: list[dict]) -> tuple[np.ndarray, np.ndarray, float, float]:
    total = np.zeros(27, dtype=np.float64)
    squares = np.zeros(27, dtype=np.float64)
    count = 0
    for row in fit:
        for seizure in row["seizures"]:
            values = seizure["x"][:, seizure["channel_mask"], 9:36].reshape(-1, 27).astype(np.float64)
            if len(values):
                total += values.sum(axis=0)
                squares += np.square(values).sum(axis=0)
                count += len(values)
    if count <= 0:
        raise RuntimeError("No valid FIT change-view cells")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0.0)) + 1e-6
    magnitudes = []
    for row in fit:
        for seizure in row["seizures"]:
            values = seizure["x"][:, seizure["channel_mask"], 9:36].reshape(-1, 27).astype(np.float64)
            if len(values):
                magnitudes.append(np.sqrt(np.mean(np.square((values - mean) / std), axis=-1)))
    all_magnitudes = np.concatenate(magnitudes)
    q80, q90 = np.quantile(all_magnitudes, [0.8, 0.9])
    return mean, std, float(q80), float(q90)


def first_time(condition: np.ndarray) -> float:
    where = np.flatnonzero(condition)
    return float(where[0] / max(len(condition) - 1, 1)) if len(where) else 1.0


def small_linear_quantile(values: np.ndarray, fraction: float) -> float:
    """NumPy's linear-quantile semantics for short strided arrays, without Windows native partition crash."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise RuntimeError("Empty small quantile")
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def rank_and_magnitude(seizure: dict, mean: np.ndarray, std: np.ndarray,
                       q80: float, q90: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = seizure["x"].astype(np.float64)
    if x.shape[-1] != 36:
        raise RuntimeError("Expected exact four-view normalized A1 input")
    magnitude = np.sqrt(np.mean(np.square((x[..., 9:36] - mean) / std), axis=-1))
    n_windows, n_channels = magnitude.shape
    valid_channels = seizure["channel_mask"]
    percentile = np.zeros_like(magnitude)
    top20 = np.zeros_like(magnitude, dtype=bool)
    top10 = np.zeros_like(magnitude, dtype=bool)
    n_valid = int(valid_channels.sum())
    if n_valid == 0:
        return (np.zeros((n_channels, 10), np.float32), np.zeros((n_channels, 10), np.float32),
                np.zeros(n_channels, dtype=bool))
    for t in range(n_windows):
        descending = rankdata(-magnitude[t, valid_channels], method="average")
        percentile[t, valid_channels] = (n_valid - descending) / max(n_valid - 1, 1) if n_valid > 1 else 1.0
        top20[t, valid_channels] = descending <= max(1, math.ceil(0.2 * n_valid))
        top10[t, valid_channels] = descending <= max(1, math.ceil(0.1 * n_valid))
    k = max(1, math.ceil(0.2 * n_windows))
    rank_features = np.zeros((n_channels, 10), dtype=np.float32)
    magnitude_features = np.zeros((n_channels, 10), dtype=np.float32)
    for c in np.flatnonzero(valid_channels):
        p, m = percentile[:, c], magnitude[:, c]
        rank_features[c] = [p[:k].mean(), p[:k].max(), top20[:k, c].mean(), p.mean(),
                            first_time(top20[:, c]), first_time(top10[:, c]), top20[:, c].mean(),
                            p.max(), linear_slope(p[:, None])[0], np.abs(np.diff(p)).mean() if n_windows > 1 else 0.0]
        above20, above10 = m >= q80, m >= q90
        magnitude_features[c] = [m[:k].mean(), m[:k].max(), above20[:k].mean(), m.mean(),
                                 first_time(above20), first_time(above10), above20.mean(),
                                 m.max(), linear_slope(m[:, None])[0], np.abs(np.diff(m)).mean() if n_windows > 1 else 0.0]
    return rank_features, magnitude_features, top10.any(axis=0)


def aggregate_seizures(row: dict, mean: np.ndarray, std: np.ndarray, q80: float, q90: float) -> dict[str, np.ndarray]:
    channels = len(row["logits"])
    rank_lists: list[list[np.ndarray]] = [[] for _ in range(channels)]
    magnitude_lists: list[list[np.ndarray]] = [[] for _ in range(channels)]
    top10_lists: list[list[bool]] = [[] for _ in range(channels)]
    for seizure in row["seizures"]:
        rank, magnitude, top10_entry = rank_and_magnitude(seizure, mean, std, q80, q90)
        for c in np.flatnonzero(seizure["channel_mask"]):
            rank_lists[c].append(rank[c])
            magnitude_lists[c].append(magnitude[c])
            top10_lists[c].append(bool(top10_entry[c]))
    b_rank, b_magnitude, c_simple, c_persistence = [], [], [], []
    for rank_items, mag_items, top10_items in zip(rank_lists, magnitude_lists, top10_lists, strict=True):
        if not rank_items:
            b_rank.append(np.zeros(20)); b_magnitude.append(np.zeros(20))
            c_simple.append(np.zeros(30)); c_persistence.append(np.zeros(30))
            continue
        r = np.stack(rank_items).astype(np.float64)
        m = np.stack(mag_items).astype(np.float64)
        rank_simple = np.concatenate([r.mean(axis=0), r.std(axis=0)])
        magnitude_simple = np.concatenate([m.mean(axis=0), m.std(axis=0)])
        p = r[:, 3]
        if len(p) == 1:
            iqr, stability, loo_variance = 0.0, 1.0, 0.0
        else:
            iqr = small_linear_quantile(p, .75) - small_linear_quantile(p, .25)
            stability = float(1 / (1 + p.std()))
            loo_variance = float(np.var([(p.sum() - p[s]) / (len(p) - 1) for s in range(len(p))]))
        extras = np.asarray([small_linear_quantile(p, .5), iqr, stability, np.mean(r[:, 6] > 0),
                             np.mean(top10_items), np.mean(r[:, 2] > 0), loo_variance,
                             len(p) / len(row["seizures"]), p.min(), p.max()], dtype=np.float64)
        b_rank.append(rank_simple)
        b_magnitude.append(magnitude_simple)
        c_simple.append(np.concatenate([rank_simple, np.zeros(10)]))
        c_persistence.append(np.concatenate([rank_simple, extras]))
    return {"B1": np.stack(b_magnitude).astype(np.float32), "B2": np.stack(b_rank).astype(np.float32),
            "C1": np.stack(c_simple).astype(np.float32), "C2": np.stack(c_persistence).astype(np.float32)}


def make_features(fit: list[dict], val: list[dict], audit: str, fold: int, epoch: int) -> tuple[dict, dict, dict]:
    if audit == "A":
        seed = 42 + fold * 1000 + epoch * 10
        ordered_fit, ordered_val = trajectory(fit, fold, epoch, False), trajectory(val, fold, epoch, False)
        shuffled_fit, shuffled_val = trajectory(fit, fold, epoch, True), trajectory(val, fold, epoch, True)
        f1, v1, d1 = project_trajectory(ordered_fit, ordered_val, seed)
        f2, v2, d2 = project_trajectory(shuffled_fit, shuffled_val, seed)
        return {"A1": f1, "A2": f2}, {"A1": v1, "A2": v2}, {"A1_PCA": d1, "A2_PCA": d2}
    payload = cached_recruitment_features(fit, val, fold)
    return ({variant: payload["fit_features"][variant] for variant in VARIANTS[audit]},
            {variant: payload["val_features"][variant] for variant in VARIANTS[audit]},
            payload["diagnostics"])


def cached_recruitment_features(fit: list[dict], val: list[dict], fold: int) -> dict:
    """Input views/normalizer do not depend on frozen A1 epoch; cache exact FIT-only features once per fold."""
    path = RUNTIME / "private" / f"fold_{fold}" / "recruitment_fitval.pt"
    fit_identity = [(r["subject_id"], len(r["logits"]), len(r["seizures"])) for r in fit]
    val_identity = [(r["subject_id"], len(r["logits"]), len(r["seizures"])) for r in val]
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["lock_sha256"] != LOCK_SHA256 or payload["fit_identity"] != fit_identity or payload["val_identity"] != val_identity:
            raise RuntimeError("Fold-level recruitment feature cache provenance changed")
        return payload
    mean, std, q80, q90 = fit_change_scaler(fit)
    fit_dicts = [aggregate_seizures(row, mean, std, q80, q90) for row in fit]
    val_dicts = [aggregate_seizures(row, mean, std, q80, q90) for row in val]
    variants = ("B1", "B2", "C1", "C2")
    payload = {"fit_features": {v: [r[v] for r in fit_dicts] for v in variants},
               "val_features": {v: [r[v] for r in val_dicts] for v in variants},
               "diagnostics": {"fit_change_count": int(sum(len(row["logits"]) for row in fit)),
                               "fit_change_std_min": float(std.min()), "fit_change_std_max": float(std.max()),
                               "fit_magnitude_q80": q80, "fit_magnitude_q90": q90},
               "fit_identity": fit_identity, "val_identity": val_identity, "lock_sha256": LOCK_SHA256}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def tensors(rows: list[dict], features: list[np.ndarray], device) -> tuple[torch.Tensor, ...]:
    if len(rows) != len(features):
        raise RuntimeError("Feature patient count changed")
    x, base, y_ez, patient_index = [], [], [], []
    for i, (row, feature) in enumerate(zip(rows, features, strict=True)):
        if len(feature) != len(row["logits"]) or not np.isfinite(feature).all():
            raise RuntimeError("Probe feature alignment or finiteness changed")
        x.append(feature)
        base.append(row["logits"])
        y_ez.append(row["y_ez"])
        patient_index.append(np.full(len(feature), i, dtype=np.int64))
    return (torch.as_tensor(np.concatenate(x), device=device, dtype=torch.float32),
            torch.as_tensor(np.concatenate(base), device=device, dtype=torch.float32),
            torch.as_tensor(np.concatenate(y_ez), device=device, dtype=torch.float32),
            torch.as_tensor(np.concatenate(patient_index), device=device, dtype=torch.int64))


def train_head(fit: list[dict], features: list[np.ndarray], initial: dict, seed: int, device) -> tuple[Head, list[float]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dim = features[0].shape[1]
    head = Head(dim).to(device)
    head.load_state_dict(initial, strict=True)
    x, base, y_ez, patient_idx = tensors(fit, features, device)
    if torch.count_nonzero(head.eval()(x)).item() != 0:
        raise RuntimeError("Zero-init probe does not exactly replay A1")
    weight = torch.where(y_ez > 0.5, 2.0, 1.0)
    denominator = torch.zeros(len(fit), device=device).index_add(0, patient_idx, weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    for _epoch in range(15):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logit = base + head(x)
        channel_loss = F.binary_cross_entropy_with_logits(logit, 1 - y_ez, reduction="none")
        numerator = torch.zeros(len(fit), device=device).index_add(0, patient_idx, channel_loss * weight)
        loss = (numerator / denominator).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite frozen-probe loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return head.eval(), losses


def validation_grid(rows: list[dict], features: list[np.ndarray], head: Head, epoch: int, device) -> dict:
    records = []
    with torch.no_grad():
        for row, feature in zip(rows, features, strict=True):
            delta = head(torch.as_tensor(feature, device=device, dtype=torch.float32)).cpu().numpy()
            logit = row["logits"] + delta
            p = torch.sigmoid(torch.from_numpy(logit)).numpy().astype(np.float32)
            records.append({"subject_id": row["subject_id"], "labels": row["y_nez"],
                            "labels_nez": row["y_nez"], "labels_ez": row["y_ez"],
                            "score_nez": p, "score_ez": 1 - p,
                            "channel_mask": np.ones(len(p), dtype=bool)})
    return epoch_grid(records, epoch)


def process_epoch(exp, model, fit_loader, val_loader, normalizer, fold: int, epoch: int,
                  audits: tuple[str, ...]) -> None:
    folder = RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}"
    checkpoint_path = A1_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}.pt"
    source_hash = sha256(checkpoint_path)
    if all((folder / f"{variant}_probe.pt").exists() and (folder / f"{variant}_validation_grid.json").exists()
           for audit in audits for variant in VARIANTS[audit]):
        for audit in audits:
            for variant in VARIANTS[audit]:
                saved = torch.load(folder / f"{variant}_probe.pt", map_location="cpu", weights_only=False)
                if saved["source_checkpoint_sha256"] != source_hash or saved["lock_sha256"] != LOCK_SHA256 or saved["probe_epochs"] != 15 or saved["variant"] != variant:
                    raise RuntimeError("Complete probe resume provenance changed")
        print(f"[RESUME] fold={fold} epoch={epoch} {','.join(audits)} complete", flush=True)
        return
    checkpoint = torch.load(checkpoint_path, map_location=exp.device, weights_only=False)
    if (checkpoint["variant"], checkpoint["fold"], checkpoint["epoch"]) != ("A1", fold, epoch):
        raise RuntimeError("Frozen A1 checkpoint identity changed")
    if not np.array_equal(normalizer.mean, checkpoint["normalizer_mean"]) or not np.array_equal(normalizer.std, checkpoint["normalizer_std"]):
        raise RuntimeError("A1 FIT normalizer changed")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    fit, val = get_repr(exp, model, fit_loader, val_loader, folder, source_hash)
    print(f"[TRACE] representations ready fold={fold} epoch={epoch}", flush=True)
    for audit_index, audit in enumerate(("A", "B", "C"), start=1):
        if audit not in audits:
            continue
        pair = VARIANTS[audit]
        if all((folder / f"{v}_probe.pt").exists() and (folder / f"{v}_validation_grid.json").exists() for v in pair):
            continue
        print(f"[TRACE] deriving {audit} features fold={fold} epoch={epoch}", flush=True)
        fit_features, val_features, diagnostics = make_features(fit, val, audit, fold, epoch)
        print(f"[TRACE] derived {audit} features fold={fold} epoch={epoch}", flush=True)
        seed = 42 + fold * 1000 + epoch * 10 + audit_index
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        initialized = Head(INPUT_DIMS[audit]).to(exp.device)
        initial = {key: value.detach().clone() for key, value in initialized.state_dict().items()}
        parameter_count = sum(p.numel() for p in initialized.parameters())
        atomic_json(folder / f"{audit}_feature_diagnostics.json", diagnostics)
        for variant in pair:
            probe_path = folder / f"{variant}_probe.pt"
            grid_path = folder / f"{variant}_validation_grid.json"
            if probe_path.exists():
                payload = torch.load(probe_path, map_location="cpu", weights_only=False)
                if payload["source_checkpoint_sha256"] != source_hash or payload["lock_sha256"] != LOCK_SHA256 or payload["variant"] != variant or payload["probe_epochs"] != 15 or payload["parameter_count"] != parameter_count:
                    raise RuntimeError("Partial probe resume provenance changed")
                head = Head(INPUT_DIMS[audit]).to(exp.device)
                head.load_state_dict(payload["head_state_dict"], strict=True)
                head.eval()
            else:
                head, losses = train_head(fit, fit_features[variant], initial, seed, exp.device)
                temp_probe = probe_path.with_suffix(".pt.tmp")
                torch.save({"variant": variant, "audit": audit, "fold": fold, "base_epoch": epoch,
                            "source_checkpoint_sha256": source_hash, "lock_sha256": LOCK_SHA256,
                            "probe_epochs": 15, "parameter_count": parameter_count,
                            "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                            "training_losses": losses, "outer_test_accessed": False}, temp_probe)
                temp_probe.replace(probe_path)
            if not grid_path.exists():
                atomic_json(grid_path, validation_grid(val, val_features[variant], head, epoch, exp.device))
            print(f"[{variant}] fold={fold} base_epoch={epoch}/30 probe_epoch=15", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--epoch", type=int, choices=tuple(range(1, 31)))
    parser.add_argument("--audit", choices=("A", "B", "C"))
    opts = parser.parse_args()
    verify_source()
    install_interleaved_hlv_view()
    args = make_args("R0", RUNTIME)
    exp = core.Exp_EZHybridLocalization(args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5 or len(exp.run_records) != 256:
        raise RuntimeError("A1 cohort changed")
    for split in exp.outer_splits:
        if opts.fold is not None and int(split["fold_idx"]) != opts.fold:
            continue
        fold, train_set, train_loader, val_loader, test_loader, normalizer = build_fold(exp, split, "validation")
        if test_loader is not None or len(val_loader.dataset) != 13:
            raise RuntimeError("Outer-test loader unexpectedly built")
        fit_loader = exp._make_loader(train_set, shuffle=False, batch_size=2)
        model = exp.runtime["model_cls"](args).to(exp.device)
        exp._dry_initialize_lazy_layers(model, train_loader)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("A1 parameter is trainable")
        audits = (opts.audit,) if opts.audit else ("A", "B", "C")
        for epoch in ((opts.epoch,) if opts.epoch else range(1, 31)):
            process_epoch(exp, model, fit_loader, val_loader, normalizer, fold, epoch, audits)
    print("[PROBES] requested cells complete", flush=True)


if __name__ == "__main__":
    main()
