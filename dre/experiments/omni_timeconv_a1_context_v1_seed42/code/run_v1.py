#!/usr/bin/env python3
"""V1: frozen TimeConv R4 plus the audited A1 patient-context residual.

Only hash-verified R4/segment-logit caches are read.  The official CNN, raw
signal cache, and checkpoint are never imported into the optimizer.  Training
is CPU-only by design: this is a small residual-head experiment and avoiding
the host's intermittent CUDA runtime failures is an engineering choice that
does not alter data, model equations, optimizer settings, or selection rules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from a1_patient_channel_ranker import PatientChannelClassifier, _patient_relative_zscore


SEED = 42
CHECKPOINT_SHA256 = "442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852"
OFFICIAL_SOURCE_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"
TEST_CACHE_SHA256 = "3f5cf9a9111fad6c5843eeaba43108057df961dd6a918d83fc40cc34bb5053ad"
# This is the hash from the PASSed full-record representation gate, not the
# invalid legacy five-clip cache.  It binds the exact 296-file artifact.
TRAIN_CACHE_SHA256 = "7260eb341ea3361bb60b18653668a73fdebda56167e6fb5bb415989b05e53da2"
EXPECTED_TEST_AUROC = 0.7987673466324111
EXPECTED_TEST_AP = 0.33030115683259276
EXPECTED_TEST_F1 = 0.6597535251408214
EPS = 1e-6
THRESHOLD = 0.5
FOLDS = 5
MAX_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
DROP = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.npz")):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(value >= 0, 1.0 / (1.0 + np.exp(-value)), np.exp(value) / (1.0 + np.exp(value)))


def center_name(value: str) -> str:
    key = str(value).casefold().replace("-", "")
    table = {"hup": "HUP", "openieeg": "Open-iEEG", "sourcesink": "SourceSink",
             "zurich": "Zurich", "multicenter": "Multicenter"}
    if key not in table:
        raise RuntimeError(f"Unknown split center: {value!r}")
    return table[key]


def load_centers(split_csv: Path) -> dict[str, str]:
    centers: dict[str, str] = {}
    with split_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            patient, center = row["patient_name"], center_name(row["dataset"])
            old = centers.setdefault(patient, center)
            if old != center:
                raise RuntimeError(f"Conflicting center for patient {patient}")
    return centers


@dataclass
class Patient:
    patient: str
    center: str
    channels: np.ndarray             # all observable good context channels
    anchors: np.ndarray              # [all_context_channels, 32]
    obs_channel_index: np.ndarray    # all official EDF-channel observations
    obs_edf: np.ndarray
    obs_label: np.ndarray            # 1 pathology, 0 normal, -1 unknown
    obs_base_normal: np.ndarray      # fixed TimeConv normal probability


def load_patients(root: Path, mode: str, centers: dict[str, str]) -> tuple[dict[str, Patient], dict]:
    paths = sorted(root.glob("*.npz"))
    expected_files = 296 if mode == "train" else 237
    if len(paths) != expected_files:
        raise RuntimeError(f"Expected {expected_files} {mode} cache files; got {len(paths)}")
    raw: dict[str, list[dict]] = defaultdict(list)
    total_segments = 0
    for path in paths:
        marker = path.with_suffix(".json")
        if not marker.is_file():
            raise RuntimeError(f"Missing marker {marker.name}")
        binding = json.loads(marker.read_text(encoding="utf-8"))
        if binding.get("output_sha256") != sha256(path):
            raise RuntimeError(f"Marker/output hash mismatch: {path.name}")
        marker_source_hash = binding.get("official_cnn_source_sha256", binding.get("official_source_sha256"))
        if binding.get("checkpoint_sha256") != CHECKPOINT_SHA256 or marker_source_hash != OFFICIAL_SOURCE_SHA256:
            raise RuntimeError(f"Frozen provenance mismatch: {path.name}")
        # The audited full-record extractors predate a ``mode`` marker field.
        # Their root-specific 296/237 count, full content digest, frozen source
        # hashes, and per-file checksum bind the split instead.  If a newer
        # artifact declares a mode, still reject a contradiction explicitly.
        declared_mode = binding.get("mode")
        if declared_mode is not None and declared_mode != mode:
            raise RuntimeError(f"Unexpected cache mode: {path.name}")
        with np.load(path, allow_pickle=False) as source:
            patient, edf = str(source["patient"]), str(source["edf"])
            names = np.asarray(source["channel_names"]).astype(str)
            labels = np.asarray(source["pathological_labels"], dtype=np.int8)
            r4 = np.asarray(source["r4_mean"], dtype=np.float32)
            logits = np.asarray(source["segment_logits"], dtype=np.float64)
            offsets = np.asarray(source["segment_offsets"], dtype=np.int64)
        if patient not in centers:
            raise RuntimeError(f"Missing split center for cached patient {patient}")
        if r4.shape != (len(names), 32) or len(labels) != len(names) or len(offsets) != len(names) + 1:
            raise RuntimeError(f"Malformed R4 cache {path.name}")
        if offsets[0] != 0 or offsets[-1] != len(logits) or np.any(np.diff(offsets) < 1):
            raise RuntimeError(f"Malformed segment offsets {path.name}")
        normal = np.asarray([sigmoid(logits[offsets[i]:offsets[i + 1]]).mean()
                             for i in range(len(names))], dtype=np.float64)
        raw[patient].append({"edf": edf, "channels": names, "labels": labels, "r4": r4,
                             "normal": normal})
        total_segments += len(logits)
    patients: dict[str, Patient] = {}
    for patient, edfs in raw.items():
        anchor_values: dict[str, list[np.ndarray]] = defaultdict(list)
        all_observations: list[tuple[str, str, int, float]] = []
        supervised: dict[str, set[int]] = defaultdict(set)
        for edf in edfs:
            for channel, label, r4, normal in zip(edf["channels"], edf["labels"], edf["r4"], edf["normal"]):
                channel = str(channel)
                anchor_values[channel].append(np.asarray(r4, dtype=np.float32))
                all_observations.append((edf["edf"], channel, int(label), float(normal)))
                if label in (0, 1):
                    supervised[channel].add(int(label))
        conflicts = [channel for channel, values in supervised.items() if len(values) != 1]
        if conflicts:
            raise RuntimeError(f"Inconsistent supervised label across EDFs for {patient}: {len(conflicts)} channels")
        channels = np.asarray(sorted(anchor_values), dtype=str)
        lookup = {channel: index for index, channel in enumerate(channels)}
        patients[patient] = Patient(
            patient=patient, center=centers[patient], channels=channels,
            anchors=np.stack([np.mean(anchor_values[channel], axis=0) for channel in channels]).astype(np.float32),
            obs_channel_index=np.asarray([lookup[channel] for _, channel, _, _ in all_observations], dtype=np.int64),
            obs_edf=np.asarray([edf for edf, _, _, _ in all_observations], dtype=str),
            obs_label=np.asarray([label for _, _, label, _ in all_observations], dtype=np.int8),
            obs_base_normal=np.asarray([normal for _, _, _, normal in all_observations], dtype=np.float64),
        )
    audit = {"cache_files": len(paths), "patients": len(patients), "all_good_context_edf_channels": int(sum(len(e["channels"]) for values in raw.values() for e in values)),
             "all_good_context_patient_channels": int(sum(len(p.channels) for p in patients.values())),
             "segments": int(total_segments), "labeled_edf_channels": int(sum(np.sum(p.obs_label >= 0) for p in patients.values()))}
    return patients, audit


class LocalResidual(torch.nn.Module):
    """Capacity-matched local-only control: 32→80→32→1, no set information."""
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(32, 80), torch.nn.GELU(), torch.nn.Dropout(DROP),
            torch.nn.Linear(80, 32), torch.nn.GELU(), torch.nn.Dropout(DROP), torch.nn.Linear(32, 1))
        self.alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def raw_delta(self, anchors: torch.Tensor, intervention: str = "full", donor: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(anchors).squeeze(-1)


class A1ContextResidual(torch.nn.Module):
    """Exact A1 ranker for delta, plus an alpha=0 residual connection."""
    def __init__(self) -> None:
        super().__init__()
        self.ranker = PatientChannelClassifier(32, num_heads=2, dropout=DROP, use_patient_relative_z=True)
        self.alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def raw_delta(self, anchors: torch.Tensor, intervention: str = "full", donor: torch.Tensor | None = None) -> torch.Tensor:
        mask = torch.ones((1, anchors.shape[0]), dtype=torch.bool, device=anchors.device)
        base = anchors.unsqueeze(0)
        if intervention == "full":
            return self.ranker(base, mask)["logits"].squeeze(0)
        if intervention == "no_attention":
            h = _patient_relative_zscore(base, mask)
            return self.ranker.classifier(h).squeeze(-1).squeeze(0)
        if intervention == "relative_zero":
            h = torch.zeros_like(base)
            context, _ = self.ranker.channel_attn(h, h, h, key_padding_mask=~mask)
            h = self.ranker.attn_norm(h + self.ranker.dropout(context))
            return self.ranker.classifier(h).squeeze(-1).squeeze(0)
        if intervention == "cross_patient_context":
            if donor is None:
                raise ValueError("cross_patient_context requires donor anchors")
            donor_mask = torch.ones((1, donor.shape[0]), dtype=torch.bool, device=anchors.device)
            query = _patient_relative_zscore(base, mask)
            kv = _patient_relative_zscore(donor.unsqueeze(0), donor_mask)
            context, _ = self.ranker.channel_attn(query, kv, kv, key_padding_mask=~donor_mask)
            h = self.ranker.attn_norm(query + self.ranker.dropout(context))
            return self.ranker.classifier(h).squeeze(-1).squeeze(0)
        raise ValueError(intervention)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def base_metrics(rows: pd.DataFrame) -> dict:
    y = rows.y.to_numpy(dtype=np.int8); score = rows.pathological_score.to_numpy(dtype=np.float64)
    pred = score >= THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "pooled_auroc": float(roc_auc_score(y, score)), "pooled_ap": float(average_precision_score(y, score)),
        "macro_f1_0_5": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "pathological_f1_0_5": float(f1_score(y, pred, labels=[0, 1], pos_label=1, zero_division=0)),
        "balanced_accuracy_0_5": float(balanced_accuracy_score(y, pred)),
        "sensitivity_0_5": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity_0_5": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "n_units": int(len(rows)), "n_pathological": int(np.sum(y == 1)), "n_normal": int(np.sum(y == 0)),
    }


def ranking_metrics(rows: pd.DataFrame) -> dict:
    ap, mrr, top1, ndcg, auc = [], [], [], [], []
    for _, group in rows.groupby("patient", sort=False):
        channel = group.groupby("channel", sort=False).agg(y=("y", "first"), pathological_score=("pathological_score", "mean")).reset_index()
        y = channel.y.to_numpy(dtype=np.int8); score = channel.pathological_score.to_numpy(dtype=np.float64)
        if len(np.unique(y)) == 2:
            auc.append(float(roc_auc_score(y, score)))
        if np.any(y == 1):
            ap.append(float(average_precision_score(y, score)))
            order = np.argsort(-score, kind="stable")
            positive = np.flatnonzero(y[order] == 1)
            mrr.append(float(1.0 / (positive[0] + 1)))
            top1.append(float(y[order[0]] == 1))
            discount = 1 / np.log2(np.arange(2, len(y) + 2))
            ideal = float((np.sort(y)[::-1] * discount).sum())
            ndcg.append(float((y[order] * discount).sum() / ideal))
    mean = lambda x: float(np.mean(x)) if x else float("nan")
    return {"patient_equal_auroc": mean(auc), "patient_equal_ap": mean(ap), "mrr": mean(mrr), "top1": mean(top1), "ndcg": mean(ndcg),
            "n_patient_auroc_estimable": len(auc), "n_patient_ranking_estimable": len(ap)}


def scores_for(model: torch.nn.Module | None, patients: dict[str, Patient], intervention: str = "full",
               donors: dict[str, str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict] = []; all_delta: list[np.ndarray] = []
    if model is not None:
        model.eval()
    with torch.inference_mode():
        for patient in sorted(patients):
            item = patients[patient]
            if model is None:
                delta = np.zeros(len(item.channels), dtype=np.float64); alpha = 0.0
            else:
                anchors = torch.from_numpy(item.anchors)
                donor = torch.from_numpy(patients[donors[patient]].anchors) if donors else None
                delta = model.raw_delta(anchors, intervention=intervention, donor=donor).detach().cpu().numpy().astype(np.float64)
                alpha = float(model.alpha.detach().cpu())
            # Base evidence stays EDF-specific.  The same anchor correction is
            # used for repeated EDF observations of each patient/channel.
            z0 = np.log(np.clip(item.obs_base_normal, EPS, 1 - EPS) / np.clip(1 - item.obs_base_normal, EPS, 1 - EPS))
            final_normal = sigmoid(z0 + alpha * delta[item.obs_channel_index])
            selected = item.obs_label >= 0
            for edf, channel_idx, label, normal in zip(item.obs_edf[selected], item.obs_channel_index[selected], item.obs_label[selected], final_normal[selected]):
                rows.append({"patient": item.patient, "center": item.center, "edf": str(edf), "channel": str(item.channels[channel_idx]),
                             "y": int(label), "pathological_score": float(1 - normal)})
            all_delta.append(delta)
    return pd.DataFrame(rows), np.concatenate(all_delta) if all_delta else np.empty(0)


def train_epoch(model: torch.nn.Module, patients: dict[str, Patient], ids: list[str], optimizer: torch.optim.Optimizer,
                rng: np.random.Generator) -> float:
    model.train(); order = list(np.asarray(ids)[rng.permutation(len(ids))]); losses = []
    for start in range(0, len(order), 2):
        batch_losses = []
        for patient in order[start:start + 2]:
            item = patients[patient]; supervised = np.flatnonzero(item.obs_label >= 0)
            if not len(supervised):
                continue
            delta = model.raw_delta(torch.from_numpy(item.anchors))
            base = torch.from_numpy(item.obs_base_normal[supervised]).to(torch.float64)
            base_logit = torch.logit(base.clamp(EPS, 1 - EPS))
            correction = (model.alpha * delta[item.obs_channel_index[supervised]]).to(torch.float64)
            normal_logit = base_logit + correction
            target = torch.from_numpy((1 - item.obs_label[supervised]).astype(np.float64))
            weight = torch.where(target < .5, torch.tensor(2.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
            loss = (F.binary_cross_entropy_with_logits(normal_logit, target, reduction="none") * weight).sum() / weight.sum()
            batch_losses.append(loss)
        if not batch_losses:
            continue
        loss = torch.stack(batch_losses).mean(); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step(); losses.append(float(loss.detach()))
    if not losses:
        raise RuntimeError("No supervised patient in training epoch")
    return float(np.mean(losses))


def patient_stratified_folds(patients: dict[str, Patient]) -> dict[str, int]:
    stats = []
    for patient, item in sorted(patients.items()):
        labels = item.obs_label[item.obs_label >= 0]
        stats.append({"patient": patient, "center": item.center, "has_pathology": int(np.any(labels == 1)), "n_labeled": len(labels)})
    frame = pd.DataFrame(stats)
    frame["stratum"] = frame.center + "|" + frame.has_pathology.astype(str)
    # Merge only too-small strata, deterministically.  The stored audit makes
    # every fallback visible; no test information enters this operation.
    for stage in (0, 1, 2):
        small = set(frame.stratum.value_counts()[lambda x: x < FOLDS].index)
        if not small: break
        if stage == 0:
            frame.loc[frame.stratum.isin(small), "stratum"] = frame.loc[frame.stratum.isin(small), "center"]
        elif stage == 1:
            small_centers = set(frame.loc[frame.stratum.isin(small), "center"])
            frame.loc[frame.center.isin(small_centers), "stratum"] = "other|" + frame.has_pathology.astype(str)
        else:
            frame["stratum"] = "all"
    if (frame.stratum.value_counts() < FOLDS).any():
        raise RuntimeError("Could not construct 5-fold patient-stratified split")
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    frame["fold"] = -1
    for fold, (_, held) in enumerate(splitter.split(frame.patient, frame.stratum), 1):
        frame.loc[held, "fold"] = fold
    if (frame.fold < 1).any() or frame.patient.duplicated().any():
        raise RuntimeError("Invalid patient split")
    return dict(zip(frame.patient, frame.fold)), frame


def selected_key(metrics: dict, epoch: int) -> tuple[float, float, int]:
    return (metrics["pooled_auroc"], metrics["pooled_ap"], -epoch)


def fit_variant(variant: str, maker, patients: dict[str, Patient], fit_ids: list[str], val_ids: list[str],
                seed: int, runtime: Path, fold: int | str) -> tuple[torch.nn.Module, int, dict, list[dict]]:
    seed_everything(seed); model = maker(); optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed); best = None; history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, patients, fit_ids, optimizer, rng)
        rows, _ = scores_for(model, {p: patients[p] for p in val_ids})
        metrics = base_metrics(rows); chosen = best is None or selected_key(metrics, epoch) > best[0]
        history.append({"fold": fold, "model": variant, "epoch": epoch, "train_patient_equal_weighted_bce": loss,
                        "validation_pooled_auroc": metrics["pooled_auroc"], "validation_pooled_ap": metrics["pooled_ap"], "selected": chosen})
        if chosen:
            path = runtime / "cv_checkpoints" / f"fold{fold}_{variant}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".pt.tmp")
            torch.save({"state": model.state_dict(), "epoch": epoch, "metrics": metrics, "variant": variant}, tmp); os.replace(tmp, path)
            best = (selected_key(metrics, epoch), epoch, metrics, path)
    assert best is not None
    saved = torch.load(best[3], map_location="cpu", weights_only=False); model.load_state_dict(saved["state"])
    return model, int(best[1]), best[2], history


def bootstrap(test: pd.DataFrame, columns: dict[str, str]) -> list[dict]:
    patients = np.asarray(sorted(test.patient.unique())); index = {p: np.flatnonzero(test.patient.to_numpy() == p) for p in patients}
    y = test.y.to_numpy(dtype=np.int8); rng = np.random.default_rng(SEED); outputs = []
    for comparison, left, right in (("V1-C0", "V1", "C0"), ("V1-C1", "V1", "C1")):
        point = float(roc_auc_score(y, test[columns[left]]) - roc_auc_score(y, test[columns[right]])); draws = []
        for _ in range(10000):
            selected = rng.integers(0, len(patients), size=len(patients))
            ids = np.concatenate([index[patients[i]] for i in selected])
            draws.append(float(roc_auc_score(y[ids], test[columns[left]].to_numpy()[ids]) - roc_auc_score(y[ids], test[columns[right]].to_numpy()[ids])))
        draws = np.asarray(draws)
        outputs.append({"comparison": comparison, "bootstrap_draws": len(draws), "seed": SEED, "point_delta_auroc": point,
                        "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)),
                        "probability_delta_gt_zero": float(np.mean(draws > 0))})
    return outputs


def percentile(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0)), **{f"p{x}": float(np.percentile(values, x)) for x in (5, 25, 50, 75, 95)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runtime.exists() or args.output.exists():
        raise RuntimeError("Refusing reuse: runtime/output paths must be new")
    args.runtime.mkdir(parents=True); args.output.mkdir(parents=True)
    centers = load_centers(args.split_csv)
    if cache_digest(args.train_cache) != TRAIN_CACHE_SHA256 or cache_digest(args.test_cache) != TEST_CACHE_SHA256:
        raise RuntimeError("Frozen R4 cache digest mismatch")
    train, train_audit = load_patients(args.train_cache, "train", centers)
    test, test_audit = load_patients(args.test_cache, "test", centers)
    if train_audit["cache_files"] != 296 or train_audit["labeled_edf_channels"] != 13350 or train_audit["segments"] != 316364:
        raise RuntimeError("Full-record TRAIN cache gate failed")
    if test_audit["cache_files"] != 237 or test_audit["labeled_edf_channels"] != 8104 or test_audit["segments"] != 240074:
        raise RuntimeError("Frozen TEST cache gate failed")
    # Mandatory frozen baseline replay before any residual-head optimization.
    test_base, _ = scores_for(None, test)
    baseline = base_metrics(test_base)
    if abs(baseline["pooled_auroc"] - EXPECTED_TEST_AUROC) > 1e-5 or abs(baseline["macro_f1_0_5"] - EXPECTED_TEST_F1) > 1e-5:
        atomic_json(args.output / "BASELINE_REPLAY_AUDIT.json", {"status": "STOP_BASELINE_REPLAY_FAILED", **baseline})
        raise RuntimeError("STOP_BASELINE_REPLAY_FAILED")
    atomic_json(args.output / "BASELINE_REPLAY_AUDIT.json", {"status": "PASS", **baseline, "expected_auroc": EXPECTED_TEST_AUROC,
                                                                 "expected_ap": EXPECTED_TEST_AP, "expected_macro_f1": EXPECTED_TEST_F1,
                                                                 "checkpoint_sha256": CHECKPOINT_SHA256, "neural_inference_run": False})
    # Mathematical identity gate, exercised on both cached populations before training.
    seed_everything(SEED); identity = {}
    for name, maker in (("C1", LocalResidual), ("V1", A1ContextResidual)):
        model = maker(); max_error = 0.0
        for population in (train, test):
            rows, _ = scores_for(model, population); base, _ = scores_for(None, population)
            max_error = max(max_error, float(np.max(np.abs(rows.pathological_score.to_numpy() - base.pathological_score.to_numpy()))))
        identity[name] = max_error
    if max(identity.values()) >= 1e-7:
        atomic_json(args.output / "ZERO_INIT_IDENTITY_AUDIT.json", {"status": "STOP_ZERO_INIT_IDENTITY_FAILED", "max_abs": identity})
        raise RuntimeError("STOP_ZERO_INIT_IDENTITY_FAILED")
    atomic_json(args.output / "ZERO_INIT_IDENTITY_AUDIT.json", {"status": "PASS", "max_abs_score_difference": identity,
                                                                   "threshold": 1e-7, "alpha_initial": 0.0})
    fold_of, split_table = patient_stratified_folds(train)
    if len(fold_of) != len(train): raise RuntimeError("TRAIN split does not cover cache patients")
    atomic_csv(args.output / "TRAIN_SPLIT_AUDIT.csv", split_table[["center", "has_pathology", "n_labeled", "stratum", "fold"]].groupby(["fold", "center", "has_pathology", "stratum"], as_index=False).agg(n_patients=("n_labeled", "size"), labeled_edf_channels=("n_labeled", "sum")))
    def coverage(population: dict[str, Patient], split: str):
        rows = []
        for item in population.values():
            supervised = defaultdict(set)
            for idx, label in zip(item.obs_channel_index, item.obs_label):
                if label >= 0: supervised[int(idx)].add(int(label))
            rows.append({"split": split, "all_good_context_channels": len(item.channels), "labeled_context_channels": len(supervised), "unlabeled_context_only_channels": len(item.channels) - len(supervised)})
        return rows
    coverage_rows = coverage(train, "TRAIN") + coverage(test, "TEST")
    coverage_public = []
    for split, group in pd.DataFrame(coverage_rows).groupby("split"):
        coverage_public.append({"split": split, "n_patients": len(group), **{f"mean_{k}": float(group[k].mean()) for k in group.columns if k != "split"},
                                **{f"median_{k}": float(group[k].median()) for k in group.columns if k != "split"},
                                "construction": "all observable cached good channels; label=-1 context-only channels retained"})
    atomic_csv(args.output / "CONTEXT_CHANNEL_COVERAGE_AUDIT.csv", coverage_public)
    r4_audit = {"status": "PASS_REUSED_HASH_VERIFIED_CACHES", "r4_dimension": 32,
                "hook_location": "frozen official CNN cnn block output immediately before later low-dimensional classifier layers",
                "r4_edf_channel_pooling": "plain mean of frozen segment R4 vectors", "train_cache_sha256": TRAIN_CACHE_SHA256,
                "test_cache_sha256": TEST_CACHE_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256,
                "official_source_sha256": OFFICIAL_SOURCE_SHA256, "historical_five_clip_train_npzs_read": False,
                "full_record_train_gate": "13350 labeled EDF-channel; 316364 segments; 296 EDF"}
    atomic_json(args.output / "R4_EXTRACTION_AUDIT.json", r4_audit)
    # Five-fold TRAIN-only patient-CV. C0 has no fitted parameter; C1/V1 share every optimization setting.
    histories: list[dict] = []; cv_rows: list[dict] = []; selected_epochs = {"C1": [], "V1": []}
    makers = {"C1": LocalResidual, "V1": A1ContextResidual}
    for fold in range(1, FOLDS + 1):
        fit = [p for p in sorted(train) if fold_of[p] != fold]; val = [p for p in sorted(train) if fold_of[p] == fold]
        c0_rows, _ = scores_for(None, {p: train[p] for p in val}); c0 = {**base_metrics(c0_rows), **ranking_metrics(c0_rows)}
        cv_rows.append({"fold": fold, "model": "C0", "selected_epoch": 0, **c0})
        for variant, maker in makers.items():
            model, epoch, metrics, history = fit_variant(variant, maker, train, fit, val, SEED + 1000 * fold + (0 if variant == "C1" else 1), args.runtime, fold)
            histories.extend(history); val_rows, _ = scores_for(model, {p: train[p] for p in val})
            cv_rows.append({"fold": fold, "model": variant, "selected_epoch": epoch, **base_metrics(val_rows), **ranking_metrics(val_rows), "alpha": float(model.alpha.detach())})
            selected_epochs[variant].append(epoch)
            print(json.dumps({"stage": "cv_fold_complete", "fold": fold, "model": variant, "epoch": epoch, "auroc": metrics["pooled_auroc"]}), flush=True)
    atomic_csv(args.output / "CV_FOLD_METRICS.csv", cv_rows)
    cv_summary = []
    for model, group in pd.DataFrame(cv_rows).groupby("model", sort=False):
        cv_summary.append({"model": model, "n_folds": len(group), "mean_pooled_auroc": float(group.pooled_auroc.mean()), "sd_pooled_auroc": float(group.pooled_auroc.std(ddof=1)),
                           "mean_pooled_ap": float(group.pooled_ap.mean()), "mean_patient_equal_auroc": float(group.patient_equal_auroc.mean()),
                           "mean_selected_epoch": float(group.selected_epoch.mean()), "nonnegative_vs_C0_folds": int(sum(group.pooled_auroc.to_numpy() >= pd.DataFrame(cv_rows).query("model == 'C0'").pooled_auroc.to_numpy())) if model != "C0" else FOLDS})
    atomic_csv(args.output / "CV_SUMMARY.csv", cv_summary)
    # Deterministic median selected epoch, then one full TRAIN refit.  The
    # resulting hashes are persisted before model scores are evaluated on TEST.
    final_epochs = {variant: int(round(float(np.median(epochs)))) for variant, epochs in selected_epochs.items()}
    final_models = {}; selections = {}
    for variant, maker in makers.items():
        seed_everything(SEED + (0 if variant == "C1" else 1)); model = maker(); opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY); rng = np.random.default_rng(SEED + (0 if variant == "C1" else 1))
        for _ in range(final_epochs[variant]): train_epoch(model, train, sorted(train), opt, rng)
        checkpoint = args.runtime / f"{variant}_frozen_final.pt"; torch.save({"state": model.state_dict(), "epoch": final_epochs[variant], "variant": variant}, checkpoint)
        final_models[variant] = model; selections[variant] = {"epoch": final_epochs[variant], "checkpoint_sha256": sha256(checkpoint), "alpha": float(model.alpha.detach())}
    model_audit = {"frozen_timeconv_trainable_parameters": 0, "timeconv_loaded_into_optimizer": False,
                   "C1_trainable_parameters": int(sum(p.numel() for p in final_models["C1"].parameters())),
                   "V1_trainable_parameters": int(sum(p.numel() for p in final_models["V1"].parameters())),
                   "a1_ranker_source_sha256": sha256(Path(__file__).with_name("a1_patient_channel_ranker.py")),
                   "a1_ranker_source_reference": "audited 2026-06-20 e3726f64d1_patient_channel_ranker.py", "dropout": DROP, "heads": 2}
    atomic_json(args.output / "MODEL_PARAMETER_AUDIT.json", model_audit)
    lock = {"experiment": "omni_timeconv_a1_context_v1_seed42", "seed": SEED, "checkpoint_sha256": CHECKPOINT_SHA256,
            "official_source_sha256": OFFICIAL_SOURCE_SHA256, "train_cache_sha256": TRAIN_CACHE_SHA256, "test_cache_sha256": TEST_CACHE_SHA256,
            "r4_dimension": 32, "context": "all observable cached good channels, including label=-1 context-only", "edf_prediction_unit": "EDF-channel",
            "anchor": "mean R4 over EDFs within patient/channel", "residual": "logit(mean segment normal probability) + alpha * patient-channel delta",
            "alpha_initial": 0.0, "models": selections, "final_epoch_rule": "rounded median of 5 TRAIN-CV validation-selected epochs",
            "optimizer": {"AdamW": True, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "max_epochs_cv": MAX_EPOCHS, "patient_batch": 2, "gradient_clip": GRAD_CLIP},
            "loss": "patient-equal weighted BCE; pathology 2, normal 1", "test_tuning": False, "test_previously_viewed": True}
    atomic_json(args.output / "PROTOCOL_LOCK.json", lock)
    atomic_json(args.runtime / "FROZEN_BEFORE_TEST_PRIVATE.json", {"status": "FROZEN_BEFORE_TEST", "protocol_sha256": sha256(args.output / "PROTOCOL_LOCK.json"), "models": selections})
    # Single post-freeze official TEST evaluation.
    score_frames = {"C0": test_base}
    delta_rows = []
    for variant, model in final_models.items():
        frame, delta = scores_for(model, test); score_frames[variant] = frame
        delta_rows.append({"model": variant, "split": "TEST", "alpha": float(model.alpha.detach()), "residual_logit": "raw_delta", **percentile(delta)})
    test_rows = []
    for variant, frame in score_frames.items():
        test_rows.append({"model": variant, **base_metrics(frame), **ranking_metrics(frame), "threshold": THRESHOLD,
                          "delta_auroc_vs_C0": float(base_metrics(frame)["pooled_auroc"] - baseline["pooled_auroc"])})
    atomic_csv(args.output / "TEST_PRIMARY_METRICS.csv", test_rows)
    atomic_csv(args.output / "ALPHA_DELTA_AUDIT.csv", delta_rows)
    center_rows = []
    for center in ("HUP", "Open-iEEG", "SourceSink", "Zurich"):
        for variant, frame in score_frames.items():
            subset = frame.loc[frame.center == center]
            if len(np.unique(subset.y)) < 2:
                center_rows.append({"center": center, "model": variant, "n_units": len(subset), "auroc": np.nan, "ap": np.nan, "status": "not_estimable"})
            else:
                center_rows.append({"center": center, "model": variant, "n_units": len(subset), "auroc": float(roc_auc_score(subset.y, subset.pathological_score)), "ap": float(average_precision_score(subset.y, subset.pathological_score)), "status": "estimable"})
    atomic_csv(args.output / "TEST_CENTER_METRICS.csv", center_rows)
    # Predeclared evaluation-only interventions.  They do not retrain or select a model.
    names = sorted(test); donor = {p: names[(idx + 1) % len(names)] for idx, p in enumerate(names)}
    interventions = []
    for name, intervention, donors in (("V1 full", "full", None), ("V1 patient-relative input zeroed", "relative_zero", None),
                                       ("V1 cross-patient context shuffled", "cross_patient_context", donor), ("V1 channel attention disabled", "no_attention", None)):
        frame, _ = scores_for(final_models["V1"], test, intervention=intervention, donors=donors)
        interventions.append({"intervention": name, **base_metrics(frame), **ranking_metrics(frame), "delta_auroc_vs_full": float(base_metrics(frame)["pooled_auroc"] - base_metrics(score_frames["V1"])["pooled_auroc"])})
    atomic_csv(args.output / "PATIENT_CONTEXT_INTERVENTIONS.csv", interventions)
    combined = score_frames["C0"][["patient", "edf", "channel", "y", "center"]].copy()
    for name, frame in score_frames.items(): combined[name] = frame.pathological_score.to_numpy()
    boot = bootstrap(combined, {"C0": "C0", "C1": "C1", "V1": "V1"})
    atomic_csv(args.output / "PAIRED_PATIENT_BOOTSTRAP.csv", boot)
    c0, c1, v1 = (next(row for row in test_rows if row["model"] == name) for name in ("C0", "C1", "V1"))
    cv = {r["model"]: r for r in cv_summary}; b0 = next(r for r in boot if r["comparison"] == "V1-C0"); b1 = next(r for r in boot if r["comparison"] == "V1-C1")
    v1_improves = v1["pooled_auroc"] > c0["pooled_auroc"] and b0["ci95_low"] > 0
    terminal = "A1_CONTEXT_TRANSFER_SUPPORTED" if v1_improves and v1["pooled_auroc"] > c1["pooled_auroc"] else ("A1_CONTEXT_NOT_SUPPORTED" if v1["pooled_auroc"] <= c0["pooled_auroc"] else "CONTEXT_GAIN_NOT_ISOLATED_FROM_LOCAL_CAPACITY")
    report = f"""# Frozen TimeConv R4 + A1 patient-level context on Omni-iEEG

**Terminal: `{terminal}`.** This is one exploratory/repeated official TEST pass: Omni outcomes were historically viewed. TimeConv was never retrained or loaded into the optimizer; C1/V1 were selected entirely with TRAIN-only patient CV before the final head hashes were frozen.

| Model | TEST pooled AUROC | AP | Macro-F1 @0.5 | Patient-equal AUROC | Patient AP | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 frozen TimeConv | {c0['pooled_auroc']:.6f} | {c0['pooled_ap']:.6f} | {c0['macro_f1_0_5']:.6f} | {c0['patient_equal_auroc']:.6f} | {c0['patient_equal_ap']:.6f} | {c0['mrr']:.6f} | {c0['top1']:.6f} |
| C1 local residual | {c1['pooled_auroc']:.6f} | {c1['pooled_ap']:.6f} | {c1['macro_f1_0_5']:.6f} | {c1['patient_equal_auroc']:.6f} | {c1['patient_equal_ap']:.6f} | {c1['mrr']:.6f} | {c1['top1']:.6f} |
| V1 A1-context residual | {v1['pooled_auroc']:.6f} | {v1['pooled_ap']:.6f} | {v1['macro_f1_0_5']:.6f} | {v1['patient_equal_auroc']:.6f} | {v1['patient_equal_ap']:.6f} | {v1['mrr']:.6f} | {v1['top1']:.6f} |

## Direct answers

1. Frozen TimeConv replayed `{baseline['pooled_auroc']:.10f}` on 8,104 labeled EDF-channel units: **yes**.
2. Zero-init identity passed: maximum absolute score difference `{max(identity.values()):.3g}` (<1e-7).
3. Context coverage is in `CONTEXT_CHANNEL_COVERAGE_AUDIT.csv`; it counts all cached good channels, labelled and `-1` context-only, separately.
4. Yes: context uses every cached observable good channel before any label filter; `-1` channels have no BCE term.
5. C0 TRAIN-CV pooled AUROC: `{cv['C0']['mean_pooled_auroc']:.6f}`.
6. C1 TRAIN-CV pooled AUROC: `{cv['C1']['mean_pooled_auroc']:.6f}`.
7. V1 TRAIN-CV pooled AUROC: `{cv['V1']['mean_pooled_auroc']:.6f}`.
8. V1 nonnegative versus C0 folds: `{cv['V1']['nonnegative_vs_C0_folds']}/5`.
9. V1 TRAIN-CV versus C1: `{cv['V1']['mean_pooled_auroc'] - cv['C1']['mean_pooled_auroc']:+.6f}`.
10. Final V1 alpha: `{selections['V1']['alpha']:.6f}`.
11. Delta distribution is recorded in `ALPHA_DELTA_AUDIT.csv`; it is not called collapsed unless its observed spread is near zero.
12. Context-zero, cross-patient-context, and no-attention interventions are reported without retraining in `PATIENT_CONTEXT_INTERVENTIONS.csv`.
13. C0 official TEST AUROC: `{c0['pooled_auroc']:.6f}`.
14. C1 official TEST AUROC: `{c1['pooled_auroc']:.6f}`.
15. V1 official TEST AUROC: `{v1['pooled_auroc']:.6f}`.
16. V1−C0 `{b0['point_delta_auroc']:+.6f}`, patient-cluster bootstrap 95% CI [{b0['ci95_low']:+.6f}, {b0['ci95_high']:+.6f}]. V1−C1 `{b1['point_delta_auroc']:+.6f}`, CI [{b1['ci95_low']:+.6f}, {b1['ci95_high']:+.6f}].
17. V1 exceeds published 0.8061: `{v1['pooled_auroc'] > .8061}`.
18. V1 reaches 0.815 / 0.820 / 0.830: `{v1['pooled_auroc'] >= .815}` / `{v1['pooled_auroc'] >= .820}` / `{v1['pooled_auroc'] >= .830}`.
19. Center changes are listed in `TEST_CENTER_METRICS.csv`; Zurich remains not-estimable where it has one label class.
20. The evidence supports the claimed transfer **only** if the terminal states `A1_CONTEXT_TRANSFER_SUPPORTED`; otherwise this V1 does not justify that claim.

## Scope distinction

This is not A1-interictal v1/v2 (new 36-D handcrafted descriptors), A1-TF fusion, PC-CNN raw re-encoding, or PR-Residual's small per-EDF relative head. V1 uses the immutable TimeConv R4 anchor, mean-EDF patient-channel anchors, exact A1 patient-relative z and two-head channel attention, then adds one shared residual correction back to the original EDF-channel TimeConv evidence.

The residual-head development split is patient-disjoint, but the frozen TimeConv encoder had already been trained on the official TRAIN population. Therefore this is not an independently held-out backbone-validation estimate. No test result changed architecture, hyperparameters, epoch rule, checkpoint, threshold, or intervention.
"""
    (args.output / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    atomic_json(args.output / "REQUIRED_OUTPUTS_STATUS.json", {"status": "COMPLETE", "terminal": terminal, "test_evaluated_once_after_freeze": True, "private_predictions_committed": False})
    print(json.dumps({"status": "COMPLETE", "terminal": terminal, "C0": c0["pooled_auroc"], "C1": c1["pooled_auroc"], "V1": v1["pooled_auroc"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
