"""Original A1 architecture and patient-equal objective on Omni official train.

All checkpoints and channel/label tensors stay under a private runtime.  The
official test feature directory is never opened by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


SOURCE = Path("E:/DRE-nips/new-pipeline/7-11")
EXPECTED_HASHES = {
    "neuroez_c/model.py": "c3413ff6d3e7e226919b7b3b1c779c3b62cd65ba78900129540006006fd27e6c",
    "neuroez_c/dataset.py": "ccb4fadd9416a63bdceb46fe7edb515a67008f896435e622059b5713cbb42702",
    "neuroez_c/evidence_views.py": "201f888411a8f45fa363cfb2ac11c165a04ae2eef5abcd3aad64aedfe3098b03",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def check_source():
    for relative, expected in EXPECTED_HASHES.items():
        if sha256(SOURCE / relative) != expected:
            raise RuntimeError(f"Historical A1 code changed: {relative}")
    sys.path.insert(0, str(SOURCE))
    from neuroez_c.dataset import collate_patient_ez_batch
    from neuroez_c.model import NeuroEZCModel
    return collate_patient_ez_batch, NeuroEZCModel


def model_args():
    return SimpleNamespace(model_dim=32, num_heads=2, dropout=0.4,
                           temporal_pooling="mean", record_pooling="mean",
                           use_patient_relative_z=True, positive_label="nez",
                           use_physics_dynamics=False, use_diffusion_residual=False,
                           use_channel_attention=True, use_edf_quality_weighting=False)


def patient_equal_weighted_bce(logits, labels_nez, labels_ez, channel_mask):
    """Exact A1 objective from objectives.py at source commit b2871b3."""
    if logits.shape != labels_nez.shape or logits.shape != labels_ez.shape or logits.shape != channel_mask.shape:
        raise ValueError("Patient BCE tensor shapes disagree")
    valid = channel_mask.bool() & (labels_nez >= 0) & (labels_ez >= 0)
    safe_labels = torch.where(valid, labels_nez, torch.zeros_like(labels_nez))
    channel_loss = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none")
    weights = torch.where(labels_ez > 0.5, 2.0, 1.0).to(logits.dtype) * valid.to(logits.dtype)
    denominator = weights.sum(dim=1)
    active = denominator > 0
    per_patient = (channel_loss * weights).sum(dim=1) / denominator.clamp_min(1e-6)
    if not torch.any(active):
        return logits.sum() * 0.0
    return per_patient[active].mean()


class Bank:
    def __init__(self, cohort: Path, features_root: Path, split: Path | None,
                 official_split: str = "train"):
        rows = pd.read_csv(cohort)
        rows = rows.loc[(rows["official_split"] == official_split) & (rows["valid_channels"] > 0)]
        roles = pd.read_csv(split).set_index("patient")["role"].to_dict() if split is not None else {}
        if split is not None and set(rows["patient"]) != set(roles):
            raise RuntimeError("Patient split differs from official train cohort")
        self.roles = roles
        self.by_patient = defaultdict(list)
        self.labels = defaultdict(dict)
        for row in rows.itertuples(index=False):
            path = features_root / Path(row.edf).with_suffix(".npz")
            marker = path.with_suffix(".json")
            if not path.exists() or not marker.exists():
                raise RuntimeError(f"Train feature cache incomplete: {row.edf}")
            with np.load(path, allow_pickle=False) as payload:
                features = np.asarray(payload["features"], dtype=np.float32)
                names = [str(name) for name in payload["channel_names"]]
                soz = np.asarray(payload["soz"], dtype=np.int8)
            if features.ndim != 4 or features.shape[1:] != (59, len(names), 36):
                raise RuntimeError(f"Unexpected A1 feature tensor: {row.edf} {features.shape}")
            if len(names) != row.valid_channels or len(soz) != len(names):
                raise RuntimeError(f"Feature/label channel count mismatch: {row.edf}")
            for name, label in zip(names, soz):
                prior = self.labels[row.patient].get(name)
                if prior is not None and prior != int(label):
                    raise RuntimeError(f"Conflicting SOZ label: {row.patient}/{name}")
                self.labels[row.patient][name] = int(label)
            self.by_patient[row.patient].append({"edf": row.edf, "features": features, "names": names})
        self.patients = sorted(self.by_patient)
        self.canonical = {patient: sorted(self.labels[patient]) for patient in self.patients}

    def fit_normalizer(self, patients: list[str]) -> tuple[np.ndarray, np.ndarray]:
        total = np.zeros(36, dtype=np.float64)
        total_sq = np.zeros(36, dtype=np.float64)
        count = 0
        for patient in patients:
            for record in self.by_patient[patient]:
                values = record["features"].reshape(-1, 36).astype(np.float64, copy=False)
                total += values.sum(axis=0)
                total_sq += np.square(values).sum(axis=0)
                count += values.shape[0]
        if count == 0:
            raise RuntimeError("No train features for normalizer")
        mean = total / count
        std = np.sqrt(np.clip(total_sq / count - mean.square(), 1e-8, None))
        return mean.astype(np.float32), std.astype(np.float32)

    def example(self, patient: str, mean: np.ndarray, std: np.ndarray,
                epoch: int | None) -> dict:
        canonical = self.canonical[patient]
        channel_index = {name: idx for idx, name in enumerate(canonical)}
        soz = np.asarray([self.labels[patient][name] for name in canonical], dtype=np.float32)
        features_list, masks, ids = [], [], []
        for record in self.by_patient[patient]:
            tensor = record["features"]
            if epoch is None:
                selected = range(len(tensor))
            else:
                key = int.from_bytes(hashlib.sha256((record["edf"] + "|" + str(epoch)).encode()).digest()[:8], "big")
                selected = [key % len(tensor)]
            for segment_idx in selected:
                local = (tensor[segment_idx] - mean) / std
                aligned = np.zeros((59, len(canonical), 36), dtype=np.float32)
                active = np.zeros(len(canonical), dtype=bool)
                for local_idx, name in enumerate(record["names"]):
                    idx = channel_index[name]
                    aligned[:, idx] = local[:, local_idx]
                    active[idx] = True
                features_list.append(aligned)
                masks.append(active)
                ids.append(f"{record['edf']}#{segment_idx}")
        return {
            "subject_id": patient, "center": "unknown", "center_id": 4,
            "canonical_channels": canonical, "labels": 1.0 - soz,
            "labels_nez": 1.0 - soz, "labels_ez": soz,
            "channel_mask": np.ones(len(canonical), dtype=bool),
            "b0_features": features_list,
            "physics_features": [np.zeros((59, len(canonical), 1), dtype=np.float32) for _ in features_list],
            "window_centers": [np.arange(59, dtype=np.float32) + 1 for _ in features_list],
            "window_mask": [np.ones(59, dtype=bool) for _ in features_list],
            "seizure_channel_mask": masks, "run_ids": ids, "sample_ids": ids,
            "label_semantics": "1=NEZ,0=EZ",
        }


def to_device(batch: dict, device: str) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def make_model(Model, collate, bank: Bank, patient: str, mean, std, device: str):
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)
    model = Model(model_args()).to(device)
    first = to_device(collate([bank.example(patient, mean, std, epoch=1)]), device)
    with torch.no_grad():
        model.eval()(first)
    params = sum(p.numel() for p in model.parameters())
    if params != 27713:
        raise RuntimeError(f"A1 architecture parameter count changed: {params}")
    return model


def score_validation(model, collate, bank: Bank, patients: list[str], mean, std, device: str):
    model.eval()
    ap = []
    label_free = 0
    with torch.inference_mode():
        for patient in patients:
            batch = to_device(collate([bank.example(patient, mean, std, epoch=None)]), device)
            logits = model(batch)["logits"][0].detach().cpu().numpy()
            soz = np.asarray([bank.labels[patient][name] for name in bank.canonical[patient]], dtype=np.int8)
            score = 1.0 - torch.sigmoid(torch.from_numpy(logits)).numpy()
            if int(soz.sum()) == 0:
                label_free += 1
                ap.append(0.0)
            else:
                ap.append(float(average_precision_score(soz, score)))
    return float(np.mean(ap)), label_free


def fit_stage(name: str, Model, collate, bank: Bank, patients: list[str],
              val_patients: list[str] | None, epochs: int, runtime: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage = runtime / name
    stage.mkdir(parents=True, exist_ok=True)
    normalizer_path = stage / "normalizer.npz"
    if normalizer_path.exists():
        with np.load(normalizer_path) as norm:
            mean, std = norm["mean"], norm["std"]
    else:
        mean, std = bank.fit_normalizer(patients)
        np.savez(normalizer_path, mean=mean, std=std)
    model = make_model(Model, collate, bank, patients[0], mean, std, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    last = stage / "last.pt"
    start = 1
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["stage"] != name or state["epoch"] > epochs:
            raise RuntimeError("Resume checkpoint identity mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["epoch"]) + 1
    history = stage / "epochs.csv"
    for epoch in range(start, epochs + 1):
        epoch_patients = sorted(patients)
        random.Random(42 * 100000 + epoch).shuffle(epoch_patients)
        torch.manual_seed(42 * 100000 + epoch)
        model.train()
        losses = []
        for offset in range(0, len(epoch_patients), 2):
            chosen = epoch_patients[offset : offset + 2]
            batch = to_device(collate([bank.example(p, mean, std, epoch=epoch) for p in chosen]), device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = patient_equal_weighted_bce(output["logits"], batch["labels_nez"],
                                              batch["labels_ez"], batch["channel_mask"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite A1 training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_ap, val_no_soz = (score_validation(model, collate, bank, val_patients, mean, std, device)
                              if val_patients is not None else (float("nan"), -1))
        with history.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(["stage", "epoch", "train_loss", "patient_equal_ez_ap", "val_no_soz_patients"])
            writer.writerow([name, epoch, float(np.mean(losses)), val_ap, val_no_soz])
        payload = {"stage": name, "epoch": epoch, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict()}
        tmp = stage / "last.pt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, last)
        if val_patients is not None:
            selected = stage / "best.json"
            prior = json.loads(selected.read_text()) if selected.exists() else None
            if prior is None or val_ap > prior["patient_equal_ez_ap"] + 1e-12:
                best = stage / "best.pt"
                tmp = stage / "best.pt.tmp"
                torch.save(payload, tmp)
                os.replace(tmp, best)
                atomic_json(selected, {"epoch": epoch, "patient_equal_ez_ap": val_ap,
                                       "no_soz_val_patients": val_no_soz,
                                       "checkpoint_sha256": sha256(best)})
        print(f"{name} epoch={epoch}/{epochs} train_loss={np.mean(losses):.5f} val_AP={val_ap}", flush=True)
    return {"normalizer_sha256": sha256(normalizer_path),
            "model_parameters": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--train-val-split", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    if (args.runtime / "official_test_predictions/TEST_ACCESS_STARTED.json").exists():
        raise RuntimeError("Official test access already started; training is permanently frozen")
    collate, Model = check_source()
    args.runtime.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    bank = Bank(args.cohort, args.features, args.train_val_split)
    fit = sorted(patient for patient in bank.patients if bank.roles[patient] == "inner_train")
    val = sorted(patient for patient in bank.patients if bank.roles[patient] == "inner_val")
    if len(fit) != 111 or len(val) != 28:
        raise RuntimeError(f"Unexpected train/val patients: {len(fit)}, {len(val)}")
    inner = fit_stage("inner", Model, collate, bank, fit.copy(), val, 30, args.runtime)
    selection = json.loads((args.runtime / "inner/best.json").read_text())
    if selection["epoch"] not in range(1, 31):
        raise RuntimeError("Invalid selected epoch")
    full = fit_stage("final", Model, collate, bank, bank.patients.copy(), None,
                     selection["epoch"], args.runtime)
    final_checkpoint = args.runtime / "final/last.pt"
    freeze = {"model_frozen_before_official_test": True, "official_test_accessed": False,
              "selected_inner_epoch": selection["epoch"],
              "inner_validation_patient_equal_ez_ap": selection["patient_equal_ez_ap"],
              "inner_checkpoint_sha256": selection["checkpoint_sha256"],
              "final_checkpoint_sha256": sha256(final_checkpoint),
              "final_normalizer_sha256": full["normalizer_sha256"],
              "protocol_sha256": sha256(args.protocol), "train_val_split_sha256": sha256(args.train_val_split),
              "train_patients": len(bank.patients), "inner_train_patients": len(fit),
              "inner_val_patients": len(val), "model_parameters": full["model_parameters"]}
    existing_freeze = args.output / "TEST_SCORE_FREEZE_AUDIT.json"
    if existing_freeze.exists() and json.loads(existing_freeze.read_text(encoding="utf-8")) != freeze:
        raise RuntimeError("Refusing to replace an existing checkpoint freeze")
    selection_rows = pd.read_csv(args.runtime / "inner/epochs.csv")
    selection_rows["selected"] = selection_rows["epoch"].eq(selection["epoch"])
    selection_rows.to_csv(args.output / "CHECKPOINT_SELECTION.csv", index=False)
    atomic_json(args.output / "TEST_SCORE_FREEZE_AUDIT.json", freeze)
    atomic_json(args.output / "TRAINING_AUDIT.json", {"inner": inner, "final": full,
                                                      "selection": selection, "freeze": freeze})
    print(json.dumps({"status": "FROZEN", "selected_epoch": selection["epoch"],
                      "final_checkpoint_sha256": freeze["final_checkpoint_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
