"""N1 official-train-only selection, threshold freeze, and final refit.

The official test feature tree is never opened here. All TF normalizers are
fit exclusively on the corresponding train patient set. Private patient rows,
optimizer states and checkpoints stay in the runtime directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from a1_tf import A1TFModel
from tf_preprocess import fit_train_frequency_normalizer


V2_CODE = Path(__file__).resolve().parents[2] / "omni_ieeg_a1_interictal_v2_seed42/code"
if not V2_CODE.is_dir():
    V2_CODE = Path("E:/DRE-nips/new-pipeline/7-11/omni_a1_v2/code")
sys.path.insert(0, str(V2_CODE))
from train_v2 import (V2Bank, select_threshold, sha256, check_source,
                      patient_equal_weighted_bce, to_device, atomic_json)  # noqa: E402

V1_CODE = Path(__file__).resolve().parents[2] / "omni_ieeg_a1_interictal_seed42_v1/code"
if not V1_CODE.is_dir():
    V1_CODE = Path("E:/DRE-nips/new-pipeline/7-11/omni_a1_v1/code")
sys.path.insert(0, str(V1_CODE))
from train_a1_omni import model_args  # noqa: E402

EXPECTED_LOCK = "ace2017e01d0d21e3dda8ddd4d42551366704430b8a10f3666f955150150630d"


class V2TFBank(V2Bank):
    def __init__(self, cohort: Path, features: Path, tf_root: Path, split: Path,
                 v2_protocol: Path, tf_protocol: Path):
        super().__init__(cohort, features, split, protocol=v2_protocol)
        self.tf_mean = None
        self.tf_std = None
        protocol_sha = sha256(tf_protocol)
        for records in self.by_patient.values():
            for record in records:
                path = tf_root / Path(record["edf"]).with_suffix(".npz")
                marker = path.with_suffix(".json")
                if not path.is_file() or not marker.is_file():
                    raise RuntimeError(f"Incomplete train-only TF cache: {record['edf']}")
                meta = json.loads(marker.read_text(encoding="utf-8"))
                if meta["protocol_sha256"] != protocol_sha or meta["sha256"] != sha256(path):
                    raise RuntimeError(f"TF cache provenance mismatch: {record['edf']}")
                with np.load(path, allow_pickle=False) as data:
                    tf = np.asarray(data["tf_log_power"], dtype=np.float32)
                    names = [str(value) for value in data["channel_names"]]
                if names != record["names"] or tf.shape != (*record["features"].shape[:3], 32):
                    raise RuntimeError(f"TF/A1 feature alignment mismatch: {record['edf']}")
                if not np.isfinite(tf).all():
                    raise RuntimeError("Nonfinite cached TF features")
                record["tf"] = tf

    def fit_tf_normalizer(self, patients: list[str]):
        values = [record["tf"] for p in patients for record in self.by_patient[p]]
        return fit_train_frequency_normalizer(values)

    def set_tf_normalizer(self, mean, std):
        self.tf_mean = np.asarray(mean, dtype=np.float32)
        self.tf_std = np.asarray(std, dtype=np.float32)
        if self.tf_mean.shape != (32,) or self.tf_std.shape != (32,):
            raise ValueError("TF normalizer must have 32 frequencies")

    def example(self, patient: str, mean, std, epoch: int | None):
        if self.tf_mean is None or self.tf_std is None:
            raise RuntimeError("Train-only TF normalizer has not been fitted")
        result = super().example(patient, mean, std, epoch)
        by_edf = {record["edf"]: record for record in self.by_patient[patient]}
        canonical = self.canonical[patient]
        by_channel = {name: index for index, name in enumerate(canonical)}
        tf_list = []
        for sample in result["run_ids"]:
            edf, clip_index = sample.rsplit("#", 1)
            record = by_edf[edf]
            local = (record["tf"][int(clip_index)] - self.tf_mean) / self.tf_std
            aligned = np.zeros((59, len(canonical), 32), dtype=np.float32)
            for local_index, name in enumerate(record["names"]):
                aligned[:, by_channel[name]] = local[:, local_index]
            tf_list.append(aligned)
        if len(tf_list) != len(result["b0_features"]):
            raise RuntimeError("TF/A1 clip count mismatch")
        result["tf_log_power"] = tf_list
        return result


def wrap_collate(historical_collate):
    def collate(examples: list[dict]):
        batch = historical_collate(examples)
        b, r, w, c, _ = batch["b0_features"].shape
        tf = torch.zeros((b, r, w, c, 32), dtype=torch.float32)
        for patient_index, example in enumerate(examples):
            for record_index, data in enumerate(example["tf_log_power"]):
                if data.shape[0] != w or data.shape[1] > c:
                    raise RuntimeError("TF padding shape mismatch")
                tf[patient_index, record_index, :, :data.shape[1]] = torch.from_numpy(data)
        batch["tf_log_power"] = tf
        return batch
    return collate


def make_model(Model, collate, bank, patient, feature_mean, feature_std, device):
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)
    model = A1TFModel(Model(model_args())).to(device)
    first = to_device(collate([bank.example(patient, feature_mean, feature_std, epoch=1)]), device)
    with torch.no_grad():
        model.eval()(first)
    if sum(p.numel() for p in model.parameters()) != 67403:
        raise RuntimeError("Unified A1-TF parameter count changed")
    return model


def descriptor_loss(prediction, batch):
    target = batch["b0_features"][..., :9]
    valid = (batch["window_mask"][:, :, :, None] &
             batch["seizure_channel_mask"][:, :, None, :])
    if not valid.any():
        raise RuntimeError("No valid A1 windows for descriptor reconstruction")
    return F.smooth_l1_loss(prediction[valid], target[valid])


def validation_patient_ap(model, collate, bank, patients, mean, std, device):
    model.eval()
    ap = []
    with torch.inference_mode():
        for patient in patients:
            batch = to_device(collate([bank.example(patient, mean, std, epoch=None)]), device)
            logits = model(batch)["logits"][0].detach().cpu().numpy()
            y = np.asarray([bank.labels[patient][name] for name in bank.canonical[patient]], dtype=np.int8)
            if y.any():
                ap.append(float(average_precision_score(y, 1.0 - torch.sigmoid(torch.from_numpy(logits)).numpy())))
    if not ap:
        raise RuntimeError("No estimable validation pathology patient")
    return float(np.mean(ap))


def fit_stage(stage_name, Model, collate, bank, patients, val_patients, epochs, runtime):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage = runtime / stage_name
    stage.mkdir(parents=True, exist_ok=True)
    normalizer = stage / "normalizer.npz"
    if normalizer.is_file():
        with np.load(normalizer) as values:
            mean, std, tf_mean, tf_std = [values[k] for k in ("mean", "std", "tf_mean", "tf_std")]
    else:
        mean, std = bank.fit_normalizer(patients)
        tf_mean, tf_std = bank.fit_tf_normalizer(patients)
        np.savez(normalizer, mean=mean, std=std, tf_mean=tf_mean, tf_std=tf_std)
    bank.set_tf_normalizer(tf_mean, tf_std)
    model = make_model(Model, collate, bank, patients[0], mean, std, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    last = stage / "last.pt"
    start = 1
    if last.is_file():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["stage"] != stage_name or state["epoch"] > epochs:
            raise RuntimeError("Resume checkpoint stage/epoch mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["epoch"]) + 1
    for epoch in range(start, epochs + 1):
        ordered = sorted(patients)
        random.Random(42 * 100000 + epoch).shuffle(ordered)
        torch.manual_seed(42 * 100000 + epoch)
        model.train()
        losses, reconstructions = [], []
        for offset in range(0, len(ordered), 2):
            batch = to_device(collate([bank.example(p, mean, std, epoch=epoch)
                                       for p in ordered[offset:offset + 2]]), device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            bce = patient_equal_weighted_bce(out["logits"], batch["labels_nez"],
                                             batch["labels_ez"], batch["channel_mask"])
            reconstruction = descriptor_loss(out["tf_descriptor_prediction"], batch)
            loss = bce + 0.1 * reconstruction
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite A1-TF loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            reconstructions.append(float(reconstruction.detach().cpu()))
        val_ap = (validation_patient_ap(model, collate, bank, val_patients,
                                        mean, std, device) if val_patients else float("nan"))
        row = [stage_name, epoch, float(np.mean(losses)), float(np.mean(reconstructions)),
               val_ap, float(model.alpha.detach().cpu())]
        with (stage / "epochs.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(["stage", "epoch", "train_loss", "train_reconstruction_smoothl1",
                                 "validation_patient_equal_AP", "alpha"])
            writer.writerow(row)
        payload = {"stage": stage_name, "epoch": epoch,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "protocol_sha256": EXPECTED_LOCK}
        temporary = stage / "last.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, last)
        if val_patients:
            best_json = stage / "best.json"
            previous = json.loads(best_json.read_text(encoding="utf-8")) if best_json.is_file() else None
            if previous is None or val_ap > previous["validation_patient_equal_AP"] + 1e-12:
                best = stage / "best.pt"
                temporary = stage / "best.pt.tmp"
                torch.save(payload, temporary)
                os.replace(temporary, best)
                atomic_json(best_json, {"epoch": epoch, "validation_patient_equal_AP": val_ap,
                                        "checkpoint_sha256": sha256(best), "alpha": row[-1]})
        print(f"{stage_name} epoch={epoch}/{epochs} loss={row[2]:.5f} "
              f"recon={row[3]:.5f} val_ap={val_ap:.5f} alpha={row[-1]:.6f}", flush=True)
    return model


def validation_channel_scores(bank, patients, collate, Model, checkpoint, normalizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(normalizer) as values:
        mean, std, tf_mean, tf_std = [values[k] for k in ("mean", "std", "tf_mean", "tf_std")]
    bank.set_tf_normalizer(tf_mean, tf_std)
    model = make_model(Model, collate, bank, patients[0], mean, std, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state["protocol_sha256"] != EXPECTED_LOCK:
        raise RuntimeError("Checkpoint protocol hash mismatch")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    rows = []
    with torch.inference_mode():
        for patient in patients:
            batch = to_device(collate([bank.example(patient, mean, std, epoch=None)]), device)
            logits = model(batch)["logits"][0].detach().cpu().numpy()
            scores = 1.0 - torch.sigmoid(torch.from_numpy(logits)).numpy()
            score_by_name = dict(zip(bank.canonical[patient], scores.tolist()))
            for record in bank.by_patient[patient]:
                for name in record["names"]:
                    rows.append({"patient": patient, "dataset": bank.patient_dataset[patient],
                                 "edf": record["edf"], "channel": name,
                                 "pathology": bank.edf_labels[record["edf"]][name],
                                 "score_pathology": score_by_name[name]})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--train-val-split", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--tf-features", type=Path, required=True)
    p.add_argument("--v2-protocol", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if sha256(a.protocol) != EXPECTED_LOCK:
        raise RuntimeError("A1-TF protocol lock changed")
    if (a.runtime / "official_test_access_started.json").is_file():
        raise RuntimeError("Official test access already began; training cannot resume")
    a.output.mkdir(parents=True, exist_ok=True)
    a.runtime.mkdir(parents=True, exist_ok=True)
    historical_collate, Model = check_source()
    bank = V2TFBank(a.cohort, a.features, a.tf_features, a.train_val_split,
                    a.v2_protocol, a.protocol)
    fit = sorted(p for p in bank.patients if bank.roles[p] == "inner_train")
    val = sorted(p for p in bank.patients if bank.roles[p] == "inner_val")
    if len(bank.patients) != 141 or len(fit) != 112 or len(val) != 29:
        raise RuntimeError("Official train/validation patient split changed")
    collate = wrap_collate(historical_collate)
    fit_stage("inner", Model, collate, bank, fit, val, 30, a.runtime)
    selected = json.loads((a.runtime / "inner/best.json").read_text(encoding="utf-8"))
    history = pd.read_csv(a.runtime / "inner/epochs.csv")
    history["selected"] = history["epoch"].eq(selected["epoch"])
    history.to_csv(a.output / "INTERICTAL_CHECKPOINT_SELECTION.csv", index=False)
    scores = validation_channel_scores(bank, val, collate, Model,
                                       a.runtime / "inner/best.pt",
                                       a.runtime / "inner/normalizer.npz")
    threshold = select_threshold(scores, a.output, selected["checkpoint_sha256"], EXPECTED_LOCK)
    fit_stage("final", Model, collate, bank, bank.patients, None,
              selected["epoch"], a.runtime)
    final_checkpoint = a.runtime / "final/last.pt"
    freeze = {"model_frozen_before_official_test": True,
              "threshold_frozen_before_official_test": True,
              "protocol_sha256": EXPECTED_LOCK,
              "selected_inner_epoch": selected["epoch"],
              "selected_inner_patient_equal_AP": selected["validation_patient_equal_AP"],
              "threshold": threshold["threshold"],
              "threshold_json_sha256": sha256(a.output / "FROZEN_THRESHOLD.json"),
              "final_checkpoint_sha256": sha256(final_checkpoint),
              "official_test_accessed_by_this_run": False}
    atomic_json(a.output / "INTERICTAL_MODEL_FREEZE.json", freeze)
    print(json.dumps(freeze, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
