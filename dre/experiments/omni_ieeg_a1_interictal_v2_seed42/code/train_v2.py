"""A1-Omni v2 training: same A1 model/objective, official labels and AP selection.

All large tensors/checkpoints remain on the server. Official test is never
opened here. Validation predicts only official-train patients.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score


V1_LOCAL = Path(__file__).resolve().parents[2] / "omni_ieeg_a1_interictal_seed42_v1/code"
V1_SERVER = Path("E:/DRE-nips/new-pipeline/7-11/omni_a1_v1/code")
V1_CODE = V1_LOCAL if V1_LOCAL.is_dir() else V1_SERVER
sys.path.insert(0, str(V1_CODE))
from train_a1_omni import (Bank as V1Bank, atomic_json, check_source, make_model,
                           patient_equal_weighted_bce, sha256, to_device)  # noqa: E402


class V2Bank(V1Bank):
    def __init__(self, cohort: Path, features_root: Path, split: Path | None,
                 official_split: str = "train", protocol: Path | None = None):
        rows = pd.read_csv(cohort)
        rows = rows.loc[(rows["official_split"] == official_split) &
                        (rows["official_labeled_channels"] > 0)]
        roles = pd.read_csv(split).set_index("patient")["role"].to_dict() if split else {}
        if split and set(rows["patient"]) != set(roles):
            raise RuntimeError("Inner split does not equal official eligible train patients")
        self.roles = roles
        self.by_patient = defaultdict(list)
        self.labels = defaultdict(dict)
        protocol_sha = sha256(protocol) if protocol else None
        self.edf_labels = {}
        self.patient_dataset = {}
        for row in rows.itertuples(index=False):
            path = features_root / Path(row.edf).with_suffix(".npz")
            marker_path = path.with_suffix(".json")
            if not path.is_file() or not marker_path.is_file():
                raise RuntimeError(f"Incomplete v2 feature cache: {row.edf}")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if protocol_sha and marker["protocol_sha256"] != protocol_sha:
                raise RuntimeError(f"V2 feature protocol mismatch: {row.edf}")
            if path.stat().st_size != marker["bytes"]:
                raise RuntimeError(f"V2 feature length changed: {row.edf}")
            with np.load(path, allow_pickle=False) as payload:
                features = np.asarray(payload["features"], dtype=np.float32)
                names = [str(name) for name in payload["channel_names"]]
                pathology = np.asarray(payload["pathology"], dtype=np.int8)
                reference_channels = int(payload["good_reference_channels"])
            if features.ndim != 4 or features.shape[1:] != (59, len(names), 36):
                raise RuntimeError(f"Unexpected A1 feature tensor: {row.edf} {features.shape}")
            if len(names) != row.official_labeled_channels or \
                    int(pathology.sum()) != row.pathological_channels or \
                    reference_channels != row.good_signal_channels or \
                    not np.isin(pathology, [0, 1]).all() or \
                    not np.isfinite(features).all():
                raise RuntimeError(f"Official label/reference count mismatch: {row.edf}")
            if len(names) != len(set(names)):
                raise RuntimeError(f"Repeated labeled channel in EDF: {row.edf}")
            self.patient_dataset[str(row.patient)] = str(row.dataset)
            self.edf_labels[str(row.edf)] = dict(zip(names, pathology.tolist()))
            for name, label in zip(names, pathology):
                prior = self.labels[str(row.patient)].get(name)
                if prior is not None and prior != int(label):
                    raise RuntimeError(f"Patient-channel pathology label conflict: {row.patient}/{name}")
                self.labels[str(row.patient)][name] = int(label)
            self.by_patient[str(row.patient)].append({"edf": str(row.edf),
                                                      "features": features, "names": names})
        self.patients = sorted(self.by_patient)
        self.canonical = {patient: sorted(self.labels[patient]) for patient in self.patients}


def validation_patient_ap(model, collate, bank: V2Bank, patients, mean, std, device):
    model.eval()
    ap = []
    non_estimable = 0
    with torch.inference_mode():
        for patient in patients:
            batch = to_device(collate([bank.example(patient, mean, std, epoch=None)]), device)
            logits = model(batch)["logits"][0].detach().cpu().numpy()
            y = np.asarray([bank.labels[patient][name] for name in bank.canonical[patient]],
                           dtype=np.int8)
            if not y.any():
                non_estimable += 1
                continue
            scores = 1.0 - torch.sigmoid(torch.from_numpy(logits)).numpy()
            ap.append(float(average_precision_score(y, scores)))
    if not ap:
        raise RuntimeError("No estimable pathology patient in validation")
    return float(np.mean(ap)), non_estimable


def fit_stage(stage_name, Model, collate, bank: V2Bank, patients: list[str],
              val_patients: list[str] | None, epochs: int, runtime: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage = runtime / stage_name
    stage.mkdir(parents=True, exist_ok=True)
    normalizer = stage / "normalizer.npz"
    if normalizer.exists():
        with np.load(normalizer) as values:
            mean, std = values["mean"], values["std"]
    else:
        mean, std = bank.fit_normalizer(patients)
        np.savez(normalizer, mean=mean, std=std)
    model = make_model(Model, collate, bank, patients[0], mean, std, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    last = stage / "last.pt"
    start = 1
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["stage"] != stage_name or state["epoch"] > epochs:
            raise RuntimeError("V2 resume checkpoint identity mismatch")
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
            batch = to_device(collate([bank.example(p, mean, std, epoch=epoch)
                                      for p in chosen]), device)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch)
            loss = patient_equal_weighted_bce(result["logits"], batch["labels_nez"],
                                              batch["labels_ez"], batch["channel_mask"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite A1 v2 loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_ap, no_positive = (validation_patient_ap(model, collate, bank, val_patients,
                                                     mean, std, device)
                               if val_patients is not None else (float("nan"), -1))
        with history.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(["stage", "epoch", "train_loss", "patient_equal_pathology_ap",
                                 "val_non_estimable_patients"])
            writer.writerow([stage_name, epoch, float(np.mean(losses)), val_ap, no_positive])
        payload = {"stage": stage_name, "epoch": epoch, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict()}
        temp = stage / "last.pt.tmp"
        torch.save(payload, temp)
        os.replace(temp, last)
        if val_patients is not None:
            selected = stage / "best.json"
            previous = json.loads(selected.read_text(encoding="utf-8")) if selected.exists() else None
            if previous is None or val_ap > previous["patient_equal_pathology_ap"] + 1e-12:
                best = stage / "best.pt"
                temp = stage / "best.pt.tmp"
                torch.save(payload, temp)
                os.replace(temp, best)
                atomic_json(selected, {"epoch": epoch, "patient_equal_pathology_ap": val_ap,
                                       "val_non_estimable_patients": no_positive,
                                       "checkpoint_sha256": sha256(best)})
        print(f"{stage_name} epoch={epoch}/{epochs} train_loss={np.mean(losses):.5f} val_AP={val_ap}",
              flush=True)
    return {"normalizer_sha256": sha256(normalizer),
            "model_parameters": sum(p.numel() for p in model.parameters())}


def validation_channel_scores(bank: V2Bank, patients: list[str], collate, Model,
                              checkpoint: Path, normalizer: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(normalizer) as values:
        mean, std = values["mean"], values["std"]
    model = make_model(Model, collate, bank, patients[0], mean, std, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    rows = []
    with torch.inference_mode():
        for patient in patients:
            batch = to_device(collate([bank.example(patient, mean, std, epoch=None)]), device)
            logits = model(batch)["logits"][0].detach().cpu().numpy()
            scores = 1.0 - torch.sigmoid(torch.from_numpy(logits)).numpy()
            score_of = dict(zip(bank.canonical[patient], scores.tolist()))
            for record in bank.by_patient[patient]:
                for name in record["names"]:
                    rows.append({"patient": patient, "dataset": bank.patient_dataset[patient],
                                 "edf": record["edf"], "channel": name,
                                 "pathology": bank.edf_labels[record["edf"]][name],
                                 "score_pathology": float(score_of[name])})
    result = pd.DataFrame(rows)
    if result.duplicated(["edf", "channel"]).any():
        raise RuntimeError("Duplicate validation EDF-channel")
    return result


def select_threshold(frame: pd.DataFrame, output: Path, checkpoint_sha: str,
                     protocol_sha: str):
    y = frame["pathology"].to_numpy(dtype=np.int8)
    score = frame["score_pathology"].to_numpy(dtype=np.float64)
    if not (0 < y.sum() < len(y)) or not np.isfinite(score).all():
        raise RuntimeError("Validation threshold cohort invalid")
    order = np.argsort(-score, kind="stable")
    y, score = y[order], score[order]
    end = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score) - 1]
    tp = np.cumsum(y)[end].astype(float)
    predicted_positive = (end + 1).astype(float)
    fp = predicted_positive - tp
    positives = float(y.sum())
    negatives = float(len(y) - y.sum())
    fn = positives - tp
    tn = negatives - fp
    ez_f1 = np.divide(2 * tp, 2 * tp + fp + fn,
                      out=np.zeros_like(tp), where=2 * tp + fp + fn > 0)
    normal_f1 = np.divide(2 * tn, 2 * tn + fp + fn,
                          out=np.zeros_like(tp), where=2 * tn + fp + fn > 0)
    sensitivity = tp / positives
    specificity = tn / negatives
    table = pd.DataFrame({"threshold": score[end], "validation_macro_f1": 0.5 * (ez_f1 + normal_f1),
                          "validation_pathological_f1": ez_f1,
                          "validation_ba": 0.5 * (sensitivity + specificity),
                          "sensitivity": sensitivity, "specificity": specificity,
                          "tp": tp.astype(int), "fp": fp.astype(int),
                          "fn": fn.astype(int), "tn": tn.astype(int)})
    keys = list(zip(table["validation_macro_f1"], table["validation_pathological_f1"],
                    table["validation_ba"], -np.abs(table["threshold"] - 0.5),
                    table["threshold"]))
    selected_idx = max(range(len(keys)), key=lambda idx: keys[idx])
    table["selected"] = False
    table.loc[selected_idx, "selected"] = True
    table.to_csv(output / "THRESHOLD_SELECTION.csv", index=False)
    chosen = table.iloc[selected_idx]
    frozen = {"threshold": float(chosen["threshold"]),
              "validation_macro_f1": float(chosen["validation_macro_f1"]),
              "validation_pathological_f1": float(chosen["validation_pathological_f1"]),
              "validation_ba": float(chosen["validation_ba"]),
              "validation_sensitivity": float(chosen["sensitivity"]),
              "validation_specificity": float(chosen["specificity"]),
              "validation_edf_channel_records": int(len(frame)),
              "candidate_thresholds": int(len(table)),
              "selection": "macro-F1, pathology-F1, BA, |threshold-0.5|; final exact tie larger threshold",
              "inner_checkpoint_sha256": checkpoint_sha,
              "protocol_sha256": protocol_sha}
    atomic_json(output / "FROZEN_THRESHOLD.json", frozen)
    return frozen


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
        raise RuntimeError("V2 official test access already started; training is permanently frozen")
    args.output.mkdir(parents=True, exist_ok=True)
    args.runtime.mkdir(parents=True, exist_ok=True)
    collate, Model = check_source()
    bank = V2Bank(args.cohort, args.features, args.train_val_split, protocol=args.protocol)
    fit = sorted(p for p in bank.patients if bank.roles[p] == "inner_train")
    val = sorted(p for p in bank.patients if bank.roles[p] == "inner_val")
    if len(bank.patients) != 141 or len(fit) != 112 or len(val) != 29:
        raise RuntimeError("Unexpected v2 official-train patient counts")
    inner = fit_stage("inner", Model, collate, bank, fit.copy(), val, 30, args.runtime)
    selection = json.loads((args.runtime / "inner/best.json").read_text(encoding="utf-8"))
    if selection["epoch"] not in range(1, 31):
        raise RuntimeError("Invalid AP-selected inner epoch")
    history = pd.read_csv(args.runtime / "inner/epochs.csv")
    history["selected"] = history["epoch"].eq(selection["epoch"])
    history.to_csv(args.output / "CHECKPOINT_SELECTION.csv", index=False)
    validation = validation_channel_scores(bank, val, collate, Model,
                                           args.runtime / "inner/best.pt",
                                           args.runtime / "inner/normalizer.npz")
    frozen_threshold = select_threshold(validation, args.output,
                                        selection["checkpoint_sha256"], sha256(args.protocol))
    final = fit_stage("final", Model, collate, bank, bank.patients.copy(), None,
                      selection["epoch"], args.runtime)
    checkpoint = args.runtime / "final/last.pt"
    freeze = {"model_frozen_before_official_test": True,
              "threshold_frozen_before_official_test": True,
              "official_test_accessed": False, "test_used_for_tuning": False,
              "selected_inner_epoch": selection["epoch"],
              "inner_validation_estimable_pathology_ap": selection["patient_equal_pathology_ap"],
              "inner_checkpoint_sha256": selection["checkpoint_sha256"],
              "final_checkpoint_sha256": sha256(checkpoint),
              "final_normalizer_sha256": final["normalizer_sha256"],
              "threshold_json_sha256": sha256(args.output / "FROZEN_THRESHOLD.json"),
              "frozen_threshold": frozen_threshold["threshold"],
              "protocol_sha256": sha256(args.protocol),
              "train_val_split_sha256": sha256(args.train_val_split),
              "official_train_patients": len(bank.patients),
              "inner_train_patients": len(fit), "inner_val_patients": len(val),
              "model_parameters": final["model_parameters"]}
    freeze_path = args.output / "TEST_SCORE_FREEZE_AUDIT.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text(encoding="utf-8")) != freeze:
        raise RuntimeError("Refusing to alter an existing frozen model/threshold")
    atomic_json(freeze_path, freeze)
    atomic_json(args.output / "TRAINING_AUDIT.json", {"inner": inner, "final": final,
                                                       "selection": selection,
                                                       "threshold": frozen_threshold,
                                                       "freeze": freeze})
    print(json.dumps({"status": "FROZEN", "selected_epoch": selection["epoch"],
                      "threshold": frozen_threshold["threshold"],
                      "checkpoint_sha256": freeze["final_checkpoint_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
