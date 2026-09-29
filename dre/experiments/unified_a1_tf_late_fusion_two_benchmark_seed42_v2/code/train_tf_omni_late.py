"""Train independent Omni TF scorer on official train only, then freeze.

Reuses audited v1 physical-frequency cache alignment and v2 official cohort.
There is deliberately no A1 model in the optimizer or this forward pass.
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
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
V1 = HERE.parents[1] / "unified_a1_tf_two_benchmark_seed42_v1" / "code"
sys.path.insert(0, str(V1))
from train_tf_omni import V2TFBank, wrap_collate  # noqa: E402
from train_v2 import patient_equal_weighted_bce, to_device, check_source  # noqa: E402
from late_tf import TFScorer  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def make_model(device):
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)
    return TFScorer().to(device)


def val_ap(model, bank, patients, collate, mean, std, device):
    model.eval()
    values = []
    with torch.inference_mode():
        for patient in patients:
            batch = to_device(collate([bank.example(patient, mean, std, epoch=None)]), device)
            margin = model(batch)["margin_tf"][0].detach().cpu().numpy()
            y = np.asarray([bank.labels[patient][name] for name in bank.canonical[patient]], dtype=np.int8)
            if y.any():
                values.append(float(average_precision_score(y, margin[:len(y)])))
    if not values:
        raise RuntimeError("No estimable validation patient AP")
    return float(np.mean(values))


def fit(stage_name, bank, patients, validation, collate, epochs, runtime, lock_sha):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage = runtime / stage_name
    stage.mkdir(parents=True, exist_ok=True)
    normalizer = stage / "normalizer.npz"
    if normalizer.is_file():
        with np.load(normalizer) as data:
            mean, std, tf_mean, tf_std = [data[k] for k in ("mean", "std", "tf_mean", "tf_std")]
    else:
        mean, std = bank.fit_normalizer(patients)
        tf_mean, tf_std = bank.fit_tf_normalizer(patients)
        np.savez(normalizer, mean=mean, std=std, tf_mean=tf_mean, tf_std=tf_std)
    bank.set_tf_normalizer(tf_mean, tf_std)
    model = make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    last = stage / "last.pt"
    start = 1
    if last.is_file():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["stage"] != stage_name or state["protocol_sha256"] != lock_sha or state["epoch"] > epochs:
            raise RuntimeError("TF resume provenance mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["epoch"]) + 1
    for epoch in range(start, epochs + 1):
        ordered = sorted(patients)
        random.Random(42 * 100000 + epoch).shuffle(ordered)
        torch.manual_seed(42 * 100000 + epoch)
        model.train()
        losses = []
        for offset in range(0, len(ordered), 2):
            batch = to_device(collate([bank.example(p, mean, std, epoch=epoch)
                                       for p in ordered[offset:offset + 2]]), device)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch)
            loss = patient_equal_weighted_bce(-result["margin_tf"], batch["labels_nez"],
                                              batch["labels_ez"], batch["channel_mask"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite TF-only loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = val_ap(model, bank, validation, collate, mean, std, device) if validation else None
        with (stage / "epochs.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(["stage", "epoch", "train_loss", "validation_patient_equal_AP"])
            writer.writerow([stage_name, epoch, float(np.mean(losses)), score])
        state = {"stage": stage_name, "epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "protocol_sha256": lock_sha}
        temp = last.with_suffix(".pt.tmp")
        torch.save(state, temp)
        os.replace(temp, last)
        if validation:
            best_path = stage / "best.json"
            previous = json.loads(best_path.read_text(encoding="utf-8")) if best_path.is_file() else None
            if previous is None or score > previous["validation_patient_equal_AP"] + 1e-12:
                best = stage / "best.pt"
                temp = best.with_suffix(".pt.tmp")
                torch.save(state, temp)
                os.replace(temp, best)
                save_json(best_path, {"epoch": epoch, "validation_patient_equal_AP": score,
                                      "checkpoint_sha256": sha(best)})
        print(f"{stage_name} epoch={epoch}/{epochs} loss={np.mean(losses):.6f} val_AP={score}", flush=True)
    return {"normalizer_sha256": sha(normalizer), "model_parameters": sum(p.numel() for p in model.parameters())}


def main():
    p = argparse.ArgumentParser()
    for name in ("cohort", "split", "features", "tf-features", "v2-protocol", "tf-protocol",
                 "lock", "runtime", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if (a.runtime / "official_test_access_started.json").exists():
        raise RuntimeError("Official test already opened for this run")
    a.runtime.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=True)
    lock_sha = sha(a.lock)
    historical_collate, _ = check_source()
    bank = V2TFBank(a.cohort, a.features, a.tf_features, a.split, a.v2_protocol, a.tf_protocol)
    fit_patients = sorted(p for p in bank.patients if bank.roles[p] == "inner_train")
    val_patients = sorted(p for p in bank.patients if bank.roles[p] == "inner_val")
    if (len(bank.patients), len(fit_patients), len(val_patients)) != (141, 112, 29):
        raise RuntimeError("Omni official train/inner split changed")
    collate = wrap_collate(historical_collate)
    inner = fit("inner", bank, fit_patients, val_patients, collate, 30, a.runtime, lock_sha)
    selected = json.loads((a.runtime / "inner/best.json").read_text(encoding="utf-8"))
    final = fit("final", bank, bank.patients, None, collate, selected["epoch"], a.runtime, lock_sha)
    freeze = {"protocol_sha256": lock_sha, "selected_epoch": selected["epoch"],
              "selected_inner_patient_equal_AP": selected["validation_patient_equal_AP"],
              "inner_checkpoint_sha256": selected["checkpoint_sha256"],
              "final_checkpoint_sha256": sha(a.runtime / "final/last.pt"),
              "inner": inner, "final": final, "model_frozen_before_official_test": True,
              "official_test_accessed": False, "test_used_for_tuning": False}
    save_json(a.output / "INTERICTAL_TF_TRAINING_AUDIT.json", freeze)
    pd.read_csv(a.runtime / "inner/epochs.csv").to_csv(a.output / "TF_CHECKPOINT_SELECTION.csv", index=False)
    print(json.dumps({"status": "FROZEN", "selected_epoch": selected["epoch"]}), flush=True)


if __name__ == "__main__":
    main()
