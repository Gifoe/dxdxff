"""One-shot frozen-model official Omni evaluation; no selection on test.

Patient/EDF/channel predictions stay under the private runtime.  Public
outputs contain only aggregate rows without identities or score vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from crst_metrics import aggregate_patients, patient_metrics
from crst_model import CRSTiEEG
from patient_bank import OmniTestPatientBank, to_device
from train_crst import sha


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_stats(y, p, tau):
    y, p = np.asarray(y, int), np.asarray(p, float)
    z = p >= tau
    tp, fp = int(np.sum((y == 1) & z)), int(np.sum((y == 0) & z))
    tn, fn = int(np.sum((y == 0) & ~z)), int(np.sum((y == 1) & ~z))
    f1p = 2 * tp / max(1, 2 * tp + fp + fn)
    f1n = 2 * tn / max(1, 2 * tn + fp + fn)
    se = tp / max(1, tp + fn)
    sp = tn / max(1, tn + fp)
    return {"macro_f1": (f1p + f1n) / 2, "pathological_f1": f1p,
            "normal_f1": f1n, "sensitivity": se, "specificity": sp,
            "balanced_accuracy": (se + sp) / 2, "accuracy": (tp+tn)/max(1,len(y)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate_variant(model, bank, patients, device, threshold, intervention=None):
    model.eval()
    official, patient_rows, patient_public = [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ordinal, patient in enumerate(patients, 1):
            sample = bank.load(patient, 0, all_clips=True)
            device_sample = to_device(sample, device)
            logits = model(device_sample["patches"], device_sample["frequency_mask"],
                           device_sample["window_mask"], device_sample["edges"],
                           intervention=intervention)[0]
            scores = logits.sigmoid().float().cpu().numpy()
            y = np.asarray(sample["labels"], int)
            good = y >= 0
            patient_rows.append((y[good], scores[good]))
            pm = patient_metrics(y[good], scores[good])
            patient_public.append({"patient_private": patient, "center": sample["center"],
                                   "channels": int(good.sum()), "records": len(sample["record_edf_ids"]),
                                   "label_conflicts": sample["label_conflicts"],
                                   **pm})
            edf_ids = np.asarray(sample["record_edf_ids"])
            for edf_idx, key in enumerate(sample["edf_keys"]):
                first = int(np.flatnonzero(edf_ids == edf_idx)[0])
                ry = np.asarray(sample["record_labels"][first], int)
                for channel_idx in np.flatnonzero(ry >= 0):
                    official.append({"patient_private": patient,
                                     "edf_private": key,
                                     "channel_private": sample["channel_names"][channel_idx],
                                     "center": sample["center"],
                                     "label": int(ry[channel_idx]),
                                     "score": float(scores[channel_idx])})
            print(json.dumps({"variant": intervention or "PRIMARY", "patient_ordinal": ordinal,
                              "patients_total": len(patients), "edfs": len(sample["edf_keys"])}),
                  flush=True)
    if len(official) != 8104:
        raise RuntimeError("Official test (EDF,channel) count not 8104")
    labels = np.asarray([row["label"] for row in official], int)
    scores = np.asarray([row["score"] for row in official], float)
    result = {"official_pooled_auroc": float(roc_auc_score(labels, scores)),
              "official_pooled_ap": float(average_precision_score(labels, scores)),
              "threshold": float(threshold),
              "official_edf_channels": len(official), "patients": len(patients)}
    result.update(binary_stats(labels, scores, threshold))
    result.update({"patient_equal_" + key: value for key, value in
                   aggregate_patients(patient_rows).items()})
    return result, official, patient_public


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--test-cache", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--training-lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    audit = json.loads((args.runtime / "OMNI_TEST_CACHE_AUDIT.json").read_text())
    if (freeze["status"] != "FROZEN_BEFORE_OFFICIAL_TEST" or
            freeze["protocol_sha256"] != sha(args.protocol) or
            freeze["training_lock_sha256"] != sha(args.training_lock) or
            audit["freeze_sha256"] != sha(args.freeze) or not audit["pass"]):
        raise RuntimeError("Official test evaluation lacks exact pre-test freeze")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    bank = OmniTestPatientBank(args.test_cache, args.freeze)
    patients = sorted(bank.patient_files)
    args.output.mkdir(parents=True, exist_ok=True)
    private = args.runtime / "omni" / "private_evaluation"
    private.mkdir(parents=True, exist_ok=True)
    primary, intervention_rows, patient_rows_all, official_private = [], [], [], {}
    for variant in ("CRST-0", "CRST-FULL"):
        obj = freeze["models"][variant]
        checkpoint = Path(obj["checkpoint_path"])
        if sha(checkpoint) != obj["checkpoint_sha256"]:
            raise RuntimeError("Frozen checkpoint changed after test access")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["threshold"] != obj["threshold"]:
            raise RuntimeError("Frozen numeric threshold changed")
        model = CRSTiEEG(activation_checkpointing=False).to(device)
        model.load_state_dict(state["model"])
        tau = float(obj["threshold"]["threshold"])
        result, official, patient_rows = evaluate_variant(model, bank, patients, device, tau)
        result.update({"model": variant, "benchmark": "Omni",
                       "threshold_origin": "frozen_inner_train_validation",
                       "checkpoint_sha256": obj["checkpoint_sha256"]})
        primary.append(result)
        patient_rows_all.extend({"model": variant, **row} for row in patient_rows)
        official_private[variant] = official
        write_csv(private / f"{variant.replace('-', '')}_OFFICIAL_ROWS_PRIVATE.csv", official)
        if variant == "CRST-FULL":
            for intervention in ("T_ZERO", "C_ZERO", "R_ZERO",
                                 "CONNECTIVITY_ZERO", "NO_RECORD_ATTENTION"):
                altered, _, _ = evaluate_variant(model, bank, patients, device, tau,
                                                  intervention=intervention)
                intervention_rows.append({"intervention": intervention,
                                          "model": variant, "benchmark": "Omni",
                                          "delta_official_auroc": altered["official_pooled_auroc"] -
                                          result["official_pooled_auroc"],
                                          "delta_patient_equal_ap": altered["patient_equal_ap"] -
                                          result["patient_equal_ap"],
                                          **altered})
    write_csv(args.output / "OMNI_PRIMARY_METRICS.csv", primary)
    # Patient identities and per-channel predictions stay private.
    write_csv(private / "OMNI_PATIENT_METRICS_PRIVATE.csv", patient_rows_all)
    strata = []
    for variant in ("CRST-0", "CRST-FULL"):
        part = [row for row in patient_rows_all if row["model"] == variant]
        for axis in ("channels", "records"):
            values = np.asarray([row[axis] for row in part], float)
            cuts = np.quantile(values, [0.25, 0.5, 0.75])
            for quartile in range(4):
                selected = [row for row in part
                            if int(np.searchsorted(cuts, row[axis], side="right")) == quartile]
                if not selected:
                    continue
                strata.append({"model": variant, "axis": axis,
                               "quartile": quartile+1, "patients": len(selected),
                               "min_axis": min(row[axis] for row in selected),
                               "max_axis": max(row[axis] for row in selected),
                               "mean_channels": float(np.mean([row["channels"] for row in selected])),
                               "mean_records": float(np.mean([row["records"] for row in selected])),
                               **{metric: float(np.nanmean([row[metric] for row in selected]))
                                  for metric in ("auroc", "ap", "mrr", "top1")}})
    write_csv(args.output / "OMNI_PATIENT_METRICS.csv", strata)
    center_rows = []
    for variant, official in official_private.items():
        patient_part = [row for row in patient_rows_all if row["model"] == variant]
        for center in sorted({row["center"] for row in official}):
            part = [row for row in official if row["center"] == center]
            pp = [row for row in patient_part if row["center"] == center]
            y = np.asarray([row["label"] for row in part], int)
            score = np.asarray([row["score"] for row in part], float)
            threshold = next(row["threshold"] for row in primary if row["model"] == variant)
            center_rows.append({"model": variant, "center": center,
                                "patients": len(pp), "edf_channels": len(part),
                                "positive_prevalence": float(y.mean()),
                                "official_pooled_auroc": float(roc_auc_score(y, score))
                                if len(np.unique(y)) == 2 else float("nan"),
                                "official_pooled_ap": float(average_precision_score(y, score))
                                if y.sum() else float("nan"),
                                "patient_equal_auroc": float(np.nanmean([row["auroc"] for row in pp])),
                                "patient_equal_ap": float(np.nanmean([row["ap"] for row in pp])),
                                "mrr": float(np.nanmean([row["mrr"] for row in pp])),
                                "top1": float(np.nanmean([row["top1"] for row in pp])),
                                **binary_stats(y, score, threshold)})
    write_csv(args.output / "OMNI_CENTER_METRICS.csv", center_rows)
    write_csv(args.output / "OMNI_RELATIVE_INTERVENTION.csv",
              [row for row in intervention_rows if row["intervention"] in
               {"T_ZERO", "C_ZERO", "R_ZERO"}])
    write_csv(args.output / "OMNI_CONNECTIVITY_INTERVENTION.csv",
              [row for row in intervention_rows if row["intervention"] ==
               "CONNECTIVITY_ZERO"])
    (args.output / "OMNI_TEST_EVALUATION_AUDIT.json").write_text(json.dumps({
        "pass": True, "freeze_sha256": sha(args.freeze),
        "official_test_cache_audit_sha256": sha(args.runtime / "OMNI_TEST_CACHE_AUDIT.json"),
        "test_checkpoint_selection": False, "test_threshold_selection": False,
        "private_predictions": str(private), "test_accessed": True}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"status": "OMNI_OFFICIAL_TEST_COMPLETE", "summary": primary}), flush=True)


if __name__ == "__main__":
    main()
