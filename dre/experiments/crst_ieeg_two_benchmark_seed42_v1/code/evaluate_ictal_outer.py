"""One-shot frozen five-fold ictal OUTER test, separate from 47-ID VLOO.

All identity-bearing query/patient rows remain in private runtime.  The
historical A1 0.7464 value belongs to the 47-ID validation fixed-query
protocol and must not be compared as if it were this 80-ID outer result.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.special import logit

from crst_model import CRSTiEEG
from evaluate_ictal_vloo import METRICS, fixed_query, query_metrics, write_csv
from patient_bank import IctalPatientBank, to_device
from train_crst import sha


def summarize(rows, model, scope):
    result = {"model": model, "scope": scope, "query_rows": len(rows),
              "patients": len({row["patient_private"] for row in rows})}
    result.update({metric: float(np.nanmean([row[metric] for row in rows]))
                   for metric in METRICS})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--training-lock", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (frozen["status"] != "FROZEN_BEFORE_ICTAL_OUTER_TEST" or
            frozen["protocol_sha256"] != sha(args.protocol) or
            frozen["training_lock_sha256"] != sha(args.training_lock) or
            frozen["outer_test_accessed_before_freeze"] or
            len(frozen["models"]) != 10):
        raise RuntimeError("All ictal models/thresholds must be frozen before outer test")
    bank = IctalPatientBank(args.cache, args.manifest, args.feature_cache)
    test_ids = [patient for fold in range(1, 6) for patient in bank.folds[fold]["test"]]
    if len(test_ids) != 80 or len(set(test_ids)) != 80:
        raise RuntimeError("Historical five-fold outer patient partition is not 80 unique IDs")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    rows, attributes = [], {}
    for fold in range(1, 6):
        for variant in ("CRST-0", "CRST-FULL"):
            key = f"fold{fold}/{variant}"
            obj = frozen["models"][key]
            checkpoint = Path(obj["checkpoint_path"])
            if sha(checkpoint) != obj["checkpoint_sha256"]:
                raise RuntimeError("Frozen ictal checkpoint changed")
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            if state["threshold"] != obj["frozen_threshold"]:
                raise RuntimeError("Frozen validation threshold changed")
            tau = float(state["threshold"]["threshold"])
            model = CRSTiEEG(activation_checkpointing=False).to(device)
            model.load_state_dict(state["model"])
            model.eval()
            interventions = ([None] if variant == "CRST-0" else
                             [None, "T_ZERO", "C_ZERO", "R_ZERO",
                              "CONNECTIVITY_ZERO", "NO_RECORD_ATTENTION"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for patient in sorted(bank.folds[fold]["test"]):
                    sample = bank.load(patient)
                    attributes[patient] = {"center": sample["center"],
                                           "channels": int((sample["labels"] >= 0).sum()),
                                           "records": len(sample["patches"])}
                    inp = to_device(sample, device)
                    labels = np.asarray(sample["labels"], np.int8)
                    for intervention in interventions:
                        score = model(inp["patches"], inp["frequency_mask"],
                                      inp["window_mask"], inp["edges"],
                                      intervention=intervention)[0].sigmoid().float().cpu().numpy()
                        margin = logit(np.clip(score, 1e-8, 1-1e-8)) - logit(tau)
                        for rep in range(20):
                            query = fixed_query(len(labels), fold, patient, rep)
                            rows.append({"model": variant, "intervention": intervention or "NONE",
                                         "fold": fold, "patient_private": patient,
                                         "rep": rep, **query_metrics(labels[query], margin[query])})
                    print(json.dumps({"fold": fold, "model": variant,
                                      "completed_outer_patients": len({r["patient_private"]
                                                                       for r in rows if r["model"] == variant}),
                                      "test_used_for_selection": False}), flush=True)
    private = args.runtime / "ictal" / "private_outer_evaluation"
    write_csv(private / "OUTER_QUERY_ROWS_PRIVATE.csv", rows)
    args.output.mkdir(parents=True, exist_ok=True)
    primary = []
    for variant in ("CRST-0", "CRST-FULL"):
        part = [r for r in rows if r["model"] == variant and r["intervention"] == "NONE"]
        if len(part) != 1600 or len({r["patient_private"] for r in part}) != 80:
            raise RuntimeError("Ictal outer 80x20 fixed-query structure incomplete")
        primary.append(summarize(part, variant, "outer_test_80_ids"))
    write_csv(args.output / "ICTAL_OUTER_METRICS.csv", primary)
    intervention_rows = []
    base = primary[1]
    for intervention in ("T_ZERO", "C_ZERO", "R_ZERO",
                         "CONNECTIVITY_ZERO", "NO_RECORD_ATTENTION"):
        part = [r for r in rows if r["model"] == "CRST-FULL" and
                r["intervention"] == intervention]
        result = summarize(part, "CRST-FULL", "outer_test_80_ids")
        result["intervention"] = intervention
        for metric in METRICS:
            result["delta_" + metric] = result[metric] - base[metric]
        intervention_rows.append(result)
    write_csv(args.output / "ICTAL_RELATIVE_INTERVENTION.csv",
              [r for r in intervention_rows if r["intervention"] in
               ("T_ZERO", "C_ZERO", "R_ZERO")])
    write_csv(args.output / "ICTAL_CONNECTIVITY_INTERVENTION.csv",
              [r for r in intervention_rows if r["intervention"] == "CONNECTIVITY_ZERO"])
    write_csv(args.output / "ICTAL_RECORD_INTERVENTION.csv",
              [r for r in intervention_rows if r["intervention"] == "NO_RECORD_ATTENTION"])
    strata = []
    for variant in ("CRST-0", "CRST-FULL"):
        base_rows = [r for r in rows if r["model"] == variant and r["intervention"] == "NONE"]
        for axis in ("center", "channels", "records"):
            if axis == "center":
                groups = {name: [patient for patient, attr in attributes.items()
                                 if attr["center"] == name]
                          for name in sorted({attr["center"] for attr in attributes.values()})}
            else:
                values = np.asarray([attr[axis] for attr in attributes.values()], float)
                cuts = np.quantile(values, [0.25, 0.5, 0.75])
                groups = {f"Q{q+1}": [patient for patient, attr in attributes.items()
                                      if np.searchsorted(cuts, attr[axis], side="right") == q]
                          for q in range(4)}
            for group_name, patients in groups.items():
                if not patients:
                    continue
                membership = set(patients)
                part = [r for r in base_rows if r["patient_private"] in membership]
                result = summarize(part, variant, "outer_test_stratum")
                result.update({"axis": axis, "group": group_name,
                               "mean_channels": float(np.mean([attributes[x]["channels"]
                                                                for x in patients])),
                               "mean_records": float(np.mean([attributes[x]["records"]
                                                               for x in patients]))})
                strata.append(result)
    write_csv(args.output / "ICTAL_PATIENT_METRICS.csv", strata)
    (args.output / "ICTAL_OUTER_ACCESS_AUDIT.json").write_text(json.dumps({
        "status": "PASS", "pretest_freeze_sha256": sha(args.freeze),
        "outer_test_patients": 80, "query_repetitions": 20,
        "test_checkpoint_selection": False, "test_threshold_selection": False,
        "historical_47id_A1_not_conflated_with_80id_outer": True,
        "test_accessed": True}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ICTAL_OUTER_TEST_COMPLETE",
                      "primary": primary, "test_used_for_selection": False}), flush=True)


if __name__ == "__main__":
    main()
