"""Single post-freeze Ictal outer pass for matched B0/PC and interventions.

All patient/channel scores stay under private runtime; only fold/overall
aggregates and cluster-bootstrap summaries are suitable for Git. The 0.746382
historical A1 reference is development VLOO, not paired to this outer test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from official_spectrum import OfficialSpectrum, load_official_module
from patient_bank import IctalBank
from pc_cnn import PCCNN
from raw_metrics import binary_metrics
from train_pccnn import evaluate as evaluate_pc
from train_rawcnn import digest, evaluate as evaluate_raw, save_json


METRICS = ("auroc", "ap", "macro_f1", "mrr", "top1")


def fixed_query(n: int, fold: int, patient: str, rep: int):
    """Exact historical A1/CRST fixed-query membership, seed 42.

    Source: crst_ieeg_two_benchmark_seed42_v1/code/evaluate_ictal_vloo.py.
    This is independent of model predictions and is shared by both models.
    """
    if n < 4:
        raise RuntimeError("Too few channels for historical fixed query")
    payload = "|".join(str(value) for value in (42, fold, patient, rep, "split"))
    seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**32)
    permutation = np.random.default_rng(seed).permutation(n)
    return permutation[n // 2:]


def query_metrics(y, score, threshold=0.5):
    y, score = np.asarray(y, np.int8), np.asarray(score, np.float64)
    if len(y) != len(score) or not np.isfinite(score).all():
        raise RuntimeError("Invalid fixed-query inputs")
    result = {"macro_f1": float(f1_score(y, score > threshold,
                                          average="macro", zero_division=0))}
    if len(np.unique(y)) < 2:
        result.update({name: None for name in ("auroc", "ap", "mrr", "top1")})
        return result
    ordering = np.argsort(-score, kind="stable")
    positive_ranks = np.flatnonzero(y[ordering] == 1)
    result.update({"auroc": float(roc_auc_score(y, score)),
                   "ap": float(average_precision_score(y, score)),
                   "mrr": float(1 / (1 + positive_ranks[0])),
                   "top1": float(y[ordering[0]])})
    return result


def original_channel_order(bank, patient, row):
    """Undo training metric's alphabetical sort before historical fixed queries."""
    with np.load(bank.path(patient), allow_pickle=False) as source:
        names = [str(value) for value in source["channel_names"]]
        labels = source["labels"].astype(np.int8)
    eligible = [(name, int(label)) for name, label in zip(names, labels) if label >= 0]
    if len(eligible) != len(row["labels"]) or len({name for name, _ in eligible}) != len(eligible):
        raise RuntimeError("Ictal cache channel coverage/order is incompatible with fixed queries")
    by_name = dict(zip(sorted(name for name, _ in eligible),
                       zip(row["labels"], row["scores"])))
    if set(by_name) != {name for name, _ in eligible}:
        raise RuntimeError("Ictal channel names differ from frozen prediction order")
    for name, label in eligible:
        if by_name[name][0] != label:
            raise RuntimeError("Ictal channel labels differ after order restoration")
    return {"labels": [label for _, label in eligible],
            "scores": [by_name[name][1] for name, _ in eligible]}


def fixed_query_rows(private, fold, bank=None):
    rows = {}
    for patient, record in private.items():
        if bank is not None:
            record = original_channel_order(bank, patient, record)
        labels = np.asarray(record["labels"], np.int8)
        scores = np.asarray(record["scores"], np.float64)
        for rep in range(20):
            ids = fixed_query(len(labels), fold, patient, rep)
            rows[(patient, rep)] = query_metrics(labels[ids], scores[ids])
    return rows


def per_patient(private):
    return {patient: binary_metrics(row["labels"], row["scores"])
            for patient, row in private.items()}


def aggregate(rows):
    result = {}
    for metric in METRICS:
        values = [row[metric] for row in rows.values() if row[metric] is not None]
        result[metric] = float(np.mean(values)) if values else None
        result[f"{metric}_estimable_patients"] = len(values)
    return result


def paired_query_bootstrap(left, right, patients, draws):
    if set(left) != set(right) or set(left) != {(patient, rep)
                                               for patient in patients for rep in range(20)}:
        raise RuntimeError("Frozen fixed-query model alignment differs")
    result = {}
    for metric in METRICS:
        # Each sampled patient brings all 20 historical fixed-query repetitions.
        deltas = np.asarray([
            np.nanmean([left[(patient, rep)][metric] - right[(patient, rep)][metric]
                        if left[(patient, rep)][metric] is not None and
                        right[(patient, rep)][metric] is not None else np.nan
                        for rep in range(20)]) for patient in patients], dtype=float)
        samples = np.nanmean(deltas[draws], axis=1)
        result[metric] = {"mean_delta": float(np.nanmean(deltas)),
                          "ci_low": float(np.nanquantile(samples, 0.025)),
                          "ci_high": float(np.nanquantile(samples, 0.975)),
                          "estimable_patients": int(np.isfinite(deltas).sum())}
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--official-cnn", type=Path, required=True)
    p.add_argument("--ictal-cache", type=Path, required=True)
    p.add_argument("--ictal-manifest", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_OFFICIAL_TEST" or \
            freeze["protocol_sha256"] != digest(args.protocol) or \
            not freeze["model_frozen_before_final_test"] or freeze["final_test_accessed"]:
        raise RuntimeError("Ictal outer data cannot be opened before model freeze")
    if digest(args.ictal_manifest) != json.loads(args.protocol.read_text(encoding="utf-8"))[
            "ictal_fold_manifest_sha256"]:
        raise RuntimeError("Frozen ictal fold membership changed")
    args.output.mkdir(parents=True, exist_ok=True)
    bank = IctalBank(args.ictal_cache, args.ictal_manifest)
    module = load_official_module(args.official_cnn)
    preprocessor = OfficialSpectrum(module, channel_chunk=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    private_by_model, query_by_model, fold_rows = {key: {} for key in
                                   ("RawCNN", "PC-CNN", "PC_no_physiology",
                                    "PC_no_context", "PC_all_disabled")}, {key: {} for key in
                                   ("RawCNN", "PC-CNN", "PC_no_physiology",
                                    "PC_no_context", "PC_all_disabled")}, []
    for fold in range(1, 6):
        key = f"ictal/fold{fold}"
        models = freeze["benchmark_checkpoints"][key]
        for selected in models.values():
            if digest(Path(selected["private_path"])) != selected["sha256"]:
                raise RuntimeError("A selected Ictal checkpoint changed after freeze")
        test_patients = bank.folds[fold]["test"]
        fold_cache = args.runtime / "ictal" / f"FROZEN_OUTER_FOLD{fold}_PRIVATE.json"
        if fold_cache.is_file():
            cached = json.loads(fold_cache.read_text(encoding="utf-8"))
            if cached.get("freeze_sha256") != digest(args.freeze) or \
                    cached.get("checkpoint_sha256") != {name: item["sha256"] for name, item in models.items()} or \
                    set(cached.get("patient_scores", {})) != set(private_by_model) or \
                    any(set(value) != set(test_patients) for value in cached["patient_scores"].values()):
                raise RuntimeError("Frozen outer fold resume provenance differs")
            snapshots = {name: (None, private) for name, private in cached["patient_scores"].items()}
        else:
            raw_state = torch.load(models["RawCNN"]["private_path"],
                               map_location=device, weights_only=False)
            raw = module.NeuralCNN(in_channels=1, outputs=1).to(device)
            raw.load_state_dict(raw_state["raw_state"])
            raw_metrics, raw_private = evaluate_raw(raw, preprocessor, bank,
                                                    test_patients, "ictal", 0, device)
            normalizer_path = args.runtime / "ictal" / f"fold{fold}" / "descriptor_normalization.json"
            if digest(normalizer_path) != models["PC-CNN"]["train_fit_descriptor_norm_sha256"]:
                raise RuntimeError("Train-fit descriptor normalization changed after freeze")
            normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
            pc_state = torch.load(models["PC-CNN"]["private_path"],
                                  map_location=device, weights_only=False)
            pc = PCCNN(module.NeuralCNN(in_channels=1, outputs=1)).to(device)
            pc.load_state_dict(pc_state["model_state"])
            variants = (("PC-CNN", True, True),
                        ("PC_no_physiology", False, True),
                        ("PC_no_context", True, False),
                        ("PC_all_disabled", False, False))
            snapshots = {"RawCNN": (raw_metrics, raw_private)}
            for name, physiology, context in variants:
                snapshots[name] = evaluate_pc(pc, preprocessor, bank, test_patients,
                                              "ictal", 0, normalizer, device,
                                              physiology=physiology, context=context)
            if any(set(private) != set(test_patients) for _, private in snapshots.values()):
                raise RuntimeError("Frozen outer fold patient coverage differs before resume save")
            save_json(fold_cache, {"freeze_sha256": digest(args.freeze),
                                   "checkpoint_sha256": {name: item["sha256"] for name, item in models.items()},
                                   "patient_scores": {name: private for name, (_, private) in snapshots.items()}})
        for name, (_, private) in snapshots.items():
            if set(private) != set(test_patients):
                raise RuntimeError("Frozen outer patient coverage differs")
            queries = fixed_query_rows(private, fold, bank)
            if len(queries) != len(test_patients) * 20:
                raise RuntimeError("Historical 20-query patient coverage differs")
            row = {"benchmark": "Ictal", "outer_fold": fold, "model": name,
                   "patients": len(test_patients), "query_rows": len(queries),
                   "metric_unit": "patient_x_20_fixed_queries",
                   **aggregate(queries)}
            fold_rows.append(row)
            private_by_model[name].update(private)
            query_by_model[name].update(queries)
        print(json.dumps({"status": "FROZEN_OUTER_FOLD_COMPLETE", "fold": fold,
                          "patients": len(test_patients)}), flush=True)
    if any(len(private) != 80 for private in private_by_model.values()):
        raise RuntimeError("Frozen outer Ictal 80-person coverage incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "ICTAL_RAWCNN_METRICS.csv",
              [row for row in fold_rows if row["model"] == "RawCNN"])
    write_csv(args.output / "ICTAL_PCCNN_METRICS.csv",
              [row for row in fold_rows if row["model"] == "PC-CNN"])
    if any(len(rows) != 1600 for rows in query_by_model.values()):
        raise RuntimeError("Historical 80x20 fixed-query outer structure incomplete")
    summary = [{"benchmark": "Ictal", "model": name, "patients": 80,
                "query_rows": 1600, "metric_unit": "patient_x_20_fixed_queries",
                **aggregate(rows)} for name, rows in query_by_model.items()]
    write_csv(args.output / "ICTAL_INTERVENTION.csv", summary)
    ids = sorted(private_by_model["RawCNN"])
    draws = np.random.default_rng(42).integers(0, len(ids), size=(10000, len(ids)))
    bootstrap = []
    for candidate in ("PC-CNN", "PC_no_physiology", "PC_no_context", "PC_all_disabled"):
        paired = paired_query_bootstrap(query_by_model[candidate],
                                       query_by_model["RawCNN"], ids, draws)
        for metric, estimate in paired.items():
            bootstrap.append({"benchmark": "Ictal", "candidate": candidate,
                              "reference": "RawCNN", "metric": metric,
                              **estimate, "draws": 10000,
                              "cluster_unit": "patient_all_20_fixed_queries"})
    write_csv(args.output / "ICTAL_BOOTSTRAP.csv", bootstrap)
    private_path = args.runtime / "ictal" / "FROZEN_OUTER_SCORES_PRIVATE.json"
    save_json(private_path, {"freeze_sha256": digest(args.freeze),
                             "patient_scores": private_by_model})
    save_json(args.output / "ICTAL_OUTER_STATUS.json",
              {"status": "ONE_FROZEN_OUTER_PASS_COMPLETE", "patients": 80,
               "query_rows_per_model": 1600,
               "metric_unit": "patient_x_20_fixed_queries",
               "matched_primary_comparison": "PC-CNN_vs_RawCNN",
               "historical_a1_0_746382_is_development_vloo_not_outer_test": True,
               "test_used_for_tuning": False})


if __name__ == "__main__":
    main()
