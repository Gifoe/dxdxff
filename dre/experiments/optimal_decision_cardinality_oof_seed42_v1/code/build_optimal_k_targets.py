"""Audit historical B0 and build private patient-level optimal-K targets.

Only fit-OOF and outer-validation arrays are indexed. Outer-test arrays are not
read. Replays deterministic nested checkpoint selection solely to recover the
25 thresholds missing from the previous cross-fit audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def membership_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def macro(y_nez, pred_nez) -> float:
    return float(f1_score(y_nez, pred_nez, labels=[0, 1], average="macro", zero_division=0))


def topk_curve(a: np.ndarray, y_nez: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """All K including 0 and C, with stable original-order tie breaking."""
    order = np.argsort(-a, kind="stable")
    ez = (y_nez[order] == 0).astype(np.int64)
    n = len(a)
    k = np.arange(n + 1)
    tp_ez = np.r_[0, np.cumsum(ez)]
    fp_ez = k - tp_ez
    fn_ez = int(ez.sum()) - tp_ez
    tp_nez = int(n - ez.sum()) - fp_ez
    den_ez = 2 * tp_ez + fp_ez + fn_ez
    den_nez = 2 * tp_nez + fp_ez + fn_ez
    f_ez = np.divide(2 * tp_ez, den_ez, out=np.zeros(n + 1, float), where=den_ez != 0)
    f_nez = np.divide(2 * tp_nez, den_nez, out=np.zeros(n + 1, float), where=den_nez != 0)
    curve = (f_ez + f_nez) / 2
    # A direct implementation check at endpoints and the curve maximum.
    for j in (0, n, int(np.argmax(curve))):
        pred = np.ones(n, np.int8)
        pred[order[:j]] = 0
        if abs(macro(y_nez, pred) - curve[j]) > 1e-12:
            raise RuntimeError("Optimal-K metric implementation mismatch")
    return order, curve, len(np.unique(a)) != n


def optimal_runs(k_values: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    left = right = int(k_values[0])
    for k in k_values[1:]:
        k = int(k)
        if k == right + 1:
            right = k
        else:
            runs.append((left, right))
            left = right = k
    runs.append((left, right))
    return runs


def threshold_oracle(a: np.ndarray, y_nez: np.ndarray) -> float:
    """Best unique-score decision; tied-score groups cannot be split."""
    order = np.argsort(-a, kind="stable")
    cut = [0]
    cut.extend(int(j) for j in range(1, len(a)) if a[order[j]] != a[order[j - 1]])
    cut.append(len(a))
    return max(macro(y_nez, np.where(np.isin(np.arange(len(a)), order[:k]), 0, 1)) for k in cut)


def group_arrays(ids, channels, a, y, evidence):
    result = []
    for patient in np.unique(ids.astype(str)):
        mask = ids.astype(str) == patient
        result.append({"patient_id": str(patient), "channels": channels[mask].astype(str),
                       "a": a[mask].astype(np.float64), "y": y[mask].astype(np.int8),
                       "evidence": evidence[mask].astype(np.float64)})
    return result


def replay_nested_threshold(cell: Path, saved: dict, *, table: pd.DataFrame,
                            columns: list[str], fold: int, inner: int,
                            source_root: Path, prior_lock_sha: str) -> float:
    output = cell / "NESTED_THRESHOLD_REPLAY_PRIVATE.json"
    if output.exists():
        record = json.loads(output.read_text(encoding="utf-8"))
        if (record["prior_lock_sha256"] != prior_lock_sha or record["selected_epoch"] != saved["nested_selected_epoch"]
                or abs(record["nested_macro_f1"] - saved["nested_validation_macro_f1"]) > 1e-9):
            raise RuntimeError("Saved nested threshold replay provenance mismatch")
        return float(record["selected_threshold_nez"])
    sys.path.insert(0, str(source_root.resolve()))
    from task1_baselines.patient_controls import _ChannelMLP, _Preprocessor, _fit_torch
    member = json.loads((cell / "MEMBERSHIP_PRIVATE.json").read_text(encoding="utf-8"))
    train_ids = set(member["nested_train_patient_ids"])
    val_ids = set(member["nested_validation_patient_ids"])
    held_ids = set(member["heldout_patient_ids"])
    if (train_ids & val_ids or train_ids & held_ids or val_ids & held_ids
            or membership_hash(held_ids) != saved["heldout_membership_sha256"]):
        raise RuntimeError("Nested patient membership leakage")
    train = table[table.subject_id.astype(str).isin(train_ids)].reset_index(drop=True)
    val = table[table.subject_id.astype(str).isin(val_ids)].reset_index(drop=True)
    if set(train.subject_id.astype(str)) != train_ids or set(val.subject_id.astype(str)) != val_ids:
        raise RuntimeError("Nested feature table patient coverage mismatch")
    pre = _Preprocessor(SimpleImputer(), StandardScaler(), True)
    tx = pre.fit_transform(train, columns)
    vx = pre.transform(val, columns)
    seed = 42 + 1009 * fold + 97 * inner
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, threshold, epoch, _ = _fit_torch(
        _ChannelMLP(88), tx, train.label_nez.to_numpy(int), train.subject_id.to_numpy(str),
        val, vx, device=torch.device("cpu"), seed=seed, max_epochs=30,
        patience=6, checkpoint_path=None, rank_scores=False)
    del model
    if epoch != saved["nested_selected_epoch"] or abs(threshold.patient_macro_f1 - saved["nested_validation_macro_f1"]) > 1e-9:
        raise RuntimeError(f"Nested B0 selection did not replay at fold={fold} inner={inner}")
    record = {"prior_lock_sha256": prior_lock_sha, "selected_epoch": int(epoch),
              "nested_macro_f1": float(threshold.patient_macro_f1),
              "selected_threshold_nez": float(threshold.threshold)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return float(threshold.threshold)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "feature-table", "feature-manifest", "split-manifest",
                "b0-ledger", "b0-embeddings", "crossfit-dir", "prior-lock", "prior-crossfit-code",
                "prior-crossfit-audit", "prior-b0-replay", "protocol-lock", "output-dir"):
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    expected = {"feature_table_sha256": args.feature_table, "feature_manifest_sha256": args.feature_manifest,
                "split_manifest_sha256": args.split_manifest, "b0_ledger_sha256": args.b0_ledger,
                "previous_crossfit_lock_sha256": args.prior_lock,
                "previous_crossfit_code_sha256": args.prior_crossfit_code}
    for name, path in expected.items():
        if sha(path) != lock[name]:
            raise RuntimeError(f"Source provenance mismatch: {name}")
    old = json.loads(args.prior_crossfit_audit.read_text(encoding="utf-8"))
    replay = json.loads(args.prior_b0_replay.read_text(encoding="utf-8"))
    if old["status"] != "PASS" or old["crossfit_cells"] != 25 or old["protocol_lock_sha256"] != lock["previous_crossfit_lock_sha256"]:
        raise RuntimeError("Prior 25-cell crossfit audit missing or mismatched")
    if replay["status"] != "PASS" or max(float(f["max_score_difference"]) for f in replay["folds"]) > lock["prior_b0_replay_max_abs_score_difference"] + 1e-12:
        raise RuntimeError("Prior exact B0 replay failed")
    table = pd.read_pickle(args.feature_table)
    sys.path.insert(0, str(args.source_root.resolve()))
    from task1_baselines.patient_controls import feature_columns
    columns = feature_columns(table)
    manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    if columns != manifest["feature_names"] or len(columns) != 88 or len(table) != 7635:
        raise RuntimeError("Feature table definition changed")
    split = pd.read_csv(args.split_manifest).rename(columns={"split_role": "partition"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public, all_fit, all_val = [], [], []
    replayed = 0
    for fold in range(1, 6):
        fold_dir = args.crossfit_dir / f"fold{fold}"
        audit = json.loads((fold_dir / "FOLD_AUDIT.json").read_text(encoding="utf-8"))
        oof_path = fold_dir / "FOLD_OOF_PRIVATE.npz"
        if audit["protocol_lock_sha256"] != lock["previous_crossfit_lock_sha256"] or sha(oof_path) != audit["combined_oof_sha256"]:
            raise RuntimeError(f"Fold {fold} OOF provenance mismatch")
        fit_ids = set(split.loc[(split.outer_fold == fold) & (split.partition == "fit"), "subject_id"].astype(str))
        val_ids = set(split.loc[(split.outer_fold == fold) & (split.partition == "validation"), "subject_id"].astype(str))
        if membership_hash(fit_ids) != audit["fit_membership_sha256"] or fit_ids & val_ids:
            raise RuntimeError("Frozen split changed")
        fit_table = table[table.subject_id.astype(str).isin(fit_ids)]
        thresholds = {}
        for cell in audit["crossfit_cells"]:
            inner = int(cell["inner_fold"])
            path = fold_dir / f"inner{inner}"
            if sha(path / "OOF_PRIVATE.npz") != cell["oof_sha256"]:
                raise RuntimeError("Crossfit cell prediction hash changed")
            t = replay_nested_threshold(path, cell, table=fit_table, columns=columns,
                                        fold=fold, inner=inner, source_root=args.source_root,
                                        prior_lock_sha=lock["previous_crossfit_lock_sha256"])
            member = json.loads((path / "MEMBERSHIP_PRIVATE.json").read_text(encoding="utf-8"))
            if not 0 < t < 1:
                raise RuntimeError("Invalid replayed threshold")
            thresholds.update({str(pid): t for pid in member["heldout_patient_ids"]})
            replayed += 1
        if set(thresholds) != fit_ids:
            raise RuntimeError("Missing inner B0 threshold for a fit patient")
        with np.load(oof_path, allow_pickle=False) as a:
            fit_groups = group_arrays(a["subject_id"], a["channel_name"], a["logit_ez"], a["label_nez"], a["valid_seizure_count"])
        with np.load(args.b0_embeddings / f"fold{fold}.npz", allow_pickle=True) as a:
            val_groups = group_arrays(a["validation_subject_id"], a["validation_channel_name"],
                                      a["validation_logit_ez"], a["validation_label_nez"], a["validation_seizure_count"])
            historical_t = float(a["b0_selected_threshold_nez"])
        if {g["patient_id"] for g in fit_groups} != fit_ids or {g["patient_id"] for g in val_groups} != val_ids:
            raise RuntimeError("B0 prediction patient alignment failed")
        # Validate all selected fit and validation labels and channels against the
        # private historical feature table without touching the test partition.
        for role, groups in (("fit", fit_groups), ("validation", val_groups)):
            selected = table[table.subject_id.astype(str).isin(fit_ids if role == "fit" else val_ids)]
            lookup = selected.set_index(["subject_id", "channel_name"])["label_nez"]
            if len(lookup) != sum(len(g["a"]) for g in groups) or not lookup.index.is_unique:
                raise RuntimeError("Historical channel alignment count mismatch")
            for g in groups:
                key = pd.MultiIndex.from_arrays([np.full(len(g["a"]), g["patient_id"]), g["channels"]])
                if not np.array_equal(lookup.loc[key].to_numpy(np.int8), g["y"]):
                    raise RuntimeError("Historical channel label mismatch")
        rows = []
        for role, groups in (("fit", fit_groups), ("validation", val_groups)):
            for g in groups:
                y, scores = g["y"], g["a"]
                order, curve, tied = topk_curve(scores, y)
                optimum = float(curve.max())
                ks = np.flatnonzero(curve >= optimum - 1e-12)
                runs = optimal_runs(ks)
                threshold_best = threshold_oracle(scores, y)
                if not tied and abs(optimum - threshold_best) > 1e-9:
                    raise RuntimeError("Untied patient optimal-K / threshold oracle disagree")
                t = thresholds[g["patient_id"]] if role == "fit" else historical_t
                formal_nez = (expit(-scores) >= t).astype(np.int8)
                k0 = int((formal_nez == 0).sum())
                top_nez = np.ones(len(y), np.int8)
                top_nez[order[:k0]] = 0
                if not np.array_equal(top_nez, formal_nez):
                    raise RuntimeError("Formal B0 cannot be reproduced by top-K (tie at threshold)")
                g.update({"fold": fold, "role": role, "formal_threshold_nez": t,
                          "k0": k0, "q0": k0 / len(y), "optimal_k": ks.astype(int),
                          "optimal_runs": runs, "oracle_macro": optimum,
                          "threshold_oracle_macro": threshold_best, "formal_macro": macro(y, formal_nez),
                          "has_ties": tied, "true_k": int((y == 0).sum())})
                rows.append(g)
                (all_fit if role == "fit" else all_val).append(g)
        fit_rows = [g for g in rows if g["role"] == "fit"]
        val_rows = [g for g in rows if g["role"] == "validation"]
        private = args.output_dir / f"fold{fold}_PATIENTS_PRIVATE.pkl"
        with private.open("wb") as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        public.append({"fold": fold, "fit_patients": len(fit_rows), "validation_patients": len(val_rows),
                       "fit_oracle_k_macro": float(np.mean([g["oracle_macro"] for g in fit_rows])),
                       "fit_threshold_oracle_macro": float(np.mean([g["threshold_oracle_macro"] for g in fit_rows])),
                       "fit_k_vs_threshold_discrepancy": float(np.mean([g["oracle_macro"] - g["threshold_oracle_macro"] for g in fit_rows])),
                       "validation_oracle_k_macro": float(np.mean([g["oracle_macro"] for g in val_rows])),
                       "validation_formal_b0_macro": float(np.mean([g["formal_macro"] for g in val_rows])),
                       "fit_tie_patients": sum(g["has_ties"] for g in fit_rows),
                       "validation_tie_patients": sum(g["has_ties"] for g in val_rows),
                       "fit_tie_oracle_affected_patients": sum(abs(g["oracle_macro"] - g["threshold_oracle_macro"]) > 1e-12 for g in fit_rows),
                       "validation_tie_oracle_affected_patients": sum(abs(g["oracle_macro"] - g["threshold_oracle_macro"]) > 1e-12 for g in val_rows),
                       "fit_multiple_optima": sum(len(g["optimal_k"]) > 1 for g in fit_rows),
                       "validation_multiple_optima": sum(len(g["optimal_k"]) > 1 for g in val_rows),
                       "fit_disconnected_optima": sum(len(g["optimal_runs"]) > 1 for g in fit_rows),
                       "validation_disconnected_optima": sum(len(g["optimal_runs"]) > 1 for g in val_rows),
                       "fit_max_threshold_discrepancy": float(max(abs(g["oracle_macro"] - g["threshold_oracle_macro"]) for g in fit_rows)),
                       "validation_max_threshold_discrepancy": float(max(abs(g["oracle_macro"] - g["threshold_oracle_macro"]) for g in val_rows)),
                       "private_patient_table_sha256": sha(private)})
        print(f"K_ORACLE_FOLD_PASS fold={fold} val_oracle={public[-1]['validation_oracle_k_macro']:.6f} val_b0={public[-1]['validation_formal_b0_macro']:.6f}", flush=True)
    validation_oracle = float(np.mean([g["oracle_macro"] for g in all_val]))
    if not 0.69 <= validation_oracle <= 0.72:
        raise RuntimeError(f"Optimal-K ceiling sanity FAILED: {validation_oracle}")
    result = {"status": "PASS", "new_lock_sha256": sha(args.protocol_lock),
              "prior_crossfit_lock_sha256": sha(args.prior_lock), "nested_thresholds_replayed": replayed,
              "fit_patient_fold_episodes": len(all_fit), "validation_patient_fold_episodes": len(all_val),
              "fit_optimal_k_macro": float(np.mean([g["oracle_macro"] for g in all_fit])),
              "fit_threshold_oracle_macro": float(np.mean([g["threshold_oracle_macro"] for g in all_fit])),
              "fit_k_vs_threshold_discrepancy": float(np.mean([g["oracle_macro"] - g["threshold_oracle_macro"] for g in all_fit])),
              "validation_optimal_k_macro": validation_oracle,
              "validation_formal_b0_macro": float(np.mean([g["formal_macro"] for g in all_val])),
              "validation_threshold_oracle_macro": float(np.mean([g["threshold_oracle_macro"] for g in all_val])),
              "validation_k_vs_threshold_discrepancy": float(np.mean([g["oracle_macro"] - g["threshold_oracle_macro"] for g in all_val])),
              "fit_tie_patients": sum(g["has_ties"] for g in all_fit),
              "validation_tie_patients": sum(g["has_ties"] for g in all_val),
              "fit_tie_oracle_affected_patients": sum(abs(g["oracle_macro"] - g["threshold_oracle_macro"]) > 1e-12 for g in all_fit),
              "validation_tie_oracle_affected_patients": sum(abs(g["oracle_macro"] - g["threshold_oracle_macro"]) > 1e-12 for g in all_val),
              "fit_multiple_optima": sum(len(g["optimal_k"]) > 1 for g in all_fit),
              "validation_multiple_optima": sum(len(g["optimal_k"]) > 1 for g in all_val),
              "fit_disconnected_optima": sum(len(g["optimal_runs"]) > 1 for g in all_fit),
              "validation_disconnected_optima": sum(len(g["optimal_runs"]) > 1 for g in all_val),
              "folds": public}
    (args.output_dir / "K_ORACLE_SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("K_ORACLE_PASS " + json.dumps({k: v for k, v in result.items() if k != "folds"}), flush=True)


if __name__ == "__main__":
    main()
