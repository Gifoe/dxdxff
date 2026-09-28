"""Recompute the frozen v1 FIT-OOF Teacher before any v2 Student outcome.

The public report contains counts and digests only. Patient IDs, channel labels,
scores, and the per-file manifest stay in the private server runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-runtime", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root / "fullpool_privileged_ranking_distillation_seed42_v1"
    sys.path.insert(0, str(source / "code"))
    import protocol as p
    import teacher as t

    p.preflight()
    if p.RUNTIME.resolve() != args.private_runtime.resolve():
        raise RuntimeError("V1 runtime mismatch")
    contexts = p.all_contexts()
    if len(contexts) != 17:
        raise RuntimeError("V1 context count changed")
    manifest = {}
    n = 0
    n_heads = 0
    max_teacher_error = 0.0
    fit_ids_seen = set()
    for ctx in contexts:
        fold = ctx["fold"]
        fit_ids = sorted(p.payload(fold, ctx["epoch"])["fit"])
        for sid in fit_ids:
            fit_ids_seen.add(sid)
            grid_path, oof_path = t.grid_path(ctx, sid), t.oof_path(ctx, sid)
            if not grid_path.is_file() or not oof_path.is_file():
                raise RuntimeError("V1 Teacher file missing")
            for path in (grid_path, oof_path):
                manifest[str(path.relative_to(args.private_runtime)).replace("\\", "/")] = digest(path)
            with oof_path.open("rb") as stream:
                saved = pickle.load(stream)
            if (saved["lock_sha"] != p.LOCK_SHA or saved["sid"] != sid or
                    saved["context_id"] != ctx["context_id"] or
                    not saved["no_scored_channel_label_in_teacher_fit"] or
                    not saved["no_scored_patient_label_in_lambda_selection"]):
                raise RuntimeError("V1 Teacher leakage/provenance flag mismatch")
            choice, _, other_count = t.leave_patient_out_lambda(ctx, sid, fit_ids)
            if choice != saved["lambda_index"] or other_count != len(fit_ids) - 1:
                raise RuntimeError("V1 leave-patient-out lambda selection mismatch")
            z, m0, y = p.patient_data(ctx, sid)
            if (len(y) != saved["n_channels"] or
                    not np.array_equal(y, saved["y"]) or
                    not np.allclose(m0, saved["a1_score"], rtol=0, atol=1e-12)):
                raise RuntimeError("V1 FIT labels/channel order/source score mismatch")
            perm = np.random.default_rng(
                p.afc.stable_seed(42, fold, ctx["context_id"], sid, "teacher_channel_folds")
            ).permutation(len(y))
            groups = np.array_split(perm, 5)
            if [len(group) for group in groups] != saved["heldout_fold_sizes"]:
                raise RuntimeError("V1 channel crossfit membership mismatch")
            lw, lb = p.GRID[choice]
            reproduced = np.full(len(y), np.nan, dtype=np.float64)
            for group in groups:
                support = np.setdiff1d(np.arange(len(y)), group, assume_unique=False)
                if set(support) & set(group):
                    raise RuntimeError("Held-out channel leaked into Teacher fit")
                b, w = p.tc.fit_head(m0[support], z[support], y[support], lw, lb)
                reproduced[group] = m0[group] + b + z[group] @ w
                n_heads += 1
            error = float(np.max(np.abs(reproduced - saved["teacher_score"])))
            max_teacher_error = max(max_teacher_error, error)
            if not np.isfinite(error) or error > 1e-9:
                raise RuntimeError("V1 Teacher target numerical replay failed")
            n += 1
        print(f"[REPLAY] fold={fold} context={ctx['context_id']} patients={len(fit_ids)}", flush=True)
    if n != 869 or n_heads != 4345 or len(fit_ids_seen) != 79 or len(manifest) != 1738:
        raise RuntimeError("V1 Teacher coverage mismatch")
    private = {"files": dict(sorted(manifest.items())), "n_contexts": 17,
               "n_patient_contexts": n, "n_crossfit_heads": n_heads}
    private_path = args.private_runtime / "v2_teacher_input_hash_manifest_PRIVATE.json"
    atomic_json(private_path, private)
    audit = json.loads((source / "TEACHER_CROSSFIT_AUDIT.json").read_text(encoding="utf-8"))
    all_row = next(row for row in read_csv(source / "FIT_OOF_TEACHER_METRICS.csv") if row["fold"] == "ALL")
    boot = {row["metric"]: row for row in read_csv(source / "FIT_OOF_TEACHER_BOOTSTRAP.csv")}
    folds = [row for row in read_csv(source / "FIT_OOF_TEACHER_METRICS.csv")
             if row["fold"] in {"1", "2", "3", "4", "5"} and row["n_target_cells"] == "13"]
    if (abs(float(all_row["delta_ap"]) - audit["mean_delta_ap"]) > 1e-10 or
            abs(float(boot["ap"]["delta_mean"]) - audit["mean_delta_ap"]) > 1e-10 or
            len(folds) != 5):
        raise RuntimeError("V1 Teacher aggregate inconsistency")
    positive_folds = sum(float(row["delta_ap"]) > 0 for row in folds)
    eligible = bool(float(all_row["delta_ap"]) >= .03 and float(boot["ap"]["ci_low"]) > 0
                    and positive_folds >= 4 and max_teacher_error <= 1e-9)
    common = {"v1_source_commit": "bea299edc88cedbeb6e674ab0e4a7217b78ea509",
              "v1_terminal_preserved": "PRIVILEGED_TEACHER_SIGNAL_GATE_FAILED",
              "v1_protocol_lock_sha256": digest(source / "PROTOCOL_LOCK.json"),
              "v1_teacher_audit_sha256": digest(source / "TEACHER_CROSSFIT_AUDIT.json"),
              "v1_teacher_bootstrap_sha256": digest(source / "FIT_OOF_TEACHER_BOOTSTRAP.csv"),
              "v1_teacher_grid_files": n, "v1_oof_target_files": n,
              "private_file_manifest_sha256": digest(private_path),
              "historical_per_file_hash_manifest_present": False,
              "new_manifest_frozen_before_any_v2_student_target_outcome": True}
    provenance = {**common, "fit_oof_a1_ap": float(all_row["a1_ap"]),
                  "fit_oof_teacher_ap": float(all_row["teacher_ap"]),
                  "teacher_delta_ap": float(all_row["delta_ap"]),
                  "teacher_delta_ap_ci_low": float(boot["ap"]["ci_low"]),
                  "teacher_delta_ap_ci_high": float(boot["ap"]["ci_high"]),
                  "positive_outer_folds": positive_folds,
                  "teacher_delta_mrr": float(all_row["delta_mrr"]),
                  "teacher_delta_top1": float(all_row["delta_top1"]),
                  "own_scored_channel_label_excluded_from_teacher_fit": True,
                  "TEACHER_AP_SIGNAL_ELIGIBLE_FOR_STUDENT": eligible}
    replay = {**common, "n_source_contexts": len(contexts),
              "n_fit_patient_contexts": n, "n_unique_fit_patient_ids": len(fit_ids_seen),
              "n_recomputed_channel_crossfit_heads": n_heads,
              "max_abs_teacher_score_replay_error": max_teacher_error,
              "numerical_target_replay_pass": max_teacher_error <= 1e-9,
              "hash_exact_to_historical_per_file_manifest": "NOT_VERIFIABLE_NO_V1_MANIFEST"}
    atomic_json(args.output / "V1_TEACHER_PROVENANCE.json", provenance)
    atomic_json(args.output / "TEACHER_HASH_REPLAY.json", replay)
    print(f"TEACHER_REPLAY_COMPLETE contexts=17 targets={n} heads={n_heads} "
          f"max_error={max_teacher_error:.3g} eligible={eligible}", flush=True)


if __name__ == "__main__":
    main()
