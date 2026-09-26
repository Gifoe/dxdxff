"""Evaluate frozen C0/C1/C2/C3 on outer-validation and enforce the locked gate.

The validation mode indexes no outer-test arrays. A separate gated test call is
required if—and only if—VALIDATION_GATE.json says PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def metrics(y, pred) -> dict:
    return {"macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "ez_f1": float(f1_score(y == 0, pred == 0, zero_division=0)),
            "nez_f1": float(f1_score(y == 1, pred == 1, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def interval_distance(q: float, row: dict) -> float:
    c = len(row["a"])
    return min(max(lo / c - q, q - hi / c, 0.0) for lo, hi in row["optimal_runs"])


def representative_k(row: dict) -> int:
    """Diagnostic only: optimal K closest to formal K0, tie toward smaller K."""
    return int(min(row["optimal_k"], key=lambda k: (abs(k - row["k0"]), k)))


def predict_patient(row: dict, delta: float, variant: str) -> dict:
    n = len(row["a"])
    q = float(np.clip(row["q0"] + delta, 0, 1))
    if variant == "C0":
        k = int(row["k0"])
    else:
        k = int(np.clip(np.rint(q * n), 0, n))
    order = np.argsort(-row["a"], kind="stable")
    pred = np.ones(n, np.int8)
    pred[order[:k]] = 0
    # The logit shift below is only for single-score compatibility; patient
    # classification follows the explicitly locked stable Top-K rule.
    if k == 0:
        tau = float(np.max(row["a"]) + 1)
    elif k == n:
        tau = float(np.min(row["a"]) - 1)
    else:
        tau = float((row["a"][order[k - 1]] + row["a"][order[k]]) / 2)
    threshold_mismatch = int(np.sum(((row["a"] - tau) > 0) != (pred == 0)))
    return {"pred": pred, "k": k, "q": q, "tau": tau,
            "threshold_mismatch_channels": threshold_mismatch}


def kstar_diagnostic(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame([{"role": r["role"], "fold": r["fold"], "channels": len(r["a"]),
                           "qstar": representative_k(r) / len(r["a"]),
                           "true_q": r["true_k"] / len(r["a"]),
                           "ranking_auprc": (float(average_precision_score(r["y"] == 0, r["a"]))
                                             if 0 < r["true_k"] < len(r["a"]) else float("nan"))}
                          for r in rows])
    output = []
    for role, role_df in frame.groupby("role"):
        slices = [("overall", "all", role_df)]
        for stratifier in ("channels", "ranking_auprc", "true_q"):
            sorted_index = role_df[stratifier].rank(method="first", na_option="bottom")
            tertile = pd.qcut(sorted_index, 3, labels=["low", "middle", "high"])
            for label in ("low", "middle", "high"):
                slices.append((stratifier, label, role_df[tertile == label]))
        for stratifier, label, part in slices:
            difference = part.qstar.to_numpy() - part.true_q.to_numpy()
            output.append({"role": role, "stratifier": stratifier, "tertile": label,
                           "patient_fold_episodes": len(part),
                           "qstar_trueq_correlation": corr(part.qstar, part.true_q),
                           "mean_qstar_minus_trueq": float(np.mean(difference)),
                           "median_qstar_minus_trueq": float(np.median(difference)),
                           "q10_qstar_minus_trueq": float(np.quantile(difference, 0.1)),
                           "q90_qstar_minus_trueq": float(np.quantile(difference, 0.9)),
                           "fraction_qstar_above_trueq": float(np.mean(difference > 1e-12)),
                           "fraction_qstar_below_trueq": float(np.mean(difference < -1e-12)),
                           "fraction_within_one_channel": float(np.mean(np.abs(difference) <= 1 / part.channels.to_numpy()))})
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("context-dir", "training-dir", "target-dir", "protocol-lock", "prior-validation-results", "output-dir"):
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    target = json.loads((args.target_dir / "K_ORACLE_SUMMARY.json").read_text(encoding="utf-8"))
    context = json.loads((args.context_dir / "PATIENT_CONTEXT_AUDIT.json").read_text(encoding="utf-8"))
    training = json.loads((args.training_dir / "TRAINING_AUDIT_PRIVATE.json").read_text(encoding="utf-8"))
    lock_hash = sha(args.protocol_lock)
    if any(x["status"] != "PASS" or x["protocol_lock_sha256"] != lock_hash for x in (context, training)) or target["new_lock_sha256"] != lock_hash:
        raise RuntimeError("Prerequisite scientific audits failed")
    old = pd.read_csv(args.prior_validation_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, diagnostic_rows, all_rows = [], [], []
    for fold in range(1, 6):
        cpath = args.context_dir / f"fold{fold}_CONTEXT_PRIVATE.pkl"
        ppath = args.training_dir / f"fold{fold}_PREDICTIONS_PRIVATE.pkl"
        if sha(cpath) != context["folds"][fold - 1]["private_context_sha256"] or sha(ppath) != training["folds"][fold - 1]["private_prediction_sha256"]:
            raise RuntimeError("Private validation input hashes changed")
        with cpath.open("rb") as f:
            rows = pickle.load(f)
        with ppath.open("rb") as f:
            preds = pickle.load(f)
        key_preds = {(p["role"], p["patient_id"]): p for p in preds}
        if len(key_preds) != len(rows):
            raise RuntimeError("Patient prediction alignment count mismatch")
        all_rows.extend(rows)
        val = [r for r in rows if r["role"] == "validation"]
        if len(val) != 13:
            raise RuntimeError("Outer-validation count changed")
        private = []
        for r in val:
            p_row = key_preds[(r["role"], r["patient_id"])]
            baseline = predict_patient(r, 0, "C0")
            base_macro = metrics(r["y"], baseline["pred"])["macro_f1"]
            rep = representative_k(r)
            for variant in lock["variants"]:
                delta = float(p_row[variant])
                result = predict_patient(r, delta, variant)
                met = metrics(r["y"], result["pred"])
                private.append({"patient_id": r["patient_id"], "fold": fold, "variant": variant,
                                "channels": len(r["a"]), "predicted_k": result["k"],
                                "predicted_q": result["q"], "delta": delta,
                                "oracle_delta": (rep - r["k0"]) / len(r["a"]),
                                "true_q": r["true_k"] / len(r["a"]),
                                "oracle_interval_distance": interval_distance(result["q"], r),
                                "oracle_macro": r["oracle_macro"],
                                "base_macro": base_macro,
                                "threshold_mismatch_channels": result["threshold_mismatch_channels"], **met})
        frame = pd.DataFrame(private)
        frame.to_csv(args.output_dir / f"fold{fold}_VALIDATION_PATIENT_PRIVATE.csv", index=False)
        base = frame[frame.variant == "C0"].set_index("patient_id")
        historical = old[(old.variant == "A0") & (old.outer_fold == fold)].iloc[0]
        if abs(base.macro_f1.mean() - historical.macro_f1) > 1e-9:
            raise RuntimeError(f"Frozen formal B0 does not match prior validation fold {fold}")
        for variant in lock["variants"]:
            part = frame[frame.variant == variant].set_index("patient_id")
            diff = part.macro_f1 - base.macro_f1
            fold_oracle_gap = float(base.oracle_macro.mean() - base.macro_f1.mean())
            summary.append({"fold": fold, "variant": variant, "validation_patients": len(part),
                            "macro_f1": float(part.macro_f1.mean()), "ez_f1": float(part.ez_f1.mean()),
                            "nez_f1": float(part.nez_f1.mean()),
                            "balanced_accuracy": float(part.balanced_accuracy.mean()),
                            "mean_predicted_ez_fraction": float((part.predicted_k / part.channels).mean()),
                            "mean_continuous_predicted_q": float(part.predicted_q.mean()),
                            "mean_predicted_k": float(part.predicted_k.mean()),
                            "mean_oracle_interval_distance": float(part.oracle_interval_distance.mean()),
                            "mean_true_fraction_mae_diagnostic_only": float(np.mean(abs(part.predicted_q - part.true_q))),
                            "mean_oracle_macro": float(part.oracle_macro.mean()),
                            "improved_patients_vs_c0": int((diff > 1e-12).sum()),
                            "degraded_patients_vs_c0": int((diff < -1e-12).sum()),
                            "unchanged_patients_vs_c0": int((abs(diff) <= 1e-12).sum()),
                            "fraction_improved_patients_vs_c0": float(np.mean(diff > 1e-12)),
                            "fraction_degraded_patients_vs_c0": float(np.mean(diff < -1e-12)),
                            "fraction_unchanged_patients_vs_c0": float(np.mean(abs(diff) <= 1e-12)),
                            "fold_oracle_headroom_recovered_fraction": (float(diff.mean() / fold_oracle_gap)
                                                                         if fold_oracle_gap > 0 else float("nan")),
                            "delta_oracle_delta_correlation": corr(part.delta, part.oracle_delta),
                            "rank_cut_score_mismatch_channels": int(part.threshold_mismatch_channels.sum())})
        print(f"VALIDATION_FOLD_PASS fold={fold} c0={summary[-4]['macro_f1']:.6f} c3={summary[-1]['macro_f1']:.6f}", flush=True)
    frame = pd.DataFrame(summary)
    frame.to_csv(args.output_dir / "VALIDATION_RESULTS.csv", index=False)
    pd.DataFrame(kstar_diagnostic(all_rows)).to_csv(args.output_dir / "KSTAR_VS_TRUEK.csv", index=False)
    base = frame[frame.variant == "C0"].sort_values("fold")
    method = frame[frame.variant == "C3"].sort_values("fold")
    gain = method.macro_f1.to_numpy() - base.macro_f1.to_numpy()
    ez_gain = method.ez_f1.to_numpy() - base.ez_f1.to_numpy()
    oracle_gap = float(base.mean_oracle_macro.mean() - base.macro_f1.mean())
    recovered = float(gain.mean() / oracle_gap) if oracle_gap > 0 else float("nan")
    corrs = method.delta_oracle_delta_correlation.to_numpy(float)
    mean_corr = float(np.mean(corrs)) if np.isfinite(corrs).all() else float("nan")
    checks = {"macro_gain": bool(gain.mean() >= lock["gate"]["mean_validation_macro_gain_min"]),
              "positive_folds": bool((gain > 1e-12).sum() >= lock["gate"]["positive_folds_min"]),
              "ez_f1": bool(ez_gain.mean() >= lock["gate"]["mean_ez_f1_delta_min"]),
              "oracle_delta_correlation": bool(np.isfinite(mean_corr) and mean_corr > 0),
              "headroom_recovery": bool(np.isfinite(recovered) and recovered >= lock["gate"]["validation_oracle_headroom_recovered_min"])}
    gate = {"status": "PASS" if all(checks.values()) else "STOP", "primary": "C3",
            "checks": checks, "mean_validation_macro_f1_c0": float(base.macro_f1.mean()),
            "mean_validation_macro_f1_c3": float(method.macro_f1.mean()),
            "mean_validation_macro_gain": float(gain.mean()),
            "positive_folds": int((gain > 1e-12).sum()),
            "mean_validation_ez_f1_delta": float(ez_gain.mean()),
            "mean_validation_oracle_k_macro": float(base.mean_oracle_macro.mean()),
            "validation_oracle_headroom": oracle_gap,
            "headroom_fraction_recovered": recovered,
            "mean_fold_correlation_predicted_vs_oracle_delta": mean_corr,
            "outer_test_arrays_read": False, "protocol_lock_sha256": lock_hash}
    (args.output_dir / "VALIDATION_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print("VALIDATION_GATE " + json.dumps(gate), flush=True)


if __name__ == "__main__":
    main()
