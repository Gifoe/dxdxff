"""Fit one locked score-only linear direct-threshold learner from patient-OOF B0 logits.

This stage reads OUTER-FIT OOF labels to construct interval targets, then reads
OUTER-VALIDATION labels solely for evaluation/gating. It never reads test arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def patient_metrics(y_nez: np.ndarray, pred_nez: np.ndarray) -> dict[str, float]:
    return {"macro_f1": float(f1_score(y_nez, pred_nez, labels=[0, 1], average="macro", zero_division=0)),
            "ez_f1": float(f1_score(y_nez == 0, pred_nez == 0, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_nez, pred_nez))}


def optimal_intervals(logit_ez: np.ndarray, y_nez: np.ndarray, safe_low: float,
                      safe_high: float) -> tuple[list[tuple[float, float]], float]:
    """Enumerate every distinct decision region, retaining disconnected optima."""
    unique = np.unique(logit_ez.astype(float))
    edges = np.concatenate(([safe_low], unique, [safe_high]))
    regions = []
    for low, high in zip(edges[:-1], edges[1:]):
        if high <= low:
            continue
        threshold = (float(low) + float(high)) / 2
        prediction = (logit_ez < threshold).astype(int)
        regions.append((float(low), float(high), patient_metrics(y_nez, prediction)["macro_f1"]))
    if not regions:
        raise RuntimeError("No valid patient threshold region")
    maximum = max(item[2] for item in regions)
    optimal = []
    for low, high, score in regions:
        if abs(score - maximum) <= 1e-12:
            if optimal and abs(optimal[-1][1] - low) <= 1e-12:
                optimal[-1] = (optimal[-1][0], high)
            else:
                optimal.append((low, high))
    return optimal, maximum


def distance_to_intervals(value: float, intervals: list[tuple[float, float]]) -> float:
    return min(max(low - value, value - high, 0.0) for low, high in intervals)


def nearest_midpoint(value: float, intervals: list[tuple[float, float]]) -> float:
    low, high = min(intervals, key=lambda item: max(item[0] - value, value - item[1], 0.0))
    return (low + high) / 2


DESCRIPTOR_NAMES = [
    "logit_mean", "logit_std", "logit_median", "logit_MAD", "logit_IQR",
    "q05", "q10", "q20", "q25", "q40", "q50", "q60", "q75", "q80", "q90", "q95",
    "entropy_mean", "entropy_std", "logit_skewness", "top_tail_gap", "lower_tail_gap",
    "near_anchor_0p1", "near_anchor_0p25", "near_anchor_0p5", "log1p_channels",
    "evidence_mean", "evidence_std", "evidence_min", "evidence_max"]


def descriptor(logit_ez: np.ndarray, evidence: np.ndarray, anchor: float) -> np.ndarray:
    a = logit_ez.astype(float)
    n = evidence.astype(float)
    quantile = np.quantile(a, [0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.60, 0.75, 0.80, 0.90, 0.95])
    p = 1 / (1 + np.exp(-np.clip(a, -30, 30)))
    entropy = -(p * np.log(np.clip(p, 1e-9, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-9, 1)))
    mean = a.mean()
    std = a.std()
    values = [mean, std, np.median(a), np.median(np.abs(a - np.median(a))), quantile[7] - quantile[3],
              *quantile, entropy.mean(), entropy.std(), np.mean((a - mean) ** 3) / max(std ** 3, 1e-8),
              quantile[10] - quantile[7], quantile[3] - quantile[0],
              *(np.mean(np.abs(a - anchor) <= margin) for margin in (0.1, 0.25, 0.5)),
              math.log1p(len(a)), n.mean(), n.std(), n.min(), n.max()]
    output = np.asarray(values, dtype=np.float64)
    if len(output) != len(DESCRIPTOR_NAMES) or not np.isfinite(output).all():
        raise RuntimeError("Invalid label-blind score descriptor")
    return output


def groups_from_archive(archive, prefix: str) -> list[dict]:
    ids = archive[f"{prefix}_subject_id"].astype(str)
    output = []
    for patient in np.unique(ids):
        mask = ids == patient
        output.append({"patient_id": patient, "a": archive[f"{prefix}_logit_ez"][mask].astype(float),
                       "y": archive[f"{prefix}_label_nez"][mask].astype(int),
                       "evidence": archive[f"{prefix}_seizure_count"][mask].astype(float)})
    return output


def oof_groups(archive) -> list[dict]:
    ids = archive["subject_id"].astype(str)
    output = []
    for patient in np.unique(ids):
        mask = ids == patient
        output.append({"patient_id": patient, "a": archive["logit_ez"][mask].astype(float),
                       "y": archive["label_nez"][mask].astype(int),
                       "evidence": archive["valid_seizure_count"][mask].astype(float)})
    return output


def fit_linear(x: np.ndarray, targets: list[list[tuple[float, float]]], anchor: float) -> tuple[StandardScaler, np.ndarray, float, dict]:
    scaler = StandardScaler().fit(x)
    scaled = torch.tensor(scaler.transform(x), dtype=torch.float32)
    torch.manual_seed(42)
    layer = torch.nn.Linear(x.shape[1], 1)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01, weight_decay=0.01)
    anchor_tensor = torch.tensor(float(anchor), dtype=torch.float32)
    initial_loss, final_loss = None, None
    for step in range(500):
        optimizer.zero_grad(set_to_none=True)
        predicted = anchor_tensor + layer(scaled).squeeze(1)
        pieces = []
        for value, intervals in zip(predicted, targets):
            candidates = []
            for low, high in intervals:
                boundary = torch.clamp(value, min=low, max=high).detach()
                candidates.append(torch.nn.functional.smooth_l1_loss(value, boundary, reduction="none"))
            pieces.append(torch.stack(candidates).min())
        interval_loss = torch.stack(pieces).mean()
        penalty = 0.01 * (predicted - anchor_tensor).square().mean()
        loss = interval_loss + penalty
        if step == 0:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    weight = layer.weight.detach().numpy().reshape(-1).astype(float)
    bias = float(layer.bias.detach().item())
    return scaler, weight, bias, {"initial_objective": initial_loss, "final_objective": final_loss,
                                  "weight_l2": float(np.linalg.norm(weight)), "bias": bias}


def predict_threshold(x: np.ndarray, scaler: StandardScaler, weight: np.ndarray, bias: float, anchor: float) -> np.ndarray:
    return float(anchor) + scaler.transform(x) @ weight + bias


def evaluate_groups(groups: list[dict], thresholds: np.ndarray, base_nez_threshold: float | None,
                    *, target_bounds: tuple[float, float]) -> tuple[dict, list[dict]]:
    patient_rows = []
    for group, tau in zip(groups, thresholds):
        if base_nez_threshold is None:
            pred_nez = (group["a"] < tau).astype(int)
        else:
            pred_nez = (1 / (1 + np.exp(np.clip(group["a"], -30, 30))) >= base_nez_threshold).astype(int)
        metrics = patient_metrics(group["y"], pred_nez)
        # For diagnostics only, expand a fixed-fit range if an unseen patient's
        # label-blind logit range exceeds it; training targets never use this.
        diagnostic_bounds = (min(target_bounds[0], float(group["a"].min() - 1)),
                             max(target_bounds[1], float(group["a"].max() + 1)))
        intervals, maximum = optimal_intervals(group["a"], group["y"], *diagnostic_bounds)
        patient_rows.append({**metrics, "patient_id": group["patient_id"], "threshold": float(tau),
                             "threshold_oracle_distance": distance_to_intervals(float(tau), intervals),
                             "threshold_oracle_midpoint": nearest_midpoint(float(tau), intervals),
                             "oracle_macro_f1": maximum})
    frame = pd.DataFrame(patient_rows)
    summary = {name: float(frame[name].mean()) for name in ("macro_f1", "ez_f1", "balanced_accuracy",
                                                        "threshold_oracle_distance", "oracle_macro_f1")}
    valid = frame[["threshold", "threshold_oracle_midpoint"]].dropna()
    summary["threshold_oracle_correlation"] = (float(valid.threshold.corr(valid.threshold_oracle_midpoint))
                                                if len(valid) >= 3 and valid.threshold.std() > 1e-12
                                                and valid.threshold_oracle_midpoint.std() > 1e-12 else float("nan"))
    return summary, patient_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("crossfit-dir", "b0-embeddings", "protocol-lock", "output-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    lock = json.loads(a.protocol_lock.read_text(encoding="utf-8"))
    if lock["direct_threshold"]["variants"] != ["A0", "A1"] or lock["seed"] != 42:
        raise RuntimeError("Direct-threshold scientific lock changed")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows, target_rows = [], []
    for fold in range(1, 6):
        fold_dir = a.crossfit_dir / f"fold{fold}"
        audit = json.loads((fold_dir / "FOLD_AUDIT.json").read_text(encoding="utf-8"))
        if audit["protocol_lock_sha256"] != digest(a.protocol_lock) or not audit["heldout_from_every_predictor"]:
            raise RuntimeError("Cross-fit provenance mismatch")
        with np.load(fold_dir / "FOLD_OOF_PRIVATE.npz", allow_pickle=False) as archive:
            fit = oof_groups(archive)
        with np.load(a.b0_embeddings / f"fold{fold}.npz", allow_pickle=True) as archive:
            validation = groups_from_archive(archive, "validation")
            b0_nez_threshold = float(archive["b0_selected_threshold_nez"])
        if len(fit) != audit["fit_patients"] or len(validation) != [13, 13, 13, 13, 13][fold - 1]:
            raise RuntimeError("Fold membership changed")
        anchor = math.log((1 - b0_nez_threshold) / b0_nez_threshold)
        all_logits = np.concatenate([group["a"] for group in fit])
        safe_bounds = (float(all_logits.min() - 1), float(all_logits.max() + 1))
        targets, x, target_meta = [], [], []
        for group in fit:
            intervals, best = optimal_intervals(group["a"], group["y"], *safe_bounds)
            targets.append(intervals)
            x.append(descriptor(group["a"], group["evidence"], anchor))
            target_meta.append({"interval_count": len(intervals), "total_optimal_width": sum(high - low for low, high in intervals),
                                "oracle_macro_f1": best, "anchor_to_optimal_interval": distance_to_intervals(anchor, intervals),
                                "oracle_delta_vs_anchor": best - patient_metrics(group["y"], (group["a"] < anchor).astype(int))["macro_f1"]})
        x = np.stack(x)
        scaler, weight, bias, training = fit_linear(x, targets, anchor)
        train_tau = predict_threshold(x, scaler, weight, bias, anchor)
        fit_summary, fit_private = evaluate_groups(fit, train_tau, None, target_bounds=safe_bounds)
        val_x = np.stack([descriptor(group["a"], group["evidence"], anchor) for group in validation])
        val_tau = predict_threshold(val_x, scaler, weight, bias, anchor)
        new_summary, new_private = evaluate_groups(validation, val_tau, None, target_bounds=safe_bounds)
        base_summary, base_private = evaluate_groups(validation, np.full(len(validation), anchor), b0_nez_threshold,
                                                    target_bounds=safe_bounds)
        if [row["patient_id"] for row in new_private] != [row["patient_id"] for row in base_private]:
            raise RuntimeError("Validation patient alignment failed")
        improved = sum(new["macro_f1"] > old["macro_f1"] + 1e-12 for new, old in zip(new_private, base_private))
        degraded = sum(new["macro_f1"] < old["macro_f1"] - 1e-12 for new, old in zip(new_private, base_private))
        for variant, summary in (("A0", base_summary), ("A1", new_summary)):
            validation_rows.append({"variant": variant, "outer_fold": fold, "validation_patients": len(validation),
                                    "improved_patients_vs_a0": 0 if variant == "A0" else improved,
                                    "degraded_patients_vs_a0": 0 if variant == "A0" else degraded,
                                    "fit_oof_macro_f1_a1_optimistic": fit_summary["macro_f1"],
                                    "fit_oof_threshold_distance_a1_optimistic": fit_summary["threshold_oracle_distance"],
                                    **summary})
        target_frame = pd.DataFrame(target_meta)
        target_rows.append({"outer_fold": fold, "fit_patients": len(fit), "fit_channels": len(all_logits),
                            "global_b0_ez_logit_threshold": anchor,
                            "safe_logit_low": safe_bounds[0], "safe_logit_high": safe_bounds[1],
                            "patients_with_disconnected_optimal_intervals": int((target_frame.interval_count > 1).sum()),
                            "mean_optimal_interval_count": float(target_frame.interval_count.mean()),
                            "mean_total_optimal_width": float(target_frame.total_optimal_width.mean()),
                            "mean_oracle_macro_f1": float(target_frame.oracle_macro_f1.mean()),
                            "mean_oracle_delta_vs_anchor": float(target_frame.oracle_delta_vs_anchor.mean()),
                            "mean_anchor_distance_to_optimum": float(target_frame.anchor_to_optimal_interval.mean()),
                            "mean_fit_oof_threshold_oracle_distance_a1": fit_summary["threshold_oracle_distance"],
                            "training_initial_objective": training["initial_objective"],
                            "training_final_objective": training["final_objective"],
                            "linear_weight_l2": training["weight_l2"]})
        private_model = a.output_dir / f"fold{fold}_MODEL_PRIVATE.npz"
        np.savez_compressed(private_model, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                            weight=weight, bias=np.array(bias), anchor=np.array(anchor),
                            descriptor_names=np.asarray(DESCRIPTOR_NAMES, dtype=str))
        private_val = a.output_dir / f"fold{fold}_VALIDATION_PATIENT_PRIVATE.csv"
        pd.DataFrame([{**{"variant": "A0"}, **row} for row in base_private] +
                     [{**{"variant": "A1"}, **row} for row in new_private]).to_csv(private_val, index=False)
        print(f"VALIDATION_FOLD_COMPLETE fold={fold} a0={base_summary['macro_f1']:.6f} a1={new_summary['macro_f1']:.6f} fit_oof={fit_summary['macro_f1']:.6f}", flush=True)
    result = pd.DataFrame(validation_rows)
    result.to_csv(a.output_dir / "VALIDATION_RESULTS.csv", index=False)
    pd.DataFrame(target_rows).to_csv(a.output_dir / "THRESHOLD_TARGET_AUDIT.csv", index=False)
    base = result[result.variant == "A0"].sort_values("outer_fold")
    method = result[result.variant == "A1"].sort_values("outer_fold")
    macro_delta = method.macro_f1.to_numpy() - base.macro_f1.to_numpy()
    ez_delta = method.ez_f1.to_numpy() - base.ez_f1.to_numpy()
    gate = {"status": "PASS" if (macro_delta.mean() >= 0.01 and (macro_delta > 1e-12).sum() >= 3
                                  and ez_delta.mean() >= -0.02) else "STOP",
            "chosen_method": "A1", "mean_validation_macro_f1_a0": float(base.macro_f1.mean()),
            "mean_validation_macro_f1_a1": float(method.macro_f1.mean()),
            "mean_validation_macro_delta": float(macro_delta.mean()),
            "positive_folds": int((macro_delta > 1e-12).sum()),
            "mean_validation_ez_f1_delta": float(ez_delta.mean()),
            "mean_fit_oof_macro_f1_a1_optimistic": float(method.fit_oof_macro_f1_a1_optimistic.mean()),
            "mean_validation_oracle_interval_distance_a1": float(method.threshold_oracle_distance.mean()),
            "protocol_lock_sha256": digest(a.protocol_lock), "outer_test_arrays_read": False}
    (a.output_dir / "VALIDATION_GATE.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    print("VALIDATION_GATE " + json.dumps(gate), flush=True)


if __name__ == "__main__":
    main()
