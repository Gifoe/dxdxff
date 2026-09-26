"""One-time outer A0/A1 evaluation, executable only after locked validation gate passes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from threshold_stage import descriptor, groups_from_archive, patient_metrics


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("b0-embeddings", "protocol-lock", "validation-dir", "output-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    gate = json.loads((a.validation_dir / "VALIDATION_GATE.json").read_text(encoding="utf-8"))
    if gate["status"] != "PASS" or gate["chosen_method"] != "A1" or gate["outer_test_arrays_read"]:
        raise RuntimeError("Outer-test access denied by validation gate")
    if gate["protocol_lock_sha256"] != digest(a.protocol_lock):
        raise RuntimeError("Protocol lock differs after validation gate")
    if (a.output_dir / "OUTER_RESULTS.csv").exists() or (a.output_dir / "OUTER_PATIENT_PRIVATE.csv").exists():
        raise RuntimeError("Outer evaluation already exists; one-time access only")
    public, private = [], []
    for fold in range(1, 6):
        with np.load(a.validation_dir / f"fold{fold}_MODEL_PRIVATE.npz", allow_pickle=False) as model:
            mean, scale, weight = model["scaler_mean"], model["scaler_scale"], model["weight"]
            bias, anchor = float(model["bias"]), float(model["anchor"])
        with np.load(a.b0_embeddings / f"fold{fold}.npz", allow_pickle=True) as archive:
            # Test arrays are first read only here, after all five validation folds passed.
            test = groups_from_archive(archive, "test")
            baseline_nez_threshold = float(archive["b0_selected_threshold_nez"])
        if len(test) != [16, 16, 17, 15, 16][fold - 1]:
            raise RuntimeError("Frozen outer test membership changed")
        rows = {"A0": [], "A1": []}
        for group in test:
            features = descriptor(group["a"], group["evidence"], anchor)
            tau = float(anchor + ((features - mean) / scale) @ weight + bias)
            base_nez_score = 1 / (1 + np.exp(np.clip(group["a"], -30, 30)))
            predictions = {"A0": (base_nez_score >= baseline_nez_threshold).astype(int),
                           "A1": (group["a"] < tau).astype(int)}
            for variant, pred in predictions.items():
                metrics = patient_metrics(group["y"], pred)
                rows[variant].append(metrics)
                private.append({"subject_id": group["patient_id"], "fold": fold, "variant": variant,
                                "threshold_ez_logit": anchor if variant == "A0" else tau, **metrics})
        for variant, metrics in rows.items():
            frame = pd.DataFrame(metrics)
            public.append({"outer_fold": fold, "variant": variant, "seed": 42, "test_patients": len(frame),
                           **{metric: float(frame[metric].mean()) for metric in ("macro_f1", "ez_f1", "balanced_accuracy")}})
        print(f"OUTER_FOLD_COMPLETE fold={fold} a0={public[-2]['macro_f1']:.6f} a1={public[-1]['macro_f1']:.6f}", flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    public_frame = pd.DataFrame(public)
    patient_frame = pd.DataFrame(private)
    reference = patient_frame[patient_frame.variant == "A0"]
    method = patient_frame[patient_frame.variant == "A1"]
    paired = method.merge(reference, on="subject_id", validate="one_to_one", suffixes=("_a1", "_a0"))
    if len(paired) != 80 or reference.subject_id.nunique() != 80 or method.subject_id.nunique() != 80:
        raise RuntimeError("Outer patient pairing incomplete")
    baseline_mean = float(reference.macro_f1.mean())
    if abs(baseline_mean - 0.6161672347405565) > 1e-6:
        raise RuntimeError("Historical B0 outer Macro-F1 did not replay")
    delta = (paired.macro_f1_a1 - paired.macro_f1_a0).to_numpy(float)
    rng = np.random.default_rng(20260926)
    resampled = rng.integers(0, 80, size=(2000, 80))
    draws = delta[resampled].mean(axis=1)
    public_frame.to_csv(a.output_dir / "OUTER_RESULTS.csv", index=False)
    patient_frame.to_csv(a.output_dir / "OUTER_PATIENT_PRIVATE.csv", index=False)
    mean_a1 = float(method.macro_f1.mean())
    report = f"""# Gated outer-test evaluation\n\nValidation gate passed before any A1 outer-test array was read. The locked linear score-only threshold predictor and historical B0 were evaluated once on the same 80 patients. No model, descriptor or hyperparameter was changed after validation.\n\n| Variant | Patient Macro-F1 | EZ-F1 | BA |\n|---|---:|---:|---:|\n| A0 historical B0 | {baseline_mean:.6f} | {reference.ez_f1.mean():.6f} | {reference.balanced_accuracy.mean():.6f} |\n| A1 direct threshold | {mean_a1:.6f} | {method.ez_f1.mean():.6f} | {method.balanced_accuracy.mean():.6f} |\n\nA1−A0 patient-equal Macro-F1 delta: `{delta.mean():+.6f}` (percentage points `{100*delta.mean():+.3f}`); paired patient bootstrap 95% interval `[{np.quantile(draws,0.025):+.6f}, {np.quantile(draws,0.975):+.6f}]`. Positive outer folds: `{sum(public_frame[public_frame.variant=='A1'].macro_f1.to_numpy() > public_frame[public_frame.variant=='A0'].macro_f1.to_numpy())}/5`. This is exploratory seed 42 because historical outer outcomes had been previously viewed.\n"""
    (a.output_dir / "OUTER_SUMMARY.md").write_text(report, encoding="utf-8")
    (a.output_dir / "OUTER_STATUS.json").write_text(json.dumps({"status": "COMPLETE", "patients": 80,
        "protocol_lock_sha256": digest(a.protocol_lock), "test_used_for_tuning": False}, indent=2), encoding="utf-8")
    print(f"OUTER_COMPLETE a0={baseline_mean:.6f} a1={mean_a1:.6f} delta={delta.mean():+.6f}", flush=True)


if __name__ == "__main__":
    main()
