"""One locked, seed-42, five-fold outer evaluation; no tuning on test."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


METHODS = ("full", "base", "prq", "bcr", "cdel")
METRICS = ("macro_f1", "ez_f1", "nez_f1", "balanced_accuracy", "ez_auprc", "ez_auroc", "accuracy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_lock(path: Path) -> dict:
    expected = path.with_suffix(".sha256").read_text(encoding="ascii").strip()
    if sha256(path) != expected:
        raise RuntimeError("Outer test lock changed after freezing")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN_BEFORE_OUTER_ACCESS" or lock.get("seed") != 42 or lock.get("folds") != [1, 2, 3, 4, 5]:
        raise RuntimeError("Outer test lock has unexpected protocol")
    if lock.get("outer_test_used_for_tuning") is not False:
        raise RuntimeError("Outer test tuning is not permitted")
    for name, value in lock["paths"].items():
        if sha256(Path(value)) != lock["sha256"][name]:
            raise RuntimeError(f"Source changed after locking: {name}")
    if sha256(Path(__file__)) != lock["sha256"]["outer_evaluator"]:
        raise RuntimeError("Evaluator changed after locking")
    for fold in range(1, 6):
        for name in ("full", "base", "prq", "bcr"):
            saved = lock["checkpoints"][f"fold{fold}"][name]
            if sha256(Path(saved["path"])) != saved["sha256"]:
                raise RuntimeError(f"Checkpoint changed after locking: fold={fold}, method={name}")
    return lock


def load_model(name: str, checkpoint_path: Path, fold: int, lock: dict, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if name in ("full", "base"):
        from pareset_ez import ModelConfig, PaReSetEZ
        expected = ModelConfig() if name == "full" else ModelConfig(use_temporal_update=False, use_adaptive_seizure_pool=False)
        if checkpoint["model_config"] != asdict(expected):
            raise RuntimeError(f"PaReSet config mismatch: fold={fold}, method={name}")
        provenance = checkpoint["provenance"]
        if provenance["seed"] != 42 or provenance["outer_fold"] != fold or provenance["variant"] != name:
            raise RuntimeError("PaReSet checkpoint provenance mismatch")
        if provenance["runner_arguments"]["evaluate_test"] is not False:
            raise RuntimeError("PaReSet checkpoint was previously test-evaluated")
        model = PaReSetEZ(expected)
        threshold = float(checkpoint["threshold"])
    else:
        import epilens.models as models
        provenance = checkpoint["provenance"]
        if provenance["seed"] != 42 or provenance["fold"] != fold or provenance["control"] != f"supplementary_{name}_finite_variance_repair":
            raise RuntimeError("Control checkpoint provenance mismatch")
        if provenance["test_evaluated"] is not False:
            raise RuntimeError("Control checkpoint was previously test-evaluated")
        model = models.PRQNet(dropout=0.4) if name == "prq" else models.BCRNet(dropout=0.4)
        threshold = float(checkpoint["selection"]["threshold"])
    if abs(threshold - float(lock["thresholds"][f"fold{fold}"][name])) > 1e-12:
        raise RuntimeError("Frozen threshold mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    from epilens.features import Standardizer
    standardizer = Standardizer(
        np.asarray(checkpoint["standardizer_mean"], dtype=np.float32),
        np.asarray(checkpoint["standardizer_scale"], dtype=np.float32),
    )
    return model, standardizer, threshold


@torch.no_grad()
def predict(model, standardizer, records, name: str, device: torch.device):
    from epilens.features import four_view_expansion
    rows = []
    timings = []
    for record in records:
        x, valid = four_view_expansion(record.descriptors, record.window_times, record.valid)
        x = standardizer.apply(x, valid)
        inputs = torch.as_tensor(x, device=device)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if name in ("full", "base"):
            times = torch.as_tensor(record.window_times, device=device)
            result = model(inputs, valid_tensor, times)
        else:
            result = model(inputs, valid_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        timings.append({"patient_id": record.patient_id, "method": name, "seconds": elapsed})
        probability = result["probability_nez"].detach().cpu().numpy()
        channel_valid = result["channel_valid"].detach().cpu().numpy().astype(bool)
        if channel_valid.shape != record.label_nez.shape or not np.all(np.isfinite(probability[channel_valid])):
            raise RuntimeError("Invalid model output shape/probability")
        for index in np.flatnonzero(channel_valid).tolist():
            rows.append({"patient_id": record.patient_id, "center": record.center,
                         "channel_name": record.channel_names[index],
                         "label_nez": int(record.label_nez[index]),
                         "probability_nez": float(probability[index])})
    return pd.DataFrame(rows), pd.DataFrame(timings)


def fuse_controls(prq: pd.DataFrame, bcr: pd.DataFrame) -> pd.DataFrame:
    keys = ["patient_id", "center", "channel_name", "label_nez"]
    merged = prq.merge(bcr, on=keys, how="outer", validate="one_to_one", suffixes=("_prq", "_bcr"), indicator=True)
    if len(merged) != len(prq) or len(merged) != len(bcr) or not (merged["_merge"] == "both").all():
        raise RuntimeError("Outer PRQ/BCR channel alignment mismatch")
    frame = merged[keys].copy()
    frame["probability_nez"] = 0.8 * merged["probability_nez_prq"] + 0.2 * merged["probability_nez_bcr"]
    return frame


def paired_bootstrap(patient: pd.DataFrame, primary: str, comparator: str, metric: str, seed: int = 4201):
    left = patient.loc[patient.method == primary, ["patient_id", metric]].rename(columns={metric: "primary"})
    right = patient.loc[patient.method == comparator, ["patient_id", metric]].rename(columns={metric: "control"})
    pair = left.merge(right, on="patient_id", how="inner", validate="one_to_one")
    if len(pair) != 80:
        raise RuntimeError("Paired comparison does not have 80 patients")
    delta = pair.primary.to_numpy(dtype=float) - pair.control.to_numpy(dtype=float)
    if np.isnan(delta).all():
        raise RuntimeError(f"All paired deltas undefined for {metric}")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(10000, len(delta)))
    estimates = np.nanmean(delta[draws], axis=1)
    return {"primary": primary, "comparator": comparator, "metric": metric,
            "mean_delta": float(np.nanmean(delta)),
            "ci95_low": float(np.nanquantile(estimates, 0.025)),
            "ci95_high": float(np.nanquantile(estimates, 0.975)),
            "positive_patients": int(np.sum(delta > 0)),
            "negative_patients": int(np.sum(delta < 0)),
            "zero_patients": int(np.sum(delta == 0)),
            "valid_patients": int(np.isfinite(delta).sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    lock = verify_lock(args.lock)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Outer output is nonempty; refusing to overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(lock["paths"]["pareset_model"]).parent))
    sys.path.insert(0, str(Path(lock["paths"]["supplement_models"]).parent.parent))
    sys.path.insert(0, str(Path(lock["paths"]["control_repair"]).parent))
    from epilens.data import load_records, load_partition_manifest, validate_protocol, select_records
    from epilens.evaluation import patient_metrics
    from train_repaired_control import repaired_mean_std
    import epilens.models as models
    models._masked_mean_std = repaired_mean_std

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    records = load_records(Path(lock["paths"]["data"]))
    manifest = load_partition_manifest(Path(lock["paths"]["manifest"]))
    validate_protocol(records, manifest, expected_patients=80)
    all_test = [select_records(records, manifest, fold, "test") for fold in range(1, 6)]
    test_ids = [record.patient_id for group in all_test for record in group]
    if len(test_ids) != 80 or len(set(test_ids)) != 80 or set(test_ids) != {record.patient_id for record in records}:
        raise RuntimeError("Five outer test folds are not a disjoint 80-patient partition")
    patient_frames = []
    fold_rows = []
    timing_frames = []
    for fold, test in enumerate(all_test, start=1):
        ledgers = {}
        timings = {}
        for name in ("full", "base", "prq", "bcr"):
            checkpoint_path = Path(lock["checkpoints"][f"fold{fold}"][name]["path"])
            model, standardizer, threshold = load_model(name, checkpoint_path, fold, lock, device)
            ledger, timing = predict(model, standardizer, test, name, device)
            ledgers[name] = ledger
            timings[name] = timing
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ledgers["cdel"] = fuse_controls(ledgers["prq"], ledgers["bcr"])
        combined_timing = timings["prq"].merge(timings["bcr"], on="patient_id", validate="one_to_one", suffixes=("_prq", "_bcr"))
        timings["cdel"] = pd.DataFrame({"patient_id": combined_timing.patient_id, "method": "cdel", "seconds": combined_timing.seconds_prq + combined_timing.seconds_bcr})
        for name in METHODS:
            threshold = float(lock["thresholds"][f"fold{fold}"][name])
            ledger = ledgers[name]
            ledger["fold"] = fold
            ledger["seed"] = 42
            ledger["method"] = name
            ledger["threshold"] = threshold
            ledger["predicted_nez"] = (ledger.probability_nez >= threshold).astype(int)
            ledger.to_csv(args.output / f"fold{fold}_{name}_channel_predictions_PRIVATE.csv", index=False)
            metrics = patient_metrics(ledger, threshold)
            metrics["fold"] = fold
            metrics["seed"] = 42
            metrics["method"] = name
            metrics["ez_fraction_signed_error"] = metrics.predicted_ez_fraction - metrics.true_ez_fraction
            metrics["ez_fraction_absolute_error"] = metrics.ez_fraction_signed_error.abs()
            patient_frames.append(metrics)
            row = {"fold": fold, "method": name, "patients": len(metrics), "threshold": threshold}
            row.update({metric: float(metrics[metric].mean()) for metric in METRICS})
            fold_rows.append(row)
            timing_frames.append(timings[name].assign(fold=fold, seed=42))
        print(f"OUTER_FOLD_COMPLETE={fold}", flush=True)
    patient = pd.concat(patient_frames, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    times = pd.concat(timing_frames, ignore_index=True)
    patient.to_csv(args.output / "OUTER_PATIENT_METRICS_PRIVATE.csv", index=False)
    folds.to_csv(args.output / "OUTER_FOLD_RESULTS.csv", index=False)
    times.to_csv(args.output / "OUTER_TIMINGS_PRIVATE.csv", index=False)
    summary_rows = []
    for name in METHODS:
        group = patient.loc[patient.method == name]
        if len(group) != 80 or group.patient_id.nunique() != 80:
            raise RuntimeError(f"Method does not have 80 disjoint test patients: {name}")
        timed = times.loc[times.method == name, "seconds"].to_numpy(dtype=float)
        row = {"method": name, "patients": 80, "seed": 42,
               "inference_seconds_median": float(np.median(timed)),
               "inference_seconds_p95": float(np.quantile(timed, 0.95)),
               "ez_fraction_bias": float(group.ez_fraction_signed_error.mean()),
               "ez_fraction_mae": float(group.ez_fraction_absolute_error.mean())}
        row.update({metric: float(group[metric].mean()) for metric in METRICS})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "OUTER_SUMMARY.csv", index=False)
    paired_rows = [paired_bootstrap(patient, "full", control, metric) for control in ("base", "bcr", "cdel") for metric in ("macro_f1", "ez_f1", "balanced_accuracy", "ez_auprc")]
    pd.DataFrame(paired_rows).to_csv(args.output / "OUTER_PAIRED_BOOTSTRAP.csv", index=False)
    center = patient.groupby(["method", "center"], as_index=False).agg(patients=("patient_id", "nunique"), macro_f1=("macro_f1", "mean"), ez_f1=("ez_f1", "mean"), balanced_accuracy=("balanced_accuracy", "mean"))
    center.to_csv(args.output / "OUTER_COHORT_SUMMARY.csv", index=False)
    result = {"status": "COMPLETE", "outer_test_accessed": True,
              "additional_final_heldout_accessed": False,
              "model_frozen_before_outer_test": True,
              "outer_test_used_for_tuning": False,
              "seed": 42, "patients": 80, "folds": 5,
              "lock_sha256": sha256(args.lock),
              "note": "New matched 36-D control reruns; not historical 28-D paper results."}
    (args.output / "OUTER_STATUS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(paired_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
