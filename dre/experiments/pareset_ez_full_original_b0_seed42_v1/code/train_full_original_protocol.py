"""PaReSet-EZ full model on the historical CDEL data/evaluation protocol.

The architecture and objective remain the supplied PaReSet-EZ full method.
Historical P2 code constructs the B0 evidence and fit-only normalizer; the
historical CDEL evaluator chooses the validation threshold and patient metrics.
Existing simplified-control outputs are never read or overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


FOLDS = (1, 2, 3, 4, 5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--historical-cdel-overall", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    requested = tuple(int(value) for value in args.folds.split(","))
    if not requested or any(fold not in FOLDS for fold in requested) or len(set(requested)) != len(requested):
        parser.error("folds must be distinct members of 1,2,3,4,5")

    sys.path.insert(0, str(args.supplement_root.resolve()))
    from epilens.data import load_partition_manifest, load_records, select_records, validate_protocol

    sys.path.insert(0, str(args.production_root.resolve()))
    from neuroez_c.evidence_views import (
        BASE_SPECTRAL_FEATURE_NAMES,
        PRUNED_SPECTRAL_FEATURE_NAMES,
        b0_self_reference_features,
        fit_normalizer,
    )
    sys.path.insert(0, str(args.model_root.resolve()))
    sys.path.insert(0, str(args.model_root.parent.resolve()))
    from pareset_ez import ModelConfig, PaReSetEZ, patient_objective, parameter_count
    from build_patient_records import DESCRIPTORS

    if tuple(DESCRIPTORS) != tuple(PRUNED_SPECTRAL_FEATURE_NAMES):
        raise RuntimeError("Adapted descriptor order differs from historical B0 feature order")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source_view = args.production_root / "neuroez_c" / "evidence_views.py"
    source_report = args.production_root.parent / "neuroez_c" / "p2_v3_fusion_reporting.py"
    if not source_report.is_file():
        raise FileNotFoundError(source_report)
    report_spec = importlib.util.spec_from_file_location("historical_cdel_reporting", source_report)
    if report_spec is None or report_spec.loader is None:
        raise RuntimeError("Cannot load historical CDEL reporting module")
    historical_reporting = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(historical_reporting)
    aggregate_patients = historical_reporting.aggregate_patients
    evaluate_probability_predictions = historical_reporting.evaluate_probability_predictions
    select_validation_threshold = historical_reporting.select_validation_threshold
    config = ModelConfig(dropout=0.4)
    protocol = {
        "status": "FROZEN_FULL_HISTORICAL_B0_SEED42",
        "seed": args.seed,
        "folds": list(requested),
        "model": "PaReSetEZ_FULL_UNCHANGED",
        "model_config": asdict(config),
        "boundary_weight": 0.05,
        "maximum_epochs": 45,
        "minimum_epochs_before_stop": 18,
        "patience": 6,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "patient_batch_size": 4,
        "gradient_clip": 1.0,
        "feature_source": "historical P2 B0 nine descriptors x abs/delta/zdelta/ratio",
        "normalizer": "historical fit_normalizer on outer-fit only",
        "checkpoint_selection": "historical PRQ per-epoch 0.05-grid validation patient Macro-F1, strictly improving",
        "threshold_selection": "historical CDEL final validation-only 0.005 grid",
        "metric": "historical CDEL patient-equal Macro-F1",
        "runner_sha256": sha256(Path(__file__)),
        "data_sha256": sha256(args.data),
        "manifest_sha256": sha256(args.manifest),
        "historical_cdel_overall_sha256": sha256(args.historical_cdel_overall),
        "adapter_sha256": sha256(args.model_root.parent / "build_patient_records.py"),
        "model_sha256": sha256(args.model_root / "pareset_ez.py"),
        "historical_evidence_views_sha256": sha256(source_view),
        "historical_reporting_sha256": sha256(source_report.resolve()),
    }
    lock_path = args.output_root / "PROTOCOL_LOCK.json"
    if not args.preflight_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
        if lock_path.exists():
            if json.loads(lock_path.read_text(encoding="utf-8")) != protocol:
                raise RuntimeError("Existing protocol lock differs; refusing to overwrite")
        else:
            atomic_json(lock_path, protocol)

    records = load_records(args.data)
    manifest = load_partition_manifest(args.manifest)
    validate_protocol(records, manifest, 80)
    historical = pd.read_csv(args.historical_cdel_overall)
    historical_row = historical.loc[historical.experiment.eq("CDEL")]
    if len(historical_row) != 1 or int(historical_row.iloc[0].n_patients) != 80:
        raise RuntimeError("Historical CDEL summary is not a unique 80-patient result")
    historical_macro = float(historical_row.iloc[0].patient_macro_f1)
    if len(records) != 80:
        raise RuntimeError("Expected exactly 80 patients")
    view_args = SimpleNamespace(
        window_feature_names=list(BASE_SPECTRAL_FEATURE_NAMES),
        b0_feature_groups="spectral_classical",
        b0_feature_parts="abs,delta,zdelta,ratio",
        self_compare_eps=1e-5,
    )
    selected = [BASE_SPECTRAL_FEATURE_NAMES.index(name) for name in DESCRIPTORS]

    # Precompute subject-local historical B0 evidence, without fitting on val/test.
    source = {}
    fit_arrays = {}
    for record in records:
        if record.descriptors.shape[-1] != len(DESCRIPTORS):
            raise RuntimeError(f"Descriptor shape mismatch: {record.patient_id}")
        expanded = np.zeros((*record.descriptors.shape[:-1], 36), dtype=np.float32)
        arrays = []
        for seizure in range(record.descriptors.shape[0]):
            mask = np.asarray(record.valid[seizure], dtype=bool)
            active = np.flatnonzero(mask.any(axis=1))
            if not len(active) or not np.array_equal(active, np.arange(len(active))):
                raise RuntimeError(f"Non-prefix window mask: {record.patient_id}")
            length = len(active)
            local = mask[:length].any(axis=0)
            if not np.all(mask[:length, local]) or np.any(mask[:length, ~local]):
                raise RuntimeError(f"Nonuniform local-channel mask: {record.patient_id}")
            raw = np.zeros((length, mask.shape[1], len(BASE_SPECTRAL_FEATURE_NAMES)), dtype=np.float32)
            raw[..., selected] = record.descriptors[seizure, :length]
            evidence = b0_self_reference_features(raw, record.window_times[seizure, :length], view_args)
            if evidence.shape[-1] != 36 or not np.isfinite(evidence).all():
                raise RuntimeError(f"Historical B0 evidence invalid: {record.patient_id}")
            expanded[seizure, :length] = evidence
            arrays.append(evidence[:, local, :])
        source[record.patient_id] = expanded
        fit_arrays[record.patient_id] = arrays

    if args.preflight_only:
        partitions = {part: select_records(records, manifest, requested[0], part) for part in ("fit", "validation", "test")}
        normalizer = fit_normalizer(array for record in partitions["fit"] for array in fit_arrays[record.patient_id])
        sample = partitions["fit"][0]
        valid = np.asarray(sample.valid, dtype=bool)
        values = np.where(valid[..., None], normalizer.transform(source[sample.patient_id]), 0.0).astype(np.float32)
        model = PaReSetEZ(config).to(device)
        output_values = model(torch.from_numpy(values).to(device), torch.from_numpy(valid).to(device),
                              torch.from_numpy(np.asarray(sample.window_times, dtype=np.float32)).to(device))
        loss, _ = patient_objective(output_values["logit_ez"],
                                    torch.from_numpy(np.asarray(sample.label_nez, dtype=np.float32)).to(device),
                                    output_values["channel_valid"], 0.05)
        loss.backward()
        if not torch.isfinite(loss) or not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        ):
            raise FloatingPointError("Preflight produced nonfinite loss or gradient")
        print(json.dumps({"status": "PREFLIGHT_PASS", "seed": args.seed, "fold": requested[0],
                          "fit": len(partitions["fit"]), "validation": len(partitions["validation"]),
                          "test": len(partitions["test"]), "input_shape": list(values.shape),
                          "valid_channels": int(valid.any(axis=(0, 1)).sum()),
                          "feature_dim": int(normalizer.mean.shape[0]), "loss": float(loss.detach().cpu())}), flush=True)
        return

    fold_results = []
    for fold in requested:
        output = args.output_root / f"fold{fold}"
        complete = output / "COMPLETE.json"
        if complete.exists():
            saved = json.loads(complete.read_text(encoding="utf-8"))
            if saved.get("fold") != fold or saved.get("seed") != args.seed:
                raise RuntimeError(f"Invalid completed fold: {output}")
            fold_results.append(saved)
            print(f"SKIP_COMPLETE fold={fold}", flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Incomplete nonempty fold; preserve for diagnosis: {output}")
        output.mkdir(parents=True)
        partitions = {part: select_records(records, manifest, fold, part) for part in ("fit", "validation", "test")}
        if set(r.patient_id for group in partitions.values() for r in group) != set(source):
            raise RuntimeError("Fold does not cover all 80 subjects")
        if len(partitions["validation"]) != 13:
            raise RuntimeError("Historical fixed validation must have 13 patients")
        normalizer = fit_normalizer(array for record in partitions["fit"] for array in fit_arrays[record.patient_id])
        tensors = {}
        for record in records:
            valid = np.asarray(record.valid, dtype=bool)
            values = normalizer.transform(source[record.patient_id])
            values = np.where(valid[..., None], values, 0.0).astype(np.float32)
            tensors[record.patient_id] = (
                torch.from_numpy(values), torch.from_numpy(valid),
                torch.from_numpy(np.asarray(record.window_times, dtype=np.float32)),
                torch.from_numpy(np.asarray(record.label_nez, dtype=np.float32)),
            )

        def tensorize(record):
            return tuple(tensor.to(device) for tensor in tensors[record.patient_id])

        model = PaReSetEZ(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

        @torch.no_grad()
        def predict(group):
            model.eval()
            rows = []
            for record in group:
                x, valid, times, labels = tensorize(record)
                output_values = model(x, valid, times)
                for index in output_values["channel_valid"].nonzero(as_tuple=False).flatten().tolist():
                    rows.append({
                        "subject_id": record.patient_id,
                        "center": record.center,
                        "outer_fold": fold,
                        "channel_name": record.channel_names[index],
                        "label_nez": int(labels[index].item()),
                        "probability_nez": float(output_values["probability_nez"][index].item()),
                    })
            return pd.DataFrame(rows)

        def epoch_threshold(validation):
            # PRQ's training-time decision rule uses 0.05 ... 0.95 and
            # patient Macro-F1; only its final CDEL comparison uses 0.005.
            chosen = 0.5
            best = -float("inf")
            for candidate in np.linspace(0.05, 0.95, 19, dtype=np.float32):
                score = historical_reporting._threshold_patient_summary(
                    validation, score_nez_column="probability_nez", threshold=float(candidate)
                )[0]
                if score > best + 1e-12 or (
                    abs(score - best) <= 1e-12 and abs(float(candidate) - 0.5) < abs(chosen - 0.5)
                ):
                    chosen, best = float(candidate), float(score)
            return chosen, best

        best_score = -float("inf")
        best_epoch = 0
        best_state = None
        stale = 0
        history = []
        for epoch in range(1, 46):
            ordered = list(partitions["fit"])
            random.Random(args.seed + epoch).shuffle(ordered)
            model.train()
            losses = []
            for start in range(0, len(ordered), 4):
                batch = ordered[start:start + 4]
                optimizer.zero_grad(set_to_none=True)
                for record in batch:
                    x, valid, times, labels = tensorize(record)
                    output_values = model(x, valid, times)
                    loss, _ = patient_objective(output_values["logit_ez"], labels, output_values["channel_valid"], 0.05)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss: fold={fold} epoch={epoch}")
                    (loss / len(batch)).backward()
                    losses.append(float(loss.detach().item()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            validation = predict(partitions["validation"])
            threshold, score = epoch_threshold(validation)
            improved = score > best_score + 1e-6
            if improved:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
                torch.save({"model": best_state, "epoch": epoch, "threshold": threshold,
                            "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std,
                            "protocol": protocol}, output / "best.pt")
            else:
                stale += 1
            row = {"fold": fold, "epoch": epoch, "training_loss": float(np.mean(losses)),
                   "validation_patient_macro_f1": score, "validation_threshold": threshold,
                   "improved": improved, "stale": stale}
            history.append(row)
            pd.DataFrame(history).to_csv(output / "HISTORY.csv", index=False)
            print(json.dumps(row), flush=True)
            if epoch >= 18 and stale >= 6:
                break
        if best_state is None:
            raise RuntimeError("No selected checkpoint")
        model.load_state_dict(best_state)
        validation = predict(partitions["validation"])
        final_threshold, search = select_validation_threshold(validation, score_nez_column="probability_nez")
        search.to_csv(output / "VALIDATION_THRESHOLD_SEARCH.csv", index=False)
        test = predict(partitions["test"])
        patient = evaluate_probability_predictions(test, score_nez_column="probability_nez", threshold=final_threshold)
        patient.to_csv(output / "TEST_PATIENT_PRIVATE.csv", index=False)
        summary = aggregate_patients(patient).iloc[0].to_dict()
        result = {"fold": fold, "seed": args.seed, "fit_patients": len(partitions["fit"]),
                  "validation_patients": len(partitions["validation"]), "test_patients": len(partitions["test"]),
                  "parameters": parameter_count(model), "selected_epoch": best_epoch,
                  "selected_validation_macro_f1": best_score, "selected_threshold": final_threshold,
                  "test_patient_macro_f1": float(summary["patient_macro_f1"]),
                  "test_patient_ez_f1": float(summary["patient_macro_ez_f1"]),
                  "test_patient_nez_f1": float(summary["patient_macro_nez_f1"]),
                  "test_patient_balanced_accuracy": float(summary["patient_macro_balanced_accuracy"])}
        atomic_json(complete, result)
        fold_results.append(result)
        print("FOLD_COMPLETE " + json.dumps(result), flush=True)
    if set(requested) == set(FOLDS):
        rows = [json.loads((args.output_root / f"fold{fold}" / "COMPLETE.json").read_text(encoding="utf-8")) for fold in FOLDS]
        pd.DataFrame(rows).to_csv(args.output_root / "FULL_FOLD_RESULTS.csv", index=False)
        mean_macro = sum(row["test_patient_macro_f1"] * row["test_patients"] for row in rows) / 80
        atomic_json(args.output_root / "FULL_SUMMARY.json", {
            "seed": args.seed, "patients": 80, "folds": 5,
            "test_patient_macro_f1": mean_macro,
            "historical_cdel_seed42_macro_f1": historical_macro,
            "delta_vs_historical_cdel_seed42": mean_macro - historical_macro,
            "protocol_sha256": sha256(lock_path),
        })
        print(f"FULL_COMPLETE patient_macro_f1={mean_macro:.9f}", flush=True)


if __name__ == "__main__":
    main()
