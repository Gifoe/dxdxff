"""Exploratory, no-inner, patient-disjoint five-fold PaReSet-EZ evaluation.

The historical 80-patient outer results have already been viewed. This rerun is
NOT an independent final test. Every fold trains on its frozen fit+validation
patients, evaluates its frozen test patients once, and uses epoch 45 and a
threshold of 0.5 without any outcome-based selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch


SEED = 42
FOLDS = (1, 2, 3, 4, 5)
METHODS = ("full", "base", "bcr")
EPOCHS = 45
THRESHOLD = 0.5
METRICS = ("macro_f1", "ez_f1", "balanced_accuracy", "ez_auprc", "accuracy")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def dependencies(args):
    sys.path.insert(0, str(args.supplement.resolve()))
    sys.path.insert(0, str(args.model.parent.resolve()))
    sys.path.insert(0, str(args.repair.parent.resolve()))
    from epilens.data import load_records, load_partition_manifest, validate_protocol, select_records
    from epilens.features import fit_standardizer, four_view_expansion
    from epilens.evaluation import patient_metrics
    from pareset_ez import ModelConfig, PaReSetEZ, patient_objective
    from train_repaired_control import repaired_mean_std
    import epilens.models as models
    from epilens.objectives import bcr_loss
    models._masked_mean_std = repaired_mean_std
    return (load_records, load_partition_manifest, validate_protocol, select_records,
            fit_standardizer, four_view_expansion, patient_metrics,
            ModelConfig, PaReSetEZ, patient_objective, models, bcr_loss)


def partition(args, dep):
    load_records, load_manifest, validate, select = dep[:4]
    records = load_records(args.data)
    manifest = load_manifest(args.manifest)
    validate(records, manifest, expected_patients=80)
    all_ids = {r.patient_id for r in records}
    if len(records) != 80 or len(all_ids) != 80:
        raise RuntimeError("Expected exactly 80 distinct patients")
    groups = {}
    test_ids = []
    for fold in FOLDS:
        fit = select(records, manifest, fold, "fit")
        val = select(records, manifest, fold, "validation")
        test = select(records, manifest, fold, "test")
        train = fit + val
        train_ids = [r.patient_id for r in train]
        fold_test_ids = [r.patient_id for r in test]
        if not train or not test or len(set(train_ids + fold_test_ids)) != 80:
            raise RuntimeError(f"Fold {fold} does not partition all patients")
        if set(train_ids) & set(fold_test_ids):
            raise RuntimeError(f"Fold {fold} has train/test overlap")
        groups[fold] = (train, test)
        test_ids.extend(fold_test_ids)
    if len(test_ids) != 80 or set(test_ids) != all_ids:
        raise RuntimeError("Five test folds are not a disjoint patient partition")
    return groups


def expected_lock(args, groups):
    sources = {
        "runner": Path(__file__), "model": args.model, "repair": args.repair,
        "data": args.data, "manifest": args.manifest,
        "supplement_features": args.supplement / "epilens" / "features.py",
        "supplement_models": args.supplement / "epilens" / "models.py",
        "supplement_objectives": args.supplement / "epilens" / "objectives.py",
        "supplement_evaluation": args.supplement / "epilens" / "evaluation.py",
    }
    return {
        "status": "FROZEN_BEFORE_FLAT_FIVEFOLD",
        "interpretation": "exploratory re-evaluation; same 80 historical test patients previously viewed",
        "seed": SEED, "folds": list(FOLDS), "methods": list(METHODS),
        "epochs": EPOCHS, "checkpoint": "final epoch, no validation or early stopping",
        "threshold": THRESHOLD, "threshold_selection": "none",
        "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-3,
        "dropout": 0.4, "patient_batch_size": 4, "gradient_clip": 1.0,
        "train_partition": "frozen fit union frozen validation",
        "test_partition": "frozen test; disjoint across five folds",
        "test_used_for_tuning": False,
        "source_sha256": {name: digest(path) for name, path in sources.items()},
        "paths": {name: str(path.resolve()) for name, path in sources.items()},
        "fold_counts": {str(fold): {"train": len(groups[fold][0]), "test": len(groups[fold][1])} for fold in FOLDS},
    }


def verify_or_create_lock(args, groups, create):
    path = args.output / "FLAT5_PROTOCOL_LOCK.json"
    marker = args.output / "FLAT5_PROTOCOL_LOCK.sha256"
    expected = expected_lock(args, groups)
    if create:
        if path.exists() or marker.exists():
            raise RuntimeError("Lock already exists; refusing replacement")
        args.output.mkdir(parents=True, exist_ok=True)
        json_write(path, expected)
        marker.write_text(digest(path) + "\n", encoding="ascii")
        print(f"FROZEN_LOCK_SHA256={digest(path)}", flush=True)
        return expected
    if not path.exists() or not marker.exists():
        raise RuntimeError("Prepare and freeze protocol before run")
    if digest(path) != marker.read_text(encoding="ascii").strip():
        raise RuntimeError("Protocol lock digest mismatch")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("Data/code/protocol changed after freeze")
    return actual


def build_model(method, dep, device):
    ModelConfig, PaReSetEZ, models = dep[7], dep[8], dep[10]
    if method in ("full", "base"):
        config = ModelConfig(dropout=0.4)
        if method == "base":
            config = replace(config, use_temporal_update=False, use_adaptive_seizure_pool=False)
        model = PaReSetEZ(config)
    else:
        model = models.BCRNet(dropout=0.4)
    return model.to(device)


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def fit_one(args, dep, fold, method, train, test, device):
    cell = args.output / f"fold{fold}" / method
    done = cell / "DONE.json"
    if done.exists():
        state = json.loads(done.read_text(encoding="utf-8"))
        if state.get("fold") != fold or state.get("method") != method or state.get("epoch") != EPOCHS:
            raise RuntimeError(f"Invalid completion marker: {done}")
        print(f"SKIP_COMPLETE fold={fold} method={method}", flush=True)
        return
    cell.mkdir(parents=True, exist_ok=True)
    standardizer = dep[4](train)
    four_view = dep[5]
    seed_all()
    model = build_model(method, dep, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    # Fit normalization exclusively on this fold's four training partitions.
    # Keep test tensors unavailable to the training loop and optimizer.
    tensors = {}
    for record in train:
        x, valid = four_view(record.descriptors, record.window_times, record.valid)
        x = standardizer.apply(x, valid)
        tensors[record.patient_id] = (torch.as_tensor(x, device=device),
                                      torch.as_tensor(valid, dtype=torch.bool, device=device),
                                      torch.as_tensor(record.window_times, device=device),
                                      torch.as_tensor(record.label_nez, device=device))
    resume_path = cell / "resume_PRIVATE.pt"
    start_epoch = 1
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["fold"] != fold or saved["method"] != method:
            raise RuntimeError("Resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        start_epoch = int(saved["epoch"]) + 1
        print(f"RESUME fold={fold} method={method} epoch={start_epoch}", flush=True)
    history_path = cell / "training_history_PRIVATE.csv"
    for epoch in range(start_epoch, EPOCHS + 1):
        order = list(train)
        random.Random(SEED + epoch).shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), 4):
            batch = order[start:start + 4]
            optimizer.zero_grad(set_to_none=True)
            for record in batch:
                x, valid, times, labels = tensors[record.patient_id]
                if method in ("full", "base"):
                    output = model(x, valid, times)
                    loss, _ = dep[9](output["logit_ez"], labels, output["channel_valid"], .05)
                else:
                    output = model(x, valid)
                    loss = dep[11](output["logit_ez"], labels, output["channel_valid"], .05, .08)[0]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite training loss")
                (loss / len(batch)).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        row = {"fold": fold, "method": method, "epoch": epoch, "train_loss": float(np.mean(losses))}
        pd.DataFrame([row]).to_csv(history_path, index=False, mode="a", header=not history_path.exists())
        saved = {"fold": fold, "method": method, "epoch": epoch,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state().cpu() if device.type == "cuda" else None}
        temporary = cell / "resume_PRIVATE.tmp"
        torch.save(saved, temporary)
        temporary.replace(resume_path)
        print(f"TRAIN fold={fold} method={method} epoch={epoch}/{EPOCHS} loss={row['train_loss']:.5f}", flush=True)
    # Only here, after the fixed final epoch, is this fold's test set predicted.
    model.eval()
    rows = []
    with torch.no_grad():
        for record in test:
            x, valid = four_view(record.descriptors, record.window_times, record.valid)
            x = standardizer.apply(x, valid)
            xt = torch.as_tensor(x, device=device)
            vt = torch.as_tensor(valid, dtype=torch.bool, device=device)
            if method in ("full", "base"):
                result = model(xt, vt, torch.as_tensor(record.window_times, device=device))
            else:
                result = model(xt, vt)
            probability = result["probability_nez"].detach().cpu().numpy()
            mask = result["channel_valid"].detach().cpu().numpy().astype(bool)
            for index in np.flatnonzero(mask):
                rows.append({"patient_id": record.patient_id, "center": record.center,
                             "channel_name": record.channel_names[index],
                             "label_nez": int(record.label_nez[index]),
                             "probability_nez": float(probability[index])})
    ledger = pd.DataFrame(rows)
    ledger["predicted_nez"] = (ledger.probability_nez >= THRESHOLD).astype(int)
    ledger["threshold"] = THRESHOLD
    ledger["fold"] = fold
    ledger["method"] = method
    ledger.to_csv(cell / "test_channel_predictions_PRIVATE.csv", index=False)
    patient = dep[6](ledger, THRESHOLD)
    patient["fold"] = fold
    patient["method"] = method
    patient.to_csv(cell / "test_patient_metrics_PRIVATE.csv", index=False)
    json_write(done, {"fold": fold, "method": method, "epoch": EPOCHS,
                      "threshold": THRESHOLD, "n_train": len(train), "n_test": len(test),
                      "patient_macro_f1": float(patient.macro_f1.mean())})
    print(f"DONE fold={fold} method={method} macro_f1={patient.macro_f1.mean():.6f}", flush=True)


def aggregate(args):
    frames = []
    for fold in FOLDS:
        for method in METHODS:
            cell = args.output / f"fold{fold}" / method
            if not (cell / "DONE.json").exists():
                raise RuntimeError(f"Incomplete cell: {cell}")
            frames.append(pd.read_csv(cell / "test_patient_metrics_PRIVATE.csv"))
    patient = pd.concat(frames, ignore_index=True)
    fold_rows = []
    summary_rows = []
    for method in METHODS:
        group = patient.loc[patient.method == method]
        if len(group) != 80 or group.patient_id.nunique() != 80:
            raise RuntimeError(f"Expected 80 disjoint test patients for {method}")
        summary_rows.append({"method": method, "patients": 80, "seed": SEED,
                             **{metric: float(group[metric].mean()) for metric in METRICS}})
        for fold in FOLDS:
            selected = group.loc[group.fold == fold]
            fold_rows.append({"fold": fold, "method": method, "patients": len(selected),
                              **{metric: float(selected[metric].mean()) for metric in METRICS}})
    summary = pd.DataFrame(summary_rows)
    fold_table = pd.DataFrame(fold_rows)
    pairs = []
    for comparator in ("base", "bcr"):
        left = patient.loc[patient.method == "full"].set_index("patient_id")
        right = patient.loc[patient.method == comparator].set_index("patient_id")
        if set(left.index) != set(right.index):
            raise RuntimeError("Patient comparison not paired")
        for metric in METRICS:
            delta = (left[metric] - right[metric]).to_numpy(dtype=float)
            rng = np.random.default_rng(4201)
            draws = rng.integers(0, 80, size=(10000, 80))
            sampled = np.nanmean(delta[draws], axis=1)
            pairs.append({"primary": "full", "comparator": comparator, "metric": metric,
                          "delta": float(np.nanmean(delta)),
                          "ci95_low": float(np.nanquantile(sampled, .025)),
                          "ci95_high": float(np.nanquantile(sampled, .975)),
                          "positive_patients": int(np.sum(delta > 0))})
    paired = pd.DataFrame(pairs)
    summary.to_csv(args.output / "FLAT5_SUMMARY.csv", index=False)
    fold_table.to_csv(args.output / "FLAT5_FOLD_RESULTS.csv", index=False)
    paired.to_csv(args.output / "FLAT5_PAIRED_BOOTSTRAP.csv", index=False)
    lines = ["# PaReSet-EZ no-inner five-fold CV, seed 42", "",
             "Exploratory rerun on the same 80 patients whose historical outer results were already viewed; not an independent heldout confirmation.",
             "Frozen fit+validation train / frozen test each fold; final epoch 45; threshold 0.5; no model/threshold selection on test.", "",
             summary.to_markdown(index=False), "", paired.to_markdown(index=False), "",
             "Patient-level predictions and checkpoints remain private on the server; only compact aggregate tables are publishable."]
    (args.output / "FLAT5_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_write(args.output / "FLAT5_STATUS.json", {"status": "COMPLETE", "patients": 80,
               "folds": 5, "seed": SEED, "historical_outcomes_previously_viewed": True,
               "test_used_for_tuning": False,
               "lock_sha256": digest(args.output / "FLAT5_PROTOCOL_LOCK.json")})
    print(summary.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    dep = dependencies(args)
    groups = partition(args, dep)
    verify_or_create_lock(args, groups, create=args.phase == "prepare")
    if args.phase == "prepare":
        print("FLAT5_PROTOCOL_PREPARED", flush=True)
    elif args.phase == "run":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required but unavailable")
        for fold in FOLDS:
            train, test = groups[fold]
            for method in METHODS:
                fit_one(args, dep, fold, method, train, test, device)
        aggregate(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
