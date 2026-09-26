"""Matched 36-D supplementary PRQ/BCR controls, validation only.

The supplementary encoder's zero-variance sqrt backward is repaired here with
the same exact-zero forward / finite-gradient contract as PaReSet-EZ. These
are newly trained controls, not historical server checkpoints or Table II rows.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repaired_mean_std(values: torch.Tensor, mask: torch.Tensor, dim: int):
    weights = mask.to(values.dtype)
    safe = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
    mean = safe.sum(dim=dim) / weights.sum(dim=dim).clamp_min(1).unsqueeze(-1)
    difference = torch.where(mask.unsqueeze(-1), values - mean.unsqueeze(dim), torch.zeros_like(values))
    variance = difference.square().sum(dim=dim) / weights.sum(dim=dim).clamp_min(1).unsqueeze(-1)
    variance = variance.clamp_min(0)
    std = torch.where(variance > 0, variance.clamp_min(1e-12).sqrt(), torch.zeros_like(variance))
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=("prq", "bcr"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--minimum-epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patient-batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Output directory is nonempty; refusing to overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root.resolve()))
    from epilens.data import load_records, load_partition_manifest, validate_protocol, select_records
    from epilens.features import fit_standardizer, four_view_expansion
    from epilens.evaluation import select_threshold, patient_metrics
    import epilens.models as models
    from epilens.objectives import prq_loss, bcr_loss

    models._masked_mean_std = repaired_mean_std
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable")
    records = load_records(args.data)
    manifest = load_partition_manifest(args.manifest)
    validate_protocol(records, manifest, expected_patients=80)
    fit = select_records(records, manifest, args.fold, "fit")
    validation = select_records(records, manifest, args.fold, "validation")
    standardizer = fit_standardizer(fit)
    model = (models.PRQNet(dropout=args.dropout) if args.branch == "prq" else models.BCRNet(dropout=args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    tensors = {}
    for record in fit + validation:
        x, valid = four_view_expansion(record.descriptors, record.window_times, record.valid)
        x = standardizer.apply(x, valid)
        tensors[record.patient_id] = (
            torch.tensor(x), torch.tensor(valid, dtype=torch.bool), torch.tensor(record.label_nez)
        )

    def tensorize(record):
        return tuple(value.to(device) for value in tensors[record.patient_id])

    @torch.no_grad()
    def predict(group):
        model.eval()
        rows = []
        for record in group:
            x, valid, labels = tensorize(record)
            output = model(x, valid)
            probability = output["probability_nez"].detach().cpu()
            for channel in output["channel_valid"].nonzero(as_tuple=False).flatten().tolist():
                rows.append({
                    "patient_id": record.patient_id,
                    "center": record.center,
                    "channel_name": record.channel_names[channel],
                    "label_nez": int(labels[channel]),
                    "probability_nez": float(probability[channel]),
                })
        return pd.DataFrame(rows)

    provenance = {
        "control": f"supplementary_{args.branch}_finite_variance_repair",
        "repair": "replace supplementary sqrt(variance) with exact-zero finite-gradient masked_mean_std",
        "historical_paper_checkpoint": False,
        "test_evaluated": False,
        "seed": args.seed, "fold": args.fold,
        "parameters": sum(p.numel() for p in model.parameters()),
        "data_sha256": sha256(args.data),
        "manifest_sha256": sha256(args.manifest),
        "supplement_models_sha256": sha256(args.source_root / "epilens" / "models.py"),
        "runner_sha256": sha256(Path(__file__)),
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    best_key = None
    best_state = None
    best_selection = None
    best_epoch = None
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        order = list(fit)
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), args.patient_batch_size):
            batch = order[start:start + args.patient_batch_size]
            optimizer.zero_grad(set_to_none=True)
            for record in batch:
                x, valid, labels = tensorize(record)
                output = model(x, valid)
                loss = (
                    prq_loss(output["logit_nez"], labels, output["channel_valid"])
                    if args.branch == "prq" else
                    bcr_loss(output["logit_ez"], labels, output["channel_valid"], 0.05, 0.08)[0]
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite control loss")
                (loss / len(batch)).backward()
                losses.append(float(loss.detach().cpu()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        ledger = predict(validation)
        selection = select_threshold(ledger)
        key = (selection.patient_macro_f1, selection.patient_ez_f1, selection.patient_balanced_accuracy)
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_selection = selection
            best_epoch = epoch
            stale = 0
            torch.save({"model": best_state, "selection": asdict(selection), "epoch": epoch, "provenance": provenance,
                        "standardizer_mean": standardizer.mean.tolist(), "standardizer_scale": standardizer.scale.tolist()},
                       args.output / "best.pt")
        else:
            stale += 1
        row = {"epoch": epoch, "training_loss": float(np.mean(losses)), **asdict(selection)}
        history.append(row)
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break
    if best_state is None or best_selection is None:
        raise RuntimeError("No control checkpoint selected")
    model.load_state_dict(best_state)
    frame = predict(validation)
    frame["fold"] = args.fold
    frame["seed"] = args.seed
    frame["branch"] = args.branch
    frame["threshold"] = best_selection.threshold
    frame["predicted_nez"] = (frame["probability_nez"] >= best_selection.threshold).astype(int)
    frame.to_csv(args.output / "validation_predictions.csv", index=False)
    metrics = patient_metrics(frame, best_selection.threshold)
    metrics.to_csv(args.output / "validation_patient_metrics.csv", index=False)
    (args.output / "selection.json").write_text(json.dumps({"epoch": best_epoch, **asdict(best_selection), "test_evaluated": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
