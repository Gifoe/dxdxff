"""Train the two fixed tiny heads on frozen TRAIN embeddings only."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (atomic_json, binary_metrics, choose_threshold,
                    relative_features, score_from_segment_logits, sha256,
                    table)
from official_embedding import ResidualHead


def load_records(cache: Path):
    records = []
    paths = sorted(cache.glob("*.npz"))
    for ordinal, path in enumerate(paths, 1):
        marker = path.with_suffix(".json")
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8"))["output_sha256"] != sha256(path):
            raise RuntimeError("Private TRAIN embedding cache marker mismatch")
        with np.load(path, allow_pickle=False) as source:
            offsets = np.asarray(source["segment_offsets"], dtype=np.int64)
            flat = np.asarray(source["segment_logits"], dtype=np.float32)
            logits = [flat[offsets[i]:offsets[i + 1]].copy() for i in range(len(offsets) - 1)]
            embedding = np.asarray(source["embeddings"], dtype=np.float32)
            difference, rank, fallback = relative_features(embedding)
            records.append({
                "patient": str(source["patient"]), "edf": str(source["edf"]),
                "channel": np.asarray(source["channel_names"]).astype(str),
                "label": np.asarray(source["pathological_labels"], dtype=np.int8),
                "embedding": embedding, "difference": difference, "rank": rank,
                "segment_logits": logits, "fallback": fallback,
            })
        if ordinal % 20 == 0 or ordinal == len(paths):
            print(json.dumps({"stage": "load_train_embedding_cache",
                              "ordinal": ordinal, "of": len(paths)}), flush=True)
    return records


def features(record, variant):
    zeros = np.zeros_like(record["embedding"])
    if variant == "ABS-ONLY":
        return np.concatenate([record["embedding"], zeros, zeros], axis=1)
    if variant == "PR-CNN":
        return np.concatenate([record["embedding"], record["difference"], record["rank"]], axis=1)
    raise ValueError(variant)


def channel_probability_torch(logits, delta):
    values = []
    for current, shift in zip(logits, delta):
        fixed = torch.as_tensor(current, dtype=torch.float32, device=delta.device)
        values.append(torch.sigmoid(fixed + shift).mean())
    return torch.stack(values)


def train_patient(model, records, optimizer, variant):
    losses = []
    for record in records:
        selected = np.flatnonzero(record["label"] >= 0)
        if not len(selected):
            continue
        x = torch.from_numpy(features(record, variant)[selected])
        delta = model(x)
        probability_normal = channel_probability_torch(
            [record["segment_logits"][index] for index in selected], delta)
        pathology = torch.from_numpy(record["label"][selected].astype(np.float32))
        target_normal = 1.0 - pathology
        probability_normal = probability_normal.clamp(1e-6, 1 - 1e-6)
        per_channel = -(target_normal * torch.log(probability_normal) +
                        (1 - target_normal) * torch.log(1 - probability_normal))
        weight = torch.where(pathology > 0.5, 2.0, 1.0)
        losses.append((per_channel * weight).sum() / weight.sum())
    if not losses:
        raise RuntimeError("TRAIN patient has no labeled channels")
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def infer(model, records, variant):
    rows = []
    model.eval()
    with torch.inference_mode():
        for record in records:
            x = torch.from_numpy(features(record, variant))
            delta = model(x).cpu().numpy()
            normal_score = score_from_segment_logits(record["segment_logits"], delta)
            for channel, label, score, change in zip(record["channel"], record["label"], normal_score, delta):
                if label in (0, 1):
                    rows.append({"patient": record["patient"], "edf": record["edf"],
                                 "channel": channel, "y": int(label),
                                 "score": float(1 - score), "delta": float(change)})
    return rows


def pooled(rows, threshold=0.5):
    y = np.asarray([row["y"] for row in rows], dtype=np.int8)
    score = np.asarray([row["score"] for row in rows], dtype=np.float64)
    return {**binary_metrics(y, score, threshold), "pooled_ap": float(average_precision_score(y, score)),
            "edf_channel_units": len(y)}


def save_checkpoint(path, model, variant, epoch, metrics, protocol_sha):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state_dict": model.state_dict(), "variant": variant,
                "epoch": epoch, "validation": metrics,
                "protocol_sha256": protocol_sha}, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-val-split", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
    if baseline.get("status") != "PASS":
        raise RuntimeError("BASELINE_REPLAY_FAILED")
    if sha256(args.train_val_split) != lock["train_validation_split_sha256"]:
        raise RuntimeError("TRAIN/validation split changed")
    with args.train_val_split.open(newline="", encoding="utf-8-sig") as stream:
        split_rows = list(csv.DictReader(stream))
    roles = {row["patient"]: row["role"] for row in split_rows}
    if len(roles) != 141 or set(roles.values()) != {"inner_train", "inner_val"}:
        raise RuntimeError("Expected frozen 141-patient split")
    records = load_records(args.train_cache)
    if len(records) != 296 or {row["patient"] for row in records} != set(roles):
        raise RuntimeError("TRAIN embedding cohort differs from frozen split")
    by_patient = defaultdict(list)
    for record in records:
        by_patient[record["patient"]].append(record)
    fit = sorted(patient for patient, role in roles.items() if role == "inner_train")
    validation = [record for record in records if roles[record["patient"]] == "inner_val"]
    args.runtime.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_sha = sha256(args.protocol)
    history, thresholds, selections = [], {}, {}

    # FrozenCNN validation threshold is independently chosen on the same
    # validation rows. It is not a checkpoint-selection criterion.
    base_rows = []
    for record in validation:
        zero = np.zeros(len(record["channel"]), dtype=np.float32)
        normal = score_from_segment_logits(record["segment_logits"], zero)
        for channel, label, score in zip(record["channel"], record["label"], normal):
            if label in (0, 1):
                base_rows.append({"patient": record["patient"], "edf": record["edf"],
                                  "channel": channel, "y": int(label), "score": float(1 - score)})
    thresholds["FrozenCNN"] = choose_threshold(
        np.asarray([x["y"] for x in base_rows]), np.asarray([x["score"] for x in base_rows]))

    for variant in ("ABS-ONLY", "PR-CNN"):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model = ResidualHead()
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if parameters >= lock["head"]["maximum_trainable_parameters"]:
            raise RuntimeError(f"Residual head too large: {parameters}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        best = None
        started = time.monotonic()
        rng = np.random.default_rng(42)
        for epoch in range(1, 21):
            model.train()
            order = list(np.asarray(fit)[rng.permutation(len(fit))])
            losses = [train_patient(model, by_patient[patient], optimizer, variant)
                      for patient in order]
            rows = infer(model, validation, variant)
            metrics = pooled(rows)
            key = (metrics["auroc"], metrics["pooled_ap"], -epoch)
            selected = best is None or key > best[0]
            if selected:
                path = args.runtime / f"{variant}_selected.pt"
                save_checkpoint(path, model, variant, epoch, metrics, protocol_sha)
                best = (key, epoch, metrics, path)
            history.append({"model": variant, "epoch": epoch,
                            "train_patient_equal_bce": float(np.mean(losses)),
                            "validation_auroc": metrics["auroc"],
                            "validation_ap": metrics["pooled_ap"],
                            "selected": selected})
            print(json.dumps(history[-1]), flush=True)
        assert best is not None
        checkpoint = torch.load(best[3], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        selected_rows = infer(model, validation, variant)
        threshold = choose_threshold(np.asarray([x["y"] for x in selected_rows]),
                                     np.asarray([x["score"] for x in selected_rows]))
        thresholds[variant] = threshold
        selections[variant] = {"checkpoint": str(best[3]), "checkpoint_sha256": sha256(best[3]),
                               "selected_epoch": best[1], "validation": best[2],
                               "threshold": threshold, "train_seconds": time.monotonic() - started,
                               "trainable_parameters": parameters}
    table(args.output / "VALIDATION_METRICS.csv", history)
    table(args.output / "VALIDATION_THRESHOLD_SELECTION.csv",
          [{"model": model, **value} for model, value in thresholds.items()])
    table(args.output / "MODEL_PARAMETER_AUDIT.csv", [
        {"model": "FrozenCNN", "trainable_parameters": 0, "frozen_cnn_parameters": "unchanged", "under_2500": True},
        {"model": "ABS-ONLY", "trainable_parameters": selections["ABS-ONLY"]["trainable_parameters"], "frozen_cnn_parameters": "unchanged", "under_2500": True},
        {"model": "PR-CNN", "trainable_parameters": selections["PR-CNN"]["trainable_parameters"], "frozen_cnn_parameters": "unchanged", "under_2500": True},
    ])
    atomic_json(args.runtime / "HEAD_SELECTION_PRIVATE.json", {
        "status": "TRAIN_VALIDATION_COMPLETE", "protocol_sha256": protocol_sha,
        "baseline_audit_sha256": sha256(args.baseline_audit),
        "fit_patients": len(fit), "validation_patients": len(set(row["patient"] for row in validation)),
        "selections": selections, "thresholds": thresholds,
        "fallback_edfs": sum(record["fallback"] for record in records),
        "official_test_accessed_by_trainer": False,
    })


if __name__ == "__main__":
    main()
