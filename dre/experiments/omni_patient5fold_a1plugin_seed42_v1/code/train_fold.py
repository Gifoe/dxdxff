#!/usr/bin/env python3
"""Train one frozen patient-CV fold without reading held-out labels.

The raw cache holds one fixed 60-s label-blind window for every eligible EDF.
Each of the 30 locked epochs consumes a different fixed non-overlapping 2-s
segment, so all 60 seconds are used exactly once in training.  This avoids a
historical five-clip cache while making the official Morlet-like TimeConv
training tractable.  Baseline and plugin receive the same segment at each
patient/epoch; only the A1 residual is different.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from modeling import A1ContextResidual, FixedSegmentSpectrum, TimeConvMorphology, load_official_module, sha256


SEED = 42
EPOCHS = 30
RATE = 1000
SEGMENT_SECONDS = 2
SEGMENT_SAMPLES = RATE * SEGMENT_SECONDS
SEGMENTS_PER_RECORD = 30
# The official model's historical four-channel chunk protected its much wider
# 60-s images.  This experiment's frozen 2-s segment schedule fits 16 safely
# on the 32-GB GPU and keeps the same computation/loss while avoiding needless
# per-chunk launch overhead.
CHANNEL_BATCH = 16


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def groups(indices: torch.Tensor, width: int = CHANNEL_BATCH) -> list[torch.Tensor]:
    """Return batches with at least two examples for official BatchNorm."""
    pieces = [indices[offset:offset + width] for offset in range(0, len(indices), width)]
    if len(pieces) > 1 and len(pieces[-1]) == 1:
        pieces[-2] = torch.cat((pieces[-2], pieces[-1]))
        pieces.pop()
    if any(len(piece) < 2 for piece in pieces):
        raise RuntimeError("Patient record has fewer than two good channels")
    return pieces


class RecordBank:
    """Private cache index.  Test NPZ labels are never loaded in this trainer."""

    def __init__(self, records: Path, allowed_patients: set[str]) -> None:
        self.by_patient: dict[str, list[Path]] = defaultdict(list)
        for path in sorted(records.glob("edf_*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                patient = str(payload["patient_name"].item())
            if patient in allowed_patients:
                self.by_patient[patient].append(path)
        absent = sorted(allowed_patients - set(self.by_patient))
        if absent:
            raise RuntimeError(f"Training patients missing record cache: {len(absent)}")

    @staticmethod
    def load(path: Path) -> dict:
        with np.load(path, allow_pickle=False) as payload:
            result = {key: payload[key] for key in ("waveforms", "labels", "channel_names")}
            result["patient_name"] = str(payload["patient_name"].item())
            result["edf_name"] = str(payload["edf_name"].item())
            result["sampling_rate_hz"] = float(payload["sampling_rate_hz"].item())
        if result["waveforms"].shape[1] != RATE * 60 or result["sampling_rate_hz"] != RATE:
            raise RuntimeError("Unexpected fixed private record geometry")
        return result


def patient_forward(raw: TimeConvMorphology, plugin: A1ContextResidual | None,
                    spectrum: FixedSegmentSpectrum, paths: list[Path], segment_index: int,
                    device: torch.device, train: bool) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return patient-channel logits/labels after EDF arithmetic-mean aggregation."""
    if not 0 <= segment_index < SEGMENTS_PER_RECORD:
        raise ValueError("Fixed temporal segment out of range")
    start = segment_index * SEGMENT_SAMPLES
    observations: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    # Unknown occurrences do not override a known patient-channel label.  This
    # exactly matches the cohort audit: only contradictory *valid* labels are
    # conflicts; ``-1`` remains context-only.
    labels: dict[str, int] = {}
    for path in paths:
        record = RecordBank.load(path)
        waves = torch.from_numpy(record["waveforms"][:, start:start + SEGMENT_SAMPLES]).to(device, non_blocking=True)
        targets = np.asarray(record["labels"], dtype=np.int8)
        names = [str(name) for name in record["channel_names"].tolist()]
        selected = torch.arange(len(names), device=device)
        for ids in groups(selected):
            image = spectrum(waves[ids], RATE)
            embedding, logit = raw.embed_and_logit(image)
            for local, channel_position in enumerate(ids.detach().cpu().tolist()):
                name = names[channel_position]
                target = int(targets[channel_position])
                if target >= 0:
                    previous = labels.get(name)
                    if previous is not None and previous != target:
                        raise RuntimeError("STOP_LABEL_CONFLICT in private record cache")
                    labels[name] = target
                observations[name].append((embedding[local], logit[local]))
    names = sorted(observations)
    anchors = torch.stack([torch.stack([value[0] for value in observations[name]]).mean(dim=0)
                           for name in names], dim=0).unsqueeze(0)
    raw_logits = torch.stack([torch.stack([value[1] for value in observations[name]]).mean(dim=0)
                              for name in names], dim=0).unsqueeze(0)
    channel_mask = torch.ones((1, len(names)), dtype=torch.bool, device=device)
    logits = raw_logits if plugin is None else raw_logits + plugin(anchors, channel_mask)
    target = torch.tensor([labels.get(name, -1) for name in names], dtype=torch.float32, device=device).unsqueeze(0)
    return logits.squeeze(0), target.squeeze(0), names


def train_epoch(raw: TimeConvMorphology, plugin: A1ContextResidual | None,
                spectrum: FixedSegmentSpectrum, bank: RecordBank, train_patients: list[str],
                epoch: int, optimizer: torch.optim.Optimizer, device: torch.device) -> dict:
    raw.train()
    if plugin is not None:
        plugin.train()
    order = list(train_patients)
    random.shuffle(order)
    loss_sum, supervised = 0.0, 0
    segment = epoch - 1
    for patient in order:
        optimizer.zero_grad(set_to_none=True)
        logits, labels, _ = patient_forward(raw, plugin, spectrum, bank.by_patient[patient], segment, device, True)
        valid = labels >= 0
        if not bool(valid.any()):
            raise RuntimeError("Both-class training patient has no valid labels")
        weights = torch.where(labels[valid] > 0, 2.0, 1.0)
        loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], weight=weights,
                                                  reduction="sum") / weights.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(raw.parameters()) + ([] if plugin is None else list(plugin.parameters())), 1.0)
        optimizer.step()
        loss_sum += float(loss.detach().cpu())
        supervised += int(valid.sum())
    return {"patient_steps": len(order), "labeled_patient_channels": supervised,
            "mean_patient_loss": loss_sum / max(1, len(order)), "segment_index": segment}


def checkpoint_state(raw, plugin, optimizer, epoch: int, protocol_sha: str, fold: int, variant: str,
                     initial_raw_hash: str, train_patient_sha: str) -> dict:
    return {
        "protocol_sha256": protocol_sha, "fold": fold, "variant": variant, "epoch": epoch,
        "raw_state": raw.state_dict(), "plugin_state": None if plugin is None else plugin.state_dict(),
        "optimizer_state": optimizer.state_dict(), "initial_raw_state_sha256": initial_raw_hash,
        "train_patient_sha256": train_patient_sha, "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state().detach().cpu() if torch.cuda.is_available() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--variant", choices=("baseline", "plugin"), required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_sha = sha256(args.protocol)
    if int(protocol["epochs"]) != EPOCHS:
        raise RuntimeError("Protocol epochs disagree with the locked 60-s schedule")
    if protocol["optimizer"] != {"name": "AdamW", "learning_rate": 1e-4,
                                 "weight_decay": 1e-3, "gradient_clip": 1.0}:
        raise RuntimeError("Optimizer differs from the frozen protocol")
    if protocol["temporal_schedule"] != {"segments_per_record": 30, "segment_seconds": 2,
                                          "segment_selection": "epoch k uses non-overlapping segment k-1; all 60 seconds once"}:
        raise RuntimeError("Temporal segment schedule differs from the frozen protocol")
    if int(protocol["channel_batch"]) != CHANNEL_BATCH:
        raise RuntimeError("Channel batch differs from the frozen protocol")
    if sha256(args.private_manifest) != protocol["patient_manifest_private_sha256"]:
        raise RuntimeError("Private manifest differs from protocol lock")
    manifest = pd.read_csv(args.private_manifest)
    if manifest.patient_name.duplicated().any() or not (manifest.n_pathological_channels.gt(0) & manifest.n_normal_channels.gt(0)).all():
        raise RuntimeError("Invalid both-class patient manifest")
    train_patients = sorted(manifest.loc[manifest.fold != args.fold, "patient_name"].astype(str))
    test_patients = set(manifest.loc[manifest.fold == args.fold, "patient_name"].astype(str))
    if set(train_patients) & test_patients:
        raise RuntimeError("STOP_PROTOCOL_INVALID patient leakage")
    train_patient_sha = hashlib.sha256("\n".join(train_patients).encode()).hexdigest()

    # Both variants seed and initialize the raw morphology path identically.
    random.seed(SEED + args.fold)
    np.random.seed(SEED + args.fold)
    torch.manual_seed(SEED + args.fold)
    torch.cuda.manual_seed_all(SEED + args.fold)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = load_official_module(args.official_cnn)
    raw = TimeConvMorphology(source).to(device)
    initial_raw_hash = state_hash(raw)
    plugin = A1ContextResidual().to(device) if args.variant == "plugin" else None
    spectrum = FixedSegmentSpectrum(source)
    params = list(raw.parameters()) + ([] if plugin is None else list(plugin.parameters()))
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-3)
    bank = RecordBank(args.records, set(train_patients))
    if args.smoke_only:
        started = time.perf_counter()
        result = train_epoch(raw, plugin, spectrum, bank, [train_patients[0]], 1, optimizer, device)
        print(json.dumps({"status": "TRAINING_SMOKE_PASS", "fold": args.fold, "variant": args.variant,
                          "test_labels_or_predictions_accessed": False,
                          "seconds": time.perf_counter() - started, **result}), flush=True)
        return

    work = args.runtime / "checkpoints" / f"fold_{args.fold}" / args.variant
    last = work / "last.pt"
    start_epoch = 1
    if last.exists():
        previous = torch.load(last, map_location=device, weights_only=False)
        required = {"protocol_sha256": protocol_sha, "fold": args.fold, "variant": args.variant,
                    "initial_raw_state_sha256": initial_raw_hash, "train_patient_sha256": train_patient_sha}
        if any(previous.get(key) != value for key, value in required.items()):
            raise RuntimeError("Checkpoint provenance mismatch")
        raw.load_state_dict(previous["raw_state"])
        if plugin is not None:
            plugin.load_state_dict(previous["plugin_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        random.setstate(previous["python_rng"])
        np.random.set_state(previous["numpy_rng"])
        torch.set_rng_state(previous["torch_rng"].detach().cpu())
        if device.type == "cuda" and previous["cuda_rng"] is not None:
            torch.cuda.set_rng_state(previous["cuda_rng"].detach().cpu())
        start_epoch = int(previous["epoch"]) + 1

    audit_path = work / "TRAINING_AUDIT_PRIVATE.json"
    history = [] if not audit_path.exists() else json.loads(audit_path.read_text(encoding="utf-8"))["epochs"]
    for epoch in range(start_epoch, EPOCHS + 1):
        started = time.perf_counter()
        epoch_audit = train_epoch(raw, plugin, spectrum, bank, train_patients, epoch, optimizer, device)
        epoch_audit.update({"epoch": epoch, "seconds": time.perf_counter() - started})
        history.append(epoch_audit)
        atomic_torch(last, checkpoint_state(raw, plugin, optimizer, epoch, protocol_sha, args.fold,
                                             args.variant, initial_raw_hash, train_patient_sha))
        atomic_json(audit_path, {"status": "TRAINING_IN_PROGRESS" if epoch < EPOCHS else "COMPLETE",
                                 "protocol_sha256": protocol_sha, "fold": args.fold, "variant": args.variant,
                                 "train_patient_count": len(train_patients), "test_patient_count": len(test_patients),
                                 "test_labels_or_predictions_accessed": False,
                                 "initial_raw_state_sha256": initial_raw_hash,
                                 "final_effective_plugin_gate": None if plugin is None else float(plugin.effective_gate().detach().cpu()),
                                 "epochs": history})
        print(json.dumps({"status": "TRAINING", "fold": args.fold, "variant": args.variant,
                          **epoch_audit, "test_accessed": False}), flush=True)
    atomic_json(work / "TRAINING_COMPLETE.json", {"status": "COMPLETE", "protocol_sha256": protocol_sha,
                                                    "fold": args.fold, "variant": args.variant,
                                                    "epochs": EPOCHS, "checkpoint": str(last),
                                                    "checkpoint_sha256": sha256(last),
                                                    "initial_raw_state_sha256": initial_raw_hash,
                                                    "final_effective_plugin_gate": None if plugin is None else float(plugin.effective_gate().detach().cpu()),
                                                    "test_labels_or_predictions_accessed": False})


if __name__ == "__main__":
    main()
