"""Faithful single-model Omni CNN training gate, importing pinned cnn.py.

The official train.py imports many unrelated optional models, eagerly loads
all NPZs, and has no per-epoch resume. This runner retains its CNN, optimizer,
sampling, positive-branch augmentation and validation rule, while saving
private epoch checkpoints for safe interruption. It never reads official test.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split

PINNED_CNN_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"


def official_module(path: Path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINNED_CNN_SHA256:
        raise RuntimeError("Official CNN source SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("official_omni_cnn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class EagerOfficialDataset(Dataset):
    def __init__(self, root: Path):
        self.blocks = []
        self.labels = []
        self.flip = []
        self.offsets = [0]
        for branch in ("positive", "negative"):
            files = sorted((root / branch).rglob("*.npz"))
            if not files:
                raise RuntimeError(f"No official {branch} extraction files")
            for file in files:
                with np.load(file, allow_pickle=False) as source:
                    data = np.asarray(source["data"], dtype=np.float32)
                    label = np.asarray(source["labels"], dtype=np.int8)
                if data.ndim != 2 or data.shape[1] != 60000 or len(data) != len(label):
                    raise RuntimeError(f"Invalid 60-s official NPZ: {file}")
                expected = 1 if branch == "positive" else 0
                if not bool(np.all(label == expected)):
                    raise RuntimeError(f"Incorrect source-branch label: {file}")
                self.blocks.append(data)
                self.labels.extend(label.tolist())
                self.flip.extend([branch == "positive"] * len(label))
                self.offsets.append(self.offsets[-1] + len(label))
        self.labels = np.asarray(self.labels, dtype=np.int8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item: int):
        block = bisect.bisect_right(self.offsets, item) - 1
        waveform = self.blocks[block][item - self.offsets[block]]
        # Same 0.5 positive-branch temporal flip as pinned ChannelDataset.
        if self.flip[item] and random.random() < 0.5:
            waveform = waveform[::-1].copy()
        return torch.from_numpy(waveform), float(self.labels[item])


def epoch(model, prep, loader, optimizer, device, *, train: bool):
    model.train(train)
    losses, predictions, labels = [], [], []
    for wave, target in loader:
        wave = wave.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            image = prep(wave)
        if not bool(torch.isfinite(image).all()):
            raise RuntimeError("Nonfinite official spectrum / min-max normalized image")
        if train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(image).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(image).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        losses.append(float(loss.detach().cpu()))
        predictions.extend((torch.sigmoid(logits) > 0.5).int().detach().cpu().tolist())
        labels.extend(target.int().detach().cpu().tolist())
    # Pinned train.py's calculate_metrics calls sklearn.f1_score with its
    # default binary average, *not* macro as the paper prose says.
    return {"loss": float(np.mean(losses)),
            "normal_class_f1": float(f1_score(labels, predictions, zero_division=0)),
            "samples": len(labels)}


def atomic_save(obj, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-cnn", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    a = p.parse_args()
    if not (a.features / "EXTRACTION_TRAIN_AUDIT.json").is_file():
        raise RuntimeError("Full official train waveform extraction not complete")
    protocol_sha = hashlib.sha256(a.protocol.read_bytes()).hexdigest()
    extraction = json.loads((a.features / "EXTRACTION_TRAIN_AUDIT.json").read_text())
    if not extraction["completed"] or extraction["protocol_sha256"] != protocol_sha:
        raise RuntimeError("Training waveform cache/protocol mismatch")
    a.runtime.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    source = official_module(a.official_cnn)
    data = EagerOfficialDataset(a.features)
    if len(data) != extraction["samples"]:
        raise RuntimeError("Training sample count differs from extraction audit")
    val_size = int(0.2 * len(data))
    train_split, val_split = random_split(
        data, [len(data) - val_size, val_size], generator=torch.Generator().manual_seed(42))
    global_counts = np.bincount(data.labels.astype(int), minlength=2)
    class_weights = len(data) / (2 * global_counts)
    class_weights /= class_weights.sum()
    train_labels = data.labels[np.asarray(train_split.indices)]
    weights = torch.as_tensor(class_weights[train_labels.astype(int)], dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_split, batch_size=32, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_split, batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=True)
    device = torch.device("cuda")
    model = source.NeuralCNN(in_channels=1, outputs=1).to(device)
    prep = source.NeuralCNNPreProcessing(
        image_size=224, frequency=1000, freq_range_hz=[10, 300],
        event_length=60000, selected_window_size_ms=30000,
        selected_freq_range_hz=[10, 300], random_shift_ms=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    latest = a.runtime / "latest.pt"
    best = a.runtime / "best_model.pt"
    history = []
    start, best_f1 = 0, -1.0
    if latest.is_file():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["protocol_sha256"] != protocol_sha or checkpoint["samples"] != len(data):
            raise RuntimeError("Latest checkpoint provenance mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        history = checkpoint["history"]
        start, best_f1 = checkpoint["epoch"], checkpoint["best_f1"]
    for i in range(start, 10):
        beginning = time.monotonic()
        train_metrics = epoch(model, prep, train_loader, optimizer, device, train=True)
        val_metrics = epoch(model, prep, val_loader, optimizer, device, train=False)
        selected = val_metrics["normal_class_f1"] > best_f1
        if selected:
            best_f1 = val_metrics["normal_class_f1"]
            atomic_save({"model_state_dict": model.state_dict(), "epoch": i + 1,
                         "protocol_sha256": protocol_sha, "val_f1": best_f1}, best)
        row = {"epoch": i + 1, "train": train_metrics, "validation": val_metrics,
               "best_checkpoint": selected, "seconds": time.monotonic() - beginning}
        history.append(row)
        print(json.dumps(row), flush=True)
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": i + 1, "history": history, "best_f1": best_f1,
                     "samples": len(data), "protocol_sha256": protocol_sha,
                     "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                     "torch_rng": torch.get_rng_state().cpu(),
                     "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()]}, latest)
    report = {"completed": True, "epochs": 10, "best_validation_normal_f1": best_f1,
              "selected_epoch": max((row["epoch"] for row in history if row["best_checkpoint"]), default=None),
              "train_samples": len(train_split), "val_samples": len(val_split),
              "all_class_counts_normal_pathology": global_counts.tolist(),
              "protocol_sha256": protocol_sha, "source_cnn_sha256": PINNED_CNN_SHA256,
              "checkpoint_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
              "history": history}
    (a.runtime / "TRAINING_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
