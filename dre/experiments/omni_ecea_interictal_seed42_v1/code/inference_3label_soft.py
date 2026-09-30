#!/usr/bin/env python3
"""Official-compatible three-label inference with soft pathological evidence.

This changes only serialization: it preserves the frozen event model and adds
softmax(logits)[:, 2] alongside the original argmax output. Class 2 is the
official Spike/pathological-event class in dataloader_parquet.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from omni_ieeg.channel_model.event_model_inference.dataloader import EventInferenceDataset
from omni_ieeg.channel_model.event_model_inference.inference_3label import collate_events
from omni_ieeg.event_model.train.model_3label.configs import model_preprocessing_configs


PATHOLOGICAL_CLASS = 2


def inference_one(model, preprocessing, device, npz_path: Path, batch_size: int, num_workers: int) -> pd.DataFrame:
    dataset = EventInferenceDataset(str(npz_path))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_events,
    )
    hard_parts: list[torch.Tensor] = []
    soft_parts: list[torch.Tensor] = []
    metadata_parts: list[dict[str, list[object]]] = []
    model.eval()
    if hasattr(preprocessing, "eval"):
        preprocessing.eval()
    if hasattr(preprocessing, "disable_random_shift"):
        preprocessing.disable_random_shift()
    with torch.no_grad():
        for batch in tqdm(loader, desc=npz_path.name, leave=False):
            waveforms = batch["waveform"].to(device)
            logits = model(preprocessing(waveforms))
            probabilities = torch.softmax(logits, dim=1)
            hard_parts.append(torch.argmax(logits, dim=1).cpu())
            soft_parts.append(probabilities[:, PATHOLOGICAL_CLASS].cpu())
            metadata_parts.append(batch["metadata"])
    metadata = {
        key: [value for part in metadata_parts for value in part[key]]
        for key in metadata_parts[0]
    } if metadata_parts else {}
    metadata["3label_pred"] = torch.cat(hard_parts).numpy() if hard_parts else np.empty(0, dtype=np.int64)
    metadata["pathological_probability"] = (
        torch.cat(soft_parts).numpy() if soft_parts else np.empty(0, dtype=np.float32)
    )
    return pd.DataFrame(metadata)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="cnn", choices=["cnn", "vit", "lstm", "transformer", "timesnet"])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = model_preprocessing_configs[args.model_name]
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model = config["model_config"]["model_class"](**config["model_config"]["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    preprocessing = config["preprocessing_config"]["preprocessing_class"](
        **config["preprocessing_config"]["preprocessing_params"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for npz_path in sorted(args.feature_path.glob("*.npz")):
        frame = inference_one(
            model, preprocessing, args.device, npz_path, args.batch_size, args.num_workers
        )
        frame.to_csv(args.output_dir / f"{npz_path.stem}.csv", index=False)
        frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir / "all_results.csv", index=False)
    (args.output_dir / "SOFT_INFERENCE_METADATA.json").write_text(
        json.dumps(
            {
                "event_checkpoint": str(args.model_path),
                "pathological_class": PATHOLOGICAL_CLASS,
                "new_model_trained": False,
                "output_rows": int(sum(len(frame) for frame in frames)),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
