"""Hardware-only train-step check on an existing private ictal patient cache."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from crst_model import CRSTiEEG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", action="store_true")
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()
    candidates = sorted(args.cache.glob("patient_*.npz"))
    if len(candidates) != 80:
        raise RuntimeError("Private ictal spectral cache incomplete")
    sizes = []
    for path in candidates:
        with np.load(path) as payload:
            shape = payload["patches"].shape
        sizes.append((shape[0] * shape[1], path))
    path = sorted(sizes)[len(sizes) // 2][1]
    with np.load(path) as payload:
        patches = torch.from_numpy(payload["patches"].astype(np.float32))[None].cuda()
        mask = torch.from_numpy(payload["window_mask"])[None].cuda()
        edges = torch.from_numpy(payload["edges"].astype(np.float32))[None].cuda()
        freq = torch.from_numpy(payload["frequency_mask"])[None].cuda()
        labels = torch.from_numpy(payload["labels"].astype(np.float32))[None].cuda()
    model = CRSTiEEG(activation_checkpointing=args.checkpoint).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(patches, freq, mask, edges)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    report = {"step_seconds": elapsed,
              "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
              "records": int(patches.shape[1]), "channels": int(patches.shape[2]),
              "finite_loss": bool(torch.isfinite(loss).item()),
              "activation_checkpointing": args.checkpoint,
              "nan_gradients": sum(int(not torch.isfinite(x.grad).all().item())
                                   for x in model.parameters() if x.grad is not None)}
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
