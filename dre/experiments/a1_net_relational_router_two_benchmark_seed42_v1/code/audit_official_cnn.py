"""Source-hash and full 60-s forward smoke for the pinned official CNN."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torchvision

PINNED_CNN_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--official-cnn", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-step", action="store_true")
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    observed = sha256(args.official_cnn)
    if observed != PINNED_CNN_SHA256:
        raise RuntimeError("Pinned official cnn.py SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("official_omni_cnn", args.official_cnn)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = module.NeuralCNN(in_channels=1, outputs=1).to(device).eval()
    prep = module.NeuralCNNPreProcessing(
        image_size=224, frequency=1000, freq_range_hz=[10, 300],
        event_length=60000, selected_window_size_ms=30000,
        selected_freq_range_hz=[10, 300], random_shift_ms=0)
    with torch.inference_mode():
        x = torch.randn(2, 60000, device=device)
        image = prep(x)
        scores = net(image)
    if list(scores.shape) != [2, 1] or not bool(torch.isfinite(scores).all()):
        raise RuntimeError("Official CNN full-record forward smoke failed")
    payload = {
        "official_code_revision": protocol["official_code_revision"],
        "cnn_py_sha256": observed,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": device,
        "model_parameters": sum(x.numel() for x in net.parameters()),
        "raw_shape": [2, 60000],
        "spectrum_shape": list(image.shape),
        "output_shape": list(scores.shape),
        "finite_output": True,
        "pretrained_resnet18_requested": True,
        "normalization": "official per-spectrogram min-max * 255",
        "source_rate_hz": 1000,
        "source_frequency_hz": [10, 300],
        "source_fft_bins": 224,
    }
    if args.train_step:
        if device != "cuda":
            raise RuntimeError("Batch-32 GPU memory smoke needs CUDA")
        del x, image, scores
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        net.train()
        batch = torch.randn(32, 60000, device=device)
        with torch.no_grad():
            picture = prep(batch)
        output = net(picture).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output, torch.randint(0, 2, (32,), device=device).float())
        loss.backward()
        payload["batch32_forward_backward_pass"] = True
        payload["batch32_peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        del batch, picture, output, loss
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
