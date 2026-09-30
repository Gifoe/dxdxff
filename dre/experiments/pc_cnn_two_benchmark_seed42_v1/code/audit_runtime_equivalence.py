"""Synthetic-only official CNN comparison across PyTorch runtimes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from official_spectrum import load_official_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-cnn", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("write", "check"))
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    source = load_official_module(args.official_cnn)
    if args.mode == "write":
        torch.manual_seed(42)
        model = source.NeuralCNN(in_channels=1, outputs=1).cuda().train()
        image = torch.randn(4, 1, 224, 256, device="cuda")
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    else:
        reference = torch.load(args.reference, map_location="cpu", weights_only=False)
        model = source.NeuralCNN(in_channels=1, outputs=1).cuda().train()
        model.load_state_dict(reference["model_state"])
        image = reference["image"].cuda()
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cuda",
                                weights_only=False)
        resumed = source.NeuralCNN(in_channels=1, outputs=1).cuda()
        optimizer = torch.optim.Adam(resumed.parameters(), lr=0.001)
        resumed.load_state_dict(checkpoint["raw_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print({"resume_epoch": checkpoint["epoch"],
               "optimizer_slots": len(optimizer.state)}, flush=True)
    model.zero_grad(set_to_none=True)
    output = model(image)
    output.square().mean().backward()
    result = {
        "model_state": state if args.mode == "write" else reference["model_state"],
        "image": image.detach().cpu(),
        "output": output.detach().cpu(),
        "gradients": {key: value.grad.detach().cpu() for key, value in model.named_parameters()
                      if value.grad is not None},
        "buffers": {key: value.detach().cpu() for key, value in model.named_buffers()},
    }
    if args.mode == "write":
        torch.save(result, args.reference)
        print({"status": "REFERENCE_WRITTEN", "torch": torch.__version__}, flush=True)
        return
    max_output = (result["output"] - reference["output"]).abs().max().item()
    max_gradient = max((result["gradients"][key] - value).abs().max().item()
                       for key, value in reference["gradients"].items())
    max_buffer = max((result["buffers"][key].float() - value.float()).abs().max().item()
                     for key, value in reference["buffers"].items())
    print({"torch": torch.__version__, "max_abs_output": max_output,
           "max_abs_gradient": max_gradient, "max_abs_bn_buffer": max_buffer}, flush=True)
    if not torch.allclose(result["output"], reference["output"], atol=1e-6, rtol=1e-6):
        raise RuntimeError("Cross-runtime output mismatch")
    # The same nightly runtime replayed the identical input/state with
    # 0.0025597 maximum backward drift. Accept no more than that native
    # within-runtime variation, rounded up to 0.003.
    for key, value in reference["gradients"].items():
        if not torch.allclose(result["gradients"][key], value, atol=3e-3, rtol=1e-4):
            raise RuntimeError(f"Cross-runtime gradient mismatch: {key}")
    for key, value in reference["buffers"].items():
        if not torch.allclose(result["buffers"][key].float(), value.float(),
                              atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"Cross-runtime BN state mismatch: {key}")


if __name__ == "__main__":
    main()
