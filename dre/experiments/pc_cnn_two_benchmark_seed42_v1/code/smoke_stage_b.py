"""One train-fit-only Stage-B gradient step; no checkpoint or outcome saved."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from official_spectrum import OfficialSpectrum, load_official_module
from patient_bank import IctalBank
from pc_cnn import PCCNN


def bn_buffers(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            result[name] = (module.running_mean.detach().clone(),
                            module.running_var.detach().clone(),
                            module.num_batches_tracked.detach().clone())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-cnn", type=Path, required=True)
    p.add_argument("--ictal-cache", type=Path, required=True)
    p.add_argument("--ictal-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(42)
    module = load_official_module(args.official_cnn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pc = PCCNN(module.NeuralCNN(in_channels=1, outputs=1)).to(device)
    for parameter in pc.raw.parameters():
        parameter.requires_grad_(False)
    pc.train()
    pc.set_batch_norm_running_state(True)
    before = bn_buffers(pc)
    fit = IctalBank(args.ictal_cache, args.ictal_manifest).folds[1]["fit"]
    record = next(IctalBank(args.ictal_cache, args.ictal_manifest).records(fit[0]))
    selected = np.flatnonzero(record["channel_mask"])[:4]
    if len(selected) < 2:
        raise RuntimeError("Fit-only smoke needs at least two channels")
    waves = torch.from_numpy(record["waveforms"][selected][None]).to(device)
    desc = torch.from_numpy(record["descriptors"][selected][None]).to(device)
    available = torch.from_numpy(record["descriptor_mask"][selected][None].astype(np.float32)).to(device)
    channel = torch.ones(1, len(selected), dtype=torch.bool, device=device)
    label = torch.from_numpy(record["labels"][selected].astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW((v for v in pc.parameters() if v.requires_grad),
                                  lr=2e-4, weight_decay=5e-4)
    preprocessor = OfficialSpectrum(module, channel_chunk=2)
    logits = pc.forward_record(waves, 250.0, desc, channel, available,
                               preprocessor, channel_chunk=2, checkpoint_backbone=True)[0]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, label, weight=torch.where(label > 0, 2.0, 1.0))
    loss.backward()
    film_grad = float(pc.physiology[-1].weight.grad.abs().sum())
    context_grad = float(pc.context_out.weight.grad.abs().sum())
    optimizer.step()
    after = bn_buffers(pc)
    bn_unchanged = all(torch.equal(b, a) for key in before
                       for b, a in zip(before[key], after[key]))
    audit = {"status": "PASS" if film_grad > 0 and context_grad > 0 and bn_unchanged else "FAIL",
             "benchmark": "Ictal", "source_role": "fit_only", "one_step_only": True,
             "film_projector_gradient_l1": film_grad,
             "context_projector_gradient_l1": context_grad,
             "batch_norm_running_state_unchanged": bn_unchanged,
             "raw_cnn_parameters_frozen": all(not v.requires_grad for v in pc.raw.parameters()),
             "test_accessed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
