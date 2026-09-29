"""Phase-1 synthetic tests. No real EEG, labels, checkpoint or outcome access."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from crst_model import CRSTiEEG, parameter_count


def run() -> dict:
    torch.manual_seed(42)
    model = CRSTiEEG(dropout=0).eval()
    nparams = parameter_count(model)
    assert 1_000_000 <= nparams <= 8_000_000, nparams
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                   for m in model.modules())
    b, r, c, w = 1, 2, 4, 5
    patch = torch.randn(b, r, c, w, 64, 8)
    freq = torch.ones(b, 64)
    mask = torch.ones(b, r, c, w, dtype=torch.bool)
    mask[:, 1, 3, :] = False
    edges = torch.randn(b, r, c, c, 15)
    with torch.no_grad():
        original = model(patch, freq, mask, edges)
        perm = torch.tensor([2, 0, 3, 1])
        permuted = model(patch[:, :, perm], freq, mask[:, :, perm],
                         edges[:, :, perm][:, :, :, perm])
        equiv_err = (original[:, perm] - permuted).abs().max().item()
        # The patient MAD denominator can be 1e-4 in synthetic near-constant
        # coordinates; FP32 attention order then amplifies ~1e-6 roundoff.
        assert equiv_err < 5e-3, equiv_err
        # Padded record/channel values cannot affect a valid patient channel.
        corrupt = patch.clone()
        corrupt[:, 1, 3] = 999.0
        masked = model(corrupt, freq, mask, edges)
        mask_err = (original - masked).abs().max().item()
        assert mask_err < 1e-5, mask_err
        # Initial lambda=0 makes learned connectivity features score-neutral.
        no_net = model(patch, freq, mask, torch.zeros_like(edges))
        lambda0_err = (original - no_net).abs().max().item()
        assert lambda0_err < 1e-5, lambda0_err
        for key in ("A_ONLY", "T_ZERO", "C_ZERO", "R_ZERO",
                    "CONNECTIVITY_ZERO", "NO_CHANNEL_ATTENTION"):
            assert torch.isfinite(model(patch, freq, mask, edges,
                                        intervention=key)).all(), key
        one_record = model(patch[:, :1], freq, mask[:, :1], edges[:, :1],
                           return_aux=True)
        assert one_record["relative_rms"][2].item() == 0
        # Variable fs changes only availability, not module or parameters.
        low_fs = torch.tensor([1.0 if f <= 112.5 else 0.0
                               for f in torch.logspace(0, torch.log10(torch.tensor(300.0)), 64)])
        assert torch.isfinite(model(patch, low_fs[None], mask, edges)).all()
    return {"pass": True, "seed": 42, "parameters": nparams,
            "channel_permutation_max_abs_error": equiv_err,
            "padded_channel_max_abs_error": mask_err,
            "connectivity_lambda0_max_abs_error": lambda0_err,
            "single_record_relative_R_rms": 0.0,
            "no_batch_norm": True}


if __name__ == "__main__":
    result = run()
    path = Path(__file__).resolve().parents[1] / "PHASE1_SYNTHETIC_TEST.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
