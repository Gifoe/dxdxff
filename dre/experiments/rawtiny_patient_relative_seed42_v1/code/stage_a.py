"""One-fold FIT-only implementation checks; no target outcomes are evaluated."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import HybridA1RawModel, RawTinyPatientModel, parameter_count


def mini_batch(batch: dict) -> dict:
    """Three FIT windows, four FIT channels for bounded gradient checks."""
    result = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            result[key] = value
        elif key in ("raw_windows",):
            result[key] = value[:1, :1, :4, :3]
        elif key in ("raw_window_mask",):
            result[key] = value[:1, :1, :4, :3]
        elif key in ("b0_features", "features", "physics_features"):
            result[key] = value[:1, :1, :3, :4]
        elif key in ("seizure_channel_mask", "raw_seizure_channel_mask"):
            result[key] = value[:1, :1, :4]
        elif key in ("window_mask", "window_centers"):
            result[key] = value[:1, :1, :3]
        elif key == "seizure_mask":
            result[key] = value[:1, :1]
        elif key in ("channel_mask", "labels", "labels_ez", "labels_nez"):
            result[key] = value[:1, :4]
        elif value.ndim and value.shape[0] == batch["channel_mask"].shape[0]:
            result[key] = value[:1]
        else:
            result[key] = value
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    for key in ("R1_HLV_SOURCE_ROOT", "R1_HLV_WINDOW_CACHE", "R1_HLV_FIXED_MANIFEST", "R1_HLV_RUNTIME", "A1_A2_RUNTIME"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing private environment variable {key}")
    source = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "neuroez_c"))
    from neuroez_c.dual_view_data import RawAlignmentStore
    import exp_ez_hybrid as core
    # Source runner carries exact A1 arguments and feature-view patch.
    runner_dir = Path(os.environ.get("RAWTINY_R1_RUNNER_DIR", str(source / "r1_hlv_ictal_dynamics_seed42_v1" / "code")))
    sys.path.insert(0, str(runner_dir))
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args

    assert_sources()
    install_interleaved_hlv_view()
    args = make_args("R0", a.output.parent / "private_scratch")
    exp = core.Exp_EZHybridLocalization(args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5:
        raise RuntimeError("A1 cohort or fold count changed")
    split = next(s for s in exp.outer_splits if int(s["fold_idx"]) == 1)
    fit = list(split["fit_subjects"])
    if len(fit) != 51 or set(fit) & set(split["validation_subjects"]):
        raise RuntimeError("Stage A must use only 51 FIT patients")
    store = RawAlignmentStore(
        exp.run_records,
        feature_cache_path=os.environ["R1_HLV_WINDOW_CACHE"],
        raw_cache_path=a.raw_cache,
        raw_target_samples=500,
        raw_target_sampling_rate=250.0,
    )
    store.assert_formal_coverage(min_channel_match_rate=1, min_window_match_rate=1, expected_patients=80)
    exp.raw_alignment_store = store
    exp.use_n6_dual_view_ema = True  # dataset build only; never N6 model/loss
    train_set, _, _, _ = exp._build_datasets(fit, fit, [])
    exp.use_n6_dual_view_ema = False
    loader = exp._make_loader(train_set, shuffle=False, batch_size=1)
    batch = next(iter(loader))
    from exp_ez_hybrid import _move_tensors_to_device
    x = mini_batch(_move_tensors_to_device(batch, exp.device))
    core._set_random_seed(43)
    base = exp.runtime["model_cls"](args).to(exp.device)
    exp._dry_initialize_lazy_layers(base, loader)
    ckpt = Path(os.environ["A1_A2_RUNTIME"]) / "A1" / "fold_1" / "epoch_30.pt"
    state = torch.load(ckpt, map_location=exp.device, weights_only=False)
    if state["variant"] != "A1" or state["epoch"] != 30:
        raise RuntimeError("Wrong A1 epoch30 source checkpoint")
    base.load_state_dict(state["model_state_dict"], strict=True)
    base.eval()
    hybrid = HybridA1RawModel(base).to(exp.device).eval()
    with torch.no_grad():
        reference = base(x)["logits"]
        replay = hybrid(x)["logits"]
    error = float((reference - replay).abs().max())
    if error >= 1e-6 or not torch.equal(reference > 0, replay > 0):
        raise RuntimeError(f"M3 alpha0 failed exact A1 replay: {error}")
    counts = {"M1_RAWTINY_NOPR": parameter_count(RawTinyPatientModel(patient_relative=False)),
              "M2_RAWTINY_PR": parameter_count(RawTinyPatientModel(patient_relative=True)),
              "M3_HYBRID_PR": parameter_count(hybrid)}
    gradients = {}
    for variant, model in (
        ("M1_RAWTINY_NOPR", RawTinyPatientModel(patient_relative=False).to(exp.device)),
        ("M2_RAWTINY_PR", RawTinyPatientModel(patient_relative=True).to(exp.device)),
        ("M3_HYBRID_PR", hybrid),
    ):
        model.train()
        logits = model(x)["logits"]
        loss = logits[x["channel_mask"]].square().mean()
        loss.backward()
        gradients[variant] = sum(int(p.grad is not None and torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
        if gradients[variant] == 0:
            raise RuntimeError(f"No finite gradients: {variant}")
        model.zero_grad(set_to_none=True)
    report = {
        "stage": "A_FIT_ONLY_IMPLEMENTATION_SANITY",
        "pass": True,
        "a1_fold": 1,
        "fit_patients": len(fit),
        "raw_alignment_patient_match_rate": store.audit["patient_match_rate"],
        "raw_alignment_window_match_rate": store.audit["window_match_rate"],
        "M3_alpha0_A1_max_logit_error": error,
        "M3_alpha0_prediction_mismatch": 0,
        "parameter_counts": counts,
        "finite_gradient_parameter_tensors": gradients,
        "target_outcomes_evaluated": False,
        "outer_test_accessed": False,
        "legacy_loader_materialized_validation_labels": True,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
