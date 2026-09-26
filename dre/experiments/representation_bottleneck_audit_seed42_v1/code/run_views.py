"""Matched A1 four-view ablation; validation only, resumable at each epoch."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT, build_fold, install_interleaved_hlv_view, make_args, sha256  # noqa: E402
sys.path.insert(0, str(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import epoch_grid, finalize_fold  # noqa: E402
from run_development import custom_compute_loss, check_grid_against_core, assert_architecture_args, assert_cohort  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402
from reproduce_source import A1_RUNTIME, LOCK_SHA256, RUNTIME, ensure_source  # noqa: E402

VIEWS = {"D1": tuple(range(0, 9)), "D2": tuple(range(27, 36)), "D3": tuple(range(0, 9)) + tuple(range(27, 36))}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def zero_views(columns: tuple[int, ...]):
    def hook(_module, inputs):
        x = inputs[0]
        if x.shape[-1] != 36:
            raise RuntimeError(f"Expected full four-view 36-D input, got {tuple(x.shape)}")
        masked = x.clone()
        masked[..., columns] = 0.0
        return (masked, *inputs[1:])
    return hook


def verify_source() -> None:
    ensure_source()
    reproduction_path = EXPERIMENT / "SOURCE_REPRODUCTION.json"
    if not reproduction_path.exists():
        raise RuntimeError("Run frozen source reproduction first")
    result = json.loads(reproduction_path.read_text(encoding="utf-8"))
    if not result.get("pass") or abs(result["observed_mean"] - 0.6259962097139906) > 1e-6:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")


def run_fold(exp, split, variants: tuple[str, ...]) -> None:
    fold, _train_set, train_loader, val_loader, test_loader, normalizer = build_fold(exp, split, "validation")
    if test_loader is not None or len(val_loader.dataset) != 13:
        raise RuntimeError("Validation-only role changed")
    init_path = A1_RUNTIME / "initial" / f"fold_{fold}_initial.pt"
    initial_state = torch.load(init_path, map_location="cpu", weights_only=True)
    init_hash = sha256(init_path)
    exp._compute_loss = types.MethodType(custom_compute_loss("A1"), exp)
    ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
    for variant in variants:
        folder = RUNTIME / "private" / "views" / variant / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "development_summary.json").exists():
            print(f"[RESUME] {variant} fold={fold}", flush=True)
            continue
        core._set_random_seed(42 + fold)
        model = exp.runtime["model_cls"](exp.args).to(exp.device)
        exp._dry_initialize_lazy_layers(model, train_loader)
        model.load_state_dict(initial_state, strict=True)
        if any(not torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in initial_state.items()):
            raise RuntimeError("D variant did not start from exact shared A1 initialization")
        parameter_count = sum(p.numel() for p in model.parameters())
        handle = model.b0_encoder.register_forward_pre_hook(zero_views(VIEWS[variant]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        try:
            for epoch in range(1, 31):
                path = folder / f"epoch_{epoch:02d}.pt"
                grid_path = folder / f"epoch_{epoch:02d}_validation_grid.json"
                if path.exists():
                    saved = torch.load(path, map_location=exp.device, weights_only=False)
                    if (saved["variant"], saved["fold"], saved["epoch"], saved["lock_sha256"], saved["initial_sha256"]) != (variant, fold, epoch, LOCK_SHA256, init_hash):
                        raise RuntimeError("D resume provenance mismatch")
                    if not np.array_equal(saved["normalizer_mean"], normalizer.mean) or not np.array_equal(saved["normalizer_std"], normalizer.std):
                        raise RuntimeError("D resume normalizer mismatch")
                    model.load_state_dict(saved["model_state_dict"], strict=True)
                    optimizer.load_state_dict(saved["optimizer_state_dict"])
                    train_metrics = saved["train_metrics"]
                else:
                    exp.current_epoch = epoch
                    core._set_random_seed(42 * 100000 + fold * 1000 + epoch)
                    train_metrics = exp._train_one_epoch(model, train_loader, optimizer, ez_weight)
                    if not math.isfinite(float(train_metrics["loss"])):
                        raise RuntimeError("Nonfinite D training loss")
                    torch.save({"variant": variant, "fold": fold, "epoch": epoch, "lock_sha256": LOCK_SHA256,
                                "initial_sha256": init_hash, "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(), "train_metrics": train_metrics,
                                "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std}, path)
                if not grid_path.exists():
                    _, _, records = exp._evaluate(model, val_loader, ez_weight, split_name="val")
                    grid = epoch_grid(records, epoch)
                    check_grid_against_core(records, grid)
                    atomic_json(grid_path, grid)
                print(f"[{variant}] fold={fold} epoch={epoch}/30 train_loss={train_metrics['loss']:.6f}", flush=True)
            grids = [json.loads((folder / f"epoch_{epoch:02d}_validation_grid.json").read_text(encoding="utf-8")) for epoch in range(1, 31)]
            public_fold, fullval = finalize_fold(grids, variant, fold, RUNTIME / "private" / "views" / f"{variant}_fold_{fold}_VLOO_PATIENT.csv")
            fullval["checkpoint_sha256"] = sha256(folder / f"epoch_{fullval['selected_epoch']:02d}.pt")
            atomic_json(folder / "development_summary.json", {"vloo": public_fold, "fullval": fullval,
                                                          "parameter_count": parameter_count, "epochs": 30,
                                                          "zeroed_views": list(VIEWS[variant]), "outer_test_accessed": False})
            print(f"[{variant}] fold={fold} VLOO={public_fold['patient_macro_f1']:.6f}", flush=True)
        finally:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--variant", choices=tuple(VIEWS))
    opts = parser.parse_args()
    verify_source()
    install_interleaved_hlv_view()
    args = make_args("R0", RUNTIME)
    assert_architecture_args(args)
    exp = core.Exp_EZHybridLocalization(args)
    assert_cohort(exp)
    for split in exp.outer_splits:
        if opts.fold is None or split["fold_idx"] == opts.fold:
            run_fold(exp, split, (opts.variant,) if opts.variant else tuple(VIEWS))


if __name__ == "__main__":
    main()
