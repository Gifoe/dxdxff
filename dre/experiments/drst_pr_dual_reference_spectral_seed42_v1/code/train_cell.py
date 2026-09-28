"""One resumable, FIT-only DRST spectral model/HP cell.

Target validation and outer-test patients are excluded from dataset creation.
Public summaries must not include patient/channel-level records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import DRSTPatientModel, VARIANT_VIEWS, parameter_count
from spectral import FixedScaleRawAlignmentStore
from stage0_source import file_sha, write_json


def source_split(fit: list[str], fold: int) -> tuple[list[str], list[str]]:
    keyed = sorted(fit, key=lambda sid: hashlib.sha256(f"42|{fold}|{sid}".encode()).hexdigest())
    n_meta = round(len(keyed) * 0.2)
    train, meta = keyed[:-n_meta], keyed[-n_meta:]
    if set(train) & set(meta) or set(train) | set(meta) != set(fit):
        raise RuntimeError("FIT-only split violation")
    return train, meta


def source_loader(source: Path, raw_cache: Path, runtime: Path, fold: int, model_args, sanity: bool):
    sys.path[:0] = [str(source), str(source / "neuroez_c"),
                    str(source / "r1_hlv_ictal_dynamics_seed42_v1" / "code"),
                    str(source / "a1_a2_patient_equal_objective_seed42_v1" / "code")]
    import exp_ez_hybrid as core
    from neuroez_c.dual_view_data import RawAlignmentStore
    from run_matched import assert_sources, install_interleaved_hlv_view

    assert_sources()
    install_interleaved_hlv_view()
    exp = core.Exp_EZHybridLocalization(model_args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5:
        raise RuntimeError("Frozen A1 population changed")
    split = next(row for row in exp.outer_splits if int(row["fold_idx"]) == fold)
    fit, val, test = map(set, (split["fit_subjects"], split["validation_subjects"], split["test_subjects"]))
    train, meta = source_split(sorted(fit), fold)
    if len(val) != 13 or train and set(train) & (val | test) or set(meta) & (val | test):
        raise RuntimeError("Target patient entered FIT loader")
    if sanity:
        train = train[:2]
        meta = train
    raw = RawAlignmentStore(exp.run_records, feature_cache_path=os.environ["R1_HLV_WINDOW_CACHE"],
                            raw_cache_path=raw_cache, raw_target_samples=500, raw_target_sampling_rate=250.0)
    raw.assert_formal_coverage(min_channel_match_rate=1, min_window_match_rate=1, expected_patients=80)
    exp.raw_alignment_store = FixedScaleRawAlignmentStore(raw)
    exp.use_n6_dual_view_ema = True
    train_set, meta_set, _, _ = exp._build_datasets(train, meta, [])
    exp.use_n6_dual_view_ema = False
    return exp, exp._make_loader(train_set, shuffle=True, batch_size=1), exp._make_loader(meta_set, shuffle=False, batch_size=1), train, meta


def meta_patient_equal_ap(model, loader, device):
    from exp_ez_hybrid import _move_tensors_to_device

    model.eval()
    values = []
    with torch.no_grad():
        for raw in loader:
            batch = _move_tensors_to_device(raw, device)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(batch)
            mask = batch["channel_mask"].bool()[0]
            y = batch["labels_ez"][0][mask].detach().cpu().numpy()
            scores = out["score_ez"][0][mask].detach().cpu().numpy()
            if len(y) == 0 or not np.isfinite(scores).all() or len(np.unique(y)) < 2:
                raise RuntimeError("Invalid FIT-meta AP support")
            values.append(float(average_precision_score(y, scores)))
    if not values:
        raise RuntimeError("Empty FIT-meta set")
    return float(np.mean(values))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    p.add_argument("--variant", choices=VARIANT_VIEWS, required=True)
    p.add_argument("--lr", type=float, choices=(1e-4, 3e-4), required=True)
    p.add_argument("--wd", type=float, choices=(1e-4, 1e-3), required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--sanity", action="store_true")
    a = p.parse_args()
    for key in ("R1_HLV_SOURCE_ROOT", "R1_HLV_WINDOW_CACHE", "R1_HLV_FIXED_MANIFEST", "R1_HLV_RUNTIME", "A1_A2_RUNTIME"):
        if key not in os.environ:
            raise RuntimeError(f"Missing source environment: {key}")
    source = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    sys.path[:0] = [str(source / "r1_hlv_ictal_dynamics_seed42_v1" / "code")]
    from run_matched import make_args

    normalizer_path = a.runtime / f"fold_{a.fold}" / "FIT_GLOBAL_SPECTRAL_NORMALIZER.json"
    moments = json.loads(normalizer_path.read_text(encoding="utf-8"))
    lock_sha = file_sha(a.lock)
    if not moments["fit_only"] or moments["lock_sha256"] != lock_sha:
        raise RuntimeError("FIT normalizer/lock mismatch")
    exp, train_loader, meta_loader, train, meta = source_loader(
        source, a.raw_cache, a.runtime, a.fold, make_args("R0", a.runtime / "scratch"), a.sanity)
    torch.manual_seed(42 * 100000 + a.fold * 1000)
    torch.cuda.manual_seed_all(42 * 100000 + a.fold * 1000)
    model = DRSTPatientModel(a.variant, torch.tensor(moments["mean"]), torch.tensor(moments["std"])).to(exp.device)
    from objectives import patient_equal_weighted_bce_loss
    from exp_ez_hybrid import _move_tensors_to_device

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    cell = a.runtime / a.variant / f"fold_{a.fold}" / f"lr_{a.lr:g}_wd_{a.wd:g}"
    if a.sanity:
        cell = cell / "overfit_sanity"
    cell.mkdir(parents=True, exist_ok=True)
    state_path = cell / "resume_private.pt"
    summary_path = cell / "summary.json"
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("complete"):
            if prior.get("lock_sha256") != lock_sha or prior.get("variant") != a.variant or prior.get("fold") != a.fold:
                raise RuntimeError("Complete cell resume identity mismatch")
            print(f"ALREADY_COMPLETE {a.variant} fold={a.fold} lr={a.lr:g} wd={a.wd:g}", flush=True)
            return
    if state_path.is_file():
        state = torch.load(state_path, map_location=exp.device, weights_only=False)
        if state["lock_sha256"] != lock_sha or state["variant"] != a.variant or state["fold"] != a.fold:
            raise RuntimeError("Resume identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].detach().cpu())
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([v.detach().cpu() for v in state["cuda_rng"]])
        start_epoch = int(state["epoch"]) + 1
        best_ap, best_epoch, patience = state["best_ap"], state["best_epoch"], state["patience"]
        rows = state["rows"]
    else:
        start_epoch, best_ap, best_epoch, patience, rows = 1, float("-inf"), 0, 0, []
    for epoch in range(start_epoch, 21 if a.sanity else 51):
        model.train()
        losses = []
        started = time.perf_counter()
        for raw in train_loader:
            batch = _move_tensors_to_device(raw, exp.device)
            batch = {k: v.to(exp.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = patient_equal_weighted_bce_loss(out["logits"], batch["labels"], batch["labels_ez"], batch["channel_mask"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite source training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        ap = meta_patient_equal_ap(model, meta_loader, exp.device)
        eligible = epoch >= (1 if a.sanity else 8)
        improved = eligible and (ap > best_ap + 1e-12)
        if improved:
            best_ap, best_epoch, patience = ap, epoch, 0
            best_tmp = cell / "selected_best_private.pt.tmp"
            torch.save({"lock_sha256": lock_sha, "model": model.state_dict(),
                        "epoch": epoch, "ap": ap, "variant": a.variant, "fold": a.fold}, best_tmp)
            best_tmp.replace(cell / "selected_best_private.pt")
        elif eligible:
            patience += 1
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "fit_meta_patient_equal_ez_ap": ap,
               "best_epoch": best_epoch, "best_ap": best_ap if np.isfinite(best_ap) else None,
               "elapsed_seconds": time.perf_counter() - started}
        rows.append(row)
        latest_tmp = cell / "resume_private.pt.tmp"
        torch.save({"lock_sha256": lock_sha, "variant": a.variant, "fold": a.fold, "epoch": epoch,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "best_ap": best_ap, "best_epoch": best_epoch, "patience": patience, "rows": rows}, latest_tmp)
        latest_tmp.replace(state_path)
        write_json(summary_path, {"variant": a.variant, "fold": a.fold, "lr": a.lr, "weight_decay": a.wd,
                                  "lock_sha256": lock_sha, "train_subject_count": len(train), "meta_subject_count": len(meta),
                                  "parameter_count": parameter_count(model), "best_ap": best_ap if np.isfinite(best_ap) else None,
                                  "best_epoch": best_epoch, "epochs_run": epoch, "patience": patience,
                                  "complete": patience >= 8 or epoch == (20 if a.sanity else 50), "rows": rows,
                                  "target_labels_indexed": False, "outer_test_accessed": False})
        print(f"{a.variant} fold={a.fold} lr={a.lr:g} wd={a.wd:g} epoch={epoch} loss={row['train_loss']:.5f} metaAP={ap:.5f} sec={row['elapsed_seconds']:.1f}", flush=True)
        if patience >= 8:
            break
    if a.sanity:
        if not rows or rows[-1]["train_loss"] >= rows[0]["train_loss"] or best_ap < 0.8:
            raise RuntimeError("Source-only subset overfit sanity failed; full study must not start")
        write_json(cell / "OVERFIT_SANITY_AUDIT.json", {"pass": True, "fold": a.fold,
                   "fit_source_patients": len(train), "initial_train_loss": rows[0]["train_loss"],
                   "final_train_loss": rows[-1]["train_loss"], "best_train_patient_equal_ez_ap": best_ap,
                   "target_outcome_access": False, "lock_sha256": lock_sha})


if __name__ == "__main__":
    main()
