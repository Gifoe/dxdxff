"""Inference-only M2/M3 ablations after primary scores/checkpoints are frozen.

All patient/channel rows stay in the private runtime. Ablations are diagnostic,
not model-selection candidates; no parameter is updated or new model trained.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from evaluate_stage_b import LOCK_SHA, dependencies, score_path, sha, snap


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def query_ap(afc, y: np.ndarray, margin: np.ndarray, fold: int, sid: str) -> tuple[float, int]:
    values = np.asarray([afc.query_metrics(y[q], margin[q])["ap"] for rep in range(20)
                         for _, q in [afc.split_indices(len(y), 42, fold, sid, rep)]], dtype=float)
    return mean(values), int(np.isfinite(values).sum())


def query_weighted_mean(rows: list[dict], field: str) -> float:
    valid = [r for r in rows if np.isfinite(r[field]) and r["n_estimable_query_ap"] > 0]
    return float(sum(r[field] * r["n_estimable_query_ap"] for r in valid) /
                 sum(r["n_estimable_query_ap"] for r in valid))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    a = p.parse_args()
    audit = json.loads((a.output / "SCORE_FREEZE_AUDIT.json").read_text(encoding="utf-8"))
    if not audit["pass"] or audit["lock_sha256"] != LOCK_SHA:
        raise RuntimeError("Primary score freeze must precede posthoc interventions")
    required = ("R1_HLV_SOURCE_ROOT", "R1_HLV_WINDOW_CACHE", "R1_HLV_FIXED_MANIFEST",
                "R1_HLV_RUNTIME", "A1_A2_RUNTIME")
    if any(not os.environ.get(key) for key in required):
        raise RuntimeError("Missing source runtime paths")
    root = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "neuroez_c"))
    sys.path.insert(0, str(root / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
    from models import HybridA1RawModel, RawTinyPatientModel
    import exp_ez_hybrid as core
    from exp_ez_hybrid import _move_tensors_to_device as move
    from neuroez_c.dual_view_data import RawAlignmentStore
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args

    afc, _, _, _ = dependencies(root / "rawtiny_patient_relative_seed42_v1")
    assert_sources()
    install_interleaved_hlv_view()
    private = []
    for fold in range(1, 6):
        model_args = make_args("R0", a.runtime / "diagnostic_scratch" / f"fold_{fold}")
        exp = core.Exp_EZHybridLocalization(model_args)
        split = next(s for s in exp.outer_splits if int(s["fold_idx"]) == fold)
        fit, val = list(split["fit_subjects"]), list(split["validation_subjects"])
        store = RawAlignmentStore(exp.run_records,
                                  feature_cache_path=os.environ["R1_HLV_WINDOW_CACHE"],
                                  raw_cache_path=a.raw_cache, raw_target_samples=500,
                                  raw_target_sampling_rate=250.0)
        store.assert_formal_coverage(min_channel_match_rate=1, min_window_match_rate=1,
                                     expected_patients=80)
        exp.raw_alignment_store = store
        exp.use_n6_dual_view_ema = True
        train_set, val_set, _, _ = exp._build_datasets(fit, val, [])
        exp.use_n6_dual_view_ema = False
        train_loader = exp._make_loader(train_set, shuffle=False, batch_size=2)
        val_loader = exp._make_loader(val_set, shuffle=False, batch_size=1)
        base = exp.runtime["model_cls"](model_args).to(exp.device)
        exp._dry_initialize_lazy_layers(base, train_loader)
        source = torch.load(Path(os.environ["A1_A2_RUNTIME"]) / "A1" /
                            f"fold_{fold}" / "epoch_30.pt", map_location=exp.device,
                            weights_only=False)
        base.load_state_dict(source["model_state_dict"], strict=True)
        models = {"M2_RAWTINY_PR": RawTinyPatientModel(patient_relative=True).to(exp.device),
                  "M3_HYBRID_PR": HybridA1RawModel(base).to(exp.device)}
        selection = {variant: {r["subject_id"]: int(r["selected_epoch"]) for r in
                               csv_rows(a.runtime / "private" / variant /
                                        f"fold_{fold}_VLOO_PATIENT_PRIVATE.csv")}
                     for variant in models}
        if any(set(row) != set(val) for row in selection.values()):
            raise RuntimeError("Frozen VLOO selection membership changed")
        for raw in val_loader:
            batch = move(raw, exp.device)
            sid = batch["subject_id"][0]
            mask = batch["channel_mask"][0].bool()
            y = batch["labels_ez"][0][mask].detach().cpu().numpy().astype(np.int8)
            for variant, model in models.items():
                epoch = selection[variant][sid]
                ckpt = a.runtime / variant / f"fold_{fold}" / f"epoch_{epoch:02d}.pt"
                state = torch.load(ckpt, map_location=exp.device, weights_only=False)
                if (state["lock_sha"] != LOCK_SHA or state["variant"] != variant or
                        state["fold"] != fold or state["epoch"] != epoch):
                    raise RuntimeError("Diagnostic checkpoint identity mismatch")
                model.load_state_dict(state["model"], strict=True)
                model.eval()
                engineered_norm = []
                if variant == "M3_HYBRID_PR":
                    def capture(_module, _inputs, output):
                        engineered_norm.append(float(output.square().mean().sqrt()))
                    hook = model.a1.b0_encoder.register_forward_hook(capture)
                with torch.no_grad():
                    output = model(batch)
                    if variant == "M3_HYBRID_PR":
                        hook.remove()
                    full = output["logits"][0][mask]
                    frozen = np.asarray(snap(str(a.runtime), variant, fold, epoch)[sid]["logit_nez"])
                    error = float(np.max(np.abs(full.detach().cpu().numpy() - frozen)))
                    if error > 1e-4:
                        raise RuntimeError(f"Frozen primary score mismatch {variant} fold={fold}: {error}")
                    u = output["patient_channel_embedding"]
                    head = model.head if variant == "M2_RAWTINY_PR" else model.pr_residual
                    if variant == "M2_RAWTINY_PR":
                        logits_z = head(u, batch["channel_mask"], zero_z=True)[0][mask]
                        logits_rank = head(u, batch["channel_mask"], zero_rank=True)[0][mask]
                    else:
                        correction = head(u, batch["channel_mask"])[0][mask]
                        logits_z = full - correction + head(u, batch["channel_mask"], zero_z=True)[0][mask]
                        logits_rank = full - correction + head(u, batch["channel_mask"], zero_rank=True)[0][mask]
                    margins = {"full": -full.detach().cpu().numpy(),
                               "zero_z": -logits_z.detach().cpu().numpy(),
                               "zero_rank": -logits_rank.detach().cpu().numpy()}
                    if variant == "M3_HYBRID_PR":
                        base0 = model.a1(batch)
                        alpha0 = (base0["logits"] +
                                  head(base0["patient_channel_embedding"], batch["channel_mask"]))[0][mask]
                        margins["alpha_zero"] = -alpha0.detach().cpu().numpy()
                        alpha = float(model.alpha.detach().abs())
                        raw_norm = float(output["raw_residual_norm"].detach())
                        ratio = raw_norm / max(engineered_norm[0], 1e-12)
                    else:
                        alpha = raw_norm = ratio = float("nan")
                ap = {name: query_ap(afc, y, margin, fold, sid) for name, margin in margins.items()}
                counts = {count for _, count in ap.values()}
                if len(counts) != 1:
                    raise RuntimeError("Ablations changed estimable-query membership")
                private.append({"variant": variant, "fold": fold, "sid": sid,
                                "epoch": epoch, "alpha_abs": alpha,
                                "raw_residual_rms": raw_norm, "raw_to_engineered_rms_ratio": ratio,
                                "n_estimable_query_ap": next(iter(counts)),
                                **{f"ap_{name}": value for name, (value, _) in ap.items()}})
            print(f"[DIAG] fold={fold} patient={len([x for x in private if x['fold']==fold])//2}/13", flush=True)
    if len(private) != 130:
        raise RuntimeError("Expected 65 target cells x 2 intervention models")
    # No identifiers or channel records leave the private runtime.
    private_path = a.runtime / "private" / "POSTHOC_INTERVENTION_PATIENT_ROWS.csv"
    write_csv(private_path, private)
    intervention, utilization = [], []
    for variant in models:
        part = [r for r in private if r["variant"] == variant]
        for kind in ("zero_z", "zero_rank"):
            intervention.append({"model": variant, "inference_only_ablation": kind,
                                 "n_target_cells": len(part), "full_ap": query_weighted_mean(part, "ap_full"),
                                 "ablated_ap": query_weighted_mean(part, f"ap_{kind}"),
                                 "delta_ablated_minus_full": query_weighted_mean(part, f"ap_{kind}") - query_weighted_mean(part, "ap_full"),
                                 "retrained": False, "model_selected_from_ablation": False})
    hybrid = [r for r in private if r["variant"] == "M3_HYBRID_PR"]
    utilization.append({"model": "M3_HYBRID_PR", "n_target_cells": len(hybrid),
                        "mean_abs_alpha": mean(r["alpha_abs"] for r in hybrid),
                        "mean_raw_residual_rms": mean(r["raw_residual_rms"] for r in hybrid),
                        "mean_raw_to_engineered_rms_ratio": mean(r["raw_to_engineered_rms_ratio"] for r in hybrid),
                        "full_ap": query_weighted_mean(hybrid, "ap_full"),
                        "alpha_zero_ap": query_weighted_mean(hybrid, "ap_alpha_zero"),
                        "delta_alpha_zero_minus_full": query_weighted_mean(hybrid, "ap_alpha_zero") - query_weighted_mean(hybrid, "ap_full"),
                        "retrained": False, "model_selected_from_ablation": False})
    write_csv(a.output / "PATIENT_RELATIVE_INTERVENTION.csv", intervention)
    write_csv(a.output / "HYBRID_RAW_UTILIZATION.csv", utilization)
    print("POSTHOC_INTERVENTIONS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
