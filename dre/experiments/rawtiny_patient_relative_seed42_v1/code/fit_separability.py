"""FIT-only, posthoc linear separability diagnostic on frozen fold-1 models.

The source checkpoints already trained on all fold-FIT patients. Consequently
these probes are descriptive, not independent generalization estimates. No
validation target or outer-test labels are accessed and no model is selected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from evaluate_stage_b import LOCK_SHA


def probe(train_x, train_y, query_x):
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500,
                               random_state=42).fit(scaler.transform(train_x), train_y)
    return model.predict_proba(scaler.transform(query_x))[:, 1]


def mean(values):
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    a = p.parse_args()
    freeze = json.loads((a.output / "SCORE_FREEZE_AUDIT.json").read_text(encoding="utf-8"))
    if not freeze["pass"] or freeze["lock_sha256"] != LOCK_SHA:
        raise RuntimeError("Primary scores must be frozen")
    root = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "neuroez_c"))
    sys.path.insert(0, str(root / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
    import exp_ez_hybrid as core
    from exp_ez_hybrid import _move_tensors_to_device as move
    from neuroez_c.dual_view_data import RawAlignmentStore
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args
    from models import HybridA1RawModel, RawTinyPatientModel

    assert_sources()
    install_interleaved_hlv_view()
    args = make_args("R0", a.runtime / "separability_scratch")
    exp = core.Exp_EZHybridLocalization(args)
    split = next(s for s in exp.outer_splits if int(s["fold_idx"]) == 1)
    fit = list(split["fit_subjects"])
    if len(fit) != 51:
        raise RuntimeError("Fold-1 FIT cohort changed")
    store = RawAlignmentStore(exp.run_records,
                              feature_cache_path=os.environ["R1_HLV_WINDOW_CACHE"],
                              raw_cache_path=a.raw_cache, raw_target_samples=500,
                              raw_target_sampling_rate=250.0)
    store.assert_formal_coverage(min_channel_match_rate=1, min_window_match_rate=1,
                                 expected_patients=80)
    exp.raw_alignment_store = store
    exp.use_n6_dual_view_ema = True
    dataset, _, _, _ = exp._build_datasets(fit, fit, [])
    exp.use_n6_dual_view_ema = False
    loader = exp._make_loader(dataset, shuffle=False, batch_size=1)
    base = exp.runtime["model_cls"](args).to(exp.device)
    exp._dry_initialize_lazy_layers(base, loader)
    a1 = torch.load(Path(os.environ["A1_A2_RUNTIME"]) / "A1" / "fold_1" / "epoch_30.pt",
                    map_location=exp.device, weights_only=False)
    base.load_state_dict(a1["model_state_dict"], strict=True)
    variants = ("M1_RAWTINY_NOPR", "M2_RAWTINY_PR", "M3_HYBRID_PR")
    records = defaultdict(lambda: defaultdict(dict))
    for variant in variants:
        if variant == variants[0]:
            model = RawTinyPatientModel(patient_relative=False).to(exp.device)
        elif variant == variants[1]:
            model = RawTinyPatientModel(patient_relative=True).to(exp.device)
        else:
            model = HybridA1RawModel(base).to(exp.device)
        state = torch.load(a.runtime / variant / "fold_1" / "epoch_30.pt",
                           map_location=exp.device, weights_only=False)
        if state["lock_sha"] != LOCK_SHA or state["variant"] != variant or state["fold"] != 1:
            raise RuntimeError("Frozen FIT representation checkpoint changed")
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        head = None if variant == variants[0] else (model.head if variant == variants[1] else model.pr_residual)
        for raw in loader:
            batch = move(raw, exp.device)
            sid = batch["subject_id"][0]
            mask = batch["channel_mask"][0].bool()
            y = batch["labels_ez"][0][mask].detach().cpu().numpy().astype(int)
            captured = []
            if head is not None:
                hook = head.norm.register_forward_hook(lambda _m, _x, output: captured.append(output.detach()))
            with torch.no_grad():
                out = model(batch)
            if head is not None:
                hook.remove()
            raw_e = out["raw_window_embedding"]  # [B,S,W,C,32]
            valid = (batch["window_mask"].bool().unsqueeze(-1) &
                     batch["raw_seizure_channel_mask"].bool().unsqueeze(2))
            count = valid.sum(dim=(1, 2)).clamp_min(1)
            channel_window = (raw_e * valid.unsqueeze(-1)).sum(dim=(1, 2)) / count.unsqueeze(-1)
            stages = {"raw_window_mean": channel_window[0][mask],
                      "cross_seizure": out["patient_channel_embedding"][0][mask]}
            if head is not None:
                if len(captured) != 1:
                    raise RuntimeError("Post-PR embedding capture failed")
                stages["post_patient_relative"] = captured[0][0][mask]
            for stage, embedding in stages.items():
                x = embedding.detach().cpu().numpy().astype(np.float64)
                if len(x) != len(y) or not np.isfinite(x).all():
                    raise RuntimeError("Invalid FIT representation")
                records[variant][stage][sid] = (x, y)
            print(f"[FIT_REPR] {variant} patient={len(records[variant]['raw_window_mean'])}/51", flush=True)
    output = []
    # Deterministic 80/20 FIT patient split for the shared linear probe.
    ordered = sorted(fit, key=lambda sid: hashlib.sha256(f"42|repr|{sid}".encode()).digest())
    fit_train, fit_query = ordered[11:], ordered[:11]
    for variant, stages in records.items():
        for stage, patients in stages.items():
            train_x = np.concatenate([patients[sid][0] for sid in fit_train])
            train_y = np.concatenate([patients[sid][1] for sid in fit_train])
            if len(np.unique(train_y)) != 2:
                raise RuntimeError("Shared FIT probe lacks both classes")
            shared = []
            specific = []
            for sid in fit_query:
                x, y = patients[sid]
                if len(np.unique(y)) < 2:
                    continue
                pred = probe(train_x, train_y, x)
                shared.append(average_precision_score(y, pred))
                order = np.argsort([hashlib.sha256(f"42|repr-channel|{sid}|{i}".encode()).digest()
                                    for i in range(len(y))])
                support = order[:len(order)//2]
                query = order[len(order)//2:]
                if len(np.unique(y[support])) != 2 or len(np.unique(y[query])) != 2:
                    continue
                own = probe(x[support], y[support], x[query])
                specific.append(average_precision_score(y[query], own))
            output.append({"model": variant, "representation_stage": stage,
                           "fold": 1, "source_fit_patients": 51,
                           "shared_probe_train_patients": len(fit_train),
                           "shared_probe_query_patients": len(shared),
                           "patient_specific_probe_patients": len(specific),
                           "shared_linear_ez_ap": mean(shared),
                           "patient_specific_linear_ez_ap": mean(specific),
                           "patient_specific_minus_shared_ap":
                           (mean(specific) - mean(shared)) if shared and specific else None,
                           "patient_specific_uses_own_FIT_labels": True,
                           "independent_generalization_estimate": False,
                           "target_or_outer_labels_used": False})
    with (a.output / "REPRESENTATION_SEPARABILITY.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print("FIT_SEPARABILITY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
