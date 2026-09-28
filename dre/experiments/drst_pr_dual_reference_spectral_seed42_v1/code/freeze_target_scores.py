"""Generate and hash-freeze spectral target scores before indexing target labels.

The historical loader materializes target labels in its batch dictionary. This
script never reads those keys. The design is explicitly not blind storage.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import DRSTPatientModel
from run_training_grid import VARIANTS
from spectral import FixedScaleRawAlignmentStore
from stage0_source import file_sha, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    lock_sha = file_sha(a.lock)
    selection = json.loads((a.runtime / "FIT_SELECTION_PRIVATE.json").read_text(encoding="utf-8"))
    if selection["lock_sha256"] != lock_sha or len(selection["rows"]) != 20:
        raise RuntimeError("Complete source-only selection must precede target scoring")
    by_key = {(row["variant"], int(row["fold"])): row for row in selection["rows"]}
    source = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    sys.path[:0] = [str(source), str(source / "neuroez_c"),
                    str(source / "r1_hlv_ictal_dynamics_seed42_v1" / "code")]
    from neuroez_c.dual_view_data import RawAlignmentStore
    import exp_ez_hybrid as core
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args

    assert_sources()
    install_interleaved_hlv_view()
    exp = core.Exp_EZHybridLocalization(make_args("R0", a.runtime / "scratch"))
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5:
        raise RuntimeError("Frozen cohort changed")
    raw = RawAlignmentStore(exp.run_records, feature_cache_path=os.environ["R1_HLV_WINDOW_CACHE"],
                            raw_cache_path=a.raw_cache, raw_target_samples=500, raw_target_sampling_rate=250.0)
    raw.assert_formal_coverage(min_channel_match_rate=1, min_window_match_rate=1, expected_patients=80)
    exp.raw_alignment_store = FixedScaleRawAlignmentStore(raw)
    manifest = {}
    for split in exp.outer_splits:
        fold = int(split["fold_idx"])
        fit, val, test = map(set, (split["fit_subjects"], split["validation_subjects"], split["test_subjects"]))
        if len(val) != 13 or fit & val or test & (fit | val):
            raise RuntimeError("Frozen fold role mismatch")
        exp.use_n6_dual_view_ema = True
        _, target_set, _, _ = exp._build_datasets(sorted(fit), sorted(val), [])
        exp.use_n6_dual_view_ema = False
        loader = exp._make_loader(target_set, shuffle=False, batch_size=1)
        moments = json.loads((a.runtime / f"fold_{fold}" / "FIT_GLOBAL_SPECTRAL_NORMALIZER.json").read_text(encoding="utf-8"))
        if moments["lock_sha256"] != lock_sha or not moments["fit_only"]:
            raise RuntimeError("FIT normalizer changed")
        for variant in VARIANTS:
            entry = by_key[(variant, fold)]
            checkpoint = Path(entry["checkpoint"])
            if file_sha(checkpoint) != entry["checkpoint_sha256"]:
                raise RuntimeError("FIT-selected checkpoint changed")
            model = DRSTPatientModel(variant, torch.tensor(moments["mean"]), torch.tensor(moments["std"])).to(exp.device)
            state = torch.load(checkpoint, map_location=exp.device, weights_only=False)
            if state["lock_sha256"] != lock_sha or state["variant"] != variant or state["fold"] != fold:
                raise RuntimeError("FIT-selected checkpoint identity mismatch")
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            snapshot = {}
            with torch.no_grad():
                for raw_batch in loader:
                    # This transfer includes legacy label tensors, but the code
                    # below only reads subject ID, masks, and model scores.
                    batch = {k: v.to(exp.device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
                    out = model(batch)
                    mask = batch["channel_mask"][0].bool().detach().cpu().numpy()
                    sid = batch["subject_id"][0]
                    if sid not in val or sid in snapshot:
                        raise RuntimeError("Target score patient role violation")
                    scores = out["score_ez"][0].detach().cpu().numpy()[mask].astype(np.float32)
                    logits = out["logits"][0].detach().cpu().numpy()[mask].astype(np.float32)
                    if not np.isfinite(scores).all() or len(scores) == 0:
                        raise RuntimeError("Invalid target score vector")
                    snapshot[sid] = {"score_ez": scores, "logit_nez": logits, "n_channels": len(scores)}
            if set(snapshot) != val:
                raise RuntimeError("Incomplete target score set")
            destination = a.runtime / "frozen_target_scores" / variant / f"fold_{fold}.pkl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                with destination.open("rb") as stream:
                    prior = pickle.load(stream)
                if set(prior) != set(snapshot) or any(not np.array_equal(prior[sid]["score_ez"], snapshot[sid]["score_ez"]) for sid in snapshot):
                    raise RuntimeError("Refusing changed preexisting target scores")
            else:
                temporary = destination.with_suffix(".pkl.tmp")
                with temporary.open("wb") as stream:
                    pickle.dump(snapshot, stream, protocol=5)
                temporary.replace(destination)
            manifest[f"{variant}/fold_{fold}.pkl"] = file_sha(destination)
            print(f"SCORES_FROZEN {variant} fold={fold} n_patients={len(snapshot)}", flush=True)
    if len(manifest) != 20:
        raise RuntimeError("Expected 20 spectral target score files")
    private_manifest = a.runtime / "SPECTRAL_SCORE_FREEZE_PRIVATE.json"
    write_json(private_manifest, {"lock_sha256": lock_sha, "files": manifest,
                "target_label_keys_indexed": False, "outer_test_accessed": False})
    write_json(a.output, {"pass": True, "lock_sha256": lock_sha, "n_score_files": 20,
                "target_cells": 65, "target_labels_indexed_before_freeze": False,
                "legacy_loader_materialized_target_label_tensors": True,
                "private_manifest_sha256": file_sha(private_manifest), "outer_test_accessed": False})
    print("SPECTRAL_SCORE_FREEZE_PASS 20/20", flush=True)


if __name__ == "__main__":
    main()
