"""Real-cache source-only DRST audit and FIT spectral moment fitting.

No validation/test outcome is indexed. Each fold normalizer uses all and only
that fold's FIT subjects, including the later source-only meta split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spectral import FixedScaleRawAlignmentStore, FrequencyMoments, log_power_stft


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    a = p.parse_args()
    if not a.lock.is_file():
        raise RuntimeError("Missing protocol lock")
    for key in ("R1_HLV_SOURCE_ROOT", "R1_HLV_WINDOW_CACHE", "R1_HLV_FIXED_MANIFEST", "R1_HLV_RUNTIME", "A1_A2_RUNTIME"):
        if key not in os.environ:
            raise RuntimeError(f"Missing source environment: {key}")
    source = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    sys.path[:0] = [str(source), str(source / "neuroez_c"),
                    str(source / "r1_hlv_ictal_dynamics_seed42_v1" / "code")]
    from neuroez_c.dual_view_data import RawAlignmentStore
    import exp_ez_hybrid as core
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args

    assert_sources()
    install_interleaved_hlv_view()
    args = make_args("R0", a.runtime / "scratch")
    exp = core.Exp_EZHybridLocalization(args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5:
        raise RuntimeError("Historical A1 cohort/folds changed")
    split = next(row for row in exp.outer_splits if int(row["fold_idx"]) == a.fold)
    fit, val, test = map(set, (split["fit_subjects"], split["validation_subjects"], split["test_subjects"]))
    if len(val) != 13 or len(fit | val | test) != 80 or fit & val or fit & test or val & test:
        raise RuntimeError("Frozen fold role violation")
    base = RawAlignmentStore(exp.run_records, feature_cache_path=os.environ["R1_HLV_WINDOW_CACHE"],
                             raw_cache_path=a.raw_cache, raw_target_samples=500, raw_target_sampling_rate=250.0)
    base.assert_formal_coverage(min_channel_match_rate=1, min_window_match_rate=1, expected_patients=80)
    alignment = {key: value for key, value in base.audit.items()
                 if key not in ("feature_cache_path", "raw_cache_path", "missing_raw_by_patient")}
    alignment.update({"fold": a.fold, "outer_test_accessed": False,
                      "independent_EDF_onset_reverification": False,
                      "label_or_outcome_indexed": False})
    write_json(a.runtime / f"fold_{a.fold}" / "RAW_ALIGNMENT_AUDIT.json", alignment)
    store = FixedScaleRawAlignmentStore(base)
    exp.raw_alignment_store = store
    exp.use_n6_dual_view_ema = True
    fit_dataset, _, _, _ = exp._build_datasets(sorted(fit), [], [])
    exp.use_n6_dual_view_ema = False
    loader = exp._make_loader(fit_dataset, shuffle=False, batch_size=1)
    seen = set()
    moments = FrequencyMoments()
    valid_windows = 0
    for batch in loader:
        sid = batch["subject_id"][0]
        if sid not in fit or sid in seen:
            raise RuntimeError("FIT-only patient identity violation")
        seen.add(sid)
        raw = batch["raw_windows"].float()
        mask = batch["raw_window_mask"].bool()
        selected = raw[mask]
        for chunk in selected.split(256):
            lp = log_power_stft(chunk)
            moments.update(lp, torch.ones(lp.shape[0], dtype=torch.bool))
            valid_windows += int(lp.shape[0])
        print(f"fold={a.fold} FIT patient {len(seen)}/{len(fit)} spectral windows={valid_windows}", flush=True)
    if seen != fit:
        raise RuntimeError("FIT loader did not cover exact patient set")
    mean, std = moments.finalize()
    result = {"fold": a.fold, "fit_patients": len(fit), "validation_patients_not_fit": len(val),
              "test_patients_not_fit": len(test), "valid_channel_windows": valid_windows,
              "local_STFT_frames_per_window": 12, "frequency_bins": 51,
              "observation_count_per_frequency": moments.count,
              "mean": mean.tolist(), "std": std.tolist(),
              "fit_only": True, "target_labels_indexed": False, "outer_test_accessed": False,
              "lock_sha256": file_sha(a.lock)}
    write_json(a.runtime / f"fold_{a.fold}" / "FIT_GLOBAL_SPECTRAL_NORMALIZER.json", result)
    print(f"NORMALIZER_PASS fold={a.fold} subjects={len(seen)} windows={valid_windows}", flush=True)


if __name__ == "__main__":
    run()
