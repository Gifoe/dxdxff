from __future__ import annotations

from pathlib import Path


def write_code_audit(
    path: str | Path,
    *,
    feature_cache_path: str | Path | None = None,
    raw_cache_path: str | Path | None = None,
) -> None:
    feature_path = str(feature_cache_path) if feature_cache_path is not None else "<configured feature cache>"
    raw_path = str(raw_cache_path) if raw_cache_path is not None else "<configured raw cache>"
    content = f"""# Task 2 HiFOS-PACT code audit

## Existing call graph

`run_neuroez_c.py` applies `neuroez_c.config` defaults and constructs `Exp_EZHybridLocalization`. The experiment loads `ez_dataset.build_or_load_run_records` through `data_factory.data_provider`, builds shuffled patient KFold splits, flattens run samples, fits `neuroez_c.dataset` evidence normalizers on fit subjects, constructs label-bearing patient examples, and trains `NeuroEZCModel`. The model calls `WindowGraphSpectralEncoder`, `ChannelTemporalEncoder`, `CrossSeizureMILAggregator`, and `PatientChannelClassifier` to emit per-channel EZ/NEZ logits.

## Reusable cache and alignment code

- `ez_dataset._extract_run_records_from_cache_payload` correctly enforces the `run_records` plus `patient_index` top-level contract.
- `ez_dataset.flatten_window_samples` exposes feature arrays, channel names, centers, raw waveform metadata, and sanitized quality metadata, but also returns channel labels and therefore is not used as a Task 2 batch builder.
- `neuroez_c.dataset.build_patient_examples` contains a useful local-channel to patient-canonical-channel alignment pattern. Task 2 reimplements that pattern in `outcome_hifos.dataset` without labels, label masks, or EZ-derived counts and rejects duplicate normalized names.
- `scripts.fm_baselines` contains frozen BIOT, CBraMod, and LaBraM loading/preprocessing. `neuroez_c.raw_brainbert` and `raw_brainbert_data` contain the repository-native RawBrainBERT encoder and spectrogram patch path.

## Task 1 semantics that cannot enter Task 2

`neuroez_c.dataset.build_patient_examples`, `collate_patient_ez_batch`, `NeuroEZCModel`, `PatientChannelClassifier`, Task 1 ranking losses, LCBO/HNC teachers, negative-anchor heads, EZ-fraction filtering, center routers, and their output scores all carry EZ/NEZ or channel-level supervision. They are not imported by the Task 2 dataset/model/trainer.

## Leakage fields

The existing caches and batches may contain `labels`, `labels_ez`, `labels_nez`, `label_mask`, clinical EZ/NEZ/SOZ fields, resection fields, true-EZ counts/fractions, Task 1/V3/HNC/OOF channel scores, postoperative fields, outcome aliases, center, quality, and count metadata. Outcome aliases are read only by the target resolver. Model inputs are selected from an explicit whitelist and scanned after example construction, after collation, and at model forward.

## Why ordinary KFold is invalid

`data_factory.build_outer_splits` sorts subject IDs and applies shuffled `KFold`; it does not balance outcome, center, or center-by-outcome cells. With four-center outcome data this can create prevalence and center imbalance and even single-class folds. Task 2 writes a deterministic composite-stratified patient ledger and reuses its hash across variants.

## Why the channel classifier cannot be relabeled

The existing classifier emits one logit per channel, consumes channel-label-derived masks/counts, uses patient-relative channel normalization, and is optimized with channel-level BCE/ranking objectives. Surgery outcome is one patient target over all seizures, windows, and channels. Replacing the label would leave the sample unit, loss, masks, pooling, evaluation, and leakage semantics wrong.

## Foundation-model shapes

- BIOT adapter input: `[B,T]` or `[B,1,T]`; output: `[B,256]` under the repository default.
- CBraMod adapter input: `[B,T]` or `[B,1,T]`, reshaped internally to `[B,1,segments,points_per_patch]`; output: `[B,D_fm]` after replacing `proj_out` with identity.
- LaBraM adapter input: `[B,T]` or `[B,1,T]`; output: `[B,D_fm]` only when the external checkpoint supports safe single-contact SEEG input.
- RawBrainBERT input: spectrogram patches `[B,N,patch_dim]`; hidden output: `[B,N,D]`. Task 2 mean-pools patches only within one channel-window and retains `(subject,run,sample,window,channel)` in the disk manifest.

## Cache path finding

The task text named `all_windows_cache.pkl`, while the configured files are `{feature_path}` and `{raw_path}`. Library code receives both paths from configuration and does not hard-code either location.

## Added Task 2 files

The independent `outcome_hifos` package contains configuration, cache audit, outcome resolution, manifests, label-blind dataset/collation, leakage guards, patient folds, normalization, H1-H10 model/training components, frozen-FM adapters, reporting, and checkpoint/resume. `run_outcome_hifos.py`, `configs/outcome_hifos_*.yaml`, `scripts/run_outcome_*`, and `tests/outcome_hifos` provide the command and verification surfaces. Existing Task 1 entry points remain unchanged.
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


__all__ = ["write_code_audit"]
