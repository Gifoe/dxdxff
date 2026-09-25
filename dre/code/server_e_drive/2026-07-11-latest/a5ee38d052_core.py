"""Strict reusable components for the Task 1 cross-seizure experiment.

This module deliberately contains no optimiser, backward pass, or model update.
It filters cached run records *before* the existing patient/seizure aggregation is
called, so one/two-seizure results are real model inferences rather than edits to
all-seizure OOF probabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from neuroez_c.p2_v3_conservative_fusion import (
    LOCKED_BCR_WEIGHT,
    LOCKED_PRQ_WEIGHT,
    conservative_probability_fusion,
)
from neuroez_c.p2_v3_fusion_protocol import normalize_channel_name


SEEDS = (42, 52, 62)
SUBSAMPLE_SEEDS = tuple(range(202607240, 202607250))
MODELS = ("PRQ-Net", "BCR-Net", "CDEL")
FORMAL_THRESHOLD_SOURCE = "original_validation_only"
LABEL_SEMANTICS = "NEZ=1,EZ=0"
CROSS_SEIZURE_INFERENCE_VERSION = "v3_prq_causal_runtime"
REPRODUCTION_MEAN_ABS_TOLERANCE = 1e-5
REPRODUCTION_MAX_ABS_TOLERANCE = 2e-4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(values: Any) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def git_commit(repo: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def prepare_output_root(path: str | Path) -> dict[str, Path]:
    root = Path(path)
    keys = ("audit", "configs", "manifests", "predictions", "patient_metrics", "summaries", "figures", "tables", "logs", "checkpoints_reference", "reports")
    result = {"root": root}
    for key in keys:
        result[key] = root / key
        result[key].mkdir(parents=True, exist_ok=True)
    return result


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Unable to read CSV {path}: {exc}") from exc


def _find_column(frame: pd.DataFrame, aliases: Sequence[str], *, required: bool = True) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    if required:
        raise ValueError(f"Missing one of {list(aliases)} in {list(frame.columns)}")
    return None


def load_subjects_and_folds_by_seed(
    protocol_root: str | Path,
    seeds: Sequence[int],
) -> tuple[set[str], dict[int, dict[int, set[str]]], Path]:
    """Discover and validate one fixed five-fold partition for each seed."""
    root = Path(protocol_root)
    requested_seeds = tuple(int(seed) for seed in seeds)
    if not requested_seeds:
        raise ValueError("At least one training seed is required")
    candidates: list[tuple[int, Path, set[str], dict[int, dict[int, set[str]]]]] = []
    for path in root.rglob("*.csv"):
        name = path.name.lower()
        # These files are model outputs, even when they contain subject/fold
        # columns or happen to use "ledger" in their filename.
        if any(token in name for token in ("oof", "prediction", "summary", "metric", "by_patient", "by_channel")):
            continue
        try:
            frame = _read_csv(path)
            subject = _find_column(frame, ("subject_id", "patient_id", "subject"))
            fold = _find_column(frame, ("outer_fold", "fold_idx", "fold"))
        except ValueError:
            continue
        role = _find_column(
            frame,
            ("split_role", "partition", "split", "role"),
            required=False,
        )
        if role:
            selected = frame[
                frame[role].astype(str).str.strip().str.lower().isin(("test", "outer_test", "heldout"))
            ]
        else:
            # A result table often has subject_id + outer_fold but it is not a
            # frozen split definition.  Accept role-less files only when their
            # filename explicitly identifies a manifest or fold ledger.
            if not any(token in name for token in ("manifest", "fold_ledger", "outer_fold", "fixed_folds")):
                continue
            selected = frame
        if selected.empty:
            continue
        selected = selected.copy()
        selected[subject] = selected[subject].astype(str)
        selected[fold] = pd.to_numeric(selected[fold], errors="coerce")
        seed_column = _find_column(
            selected,
            ("training_seed", "model_seed", "random_seed", "seed"),
            required=False,
        )
        if seed_column is not None:
            selected[seed_column] = pd.to_numeric(selected[seed_column], errors="coerce")

        fold_maps: dict[int, dict[int, set[str]]] = {}
        cohort: set[str] | None = None
        candidate_valid = True
        for training_seed in requested_seeds:
            seed_frame = (
                selected[selected[seed_column].eq(training_seed)].copy()
                if seed_column is not None
                else selected.copy()
            )
            if seed_frame.empty or set(seed_frame[fold].dropna().astype(int)) != {1, 2, 3, 4, 5}:
                candidate_valid = False
                break
            # Duplicate rows across methods are legal within a seed, but one
            # patient cannot be an outer-test patient in two folds for that seed.
            subject_fold_counts = seed_frame.dropna(subset=[fold]).groupby(subject)[fold].nunique()
            if (subject_fold_counts > 1).any():
                conflicts = subject_fold_counts[subject_fold_counts > 1].index.astype(str).tolist()
                raise ValueError(
                    f"Fixed manifest assigns patients to multiple outer test folds "
                    f"for training_seed={training_seed}: {conflicts[:10]}"
                )
            seed_frame = seed_frame.drop_duplicates([subject, fold]).copy()
            seed_subjects = set(seed_frame[subject].astype(str))
            if len(seed_subjects) != 80:
                candidate_valid = False
                break
            if cohort is None:
                cohort = seed_subjects
            elif seed_subjects != cohort:
                candidate_valid = False
                break
            fold_map = {
                int(index): set(group[subject].astype(str))
                for index, group in seed_frame.groupby(fold)
            }
            if set(fold_map) != {1, 2, 3, 4, 5} or sum(map(len, fold_map.values())) != 80:
                candidate_valid = False
                break
            fold_maps[training_seed] = fold_map
        if not candidate_valid or cohort is None:
            continue
        if name == "fixed_partition_manifest.csv":
            priority = 0
        elif "audit" in {part.lower() for part in path.parts} and "fixed" in name and "manifest" in name:
            priority = 1
        elif "fixed" in name and "manifest" in name:
            priority = 2
        elif role:
            priority = 3
        else:
            priority = 4
        candidates.append((priority, path, cohort, fold_maps))
    if not candidates:
        raise FileNotFoundError(
            f"No fixed 80-patient 5-fold manifest covers training seeds "
            f"{list(requested_seeds)} under {root}"
        )
    best_priority = min(item[0] for item in candidates)
    preferred = [item for item in candidates if item[0] == best_priority]
    if len(preferred) > 1:
        mappings = []
        for _, candidate_path, _, candidate_maps in preferred:
            mapping = tuple(
                (
                    training_seed,
                    tuple(sorted(
                        (subject_id, fold_id)
                        for fold_id, subjects in candidate_maps[training_seed].items()
                        for subject_id in subjects
                    )),
                )
                for training_seed in requested_seeds
            )
            mappings.append((candidate_path, mapping))
        if any(mapping != mappings[0][1] for _, mapping in mappings[1:]):
            paths = [str(path) for path, _ in mappings]
            raise RuntimeError(f"Conflicting seed-specific fixed manifests at equal priority: {paths}")
        preferred.sort(key=lambda item: str(item[1]).lower())
    _, path, subjects, fold_maps = preferred[0]
    return subjects, fold_maps, path


def load_subjects_and_folds(
    protocol_root: str | Path,
    training_seed: int = 42,
) -> tuple[set[str], dict[int, set[str]], Path]:
    """Backward-compatible single-seed view of the fixed partition."""
    subjects, fold_maps, path = load_subjects_and_folds_by_seed(
        protocol_root,
        [training_seed],
    )
    return subjects, fold_maps[int(training_seed)], path


def load_fit_subjects_by_seed(
    manifest_path: str | Path,
    seeds: Sequence[int],
) -> dict[int, dict[int, set[str]]]:
    """Load the exact fit-only patients used to estimate each fold normalizer."""
    frame = _read_csv(Path(manifest_path))
    subject = _find_column(frame, ("subject_id", "patient_id", "subject"))
    fold = _find_column(frame, ("outer_fold", "fold_idx", "fold"))
    role = _find_column(frame, ("split_role", "partition", "split", "role"))
    seed_column = _find_column(
        frame,
        ("training_seed", "model_seed", "random_seed", "seed"),
        required=False,
    )
    normalized_role = frame[role].astype(str).str.strip().str.lower()
    fit_rows = frame[normalized_role.eq("fit")].copy()
    if fit_rows.empty:
        raise ValueError(
            f"Fixed partition manifest has no explicit fit rows: {manifest_path}"
        )
    result: dict[int, dict[int, set[str]]] = {}
    for seed in map(int, seeds):
        selected = fit_rows
        if seed_column is not None:
            numeric_seed = pd.to_numeric(fit_rows[seed_column], errors="coerce")
            selected = fit_rows[numeric_seed.eq(seed)]
        if selected.empty:
            raise ValueError(
                f"Fixed partition manifest has no fit rows for seed={seed}"
            )
        fold_map = {
            int(fold_id): set(group[subject].astype(str))
            for fold_id, group in selected.groupby(fold)
        }
        if set(fold_map) != {1, 2, 3, 4, 5}:
            raise ValueError(
                f"Fit partition does not cover all five folds for seed={seed}"
            )
        result[seed] = fold_map
    return result


def load_cache(cache_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    path = Path(cache_path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("run_records"), list) or not isinstance(payload.get("patient_index"), dict):
        raise ValueError("Cache must contain dict fields run_records and patient_index")
    return payload["run_records"], payload["patient_index"]


def seizure_id(record: dict[str, Any]) -> str:
    sample = record.get("sample", {}) if isinstance(record.get("sample"), dict) else {}
    value = sample.get("source_seizure_id", record.get("source_seizure_id"))
    # run_id is the cache's lossless record identifier and is the only permitted fallback.
    return str(value if value not in (None, "") else record.get("run_id", ""))


def patient_seizure_map(run_records: Iterable[dict[str, Any]], allowed_subjects: set[str]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {subject: [] for subject in sorted(allowed_subjects)}
    for record in run_records:
        subject = str(record.get("subject_id", ""))
        if subject not in values:
            continue
        sid = seizure_id(record)
        if not sid:
            raise ValueError(f"Empty seizure/run identifier for {subject}")
        values[subject].append(sid)
    for subject, ids in values.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate seizure/run IDs for {subject}; cache cannot distinguish seizures")
        if not ids:
            raise ValueError(f"No valid seizures for {subject}")
        values[subject] = sorted(ids)
    return values


def sample_seizure_subset(subject_id: str, available_ids: Sequence[str], *, mode: str, repeat_id: int, subsample_seed: int) -> list[str]:
    """Pure ID-based deterministic sampler; labels and scores are intentionally absent."""
    ids = sorted({str(value) for value in available_ids})
    if mode == "all":
        return ids
    needed = 1 if mode == "one" else 2 if mode == "two" else None
    if needed is None:
        raise ValueError(f"Unsupported seizure mode {mode!r}")
    if len(ids) < needed:
        return []
    digest = int(hashlib.sha256(f"{subsample_seed}|{repeat_id}|{subject_id}".encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(digest)
    return sorted(rng.sample(ids, needed))


def build_subset_manifest(
    seizures: dict[str, list[str]],
    fold_map: dict[int, set[str]] | dict[int, dict[int, set[str]]],
    seeds: Sequence[int],
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for training_seed in seeds:
        seed_fold_map = (
            fold_map[int(training_seed)]
            if training_seed in fold_map and isinstance(fold_map[training_seed], dict)
            else fold_map
        )
        subject_fold = {
            subject: fold
            for fold, subjects in seed_fold_map.items()
            for subject in subjects
        }
        if set(subject_fold) != set(seizures):
            raise ValueError(
                f"Seed {training_seed} fold manifest does not exactly cover the seizure cohort"
            )
        for mode in ("all", "one", "two"):
            mode_repeats = 1 if mode == "all" else repeats
            for repeat_id in range(mode_repeats):
                subsample_seed = SUBSAMPLE_SEEDS[repeat_id] if repeat_id < len(SUBSAMPLE_SEEDS) else 202607240 + repeat_id
                for subject, available in sorted(seizures.items()):
                    selected = sample_seizure_subset(subject, available, mode=mode, repeat_id=repeat_id, subsample_seed=subsample_seed)
                    rows.append({
                        "training_seed": int(training_seed), "outer_fold": int(subject_fold[subject]), "subject_id": subject,
                        "seizure_mode": mode, "repeat_id": int(repeat_id), "subsample_seed": int(subsample_seed),
                        "selected_seizure_ids": json.dumps(selected), "total_available_seizures": int(len(available)),
                        "n_selected_seizures": int(len(selected)), "eligible": bool(selected),
                    })
    return pd.DataFrame(rows)


def discover_checkpoint(root: str | Path, seed: int, fold: int, kind: str) -> Path:
    root_path = Path(root)
    seed_dir = root_path / f"seed_{seed}"
    if kind == "PRQ-Net":
        candidates = list((seed_dir / "p2_q10" / f"fold_{fold}").glob("best_model.pt"))
    else:
        candidates = list((seed_dir / "bcr_boundary_coverage" / f"fold_{fold}").glob("best_bcr_boundary_coverage.pt"))
        if not candidates:
            candidates = [p for p in (seed_dir / "bcr_boundary_coverage" / f"fold_{fold}").glob("*.pt") if "bcr" in p.name.lower()]
    if len(candidates) != 1:
        # A recursive fall-back is used only after directory metadata was checked by the audit.
        name = "best_model.pt" if kind == "PRQ-Net" else "best_bcr_boundary_coverage.pt"
        candidates = [p for p in root_path.rglob(name) if f"seed_{seed}" in str(p) and f"fold_{fold}" in str(p)]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {kind} checkpoint for seed={seed}, fold={fold}; found={candidates}")
    return candidates[0]


def discover_config(checkpoint: Path) -> Path:
    for parent in (checkpoint.parent, *checkpoint.parents):
        for name in ("effective_config.json", "run_args_b0_pruned.json", "run_args.json", "run_args_p23.json"):
            candidate = parent / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No effective/run args JSON above {checkpoint}")


def discover_threshold_file(root: str | Path, seed: int, *, model: str) -> Path:
    root_path = Path(root)
    candidates = [
        root_path / "thresholds" / "selected_thresholds.csv",
        root_path.parent / "thresholds" / "selected_thresholds.csv",
        root_path / "fold_thresholds.csv",
        root_path / "metrics" / "fold_thresholds.csv",
        root_path / "p23_fold_summary.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            read_thresholds(path, model=model)
        except (KeyError, ValueError):
            continue
        return path
    raise FileNotFoundError(
        f"No deterministic validation-threshold table for {model}, seed={seed}; "
        f"checked={candidates}"
    )


def read_thresholds(path: str | Path, *, model: str | None = None) -> dict[int, float]:
    frame = _read_csv(Path(path))
    model_column = _find_column(frame, ("model", "method", "experiment"), required=False)
    if model_column is not None and model is not None:
        aliases = {
            "PRQ-Net": {"prq-net", "p2-q10", "prq", "p2"},
            "BCR-Net": {"bcr-net", "v3-qbc", "bcr", "v3"},
            "CDEL": {"cdel", "p2-q10+v3-qbc", "fusion"},
        }
        accepted = aliases.get(model)
        if accepted is None:
            raise ValueError(f"Unsupported threshold model: {model}")
        normalized = frame[model_column].astype(str).str.strip().str.lower()
        frame = frame[normalized.isin(accepted)].copy()
        if frame.empty:
            raise ValueError(f"Threshold table {path} has no rows for {model}")
    fold = _find_column(frame, ("outer_fold", "fold_idx", "fold"))
    threshold = _find_column(frame, ("threshold", "classification_threshold", "selected_threshold"))
    if frame[fold].astype(int).duplicated().any():
        raise ValueError(f"Threshold table has duplicate folds for {model}: {path}")
    values = {int(key): float(value) for key, value in zip(frame[fold], frame[threshold])}
    if set(values) != {1, 2, 3, 4, 5} or not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
        raise ValueError(f"Invalid validation-only threshold table: {path}")
    return values


def _default_args(config: dict[str, Any], *, model_kind: str) -> argparse.Namespace:
    if model_kind == "PRQ-Net":
        from P23_TRN_NEZ_80.neuroez_c.config import apply_pruned_defaults
        from P23_TRN_NEZ_80.run_neuroez_c import build_parser
    elif model_kind == "BCR-Net":
        from neuroez_c.config import apply_pruned_defaults
        from run_neuroez_c import build_parser
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    args = build_parser().parse_args([])
    for key, value in config.items():
        setattr(args, key, value)
    apply_pruned_defaults(args)
    return args


def checkpoint_normalizer(payload: dict[str, Any]):
    from neuroez_c.evidence_views import EvidenceNormalizer, MultiViewEvidenceNormalizer

    mean = np.asarray(payload.get("normalizer_mean"), dtype=np.float32)
    std = np.asarray(payload.get("normalizer_std"), dtype=np.float32)
    physics_mean = payload.get("normalizer_physics_mean")
    physics_std = payload.get("normalizer_physics_std")
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError("Checkpoint lacks valid fold normalizer mean/std")
    b0 = EvidenceNormalizer(mean, np.maximum(std, 1e-6))
    if physics_mean is None or physics_std is None:
        return b0
    return MultiViewEvidenceNormalizer(
        b0=b0,
        physics=EvidenceNormalizer(np.asarray(physics_mean, dtype=np.float32), np.maximum(np.asarray(physics_std, dtype=np.float32), 1e-6)),
    )


def fit_model_normalizer(
    checkpoint: Path,
    *,
    model_kind: str,
    run_records: Sequence[dict[str, Any]],
    fit_subjects: Sequence[str],
):
    """Recreate the original fold normalizer from frozen fit patients only."""
    from ez_dataset import flatten_window_samples

    config = read_json(discover_config(checkpoint))
    args = _default_args(config, model_kind=model_kind)
    fit_samples = flatten_window_samples(
        run_records,
        subject_ids=sorted(map(str, fit_subjects)),
    )
    if not fit_samples:
        raise ValueError(
            f"No fit samples available to reconstruct normalizer for {checkpoint}"
        )
    if model_kind == "PRQ-Net":
        from P23_TRN_NEZ_80.neuroez_c.dataset import (
            fit_window_tensor_normalizer,
        )
    elif model_kind == "BCR-Net":
        from neuroez_c.dataset import fit_window_tensor_normalizer
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    return fit_window_tensor_normalizer(fit_samples, args=args)


def resolve_model_normalizer(
    checkpoint: Path,
    *,
    model_kind: str,
    run_records: Sequence[dict[str, Any]],
    fit_subjects: Sequence[str],
):
    """Prefer persisted fold statistics and otherwise rebuild from fit only."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("normalizer_mean") is not None and payload.get("normalizer_std") is not None:
        return checkpoint_normalizer(payload)
    return fit_model_normalizer(
        checkpoint,
        model_kind=model_kind,
        run_records=run_records,
        fit_subjects=fit_subjects,
    )


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    args: argparse.Namespace
    normalizer: Any
    checkpoint: Path
    model_kind: str
    causal_propagation_store: Any | None = None


def load_model(
    checkpoint: Path,
    *,
    device: str,
    model_kind: str,
    normalizer: Any | None = None,
):
    import torch

    if model_kind == "PRQ-Net":
        from P23_TRN_NEZ_80.neuroez_c.model import NeuroEZCModel
    elif model_kind == "BCR-Net":
        from neuroez_c.model import NeuroEZCModel
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint lacks model_state_dict: {checkpoint}")
    config = read_json(discover_config(checkpoint))
    args = _default_args(config, model_kind=model_kind)
    model = NeuroEZCModel(args)
    result = model.load_state_dict(payload["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint compatibility result for {checkpoint}: {result}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if normalizer is None:
        normalizer = checkpoint_normalizer(payload)
    causal_propagation_store = None
    if bool(getattr(args, "use_causal_propagation_residual", False)):
        if model_kind == "PRQ-Net":
            from P23_TRN_NEZ_80.neuroez_c.causal_propagation_features import (
                CausalPropagationFeatureStore,
            )
        else:
            from neuroez_c.causal_propagation_features import (
                CausalPropagationFeatureStore,
            )
        causal_propagation_store = CausalPropagationFeatureStore(
            getattr(args, "causal_propagation_cache_path"),
            getattr(args, "causal_cache_audit_path"),
            require_passed=True,
        )
    return LoadedModel(
        model=model,
        args=args,
        normalizer=normalizer,
        checkpoint=checkpoint,
        model_kind=model_kind,
        causal_propagation_store=causal_propagation_store,
    )


def selected_samples(run_records: Iterable[dict[str, Any]], subject_id: str, selected_ids: Sequence[str]) -> list[dict[str, Any]]:
    selected = set(map(str, selected_ids))
    rows = [record for record in run_records if str(record.get("subject_id")) == subject_id and seizure_id(record) in selected]
    if len(rows) != len(selected):
        actual = {seizure_id(record) for record in rows}
        raise ValueError(f"Selected seizures not uniquely present for {subject_id}: expected={selected}, actual={actual}")
    return rows


def infer_patient(loaded: LoadedModel, records: Sequence[dict[str, Any]], patient_index: dict[str, dict[str, Any]], *, device: str, amp: bool) -> tuple[dict[str, Any], np.ndarray]:
    """Use existing tensor construction and original model forward exactly once."""
    import torch
    from ez_dataset import flatten_window_samples
    if loaded.model_kind == "PRQ-Net":
        from P23_TRN_NEZ_80.neuroez_c.dataset import build_patient_examples, collate_patient_ez_batch
    else:
        from neuroez_c.dataset import build_patient_examples, collate_patient_ez_batch

    if not records:
        raise ValueError("No records supplied for inference")
    subject = str(records[0]["subject_id"])
    samples = flatten_window_samples(records, subject_ids=[subject])
    examples = build_patient_examples(
        samples,
        patient_index,
        normalizer=loaded.normalizer,
        subject_ids=[subject],
        args=loaded.args,
        causal_propagation_store=loaded.causal_propagation_store,
    )
    if len(examples) != 1:
        raise RuntimeError(f"Expected exactly one patient example for {subject}, got {len(examples)}")
    batch = collate_patient_ez_batch(examples)
    from exp_ez_hybrid import _move_tensors_to_device
    batch = _move_tensors_to_device(batch, torch.device(device))
    if str(device).startswith("cuda"):
        # Keep each historical runtime's matrix precision isolated. PRQ was
        # trained with TF32 enabled; BCR reference inference used strict FP32.
        use_tf32 = loaded.model_kind == "PRQ-Net"
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
    autocast = torch.autocast(device_type="cuda", enabled=bool(amp and str(device).startswith("cuda")))
    with torch.inference_mode(), autocast:
        output = loaded.model(batch)
    score = output.get("score_nez")
    if score is None:
        raise KeyError("Original model forward did not expose score_nez")
    mask = batch["channel_mask"][0].detach().cpu().numpy().astype(bool)
    labels = batch["labels_nez"][0].detach().cpu().numpy()
    values = score[0].detach().float().cpu().numpy()
    channels = list(batch["canonical_channels"][0])
    if not np.isfinite(values[mask]).all():
        raise ValueError(f"Non-finite P(NEZ) output for {subject}")
    result = {
        "subject_id": subject, "center": str(batch["center"][0]), "channel_name": [channels[i] for i in np.flatnonzero(mask)],
        "label_nez": labels[mask].astype(int), "score_nez": values[mask].astype(float),
    }
    return result, np.asarray(batch["run_ids"][0], dtype=object)


def make_prediction_rows(prq: dict[str, Any], bcr: dict[str, Any], *, seed: int, fold: int, mode: str, repeat_id: int, selected_ids: Sequence[str], total: int, thresholds: dict[str, float]) -> pd.DataFrame:
    keys_prq = [(str(s), normalize_channel_name(c)) for s, c in zip([prq["subject_id"]] * len(prq["channel_name"]), prq["channel_name"])]
    keys_bcr = [(str(s), normalize_channel_name(c)) for s, c in zip([bcr["subject_id"]] * len(bcr["channel_name"]), bcr["channel_name"])]
    if keys_prq != keys_bcr or not np.array_equal(prq["label_nez"], bcr["label_nez"]):
        raise ValueError(f"PRQ/BCR channel alignment failed for {prq['subject_id']}")
    prq_score = np.asarray(prq["score_nez"], dtype=float)
    bcr_score = np.asarray(bcr["score_nez"], dtype=float)
    cdel_score, _ = conservative_probability_fusion(prq_score, bcr_score, p2_weight=LOCKED_PRQ_WEIGHT, v3_weight=LOCKED_BCR_WEIGHT)
    label_nez = np.asarray(prq["label_nez"], dtype=int)
    base: dict[str, Any] = {
        "training_seed": int(seed), "outer_fold": int(fold), "subject_id": prq["subject_id"], "center": prq["center"],
        "channel_name": prq["channel_name"], "label_nez": label_nez, "label_ez": 1 - label_nez,
        "seizure_mode": mode, "repeat_id": int(repeat_id), "selected_seizure_ids": json.dumps(list(selected_ids)),
        "total_available_seizures": int(total), "n_selected_seizures": int(len(selected_ids)),
        "prq_score_nez": prq_score, "prq_score_ez": 1.0 - prq_score,
        "bcr_score_nez": bcr_score, "bcr_score_ez": 1.0 - bcr_score,
        "cdel_score_nez": cdel_score, "cdel_score_ez": 1.0 - cdel_score,
        "prq_threshold": float(thresholds["prq"]), "bcr_threshold": float(thresholds["bcr"]), "cdel_threshold": float(thresholds["cdel"]),
        "threshold_source": FORMAL_THRESHOLD_SOURCE, "threshold_seed": int(seed), "threshold_fold": int(fold),
    }
    for name, score in (("prq", prq_score), ("bcr", bcr_score), ("cdel", cdel_score)):
        prediction = (score >= float(thresholds[name])).astype(int)
        base[f"{name}_pred_nez"] = prediction
        base[f"{name}_pred_ez"] = 1 - prediction
    return pd.DataFrame(base)


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    required = {
        "training_seed", "outer_fold", "subject_id", "center", "channel_name", "label_nez", "label_ez", "prq_score_nez", "bcr_score_nez", "cdel_score_nez",
        "prq_threshold", "bcr_threshold", "cdel_threshold", "threshold_source",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file missing fields: {missing}")
    if not set(pd.to_numeric(frame.label_nez)).issubset({0, 1}) or not np.array_equal((1 - pd.to_numeric(frame.label_nez)).to_numpy(), pd.to_numeric(frame.label_ez).to_numpy()):
        raise ValueError("Required label semantics are NEZ=1 and EZ=0")
    if not frame.threshold_source.eq(FORMAL_THRESHOLD_SOURCE).all():
        raise ValueError("Thresholds must come from original validation only")
    for column in ("prq_score_nez", "bcr_score_nez", "cdel_score_nez"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must be finite P(NEZ) in [0,1]")


def patient_metrics(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

    prefix = {"PRQ-Net": "prq", "BCR-Net": "bcr", "CDEL": "cdel"}[model]
    rows: list[dict[str, Any]] = []
    group_keys = ["training_seed", "outer_fold", "subject_id", "center", "seizure_mode", "repeat_id"]
    for key, group in frame.groupby(group_keys, sort=True):
        y_nez = group.label_nez.to_numpy(int)
        p_nez = group[f"{prefix}_score_nez"].to_numpy(float)
        prediction = group[f"{prefix}_pred_nez"].to_numpy(int)
        y_ez = 1 - y_nez
        score_ez = 1.0 - p_nez
        reason = ""
        try:
            ez_auprc = float(average_precision_score(y_ez, score_ez))
            ez_auroc = float(roc_auc_score(y_ez, score_ez)) if len(np.unique(y_ez)) == 2 else float("nan")
        except ValueError as exc:
            ez_auprc, ez_auroc, reason = float("nan"), float("nan"), str(exc)
        order = np.argsort(-score_ez, kind="mergesort")
        ranks = np.flatnonzero(y_ez[order] == 1)
        mrr = float(1.0 / (ranks[0] + 1)) if len(ranks) else float("nan")
        gains = y_ez[order].astype(float)
        discount = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        ideal = np.sort(y_ez)[::-1].astype(float)
        ndcg = float(np.sum(gains * discount) / np.sum(ideal * discount)) if np.sum(ideal * discount) > 0 else float("nan")
        k = int(y_ez.sum())
        selected = np.zeros_like(y_ez)
        selected[order[:k]] = 1
        rows.append({
            **dict(zip(group_keys, key)), "model": model, "n_channels": int(len(group)), "undefined_reason": reason,
            "patient_macro_f1": float(f1_score(y_nez, prediction, labels=[0, 1], average="macro", zero_division=0)),
            "patient_ez_f1": float(f1_score(y_nez, prediction, labels=[0], average="macro", zero_division=0)),
            "patient_nez_f1": float(f1_score(y_nez, prediction, labels=[1], average="macro", zero_division=0)),
            "patient_accuracy": float(accuracy_score(y_nez, prediction)), "patient_ez_auprc": ez_auprc, "patient_ez_auroc": ez_auroc,
            "patient_ndcg_ez": ndcg, "patient_mrr_ez": mrr,
            "recall_at_true_ez_count": float((selected[y_ez == 1].sum() / k) if k else float("nan")),
            "precision_at_true_ez_count": float((selected[y_ez == 1].sum() / max(k, 1))),
            "ez_fraction_bias": float(selected.mean() - y_ez.mean()),
        })
    return pd.DataFrame(rows)


def aggregate_metrics(patient_frame: pd.DataFrame, *, primary_subjects: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = patient_frame[patient_frame.subject_id.isin(primary_subjects or set())].copy()
    available = patient_frame.copy()
    metric_columns = [column for column in patient_frame.columns if column.startswith("patient_") or column in {"recall_at_true_ez_count", "precision_at_true_ez_count", "ez_fraction_bias"}]

    def grouped(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        rows = []
        for keys, group in frame.groupby(["model", "seizure_mode", "training_seed", "repeat_id"], sort=True):
            row = dict(zip(["model", "seizure_mode", "training_seed", "repeat_id"], keys))
            row["cohort"] = label
            row["n_patients"] = int(group.subject_id.nunique())
            for metric in metric_columns:
                row[metric] = float(group[metric].mean(skipna=True))
            rows.append(row)
        return pd.DataFrame(rows)
    return grouped(primary, "primary_matched"), grouped(available, "available")


def summarize_seed_repeat(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in frame.columns if column.startswith("patient_") or column in {"recall_at_true_ez_count", "precision_at_true_ez_count", "ez_fraction_bias"}]
    rows = []
    for keys, group in frame.groupby(["cohort", "model", "seizure_mode", "training_seed"], sort=True):
        row = dict(zip(["cohort", "model", "seizure_mode", "training_seed"], keys))
        for metric in metric_columns:
            row[f"{metric}_repeat_mean"] = float(group[metric].mean(skipna=True))
            row[f"{metric}_repeat_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def cohort_seed_summary(seed_repeat: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in seed_repeat.columns if column.endswith("_repeat_mean")]
    rows = []
    for keys, group in seed_repeat.groupby(["cohort", "model", "seizure_mode"], sort=True):
        row = dict(zip(["cohort", "model", "seizure_mode"], keys))
        row["n_training_seeds"] = int(group.training_seed.nunique())
        for metric in metric_columns:
            base = metric.removesuffix("_repeat_mean")
            row[f"{base}_mean"] = float(group[metric].mean())
            row[f"{base}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(patient_frame: pd.DataFrame, *, repeats: int = 10_000, seed: int = 42) -> pd.DataFrame:
    primary_subjects = set(patient_frame.loc[patient_frame.seizure_mode.eq("two"), "subject_id"])
    frame = patient_frame[patient_frame.subject_id.isin(primary_subjects)].copy()
    # average repeats and training seeds per patient first, as stipulated.
    average = frame.groupby(["model", "seizure_mode", "subject_id"], as_index=False).mean(numeric_only=True)
    comparisons = (("CDEL", "all", "one"), ("CDEL", "all", "two"), ("PRQ-Net", "all", "one"), ("BCR-Net", "all", "one"))
    metrics = ("patient_macro_f1", "patient_ez_f1", "patient_ez_auprc", "patient_ndcg_ez")
    rng = np.random.default_rng(seed)
    rows = []
    for model, left, right in comparisons:
        a = average[(average.model == model) & (average.seizure_mode == left)].set_index("subject_id")
        b = average[(average.model == model) & (average.seizure_mode == right)].set_index("subject_id")
        subjects = sorted(set(a.index) & set(b.index))
        if not subjects:
            continue
        for metric in metrics:
            delta = (a.loc[subjects, metric] - b.loc[subjects, metric]).to_numpy(float)
            samples = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(repeats)])
            rows.append({"model": model, "comparison": f"{left}_minus_{right}", "metric": metric, "n_patients": len(delta),
                         "mean_delta": float(delta.mean()), "ci_2_5": float(np.quantile(samples, .025)), "ci_97_5": float(np.quantile(samples, .975)),
                         "probability_delta_gt_zero": float(np.mean(samples > 0)), "bootstrap_repeats": int(repeats), "seed": int(seed)})
    return pd.DataFrame(rows)


def compare_all_seizure(new: pd.DataFrame, reference: pd.DataFrame, *, model: str) -> dict[str, Any]:
    key = ["training_seed", "outer_fold", "subject_id", "channel_name"]
    new = new.copy(); reference = reference.copy()
    new["channel_name"] = new.channel_name.map(normalize_channel_name)
    reference["channel_name"] = reference.channel_name.map(normalize_channel_name)
    prefix = {"PRQ-Net": "prq", "BCR-Net": "bcr", "CDEL": "cdel"}[model]
    score_candidates = [f"{prefix}_score_nez", "score_nez", "p_pos", "fused_score_nez"]
    ref_score = next((column for column in score_candidates if column in reference.columns), None)
    if ref_score is None:
        raise ValueError(f"Reference ledger has no recognized NEZ score for {model}")
    merged = new.merge(reference, on=key, how="outer", suffixes=("_new", "_ref"), indicator=True)
    matched = merged[merged._merge.eq("both")]
    error = np.abs(pd.to_numeric(matched[f"{prefix}_score_nez"], errors="raise") - pd.to_numeric(matched[ref_score], errors="raise"))
    threshold_col = next((column for column in ("selected_threshold", "classification_threshold", "threshold") if column in reference.columns), None)
    threshold_mismatch = 0
    prediction_mismatch = 0
    patient_metric_mismatch = 0
    if threshold_col:
        threshold_mismatch = int((~np.isclose(pd.to_numeric(matched[f"{prefix}_threshold"], errors="raise"), pd.to_numeric(matched[threshold_col], errors="raise"), atol=1e-12)).sum())
        ref_pred = (pd.to_numeric(matched[ref_score], errors="raise") >= pd.to_numeric(matched[threshold_col], errors="raise")).astype(int)
        prediction_mismatch = int((ref_pred.to_numpy() != matched[f"{prefix}_pred_nez"].to_numpy(int)).sum())
        from sklearn.metrics import f1_score
        for _, group in matched.assign(_ref_pred=ref_pred.to_numpy()).groupby("subject_id", sort=True):
            y = pd.to_numeric(group["label_nez_new"], errors="raise").to_numpy(int)
            current_f1 = f1_score(y, group[f"{prefix}_pred_nez"].to_numpy(int), labels=[0, 1], average="macro", zero_division=0)
            reference_f1 = f1_score(y, group["_ref_pred"].to_numpy(int), labels=[0, 1], average="macro", zero_division=0)
            if not np.isclose(current_f1, reference_f1, atol=1e-12):
                patient_metric_mismatch += 1
    label_mismatch = int((pd.to_numeric(matched.get("label_nez_new"), errors="raise") != pd.to_numeric(matched.get("label_nez_ref"), errors="raise")).sum())
    center_mismatch = int((matched.get("center_new").astype(str) != matched.get("center_ref").astype(str)).sum())
    mean_error = float(error.mean()) if len(error) else float("inf")
    max_error = float(error.max()) if len(error) else float("inf")
    return {"model": model, "n_new": int(len(new)), "n_reference": int(len(reference)), "n_matched": int(len(matched)),
            "unmatched": int((merged._merge != "both").sum()), "mean_abs_probability_error": float(error.mean()) if len(error) else float("inf"),
            "max_abs_probability_error": float(error.max()) if len(error) else float("inf"), "threshold_mismatch_count": threshold_mismatch,
            "prediction_mismatch_count": prediction_mismatch, "patient_metric_mismatch_count": patient_metric_mismatch, "label_mismatch_count": label_mismatch, "center_mismatch_count": center_mismatch,
            "mean_abs_tolerance": REPRODUCTION_MEAN_ABS_TOLERANCE,
            "max_abs_tolerance": REPRODUCTION_MAX_ABS_TOLERANCE,
            "status": "PASS" if (
                len(new) == len(reference) == len(matched)
                and mean_error <= REPRODUCTION_MEAN_ABS_TOLERANCE
                and max_error <= REPRODUCTION_MAX_ABS_TOLERANCE
                and threshold_mismatch == prediction_mismatch == patient_metric_mismatch == label_mismatch == center_mismatch == 0
            ) else "FAIL"}
