"""Build S5-8 pseudo-core targets under an explicitly audited teacher protocol."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

S5_8_SIGNS_EZ = {
    "early_high_gamma_slope": 1,
    "early_line_length_slope": 1,
    "onset_latency_high_gamma": -1,
    "onset_latency_line_length": -1,
    # ez_features._rank_earlier_is_better assigns larger values to earlier channels.
    "onset_rank_high_gamma": 1,
    "onset_rank_line_length": 1,
    "high_gamma_top20pct_mean": 1,
    "line_length_top20pct_mean": 1,
}


def _norm_id(value: Any) -> str:
    return str(value).strip().lower()


def _norm_channel(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "_").replace(".", "_")


def _zscore(values: pd.Series) -> pd.Series:
    array = values.astype(float).to_numpy()
    std = float(array.std(ddof=0))
    if std <= 1e-8:
        return pd.Series(np.zeros_like(array), index=values.index)
    return pd.Series((array - float(array.mean())) / std, index=values.index)


def _q(scores: pd.Series, mass: float, tau: float) -> pd.Series:
    if scores.empty or mass <= 0.0:
        return pd.Series(0.0, index=scores.index)
    values = scores.astype(float).to_numpy()
    k = min(len(values), max(1, int(round(mass))))
    threshold = float(-np.partition(-values, k - 1)[k - 1])
    raw = 1.0 / (1.0 + np.exp(-(values - threshold) / max(float(tau), 1e-6)))
    scaled = np.clip(raw * (float(mass) / max(float(raw.sum()), 1e-12)), 0.0, 1.0)
    return pd.Series(scaled, index=scores.index)


def _teacher(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"fold_id", "center", "channel_name", "label_ez", "a9v3_oof_score"}
    if "subject_id" not in frame.columns and "patient_id" not in frame.columns:
        raise RuntimeError("Teacher must contain subject_id or patient_id.")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Teacher is missing real OOF fields: {missing}")
    if "subject_id" not in frame.columns:
        frame["subject_id"] = frame["patient_id"].astype(str)
    if "patient_id" not in frame.columns:
        frame["patient_id"] = frame["subject_id"].astype(str)
    frame = frame.drop_duplicates(["subject_id", "channel_name"]).copy()
    if len(frame) == 0 or not np.isfinite(frame["a9v3_oof_score"].astype(float)).all():
        raise RuntimeError("Teacher OOF scores are missing or non-finite.")
    return frame


def _cache_channel_features(cache_path: Path) -> tuple[pd.DataFrame, list[str]]:
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    names = [str(value) for value in payload.get("window_feature_names", [])]
    missing = [name for name in S5_8_SIGNS_EZ if name not in names]
    if missing:
        raise RuntimeError(f"S5-8 cache is missing features: {missing}")
    indices = [names.index(name) for name in S5_8_SIGNS_EZ]
    accum: dict[tuple[str, str], list[np.ndarray]] = {}
    display: dict[tuple[str, str], tuple[str, str]] = {}
    for record in payload.get("run_records", []):
        sample = record.get("sample", {}) if isinstance(record.get("sample"), dict) else record
        values = np.asarray(sample.get("window_features"), dtype=np.float32)
        channels = list(record.get("channel_names_norm", sample.get("channel_names_norm", [])))
        if values.ndim != 3 or values.shape[-1] != len(names) or len(channels) != values.shape[1]:
            raise RuntimeError(f"Invalid cache record {record.get('subject_id')} / {record.get('run_id')}")
        per_channel = values[:, :, indices].mean(axis=0)
        for channel_idx, channel in enumerate(channels):
            key = (_norm_id(record.get("subject_id")), _norm_channel(channel))
            accum.setdefault(key, []).append(per_channel[channel_idx].astype(np.float64))
            display[key] = (str(record.get("subject_id")), str(channel))
    rows = []
    for key, blocks in accum.items():
        row = {"_sid": key[0], "_channel": key[1], "subject_id": display[key][0], "channel_name": display[key][1]}
        mean = np.stack(blocks).mean(axis=0)
        row.update({name: float(mean[idx]) for idx, name in enumerate(S5_8_SIGNS_EZ)})
        rows.append(row)
    return pd.DataFrame(rows), names


def _cache_channel_metadata(cache_path: Path, *, random_seed: int = 42) -> pd.DataFrame:
    """Build labels/folds from the feature cache without reading any teacher file."""
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    patient_index = payload.get("patient_index", {})
    if not isinstance(patient_index, dict) or not patient_index:
        raise RuntimeError("Cache must contain patient_index for physiology-only targets.")
    subject_ids = sorted(str(value) for value in patient_index)
    if len(subject_ids) != 90:
        raise RuntimeError(f"Expected exactly 90 patients for physiology-only targets, got {len(subject_ids)}.")
    folds = {}
    kfold = KFold(n_splits=5, shuffle=True, random_state=int(random_seed))
    subject_array = np.asarray(subject_ids)
    for fold_id, (_, test_idx) in enumerate(kfold.split(subject_array), start=1):
        for idx in test_idx:
            folds[str(subject_array[idx])] = fold_id
    rows = []
    for subject_id in subject_ids:
        meta = patient_index[subject_id]
        channels = list(meta.get("canonical_channels", []))
        labels = np.asarray(meta.get("labels", []), dtype=float).reshape(-1)
        if len(channels) != len(labels):
            raise RuntimeError(f"Cache patient {subject_id} has inconsistent channels/labels.")
        center = str(meta.get("source_center", subject_id.split(":", 1)[0] if ":" in subject_id else "unknown"))
        for channel, label in zip(channels, labels):
            rows.append(
                {
                    "subject_id": subject_id,
                    "patient_id": subject_id,
                    "channel_name": str(channel),
                    "label_ez": float(label),
                    "fold_id": int(folds[subject_id]),
                    "center": center,
                }
            )
    return pd.DataFrame(rows)


def build(
    teacher_path: Path | None,
    cache_path: Path,
    output_dir: Path,
    *,
    target_semantics: str,
    rho_nez: float,
    tau_q: float,
    teacher_mode: str = "physiology_only",
    outer_fold: int | None = None,
    outer_train_subjects: Path | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    semantics = str(target_semantics).lower()
    if semantics not in {"ez", "nez"}:
        raise ValueError("target_semantics must be ez or nez")
    features, cache_names = _cache_channel_features(cache_path)
    mode = str(teacher_mode).lower()
    if mode not in {"physiology_only", "legacy_global_oof", "nested_outer_oof"}:
        raise ValueError("teacher_mode must be physiology_only, legacy_global_oof, or nested_outer_oof")
    if mode == "physiology_only":
        metadata = _cache_channel_metadata(cache_path, random_seed=random_seed)
        metadata["_sid"] = metadata["subject_id"].map(_norm_id)
        metadata["_channel"] = metadata["channel_name"].map(_norm_channel)
        merged = metadata.merge(
            features.drop(columns=["subject_id", "channel_name"]),
            on=["_sid", "_channel"],
            how="left",
            validate="one_to_one",
        )
        merged["a9v3_oof_score"] = np.nan
        merged["teacher_nez_score"] = np.nan
        merged["teacher_score_available"] = False
    else:
        if teacher_path is None:
            raise RuntimeError(f"teacher_mode={mode} requires --teacher-csv; no fallback is allowed.")
        teacher = _teacher(teacher_path)
        if mode == "nested_outer_oof":
            if outer_fold is None or outer_train_subjects is None:
                raise RuntimeError("nested_outer_oof requires --outer-fold and --outer-train-subjects.")
            train_subjects = {str(value).strip() for value in pd.read_csv(outer_train_subjects).iloc[:, 0].dropna()}
            observed_subjects = set(teacher["subject_id"].astype(str))
            if not observed_subjects.issubset(train_subjects) or observed_subjects != train_subjects:
                raise RuntimeError("Nested teacher must contain exactly the current outer-train patients.")
            if "outer_fold" in teacher.columns and bool((teacher["outer_fold"].astype(int) != int(outer_fold)).any()):
                raise RuntimeError("Nested teacher file contains rows for a different outer fold.")
            teacher["fold_id"] = 0
        teacher["_sid"] = teacher["subject_id"].map(_norm_id)
        teacher["_channel"] = teacher["channel_name"].map(_norm_channel)
        merged = teacher.merge(
            features.drop(columns=["subject_id", "channel_name"]),
            on=["_sid", "_channel"],
            how="left",
            validate="one_to_one",
        )
        merged["teacher_score_available"] = True
    matched = merged[list(S5_8_SIGNS_EZ)].notna().all(axis=1)
    if not bool(matched.all()):
        raise RuntimeError(f"S5-8 target match rate below 1.0: {int(matched.sum())}/{len(merged)}")
    if mode == "nested_outer_oof":
        if merged["subject_id"].nunique() == 0 or sorted(merged["fold_id"].astype(int).unique()) != [0]:
            raise RuntimeError("Nested teacher targets must contain only current outer-train rows with fold_id=0.")
    elif merged["subject_id"].nunique() != 90 or sorted(merged["fold_id"].astype(int).unique()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Expected exactly 90 patients and five OOF folds.")

    for name in S5_8_SIGNS_EZ:
        merged[f"z_{name}"] = merged.groupby("subject_id")[name].transform(_zscore)
    signed = [float(sign) * merged[f"z_{name}"] for name, sign in S5_8_SIGNS_EZ.items()]
    merged["phys_ez_score"] = sum(signed) / float(len(signed))
    merged["phys_nez_score"] = -merged["phys_ez_score"]
    merged["label_nez"] = 1.0 - merged["label_ez"].astype(float)
    if mode != "physiology_only":
        merged["teacher_nez_score"] = 1.0 - merged["a9v3_oof_score"].astype(float)
        merged["teacher_nez_score_z_patient"] = merged.groupby("subject_id")["teacher_nez_score"].transform(_zscore)
        merged["core_score_nez"] = merged["teacher_nez_score_z_patient"] + merged["phys_nez_score"]
        merged["teacher_ez_score_z_patient"] = merged.groupby("subject_id")["a9v3_oof_score"].transform(_zscore)
        merged["core_score_ez"] = merged["teacher_ez_score_z_patient"] + merged["phys_ez_score"]
    else:
        merged["core_score_nez"] = merged["phys_nez_score"]
        merged["core_score_ez"] = merged["phys_ez_score"]
    merged["pseudo_core_q"] = 0.0
    for _, group in merged.groupby("subject_id", sort=False):
        positive_label = "label_nez" if semantics == "nez" else "label_ez"
        core_column = "core_score_nez" if semantics == "nez" else "core_score_ez"
        positive = group.index[group[positive_label] > 0.5]
        if len(positive) == 0:
            continue
        mass = max(1.0, float(rho_nez) * float(len(positive)))
        merged.loc[positive, "pseudo_core_q"] = _q(merged.loc[positive, core_column], mass, tau_q)
    negative_label = "label_ez" if semantics == "nez" else "label_nez"
    if bool((merged.loc[merged[negative_label] > 0.5, "pseudo_core_q"] != 0.0).any()):
        raise RuntimeError(f"{semantics.upper()} pseudo-core q leaked onto negative channels.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "patient_id", "subject_id", "fold_id", "center", "channel_name", "label_ez", "label_nez",
        "pseudo_core_q", "a9v3_oof_score", "teacher_nez_score", "teacher_score_available",
        "phys_ez_score", "phys_nez_score", "core_score_ez", "core_score_nez",
    ]
    result = merged[output_columns].copy()
    result.to_csv(output_dir / "latent_core_targets_all_oof.csv", index=False)
    for fold_id in range(1, 6):
        train = result[result["fold_id"].astype(int) != fold_id]
        if bool((train["fold_id"].astype(int) == fold_id).any()):
            raise RuntimeError(f"Fold {fold_id} target leakage detected.")
        train.to_csv(output_dir / f"latent_core_targets_fold{fold_id}_train.csv", index=False)

    patient_summary = result.groupby(["subject_id", "fold_id", "center"], as_index=False).agg(
        n_channels=("channel_name", "size"), n_nez=("label_nez", "sum"), pseudo_core_mass=("pseudo_core_q", "sum")
    )
    center_summary = patient_summary.groupby("center", as_index=False).agg(
        n_patients=("subject_id", "nunique"), n_channels=("n_channels", "sum"), pseudo_core_mass=("pseudo_core_mass", "sum")
    )
    patient_summary.to_csv(output_dir / "latent_core_patient_summary.csv", index=False)
    center_summary.to_csv(output_dir / "latent_core_center_summary.csv", index=False)
    pd.DataFrame([
        {"feature": name, "ez_sign": sign, "nez_sign": -sign, "rank_semantics": "larger_is_earlier" if "rank" in name else ""}
        for name, sign in S5_8_SIGNS_EZ.items()
    ]).to_csv(output_dir / "phys_core_feature_audit.csv", index=False)
    audit = {
        "target_semantics": semantics,
        "teacher_mode": mode,
        "teacher_source": str(teacher_path) if teacher_path is not None else None,
        "cache_path": str(cache_path),
        "feature_mode": "S5_8_NO_HFO",
        "n_patients": int(result["subject_id"].nunique()),
        "n_channels": int(len(result)),
        "fold_ids": sorted(result["fold_id"].astype(int).unique().tolist()),
        "target_match_rate": float(matched.mean()),
        "rho": float(rho_nez),
        "tau_q": float(tau_q),
        "pseudo_core_only_on_positive_label": True,
        "teacher_is_real_oof": mode != "physiology_only",
        "teacher_used": mode != "physiology_only",
        "teacher_outer_fold_leak_free": mode in {"physiology_only", "nested_outer_oof"},
        "teacher_anchor_allowed": mode != "physiology_only",
        "selection_role": "exploratory" if mode == "legacy_global_oof" else "formal_primary_candidate",
        "formal_primary_eligible": mode != "legacy_global_oof",
        "formal_protocol_pass": mode != "legacy_global_oof",
        "formal_protocol_failure_reason": (
            "Legacy global OOF teacher is not nested within the current outer fold."
            if mode == "legacy_global_oof" else None
        ),
        "lambda_core_distill_effective": 0.0 if mode == "physiology_only" else None,
        "cache_feature_count": len(cache_names),
    }
    (output_dir / "latent_core_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-csv", type=Path, default=None)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-semantics", choices=["ez", "nez"], default="nez")
    parser.add_argument("--rho-nez", type=float, default=0.20)
    parser.add_argument("--tau-q", type=float, default=0.10)
    parser.add_argument("--teacher-mode", choices=["physiology_only", "legacy_global_oof", "nested_outer_oof"], default="physiology_only")
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--outer-train-subjects", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.teacher_csv, args.cache, args.output_dir, target_semantics=args.target_semantics, rho_nez=args.rho_nez, tau_q=args.tau_q, teacher_mode=args.teacher_mode, outer_fold=args.outer_fold, outer_train_subjects=args.outer_train_subjects, random_seed=args.random_seed), indent=2))


if __name__ == "__main__":
    main()
