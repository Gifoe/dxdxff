"""Input discovery, canonicalization, and strict P2/V3 fold alignment."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .p2_v3_conservative_fusion import bcr_ez_logit_to_nez_probability


STANDARD = ["subject_id", "center", "outer_fold", "channel_name", "label_nez", "label_ez", "score_nez", "score_ez", "source_model", "split_role"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_channel_name(value: object) -> str:
    text = str(value).strip().upper().replace("’", "'").replace("`", "'").replace("′", "'")
    return re.sub(r"\s+", "", text)


def _one_column(frame: pd.DataFrame, aliases: tuple[str, ...], *, required: bool = True) -> str | None:
    found = [name for name in aliases if name in frame.columns]
    if found:
        return found[0]
    if required:
        raise ValueError(f"Missing required column; accepted aliases: {aliases}")
    return None


def _as_probability(values: pd.Series, *, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if not np.isfinite(numeric.to_numpy()).all() or ((numeric < 0.0) | (numeric > 1.0)).any():
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return numeric


def _sigmoid(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("NEZ logit contains non-finite values")
    return 1.0 / (1.0 + np.exp(-numeric))


def canonicalize_fusion_ledger(frame: pd.DataFrame, *, source_model: str, split_role: str) -> pd.DataFrame:
    if split_role not in {"validation", "test"}:
        raise ValueError("split_role must be validation or test")
    subject = _one_column(frame, ("subject_id", "patient_id", "patient"))
    center = _one_column(frame, ("center", "center_id"), required=False)
    fold = _one_column(frame, ("outer_fold", "fold_idx", "fold"))
    channel = _one_column(frame, ("channel_name", "channel", "channel_id"))
    label_nez = _one_column(frame, ("label_nez", "true_nez", "nez_label"), required=False)
    label_ez = _one_column(frame, ("label_ez", "true_ez", "ez_label"), required=False)
    if label_nez is None and label_ez is None:
        raise ValueError("Ledger needs label_nez or label_ez")
    out = pd.DataFrame({
        "subject_id": frame[subject].astype(str).str.strip(),
        "center": frame[center].astype(str).str.strip().str.lower() if center else frame[subject].astype(str).str.split(":", n=1).str[0].str.lower(),
        "outer_fold": pd.to_numeric(frame[fold], errors="raise").astype(int),
        "channel_name": frame[channel].astype(str),
    })
    out["channel_key"] = out.channel_name.map(normalize_channel_name)
    out["label_nez"] = pd.to_numeric(frame[label_nez], errors="raise").astype(int) if label_nez else 1 - pd.to_numeric(frame[label_ez], errors="raise").astype(int)
    out["label_ez"] = pd.to_numeric(frame[label_ez], errors="raise").astype(int) if label_ez else 1 - out.label_nez
    if not out.label_nez.isin([0, 1]).all() or not out.label_ez.isin([0, 1]).all() or not (out.label_nez + out.label_ez == 1).all():
        raise ValueError("Labels must obey NEZ=1, EZ=0 exactly")
    nez_probability = _one_column(
        frame,
        ("score_nez", "score_nez_probability", "final_nez_score", "prob_nez", "predicted_nez_probability"),
        required=False,
    )
    ez_probability = _one_column(
        frame,
        ("score_ez", "score_ez_probability", "final_ez_score", "prob_ez", "predicted_ez_probability"),
        required=False,
    )
    nez_logit = _one_column(frame, ("nez_logit", "final_nez_logit", "base_nez_logit", "direct_nez_logit"), required=False)
    if nez_probability:
        out["score_nez"] = _as_probability(frame[nez_probability], field=nez_probability)
    elif ez_probability:
        out["score_nez"] = 1.0 - _as_probability(frame[ez_probability], field=ez_probability)
    elif nez_logit:
        out["score_nez"] = _sigmoid(frame[nez_logit])
    else:
        raise ValueError("Ledger needs a NEZ/EZ probability or a NEZ logit")
    out["score_ez"] = 1.0 - out.score_nez
    out["source_model"] = source_model
    out["split_role"] = split_role
    for optional in ("record_id", "channel_index", "global_channel_index", "channel_id", "n_records", "source_checkpoint", "source_run", "source_profile"):
        if optional in frame.columns and optional not in out.columns:
            out[optional] = frame[optional]
    key = ["subject_id", "outer_fold", "channel_key"]
    if out.duplicated(key).any():
        examples = out.loc[out.duplicated(key, keep=False), key].head(4).to_dict("records")
        raise ValueError(f"{source_model} {split_role} ledger has duplicate channel keys: {examples}")
    if (out.subject_id == "").any() or (out.channel_key == "").any():
        raise ValueError("Ledger contains empty subject_id or channel name")
    return out


def canonicalize_p2_fusion_ledger(
    frame: pd.DataFrame, *, split_role: str, source_model: str | None = None,
) -> pd.DataFrame:
    """Canonicalize P2 scores without inventing an unverified model identity."""
    identity = source_model
    if identity is None:
        for column in ("method_id", "source_model", "source_profile"):
            if column not in frame.columns:
                continue
            values = sorted(set(frame[column].dropna().astype(str).str.strip()) - {""})
            if len(values) == 1:
                identity = values[0]
                break
            if len(values) > 1:
                raise ValueError(f"P2 ledger contains multiple identities in {column}: {values}")
    return canonicalize_fusion_ledger(
        frame, source_model=identity or "UNVERIFIED_P2", split_role=split_role,
    )


def canonicalize_v3_fusion_ledger(frame: pd.DataFrame, *, split_role: str) -> pd.DataFrame:
    # BCR is trained with EZ-oriented logits.  Prefer the semantic logit when
    # present so the CDEL conversion remains explicitly 1 - sigmoid(e_BCR).
    if "ez_semantic_logit" in frame.columns:
        adapted = frame.copy()
        adapted["score_nez_probability"] = bcr_ez_logit_to_nez_probability(
            pd.to_numeric(adapted["ez_semantic_logit"], errors="raise").to_numpy()
        )
        return canonicalize_fusion_ledger(adapted, source_model="BCR_NET", split_role=split_role)
    return canonicalize_fusion_ledger(frame, source_model="BCR_NET", split_role=split_role)


def align_fold_ledgers(p2: pd.DataFrame, v3: pd.DataFrame, *, outer_fold: int, split_role: str) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    key = ["subject_id", "outer_fold", "channel_key"]
    merged = p2.merge(v3, on=key, how="outer", suffixes=("_p2", "_v3"), indicator=True)
    matched = merged[merged._merge.eq("both")].copy()
    audit = {
        "outer_fold": int(outer_fold), "split_role": split_role, "n_p2_rows": int(len(p2)), "n_v3_rows": int(len(v3)),
        "n_merged_rows": int(len(matched)), "n_p2_patients": int(p2.subject_id.nunique()), "n_v3_patients": int(v3.subject_id.nunique()),
        "n_merged_patients": int(matched.subject_id.nunique()), "p2_duplicate_keys": int(p2.duplicated(key).sum()), "v3_duplicate_keys": int(v3.duplicated(key).sum()),
        "p2_unmatched_rows": int((merged._merge == "left_only").sum()), "v3_unmatched_rows": int((merged._merge == "right_only").sum()),
        "label_nez_mismatch_count": int((matched.label_nez_p2 != matched.label_nez_v3).sum()),
        "label_ez_mismatch_count": int((matched.label_ez_p2 != matched.label_ez_v3).sum()),
        "center_mismatch_count": int((matched.center_p2 != matched.center_v3).sum()),
        "fold_mismatch_count": 0,
    }
    per_patient = matched.groupby("subject_id").size() if not matched.empty else pd.Series(dtype=int)
    audit["patient_channel_count_mismatch_count"] = int(
        (p2.groupby("subject_id").size().reindex(sorted(set(p2.subject_id) | set(v3.subject_id)), fill_value=-1)
         != v3.groupby("subject_id").size().reindex(sorted(set(p2.subject_id) | set(v3.subject_id)), fill_value=-1)).sum()
    )
    audit["passed"] = bool(
        len(p2) == len(v3) == len(matched) and audit["p2_unmatched_rows"] == audit["v3_unmatched_rows"] == 0
        and audit["label_nez_mismatch_count"] == audit["label_ez_mismatch_count"] == audit["center_mismatch_count"] == 0
        and audit["patient_channel_count_mismatch_count"] == 0 and (per_patient >= 2).all()
    )
    if not audit["passed"]:
        raise ValueError(f"P2/V3 alignment failed: {json.dumps(audit, sort_keys=True)}")
    output = pd.DataFrame({
        "subject_id": matched.subject_id, "center": matched.center_p2, "outer_fold": matched.outer_fold,
        "channel_name": matched.channel_name_p2, "channel_key": matched.channel_key,
        "label_nez": matched.label_nez_p2.astype(int), "label_ez": matched.label_ez_p2.astype(int),
        "p2_score_nez": matched.score_nez_p2.astype(float), "p2_score_ez": matched.score_ez_p2.astype(float),
        "v3_score_nez": matched.score_nez_v3.astype(float), "v3_score_ez": matched.score_ez_v3.astype(float),
        "split_role": split_role,
    })
    for optional in ("record_id", "channel_index", "global_channel_index", "channel_id", "n_records", "source_checkpoint", "source_run", "source_profile"):
        p2_name, v3_name = f"{optional}_p2", f"{optional}_v3"
        if p2_name in matched.columns:
            output[f"p2_{optional}"] = matched[p2_name]
        if v3_name in matched.columns:
            output[f"v3_{optional}"] = matched[v3_name]
    return output, audit, merged


def _path_from_manifest(manifest: dict[str, Any], fold: int, model: str, role: str) -> str | None:
    for item in manifest.get("folds", manifest.get("per_fold", [])):
        if int(item.get("outer_fold", item.get("fold", -1))) == fold:
            for key in (f"{model}_{role}_ledger_path", f"{model}_{role}_path", f"{role}_path"):
                if item.get(key):
                    return str(item[key])
    return None


def discover_fold_ledger(root: str | Path, fold: int, role: str, manifest: dict[str, Any] | None = None, model: str | None = None) -> Path:
    if manifest and model:
        explicit = _path_from_manifest(manifest, fold, model, role)
        if explicit:
            path = Path(explicit)
            if not path.is_file():
                raise FileNotFoundError(f"Manifest ledger does not exist: {path}")
            return path
    root = Path(root)
    stem = "val" if role == "validation" else "test"
    names = (f"{stem}_channel_predictions_neuroez_v2_fold_{fold}.csv", f"{role}_channel_predictions_fold_{fold}.csv")
    candidates = [root / f"fold_{fold}" / name for name in names] + [root / name for name in names]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {role} ledger for fold {fold} under {root}; found {found}")
    return found[0]


def read_subjects(path: str | Path) -> set[str]:
    frame = pd.read_csv(path)
    column = _one_column(frame, ("subject_id", "patient_id", "subject"))
    values = set(frame[column].astype(str))
    if not values:
        raise ValueError("Allowed-subject ledger is empty")
    return values


def validate_fold_manifest(path: str | Path, allowed_subjects: set[str], *, n_folds: int = 5) -> dict[int, set[str]]:
    frame = pd.read_csv(path)
    subject = _one_column(frame, ("subject_id", "patient_id", "subject"))
    fold = _one_column(frame, ("outer_fold", "fold_idx", "fold"))
    role = _one_column(frame, ("split", "partition", "role"), required=False)
    if role:
        frame = frame[frame[role].astype(str).str.lower().isin({"test", "outer_test", "heldout"})].copy()
    frame[fold] = pd.to_numeric(frame[fold], errors="raise").astype(int)
    if frame.duplicated([subject]).any():
        raise ValueError("Fixed fold manifest must contain every subject in exactly one outer test fold")
    result = {index: set(group[subject].astype(str)) for index, group in frame.groupby(fold)}
    if set(result) != set(range(1, n_folds + 1)) or set().union(*result.values()) != allowed_subjects:
        raise ValueError("Fixed fold manifest does not exactly cover allowed subjects once")
    return result


__all__ = [
    "STANDARD", "sha256_file", "normalize_channel_name", "canonicalize_fusion_ledger", "canonicalize_p2_fusion_ledger",
    "canonicalize_v3_fusion_ledger", "align_fold_ledgers", "discover_fold_ledger", "read_subjects", "validate_fold_manifest",
]
