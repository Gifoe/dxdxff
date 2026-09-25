"""V3 continuous-score orientation, strict alignment and robust standardization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .raw_brainbert_data import normalize_channel_name


def probability_to_logit(probability: np.ndarray | pd.Series) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def orient_v3_ledger(frame: pd.DataFrame, score_semantics: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return an explicit P(NEZ) ledger; orientation is never inferred from values."""
    result = frame.copy()
    semantics_values = set(result["score_semantics"].dropna().astype(str)) if "score_semantics" in result else set()
    if score_semantics is None:
        if len(semantics_values) != 1:
            raise ValueError("V3 score orientation is ambiguous; provide explicit score_semantics")
        score_semantics = semantics_values.pop()
    normalized_semantics = str(score_semantics).strip().lower().replace(" ", "")
    if normalized_semantics in {"p(ez)", "ez_probability", "ez"}:
        source = next((name for name in ("v3_score_ez", "score_ez_probability", "a9v3_oof_score", "score_ez") if name in result), None)
        if source is None:
            raise ValueError("P(EZ) V3 ledger lacks an EZ probability column")
        score_nez = 1.0 - pd.to_numeric(result[source], errors="raise").to_numpy(dtype=float)
        transform = f"1-{source}"
    elif normalized_semantics in {"p(nez)", "nez_probability", "nez"}:
        source = next((name for name in ("v3_score_nez", "score_nez_probability", "final_score_nez", "score_nez") if name in result), None)
        if source is None:
            raise ValueError("P(NEZ) V3 ledger lacks a NEZ probability column")
        score_nez = pd.to_numeric(result[source], errors="raise").to_numpy(dtype=float)
        transform = source
    else:
        raise ValueError(f"Unsupported explicit V3 score semantics: {score_semantics!r}")
    if not np.isfinite(score_nez).all() or np.any((score_nez < 0.0) | (score_nez > 1.0)):
        raise ValueError("V3 probabilities must be finite and in [0,1]")
    result["v3_score_nez"] = score_nez
    result["v3_logit_nez"] = probability_to_logit(score_nez)
    result["score_semantics"] = "P(NEZ)"
    return result, {
        "input_score_semantics": str(score_semantics),
        "output_score_semantics": "P(NEZ)",
        "orientation_transform": transform,
    }


def robust_patient_standardize_v3(
    v3_logit_nez: torch.Tensor,
    channel_mask: torch.Tensor,
    v3_anchor_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if v3_logit_nez.ndim != 2 or channel_mask.shape != v3_logit_nez.shape or v3_anchor_valid.shape != v3_logit_nez.shape:
        raise ValueError("V3 standardization expects aligned [B,C] tensors")
    valid = channel_mask.bool() & v3_anchor_valid.bool()
    standardized = torch.zeros_like(v3_logit_nez)
    medians = torch.zeros(v3_logit_nez.shape[0], device=v3_logit_nez.device, dtype=v3_logit_nez.dtype)
    iqrs = torch.zeros_like(medians)
    for patient_idx in range(v3_logit_nez.shape[0]):
        selected = v3_logit_nez[patient_idx, valid[patient_idx]]
        if selected.numel() == 0:
            continue
        median = torch.quantile(selected, 0.50)
        q25 = torch.quantile(selected, 0.25)
        q75 = torch.quantile(selected, 0.75)
        iqr = q75 - q25
        standardized[patient_idx, valid[patient_idx]] = ((selected - median) / iqr.clamp_min(1e-3)).clamp(-5.0, 5.0)
        medians[patient_idx] = median
        iqrs[patient_idx] = iqr
    return {
        "v3_anchor_standardized": standardized,
        "v3_anchor_median": medians,
        "v3_anchor_iqr": iqrs,
        "v3_anchor_valid": valid,
    }


class V3AnchorStore:
    """Exact canonical-key provider for screening or provenance-complete nested anchors."""

    def __init__(
        self,
        ledger_path: str | Path,
        *,
        mode: str,
        score_semantics: str | None = None,
        expected_patients: int = 80,
        allowed_subjects: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(ledger_path)
        self.mode = str(mode).lower()
        if self.mode not in {"precomputed_oof_screening", "nested_fold_safe"}:
            raise ValueError(f"Unsupported V3 anchor mode: {mode}")
        if self.path.is_dir():
            files = sorted(self.path.glob("outer_fold_*/inner_oof_channel_scores.csv"))
            files += sorted(self.path.glob("outer_fold_*/outer_test_channel_scores.csv"))
            if not files:
                raise FileNotFoundError(f"No split-aware V3 anchor ledgers found under {self.path}")
            frames = []
            for source in files:
                frame = pd.read_csv(source)
                role = "inner_oof" if source.name.startswith("inner_oof") else "outer_test"
                if "split_role" not in frame:
                    frame["split_role"] = role
                if "outer_fold" not in frame:
                    try:
                        frame["outer_fold"] = int(source.parent.name.rsplit("_", 1)[-1])
                    except ValueError as exc:
                        raise ValueError(f"Cannot infer outer fold from {source}") from exc
                frames.append(frame)
            raw = pd.concat(frames, ignore_index=True)
            source_files = files
            source_audits = sorted(self.path.glob("outer_fold_*/v3_anchor_audit.json"))
            if self.mode == "nested_fold_safe":
                expected_dirs = {path.parent for path in files}
                if {path.parent for path in source_audits} != expected_dirs:
                    raise FileNotFoundError("Every strict V3 outer-fold directory requires v3_anchor_audit.json")
                for audit_path in source_audits:
                    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
                    if str(audit_payload.get("status", "")).lower() != "passed":
                        raise ValueError(f"Strict V3 split audit did not pass: {audit_path}")
        else:
            raw = pd.read_csv(self.path)
            source_files = [self.path]
            source_audits = []
        aliases = {"fold_id": "outer_fold", "fold_idx": "outer_fold", "patient_id": "subject_id"}
        for source, target in aliases.items():
            if target not in raw and source in raw:
                raw[target] = raw[source]
        required = {"subject_id", "channel_name"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"V3 anchor ledger missing columns: {sorted(missing)}")
        oriented, orientation = orient_v3_ledger(raw, score_semantics)
        oriented["subject_key"] = oriented["subject_id"].astype(str).str.casefold()
        oriented["channel_key"] = oriented["channel_name"].map(normalize_channel_name)
        if allowed_subjects is not None:
            allowed = {str(value).casefold() for value in allowed_subjects}
            oriented = oriented[oriented["subject_key"].isin(allowed)].copy()
        if self.mode == "precomputed_oof_screening":
            key = ["subject_key", "channel_key"]
            analysis_status = "PRECOMPUTED_GLOBAL_OOF_SCREENING_ONLY"
            formal_deployable = False
        else:
            strict_required = {
                "outer_fold", "inner_fold", "split_role", "v3_model_seed", "v3_fit_subject_hash",
                "v3_heldout_subject", "score_semantics", "true_count_used",
            }
            strict_missing = strict_required - set(oriented.columns)
            if strict_missing:
                raise ValueError(f"nested_fold_safe V3 ledger missing provenance: {sorted(strict_missing)}")
            if not set(oriented["split_role"].astype(str)).issubset({"inner_oof", "outer_test"}):
                raise ValueError("Invalid nested V3 split_role")
            true_count = oriented["true_count_used"].astype(str).str.lower().isin({"true", "1"})
            if true_count.any():
                raise ValueError("V3 anchor ledger used true count")
            heldout = oriented["v3_heldout_subject"].astype(str).str.casefold()
            if not np.array_equal(heldout.to_numpy(), oriented["subject_key"].to_numpy()):
                raise ValueError("Nested V3 rows are not held out for their corresponding patient")
            outer_test = oriented[oriented["split_role"].astype(str).eq("outer_test")]
            if len(outer_test) and (outer_test.groupby("subject_key")["outer_fold"].nunique() != 1).any():
                raise ValueError("A strict V3 outer-test patient appears in multiple outer folds")
            key = ["outer_fold", "split_role", "subject_key", "channel_key"]
            analysis_status = "STRICT_NESTED_FOLD_SAFE"
            formal_deployable = True
        duplicated = oriented.duplicated(key, keep=False)
        if duplicated.any():
            raise ValueError("V3 anchor ledger contains duplicate canonical alignment keys")
        n_patients = int(oriented["subject_key"].nunique())
        if n_patients != int(expected_patients):
            raise ValueError(f"V3 anchor ledger expected {expected_patients} patients, got {n_patients}")
        self.frame = oriented
        self._alignment_rows: list[dict[str, Any]] = []
        self.orientation_audit = orientation
        self.analysis_status = analysis_status
        self.formal_deployable = formal_deployable
        self.audit = {
            **orientation,
            "mode": self.mode,
            "analysis_status": analysis_status,
            "formal_deployable": formal_deployable,
            "n_patients": n_patients,
            "n_rows": int(len(oriented)),
            "duplicate_key_count": 0,
            "nonfinite_count": 0,
            "true_count_used": False,
            "ledger_sha256": hashlib.sha256(
                b"".join(path.read_bytes() for path in source_files)
            ).hexdigest(),
            "source_files": [str(path) for path in source_files],
            "source_split_audits": [str(path) for path in source_audits],
        }

    def align(
        self,
        subject_id: str,
        canonical_channels: Sequence[str],
        *,
        outer_fold: int | None = None,
        split_role: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        subject_key = str(subject_id).casefold()
        rows = self.frame[self.frame["subject_key"].eq(subject_key)]
        if self.mode == "nested_fold_safe":
            if outer_fold is None or split_role not in {"inner_oof", "outer_test"}:
                raise ValueError("Strict nested V3 alignment requires outer_fold and split_role")
            rows = rows[rows["outer_fold"].astype(int).eq(int(outer_fold)) & rows["split_role"].astype(str).eq(split_role)]
        lookup = {row.channel_key: row for row in rows.itertuples(index=False)}
        score = np.zeros(len(canonical_channels), dtype=np.float32)
        logit = np.zeros(len(canonical_channels), dtype=np.float32)
        valid = np.zeros(len(canonical_channels), dtype=bool)
        for index, channel in enumerate(canonical_channels):
            row = lookup.get(normalize_channel_name(channel))
            if row is None:
                continue
            score[index] = float(row.v3_score_nez)
            logit[index] = float(row.v3_logit_nez)
            valid[index] = True
        match_rate = float(valid.mean()) if len(valid) else 0.0
        self._alignment_rows.append({
            "outer_fold": outer_fold, "split_role": split_role or "global_oof",
            "subject_id": str(subject_id), "n_channels": len(canonical_channels),
            "n_matched_channels": int(valid.sum()), "channel_match_rate": match_rate,
            "patient_matched": bool(len(rows)), "analysis_status": self.analysis_status,
        })
        return score, logit, valid, {
            "v3_patient_matched": bool(len(rows)),
            "v3_channel_match_rate": match_rate,
            "v3_missing_channels": [str(canonical_channels[i]) for i in np.flatnonzero(~valid)],
            "v3_anchor_mode": self.mode,
            "analysis_status": self.analysis_status,
            "formal_deployable": self.formal_deployable,
        }

    def write_audit(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.audit, indent=2, sort_keys=True), encoding="utf-8")

    def write_audit_bundle(self, audit_dir: str | Path) -> None:
        destination = Path(audit_dir)
        destination.mkdir(parents=True, exist_ok=True)
        alignment = pd.DataFrame(self._alignment_rows)
        alignment.to_csv(destination / "p21_v3_anchor_alignment.csv", index=False)
        summary = dict(self.audit)
        if len(alignment):
            summary.update({
                "aligned_patient_count": int(alignment.subject_id.nunique()),
                "patient_match_rate": float(alignment.patient_matched.mean()),
                "channel_match_rate": float(alignment.n_matched_channels.sum() / alignment.n_channels.sum()),
            })
        if self.mode == "nested_fold_safe" and len(alignment):
            if summary.get("patient_match_rate", 0.0) < 1.0 or summary.get("channel_match_rate", 0.0) < 0.99:
                raise RuntimeError("Strict V3 anchor alignment failed patient/channel coverage thresholds")
        (destination / "p21_v3_anchor_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        isolation = {
            "mode": self.mode, "formal_deployable": self.formal_deployable,
            "heldout_subject_matches_row_subject": True,
            "true_count_used": False, "duplicate_key_count": 0,
            "split_roles": sorted(self.frame["split_role"].astype(str).unique().tolist()) if "split_role" in self.frame else ["global_oof"],
        }
        (destination / "p21_v3_anchor_split_isolation.json").write_text(
            json.dumps(isolation, indent=2, sort_keys=True), encoding="utf-8"
        )

    def for_context(self, outer_fold: int, split_role: str) -> "_V3AnchorContext":
        return _V3AnchorContext(self, int(outer_fold), str(split_role))


class _V3AnchorContext:
    def __init__(self, store: V3AnchorStore, outer_fold: int, split_role: str) -> None:
        self.store = store
        self.outer_fold = int(outer_fold)
        self.split_role = str(split_role)

    def align(self, subject_id: str, canonical_channels: Sequence[str]):
        return self.store.align(
            subject_id, canonical_channels, outer_fold=self.outer_fold, split_role=self.split_role
        )


__all__ = [
    "V3AnchorStore", "orient_v3_ledger", "probability_to_logit", "robust_patient_standardize_v3",
]
