from __future__ import annotations

from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

CENTERS = ("lzu", "hup", "multicenter", "pediatric")


def _normalize_center(value: str) -> str:
    x = str(value).strip().lower().replace("-", "_")
    aliases = {"multi_center": "multicenter", "child": "pediatric", "kids": "pediatric"}
    return aliases.get(x, x)


def quality_score(frame: pd.DataFrame) -> pd.Series:
    fields = {"raw_feature_match_rate": 1, "n_seizures": 1, "valid_channel_ratio": 1,
              "raw_duration_sec": 1, "missing_field_count": -1}
    score = np.zeros(len(frame), dtype=float)
    used = 0
    for name, direction in fields.items():
        if name in frame:
            rank = pd.to_numeric(frame[name], errors="coerce").rank(pct=True, method="average")
            score += direction * rank.fillna(.5).to_numpy(); used += 1
    if not used: raise ValueError("quality table has none of the fixed non-outcome quality fields")
    return pd.Series(score / used, index=frame.index)


def select_pilot16(candidates: pd.DataFrame, manifest_path: str | Path | None = None,
                   resume: bool = True) -> pd.DataFrame:
    path = Path(manifest_path) if manifest_path else None
    if resume and path and path.exists():
        result = pd.read_csv(path); _validate_pilot(result); return result
    data = candidates.copy(); data["center"] = data.center.map(_normalize_center)
    required = {"patient_key", "center", "outcome_true", "original_outer_fold", "n_seizures",
                "n_valid_channels", "raw_feature_match_rate", "target_source"}
    missing = required - set(data.columns)
    if missing: raise ValueError(f"pilot candidate columns missing: {sorted(missing)}")
    eligibility = np.ones(len(data), dtype=bool)
    for optional in ("in_original_folds", "feature_available", "raw_available", "target_complete",
                     "p2_oof_available", "has_valid_seizure", "target_has_both_classes"):
        if optional in data: eligibility &= data[optional].astype(bool).to_numpy()
    eligibility &= pd.to_numeric(data.raw_feature_match_rate, errors="coerce").ge(.95).to_numpy()
    eligibility &= pd.to_numeric(data.n_seizures, errors="coerce").ge(1).to_numpy()
    data = data.loc[eligibility].copy(); data["quality_score"] = quality_score(data)
    selected = []
    for center, outcome in product(CENTERS, (0, 1)):
        group = data[(data.center == center) & (data.outcome_true.astype(int) == outcome)].copy()
        if len(group) < 2: raise ValueError(f"PILOT_STRATUM_TOO_SMALL: center={center}, outcome={outcome}, n={len(group)}")
        group = group.sort_values(["quality_score", "patient_key"], kind="stable")
        ranks = group.quality_score.rank(pct=True, method="average")
        picks = []
        for target_q in (.33, .67):
            pool = group.loc[~group.patient_key.astype(str).isin(picks)].copy()
            distances = (ranks.loc[pool.index] - target_q).abs()
            pick = pool.assign(_distance=distances).sort_values(["_distance", "patient_key"], kind="stable").iloc[0]
            picks.append(str(pick.patient_key)); row = pick.drop(labels=["_distance"]); row["quality_quantile"] = target_q; selected.append(row)
    result = pd.DataFrame(selected)
    result["selection_reason"] = "nearest_fixed_quality_quantile_33_or_67"
    columns = ["patient_key", "center", "outcome_true", "original_outer_fold", "quality_score",
               "quality_quantile", "n_seizures", "n_valid_channels", "raw_feature_match_rate",
               "target_source", "selection_reason"]
    result = result[columns].sort_values(["center", "outcome_true", "quality_quantile", "patient_key"]).reset_index(drop=True)
    _validate_pilot(result)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp"); result.to_csv(tmp, index=False); tmp.replace(path)
    return result


def _validate_pilot(frame: pd.DataFrame) -> None:
    if len(frame) != 16 or frame.patient_key.astype(str).duplicated().any(): raise ValueError("pilot manifest must contain 16 unique patients")
    counts = frame.assign(center=frame.center.map(_normalize_center)).groupby(["center", "outcome_true"]).size()
    if any(counts.get((center, label), 0) != 2 for center in CENTERS for label in (0, 1)):
        raise ValueError("pilot must contain two patients per center and outcome")


def build_pilot_outer_folds(pilot: pd.DataFrame) -> pd.DataFrame:
    _validate_pilot(pilot); data = pilot.copy(); data["center"] = data.center.map(_normalize_center)
    by_center = {c: sorted(data.loc[data.center == c, "patient_key"].astype(str)) for c in CENTERS}
    assignments = []
    # Lexicographic depth-first search; the first complete allocation is fixed.
    def search(fold: int, remaining: dict[str, list[str]]) -> bool:
        if fold == 4: return True
        for choice in product(*(remaining[c] for c in CENTERS)):
            labels = data.set_index(data.patient_key.astype(str)).loc[list(choice), "outcome_true"].astype(int)
            if labels.sum() != 2: continue
            assignments.append(choice)
            nxt = {c: [p for p in remaining[c] if p != choice[i]] for i, c in enumerate(CENTERS)}
            if search(fold + 1, nxt): return True
            assignments.pop()
        return False
    if not search(0, by_center): raise ValueError("no valid deterministic pilot fold assignment")
    rows = [{"patient_key": patient, "outer_fold": fold + 1}
            for fold, choice in enumerate(assignments) for patient in choice]
    result = pd.DataFrame(rows).sort_values(["outer_fold", "patient_key"]).reset_index(drop=True)
    return result.merge(data[["patient_key", "center", "outcome_true"]], on="patient_key", validate="one_to_one")


def validate_pilot_outer_folds(folds: pd.DataFrame, pilot: pd.DataFrame) -> None:
    required={"patient_key","outer_fold","center","outcome_true"}
    if required-set(folds): raise ValueError("pilot fold manifest is missing required columns")
    if set(folds.patient_key.astype(str))!=set(pilot.patient_key.astype(str)) or folds.patient_key.astype(str).duplicated().any(): raise ValueError("pilot folds must assign every pilot patient exactly once")
    if set(folds.outer_fold.astype(int))!={1,2,3,4}: raise ValueError("pilot outer folds must be 1..4")
    for _,group in folds.groupby("outer_fold"):
        if len(group)!=4 or group.center.map(_normalize_center).nunique()!=4 or group.outcome_true.astype(int).sum()!=2: raise ValueError("each pilot test fold must have four centers and 2/2 outcomes")


__all__ = ["CENTERS", "quality_score", "select_pilot16", "build_pilot_outer_folds", "validate_pilot_outer_folds"]
