from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import normalize_channel_name
from .topology_features import parse_channel_topology


@dataclass(frozen=True)
class CandidateConfig:
    selected_tail_frac: float = .30
    selected_tail_min: int = 3
    selected_tail_max: int = 12
    boundary_width: int = 20
    add_pool_max: int = 30
    max_eject_candidates: int = 12
    max_add_candidates: int = 30
    max_pairs_per_patient: int = 360


EJECT_FLAGS = ("eject_source_selected_tail", "eject_source_anchor", "eject_source_trajectory",
               "eject_source_isolated", "eject_source_segment_edge", "eject_source_hnc")
ADD_FLAGS = ("add_source_boundary", "add_source_anchor", "add_source_trajectory",
             "add_source_same_shaft", "add_source_segment_neighbor", "add_source_hnc")


def _topology_flags(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    parsed = out["channel_name_norm"].map(parse_channel_topology)
    out[["topology_shaft", "topology_contact", "topology_valid"]] = pd.DataFrame(parsed.tolist(), index=out.index)
    selected = out[(out["old_v3_selected"].astype(int) == 1) & out["topology_valid"].astype(bool)]
    by_shaft = {shaft: set(part["topology_contact"].astype(int)) for shaft, part in selected.groupby("topology_shaft")}
    isolated, edge, same_shaft, neighbor = [], [], [], []
    for _, row in out.iterrows():
        if not bool(row["topology_valid"]):
            isolated.append(False); edge.append(False); same_shaft.append(False); neighbor.append(False)
            continue
        contacts = by_shaft.get(row["topology_shaft"], set())
        contact = int(row["topology_contact"])
        selected_row = int(row["old_v3_selected"]) == 1
        adjacent_selected = {contact - 1, contact + 1} & contacts
        isolated.append(selected_row and not adjacent_selected)
        edge.append(selected_row and bool(adjacent_selected) and not ({contact - 1, contact + 1} <= contacts))
        same_shaft.append((not selected_row) and any(abs(contact - other) in {1, 2} for other in contacts))
        segments: list[tuple[int, int]] = []
        if contacts:
            start = previous = min(contacts)
            for value in sorted(contacts)[1:]:
                if value != previous + 1:
                    segments.append((start, previous)); start = value
                previous = value
            segments.append((start, previous))
        neighbor.append((not selected_row) and any(contact in {left - 1, right + 1} for left, right in segments))
    out["_isolated"] = isolated; out["_segment_edge"] = edge
    out["_same_shaft"] = same_shaft; out["_segment_neighbor"] = neighbor
    return out


def _union(parts: list[tuple[pd.DataFrame, str]], *, max_items: int) -> pd.DataFrame:
    rows: dict[str, dict[str, object]] = {}
    for frame, flag in parts:
        for _, row in frame.iterrows():
            key = str(row["channel_name_norm"])
            existing = rows.setdefault(key, row.to_dict())
            existing[flag] = 1
    out = pd.DataFrame(rows.values())
    if out.empty:
        return out
    flags = EJECT_FLAGS if any(flag.startswith("eject_") for _, flag in parts) else ADD_FLAGS
    for flag in flags:
        if flag not in out:
            out[flag] = 0
    out[list(flags)] = out[list(flags)].fillna(0).astype(int)
    out["candidate_source"] = out[list(flags)].apply(
        lambda row: "|".join(name.removeprefix("eject_source_").removeprefix("add_source_") for name, value in row.items() if value), axis=1)
    return out.sort_values(["old_v3_rank", "channel_name_norm"], kind="mergesort").head(int(max_items))


def build_candidate_pools(ledger: pd.DataFrame, config: CandidateConfig = CandidateConfig()) -> pd.DataFrame:
    """Build a label-free post-union capped pool with independent source flags."""
    rows: list[dict[str, object]] = []
    for subject_id, raw_group in ledger.groupby("subject_id", sort=True):
        group = raw_group.copy()
        if "channel_name_norm" not in group:
            group["channel_name_norm"] = group["channel_name_original"].map(normalize_channel_name)
        group = _topology_flags(group)
        selected = group[group["old_v3_selected"].astype(int) == 1].sort_values("old_v3_score_ez", ascending=True, kind="mergesort")
        outside = group[group["old_v3_selected"].astype(int) == 0].sort_values("old_v3_rank", kind="mergesort")
        tail_n = min(len(selected), int(config.selected_tail_max), max(int(config.selected_tail_min), int(np.ceil(config.selected_tail_frac * len(selected)))))
        eject_parts = [(selected.head(tail_n), "eject_source_selected_tail"),
                       (selected[selected["_isolated"]], "eject_source_isolated"),
                       (selected[selected["_segment_edge"]], "eject_source_segment_edge")]
        add_parts = [(outside.head(min(int(config.boundary_width), int(config.add_pool_max))), "add_source_boundary"),
                     (outside[outside["_same_shaft"]], "add_source_same_shaft"),
                     (outside[outside["_segment_neighbor"]], "add_source_segment_neighbor")]
        if "distance_to_patient_nez_anchor_l2" in group:
            eject_parts.append((selected.sort_values("distance_to_patient_nez_anchor_l2", kind="mergesort"), "eject_source_anchor"))
            add_parts.append((outside.sort_values("distance_to_patient_nez_anchor_l2", ascending=False, kind="mergesort"), "add_source_anchor"))
        trajectory_columns = [name for name in group if name.startswith("trajectory_stability")]
        if trajectory_columns:
            group["_trajectory_stability"] = group[trajectory_columns].median(axis=1, skipna=True)
            selected_t = group[group["old_v3_selected"].astype(int) == 1]
            outside_t = group[group["old_v3_selected"].astype(int) == 0]
            eject_parts.append((selected_t.sort_values("_trajectory_stability", kind="mergesort"), "eject_source_trajectory"))
            add_parts.append((outside_t.sort_values("_trajectory_stability", ascending=False, kind="mergesort"), "add_source_trajectory"))
        if "hnc_eject_priority" in group:
            eject_parts.append((selected.sort_values("hnc_eject_priority", ascending=False, kind="mergesort"), "eject_source_hnc"))
            add_parts.append((outside.sort_values("hnc_add_priority", ascending=False, kind="mergesort"), "add_source_hnc"))
        eject = _union(eject_parts, max_items=config.max_eject_candidates)
        add = _union(add_parts, max_items=config.max_add_candidates)
        for _, e in eject.iterrows():
            for _, a in add.iterrows():
                row = {"subject_id": str(subject_id), "outer_fold": int(e["outer_fold"]),
                       "eject_channel": str(e["channel_name_original"]), "eject_channel_norm": str(e["channel_name_norm"]),
                       "add_channel": str(a["channel_name_original"]), "add_channel_norm": str(a["channel_name_norm"]),
                       "eject_score_ez": float(e["old_v3_score_ez"]), "add_score_ez": float(a["old_v3_score_ez"]),
                       "eject_rank": int(e["old_v3_rank"]), "add_rank": int(a["old_v3_rank"]),
                       "eject_candidate_source": str(e["candidate_source"]), "add_candidate_source": str(a["candidate_source"]),
                       "candidate_source": f"eject:{e['candidate_source']}|add:{a['candidate_source']}"}
                row.update({flag: int(e.get(flag, 0)) for flag in EJECT_FLAGS})
                row.update({flag: int(a.get(flag, 0)) for flag in ADD_FLAGS})
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["subject_id", "outer_fold", "eject_channel", "eject_channel_norm", "add_channel", "add_channel_norm", *EJECT_FLAGS, *ADD_FLAGS])
    out["candidate_priority"] = out["add_score_ez"] - out["eject_score_ez"]
    return out.sort_values(["subject_id", "candidate_priority", "eject_channel_norm", "add_channel_norm"], ascending=[True, False, True, True], kind="mergesort").groupby("subject_id", group_keys=False).head(config.max_pairs_per_patient).reset_index(drop=True)
