from __future__ import annotations

from itertools import combinations

import pandas as pd

from .schemas import normalize_channel_name


def select_max_weight_swaps(edges: pd.DataFrame, *, max_swaps: int = 1, utility_threshold: float = 0.0) -> pd.DataFrame:
    eligible = edges[edges["utility"].astype(float) > float(utility_threshold)].copy()
    if eligible.empty or max_swaps <= 0:
        return eligible.iloc[0:0].copy()
    records = eligible.sort_values(["utility", "eject_channel", "add_channel"], ascending=[False, True, True], kind="mergesort").to_dict("records")
    best: tuple[float, tuple[dict[str, object], ...]] = (0.0, tuple())
    for size in range(1, min(int(max_swaps), len(records)) + 1):
        for choice in combinations(records, size):
            ejects = [normalize_channel_name(row.get("eject_channel_norm", row["eject_channel"])) for row in choice]
            adds = [normalize_channel_name(row.get("add_channel_norm", row["add_channel"])) for row in choice]
            if len(set(ejects)) != len(ejects) or len(set(adds)) != len(adds):
                continue
            score = float(sum(float(row["utility"]) for row in choice))
            if score > best[0]:
                best = (score, choice)
    selected = pd.DataFrame(best[1])
    if not selected.empty:
        selected = selected.sort_values("utility", ascending=False, kind="mergesort").reset_index(drop=True)
        selected["action_order"] = range(1, len(selected) + 1)
    return selected
