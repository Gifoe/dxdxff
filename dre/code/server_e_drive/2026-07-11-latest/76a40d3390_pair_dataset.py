from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .pair_labels import compute_pair_label
from .schemas import normalize_channel_name


PAIR_CACHE_VERSION = 1


def build_pair_dataset(ledger: pd.DataFrame, candidates: pd.DataFrame, *, include_labels: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_id, edges in candidates.groupby("subject_id", sort=True):
        patient = ledger[ledger["subject_id"].astype(str) == str(subject_id)]
        for _, edge in edges.iterrows():
            row = edge.to_dict()
            row["eject_channel_norm"] = normalize_channel_name(edge["eject_channel"])
            row["add_channel_norm"] = normalize_channel_name(edge["add_channel"])
            if include_labels:
                row.update(compute_pair_label(patient, str(edge["eject_channel"]), str(edge["add_channel"])))
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["patient_pair_weight"] = 1.0 / out.groupby("subject_id")["subject_id"].transform("size").astype(float)
    return out


def write_pair_cache(pairs: pd.DataFrame, output_dir: str | Path, *, config: dict[str, object], force_rebuild: bool = False) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = pairs.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    schema_path = root / "pair_schema.json"
    if schema_path.exists() and not force_rebuild:
        existing = json.loads(schema_path.read_text(encoding="utf-8"))
        pair_path = root / ("pairs.parquet" if existing.get("storage") == "parquet" else "pairs.pkl")
        if existing.get("sha256") == digest and existing.get("config") == config and pair_path.exists():
            return {"pairs": str(pair_path), "schema": str(schema_path), "reused": "true"}
        raise RuntimeError("pair cache fingerprint changed; pass --force-rebuild-pair-cache to rebuild")
    try:
        pair_path = root / "pairs.parquet"
        pairs.to_parquet(pair_path, index=False)
        storage = "parquet"
    except Exception:
        pair_path = root / "pairs.pkl"
        pairs.to_pickle(pair_path)
        storage = "pickle"
    schema_path.write_text(json.dumps({"version": PAIR_CACHE_VERSION, "storage": storage, "columns": list(pairs.columns), "sha256": digest, "config": config}, indent=2, sort_keys=True), encoding="utf-8")
    return {"pairs": str(pair_path), "schema": str(schema_path), "reused": "false"}
