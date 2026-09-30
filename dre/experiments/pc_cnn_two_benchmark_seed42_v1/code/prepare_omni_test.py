"""Post-freeze-only official Omni test raw/descriptor extraction.

This entry point is intentionally distinct from TRAIN extraction. It refuses
to instantiate the official test cohort until a complete, hashed model and
threshold selection freeze exists. Extracted files stay in private runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepare_omni_train import convert, exact_descriptor_module, flag, sha


HISTORICAL_COHORT_SHA256 = "e10241ce0e823ced7dd262ed6eda4eeb0ffdbe52082aa4ec771590e253ffaf00"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--official-split", type=Path, required=True)
    p.add_argument("--cohort-audit", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--signal-cache", type=Path, required=True)
    p.add_argument("--descriptor-source", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha(args.official_split) != lock["omni_official_split_sha256"]:
        raise RuntimeError("Official split hash changed")
    if sha(args.cohort_audit) != HISTORICAL_COHORT_SHA256:
        raise RuntimeError("Historical 174-EDF cohort audit hash changed")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_TEST" or \
            freeze.get("protocol_sha256") != sha(args.protocol) or \
            not freeze.get("model_frozen_before_final_test") or \
            freeze.get("final_test_accessed"):
        raise RuntimeError("A complete pre-test model/threshold freeze is required")
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("Invalid extraction shard")
    module = exact_descriptor_module(args.descriptor_source)
    rows = pd.read_csv(args.official_split)
    rows = rows.loc[(rows["split"] == "test") & (rows["dataset"] != "Multicenter") &
                    (pd.to_numeric(rows["frequency"], errors="coerce") > 900) &
                    rows["interictal"].map(flag) &
                    (pd.to_numeric(rows["length"], errors="coerce") >= 62)]
    if len(rows) != 237 or rows["patient_name"].nunique() != 102:
        raise RuntimeError("Frozen official Omni 237-EDF/102-patient metadata cohort differs")
    audited = pd.read_csv(args.cohort_audit)
    audited = audited.loc[(audited["official_split"] == "test") &
                          (pd.to_numeric(audited["official_labeled_channels"], errors="coerce") > 0)]
    if len(audited) != 174 or audited["patient"].nunique() != 96 or audited["edf"].duplicated().any():
        raise RuntimeError("Historical supervised test cohort differs")
    expected = audited.set_index("edf")["patient"].to_dict()
    rows = rows.loc[rows["edf_name"].astype(str).isin(expected)]
    if len(rows) != 174 or rows["patient_name"].nunique() != 96 or \
            any(expected[str(row.edf_name)] != str(row.patient_name)
                for row in rows.itertuples(index=False)):
        raise RuntimeError("Historical cohort is not an exact official-split subset")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = sha(args.protocol) + "|" + sha(args.freeze) + "|" + sha(args.cohort_audit)
    total = {"edfs": 0, "clips": 0, "labeled_channels": 0,
             "model_or_threshold_modified": False}
    subset = rows.iloc[args.shard_index::args.shards]
    for index, row in enumerate(subset.itertuples(index=False), 1):
        result = convert(row, args.source, args.signal_cache, args.output,
                         module, provenance)
        total["edfs"] += int(result["clips"] > 0)
        total["clips"] += result["clips"]
        total["labeled_channels"] += result["labeled_channels"]
        print(json.dumps({"ordinal": index, "of": len(subset),
                          "reused": result["reused"]}), flush=True)
    print(json.dumps({"status": "FROZEN_TEST_EXTRACTION_COMPLETE", **total}), flush=True)


if __name__ == "__main__":
    main()
