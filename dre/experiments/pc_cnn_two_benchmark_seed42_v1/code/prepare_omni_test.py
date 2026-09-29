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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--official-split", type=Path, required=True)
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
    if len(rows) != 174 or rows["patient_name"].nunique() != 96:
        raise RuntimeError("Frozen official Omni 174-EDF/96-patient test cohort differs")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = sha(args.protocol) + "|" + sha(args.freeze)
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
