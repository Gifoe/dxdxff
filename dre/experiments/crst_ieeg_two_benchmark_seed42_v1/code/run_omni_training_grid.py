"""Sequential Omni train/validation-only CRST-0/FULL; no official test access."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--spectral-cache", type=Path, required=True)
    args = p.parse_args()
    audit = json.loads((args.runtime / "OMNI_TRAIN_CACHE_AUDIT.json").read_text(encoding="utf-8"))
    if not audit["pass"] or audit["test_split_accessed"]:
        raise RuntimeError("Official Omni TRAIN cache audit not passed")
    for variant in ("CRST-0", "CRST-FULL"):
        complete = args.runtime / "omni" / "fold1" / variant / "supervised_complete.json"
        if complete.exists():
            print(json.dumps({"variant": variant, "status": "reused"}), flush=True)
            continue
        log = args.runtime / f"omni_{variant.replace('-', '')}.log"
        error = args.runtime / f"omni_{variant.replace('-', '')}.err"
        argv = [sys.executable, str(args.code / "train_crst.py"),
                "--benchmark", "omni", "--fold", "1", "--variant", variant,
                "--spectral-cache", str(args.spectral_cache),
                "--train-val-split", str(args.code / "TRAIN_VAL_SPLIT.csv"),
                "--protocol", str(args.code / "PROTOCOL_LOCK.json"),
                "--training-lock", str(args.code / "TRAINING_LOCK.json"),
                "--runtime", str(args.runtime)]
        with log.open("a", encoding="utf-8") as out, error.open("a", encoding="utf-8") as err:
            code = subprocess.call(argv, stdout=out, stderr=err)
        if code != 0 or not complete.is_file():
            raise RuntimeError(f"Omni {variant} train/validation failed; exit={code}")
        print(json.dumps({"variant": variant, "status": "complete", "test_accessed": False}),
              flush=True)
    print(json.dumps({"status": "OMNI_TRAIN_VALIDATION_COMPLETE", "test_accessed": False}),
          flush=True)


if __name__ == "__main__":
    main()
