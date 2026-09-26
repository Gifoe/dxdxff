"""Finalize validation-only summary when the two pre-locked grids finish."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def read_status(path: Path) -> dict:
    if not path.exists():
        return {"status": "PENDING"}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--timeout-hours", type=float, default=36)
    args = parser.parse_args()
    summary = args.root / "development_summary"
    summary.mkdir(parents=True, exist_ok=True)
    output = summary / "AGGREGATION_STATUS.json"
    start = time.monotonic()
    paths = {
        "stage_a": args.root / "stage_a_seed42" / "DEVELOPMENT_STATUS.json",
        "controls": args.root / "matched_controls_seed42" / "CONTROL_STATUS.json",
        "tabular": args.root / "tabular_seed42" / "TABULAR_STATUS.json",
    }
    while True:
        statuses = {name: read_status(path) for name, path in paths.items()}
        states = {name: state["status"] for name, state in statuses.items()}
        output.write_text(json.dumps({"status": "WAITING", "dependencies": states, "test_evaluated": False}, indent=2), encoding="utf-8")
        if any(value == "FAILED" for value in states.values()):
            raise RuntimeError(f"One dependency failed: {states}")
        if all(value == "COMPLETE" for value in states.values()):
            break
        if time.monotonic() - start > args.timeout_hours * 3600:
            raise TimeoutError(f"Development grid did not finish: {states}")
        time.sleep(60)
    command = [sys.executable, str(args.script),
               "--source-root", str(args.source_root),
               "--stage-a", str(args.root / "stage_a_seed42"),
               "--controls", str(args.root / "matched_controls_seed42"),
               "--tabular", str(args.root / "tabular_seed42"),
               "--output", str(summary)]
    with (summary / "aggregate.log").open("w", encoding="utf-8") as log:
        finished = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if finished.returncode != 0:
        output.write_text(json.dumps({"status": "FAILED", "aggregate_exit_code": finished.returncode}, indent=2), encoding="utf-8")
        raise RuntimeError("Validation aggregation failed")
    output.write_text(json.dumps({"status": "COMPLETE", "test_evaluated": False}, indent=2), encoding="utf-8")
    print("VALIDATION_AGGREGATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
