"""One-way model/threshold freeze gate before N1 official-test TF access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    a = p.parse_args()
    freeze_path = a.output / "INTERICTAL_MODEL_FREEZE.json"
    threshold_path = a.output / "FROZEN_THRESHOLD.json"
    checkpoint = a.runtime / "final/last.pt"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze["model_frozen_before_official_test"] or not freeze["threshold_frozen_before_official_test"]:
        raise RuntimeError("Model/threshold freeze flags missing")
    for path, expected in ((a.protocol, freeze["protocol_sha256"]),
                           (threshold_path, freeze["threshold_json_sha256"]),
                           (checkpoint, freeze["final_checkpoint_sha256"])):
        if sha(path) != expected:
            raise RuntimeError(f"Freeze hash mismatch: {path.name}")
    marker = a.runtime / "official_test_access_started.json"
    if marker.is_file():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior["checkpoint_sha256"] != freeze["final_checkpoint_sha256"]:
            raise RuntimeError("Already started official test with a different checkpoint")
        print("OFFICIAL_TEST_ALREADY_STARTED_SAME_FREEZE", flush=True)
        return
    value = {"started_utc": datetime.now(timezone.utc).isoformat(),
             "model_frozen_before_official_test": True,
             "threshold_frozen_before_official_test": True,
             "checkpoint_sha256": freeze["final_checkpoint_sha256"],
             "threshold_json_sha256": freeze["threshold_json_sha256"],
             "protocol_sha256": freeze["protocol_sha256"],
             "training_resume_forbidden": True}
    temp = marker.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, marker)
    print("OFFICIAL_TEST_ACCESS_STARTED_AFTER_FREEZE", flush=True)


if __name__ == "__main__":
    main()
