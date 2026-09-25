#!/usr/bin/env python3
"""Run the memory-heavy cache provenance audit in an isolated process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from task1_confirmatory.audit import audit_cache_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    report = audit_cache_provenance(config, output_root=args.output_root)
    print(json.dumps({
        "status": report["status"],
        "cache_path": report["caches"][0]["cache_path"],
        "cache_sha256": report["caches"][0]["cache_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
