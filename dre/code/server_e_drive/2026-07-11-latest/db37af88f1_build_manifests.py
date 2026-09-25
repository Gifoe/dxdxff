#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from task1_confirmatory.config import load_config, resolve_path
from task1_confirmatory.orchestrator import prepare_manifests

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-root", default="")
    args = parser.parse_args(); config = load_config(args.config); output = Path(args.output_root or config["output_root"])
    print(json.dumps(prepare_manifests(config, output_root=output), indent=2, default=str))
if __name__ == "__main__": main()
