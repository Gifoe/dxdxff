"""CLI for the canonical strict ReVA Mini cache verifier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reva_dlm.cache_verifier import verify_cache  # noqa: E402
from reva_dlm.config import CACHE_VERSION  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly verify a READY_FOR_G2 ReVA Mini cache."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "cache" / CACHE_VERSION,
    )
    parser.add_argument(
        "--skip-source-hashes",
        action="store_true",
        help=(
            "Skip raw trajectory hashes and live-workspace comparison; frozen "
            "cache/snapshot hashes and all semantic contracts remain mandatory."
        ),
    )
    args = parser.parse_args()
    result = verify_cache(
        args.cache_dir, verify_source_hashes=not args.skip_source_hashes
    )
    print("CACHE_VERIFY_OK")
    for key, value in result.items():
        print(f"{key.upper()}={value}")


if __name__ == "__main__":
    main()
