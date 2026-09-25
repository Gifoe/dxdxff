from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
for path in (THIS_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import VERSIONS, build_parser
from pipeline import build_caches
from train_runner import aggregate_outputs, run_training, write_manifest
from utils import log


def main() -> None:
    args = build_parser().parse_args()
    args.output_root = Path(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "run_config.json").open("w", encoding="utf-8") as fout:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, fout, indent=2, ensure_ascii=False)

    cache_root = args.output_root / "_caches"
    expected = {version: cache_root / f"{version}_window_cache.pkl" for version in VERSIONS}
    if args.reuse_caches and all(path.exists() for path in expected.values()):
        caches = expected
        log("Reusing existing caches.")
    else:
        caches = build_caches(args)

    if args.skip_training:
        write_manifest(args, caches)
        log("Skipping training because --skip_training was set.")
        return

    run_training(args, caches)
    aggregate_outputs(args)


if __name__ == "__main__":
    main()
