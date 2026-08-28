"""Download the frozen GSM8K test source needed to reconstruct prompt IDs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("NO_PROXY", "hf-mirror.com,.hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from huggingface_hub import HfApi, snapshot_download  # noqa: E402


ENDPOINT = "https://hf-mirror.com"
REPO_ID = "openai/gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
ALLOWLIST = ("README.md", "main/test-00000-of-00001.parquet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data" / "raw" / "GSM8K"
    info = HfApi(endpoint=ENDPOINT).dataset_info(REPO_ID, revision=REVISION)
    if info.sha != REVISION:
        raise RuntimeError(f"GSM8K revision mismatch: {info.sha}")
    available = {item.rfilename for item in info.siblings}
    if not set(ALLOWLIST).issubset(available):
        raise RuntimeError("GSM8K allowlist is not present at the frozen revision")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=list(ALLOWLIST),
        local_dir=output_dir,
        max_workers=2,
    )
    files = []
    for relative in ALLOWLIST:
        path = output_dir / relative
        files.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "dataset_repo": REPO_ID,
        "dataset_revision": REVISION,
        "endpoint": ENDPOINT,
        "configuration": "main",
        "split": "test",
        "allowlist": list(ALLOWLIST),
        "download_method": "hf/snapshot_download with hf-mirror endpoint and frozen revision",
        "files": files,
        "total_bytes": sum(item["size_bytes"] for item in files),
    }
    manifest_path = root / "data" / "metadata" / "gsm8k_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"GSM8K_DOWNLOAD_OK files={len(files)} bytes={manifest['total_bytes']}")
    print(f"GSM8K_REVISION={REVISION}")


if __name__ == "__main__":
    main()
