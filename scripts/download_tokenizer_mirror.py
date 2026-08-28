"""Download only the frozen LLaDA tokenizer/config allowlist via hf-mirror.

The allowlist is deliberately closed: model weight suffixes cannot be selected.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("NO_PROXY", "hf-mirror.com,.hf-mirror.com")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ENDPOINT = "https://hf-mirror.com"
REPO_ID = "GSAI-ML/LLaDA-8B-Instruct"
REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
ALLOWLIST = (
    "README.md",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
)
FORBIDDEN_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    command = [
        curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--output",
        str(part),
        url,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    part.replace(destination)


def main() -> None:
    if any(path.lower().endswith(FORBIDDEN_SUFFIXES) for path in ALLOWLIST):
        raise RuntimeError("weight-like file in tokenizer allowlist")
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data" / "raw" / "LLaDA-8B-Instruct-tokenizer"
    records = []
    for relative_path in ALLOWLIST:
        encoded = quote(relative_path, safe="/")
        url = f"{ENDPOINT}/{REPO_ID}/resolve/{REVISION}/{encoded}"
        destination = output_dir / relative_path
        download(url, destination)
        records.append(
            {
                "relative_path": relative_path,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    manifest = {
        "model_repo": REPO_ID,
        "model_revision": REVISION,
        "endpoint": ENDPOINT,
        "purpose": "tokenizer-only trajectory decoding",
        "weights_downloaded": False,
        "allowlist": list(ALLOWLIST),
        "files": records,
    }
    manifest_path = root / "data" / "metadata" / "tokenizer_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"TOKENIZER_DOWNLOAD_OK files={len(records)} bytes={sum(r['size_bytes'] for r in records)}")
    print("WEIGHTS_DOWNLOADED=False")
    print(f"TOKENIZER_PATH={output_dir}")


if __name__ == "__main__":
    main()
