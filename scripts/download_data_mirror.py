"""Download the single frozen Prophet GSM8K trajectory configuration.

This is the curl fallback used because huggingface_hub/httpx fails TLS through
the host proxy.  Every resolver request starts at hf-mirror.com, downloads are
resumable, completed files are not fetched again, and no model weights are in
scope.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import quote


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("NO_PROXY", "hf-mirror.com,.hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ENDPOINT = "https://hf-mirror.com"
REPO_ID = "YefanZhou98/DLM-Decoding-Analysis"
REVISION = "91beb881acaa0b6edfccd88e8d19c08ec5e1225b"
TARGET_FOLDER = "question_histories_low_conf_none_index_genlen_step256_blocklen32"
EXPECTED_CASES = 1319


def _curl(url: str, output: Path, resume: bool) -> None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for the mirror fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "900",
    ]
    if resume:
        command.extend(["--continue-at", "-"])
    command.extend(["--output", str(output), url])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"curl failed ({result.returncode}) for {url}: {result.stderr.strip()}"
        )


def _atomic_curl(url: str, destination: Path) -> str:
    if destination.is_file() and destination.stat().st_size > 0:
        return "skipped"
    part = destination.with_name(destination.name + ".part")
    _curl(url, part, resume=True)
    if part.stat().st_size <= 0:
        raise RuntimeError(f"empty download: {destination}")
    part.replace(destination)
    return "downloaded"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_url(path: str) -> str:
    encoded = quote(path, safe="/")
    return f"{ENDPOINT}/datasets/{REPO_ID}/resolve/{REVISION}/{encoded}"


def _load_repo_info(metadata_dir: Path) -> tuple[dict, Path]:
    destination = metadata_dir / "repo_info.json"
    api_url = f"{ENDPOINT}/api/datasets/{REPO_ID}"
    part = destination.with_name(destination.name + ".part")
    _curl(api_url, part, resume=False)
    part.replace(destination)
    info = json.loads(destination.read_text(encoding="utf-8"))
    if info.get("sha") != REVISION:
        raise RuntimeError(
            f"mirror main revision changed: expected {REVISION}, got {info.get('sha')}"
        )
    return info, destination


def _selected_paths(info: dict) -> list[str]:
    prefix = TARGET_FOLDER + "/"
    paths = sorted(
        entry["rfilename"]
        for entry in info.get("siblings", [])
        if entry.get("rfilename", "").startswith(prefix)
        and entry["rfilename"].endswith(".pt")
    )
    if len(paths) != EXPECTED_CASES:
        raise RuntimeError(
            f"target folder contains {len(paths)} .pt files; expected {EXPECTED_CASES}"
        )
    expected_names = {
        f"{TARGET_FOLDER}/question_{index:04d}_steps_256.pt"
        for index in range(EXPECTED_CASES)
    }
    if set(paths) != expected_names:
        missing = sorted(expected_names - set(paths))[:10]
        extra = sorted(set(paths) - expected_names)[:10]
        raise RuntimeError(f"non-contiguous case files; missing={missing}, extra={extra}")
    return ["README.md", *paths]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    root = args.root.resolve()
    raw_dir = root / "data" / "raw" / "DLM-Decoding-Analysis"
    metadata_dir = root / "data" / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    info, repo_info_path = _load_repo_info(metadata_dir)
    paths = _selected_paths(info)
    print(f"MIRROR={ENDPOINT}", flush=True)
    print(f"REVISION={REVISION}", flush=True)
    print(f"TARGET_FOLDER={TARGET_FOLDER}", flush=True)
    print(f"FILES_SELECTED={len(paths)}", flush=True)

    started = time.time()
    counts = {"downloaded": 0, "skipped": 0}

    def fetch(relative_path: str) -> tuple[str, str]:
        destination = raw_dir / Path(relative_path)
        status = _atomic_curl(_resolve_url(relative_path), destination)
        return relative_path, status

    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, path): path for path in paths}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            path = futures[future]
            try:
                _, status = future.result()
                counts[status] += 1
            except Exception as exc:  # keep collecting independent failures
                failures.append((path, str(exc)))
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"PROGRESS={completed}/{len(futures)} "
                    f"downloaded={counts['downloaded']} skipped={counts['skipped']} "
                    f"failed={len(failures)}",
                    flush=True,
                )

    if failures:
        for path, error in failures[:20]:
            print(f"FAILED {path}: {error}", file=sys.stderr)
        print("Re-run the same command to resume .part files.", file=sys.stderr)
        return 2

    print("Hashing selected source files...", flush=True)
    entries = []
    for relative_path in paths:
        local_path = raw_dir / Path(relative_path)
        entries.append(
            {
                "relative_path": relative_path.replace("\\", "/"),
                "size_bytes": local_path.stat().st_size,
                "sha256": _sha256(local_path),
            }
        )

    manifest = {
        "dataset_repo": REPO_ID,
        "dataset_revision": REVISION,
        "endpoint": ENDPOINT,
        "download_method": "curl mirror resolver fallback (resumable .part files)",
        "target_folder": TARGET_FOLDER,
        "expected_cases": EXPECTED_CASES,
        "repo_last_modified": info.get("lastModified"),
        "repo_info_sha256": _sha256(repo_info_path),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }
    manifest_path = metadata_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    elapsed = time.time() - started
    print(f"DOWNLOAD_OK files={len(entries)} bytes={manifest['total_bytes']} seconds={elapsed:.1f}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
