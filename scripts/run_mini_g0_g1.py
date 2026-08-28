"""Run the frozen CPU-only Mini G0/G1 protocol."""

from __future__ import annotations

import os
from pathlib import Path
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("NO_PROXY", "hf-mirror.com,.hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reva_dlm.pipeline import run_pipeline  # noqa: E402


if __name__ == "__main__":
    run_pipeline()
