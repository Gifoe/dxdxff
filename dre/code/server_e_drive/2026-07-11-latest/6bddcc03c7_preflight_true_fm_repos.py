from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fm_baselines.extractors import build_extractor


MODEL_CHOICES = ("biot", "cbramod", "labram")


def _split_models(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return [item.strip().lower() for item in text.split(",") if item.strip()]
    return [str(item).strip().lower() for item in text if str(item).strip()]


def _model_report(model_name: str, repo_path: str | None, checkpoint_path: str | None, device: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "model": model_name,
        "repo_path": str(repo_path or ""),
        "checkpoint_path": str(checkpoint_path or ""),
        "passed": False,
    }
    try:
        if not repo_path or not Path(repo_path).exists():
            raise FileNotFoundError(f"{model_name} repo path does not exist: {repo_path}")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"{model_name} checkpoint path does not exist: {checkpoint_path}")
        extractor = build_extractor(model_name, target_sfreq=200, window_sec=4.0)
        extractor.load(str(checkpoint_path), str(repo_path), device)
        output = extractor.encode_batch(np.ones((2, 800), dtype=np.float32))
        if output.ndim != 2 or output.shape[0] != 2 or output.shape[1] <= 0:
            raise RuntimeError(f"{model_name} output shape must be [2,D], got {output.shape}")
        trainable = [name for name, param in extractor.model.named_parameters() if param.requires_grad]
        if trainable:
            raise RuntimeError(f"{model_name} has trainable backbone parameters: {trainable[:5]}")
        report.update(
            {
                "passed": True,
                "embedding_shape": [int(output.shape[0]), int(output.shape[1])],
                "adapter_status": getattr(extractor, "audit_info", {}).get("adapter_status", ""),
                "all_backbone_params_frozen": True,
            }
        )
    except Exception as exc:
        report.update({"failure_reason": str(exc), "traceback": traceback.format_exc()})
    return report


def run_preflight(
    *,
    models: Sequence[str],
    repo_paths: Mapping[str, str | None],
    checkpoint_paths: Mapping[str, str | None],
    device: str,
    output_dir: str | Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = _split_models(models)
    report = {"device": str(device), "allow_partial": bool(allow_partial), "models": {}}
    for model_name in selected:
        if model_name not in MODEL_CHOICES:
            raise ValueError(f"Unknown model in --models: {model_name}")
        report["models"][model_name] = _model_report(model_name, repo_paths.get(model_name), checkpoint_paths.get(model_name), device)
    path = out / "true_fm_preflight_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    failed = {name: item for name, item in report["models"].items() if not item.get("passed")}
    if failed and not allow_partial:
        reasons = "; ".join(f"{name}: {item.get('failure_reason')}" for name, item in failed.items())
        raise RuntimeError(f"True FM preflight failed: {reasons}")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="biot,cbramod,labram")
    parser.add_argument("--biot_repo", default=None)
    parser.add_argument("--biot_ckpt", default=None)
    parser.add_argument("--cbramod_repo", default=None)
    parser.add_argument("--cbramod_ckpt", default=None)
    parser.add_argument("--labram_repo", default=None)
    parser.add_argument("--labram_ckpt", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--output_dir", default=".")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_paths = {"biot": args.biot_repo, "cbramod": args.cbramod_repo, "labram": args.labram_repo}
    checkpoint_paths = {"biot": args.biot_ckpt, "cbramod": args.cbramod_ckpt, "labram": args.labram_ckpt}
    try:
        report = run_preflight(
            models=_split_models(args.models),
            repo_paths=repo_paths,
            checkpoint_paths=checkpoint_paths,
            device=args.device,
            output_dir=args.output_dir,
            allow_partial=args.allow_partial,
        )
    except Exception as exc:
        print(f"FAIL preflight: {exc}")
        return 1
    for name, item in report["models"].items():
        status = "PASS" if item.get("passed") else "FAIL"
        print(f"{status} {name}: {item.get('failure_reason', item.get('embedding_shape', ''))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
