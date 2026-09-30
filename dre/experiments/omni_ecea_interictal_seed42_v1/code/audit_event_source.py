#!/usr/bin/env python3
"""Audit whether frozen official Omni event evidence exists.

This is deliberately an availability audit, not an event-model trainer.  ECEA is
allowed to proceed only when either an existing official three-class checkpoint
or existing per-event three-class predictions can be found.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
OUTPUT_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
EVENT_HINTS = ("event", "hfo", "3label")
OUTPUT_FIELDS = {"3label_pred", "pathological_probability"}
SKIP_DIRS = {".git", "__pycache__", "node_modules"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def walk_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [name for name in subdirs if name not in SKIP_DIRS]
        base = Path(directory)
        for name in files:
            yield base / name


def read_text_fields(path: Path) -> set[str]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
                return {str(value).strip() for value in next(csv.reader(stream, delimiter=delimiter), [])}
        if suffix in {".json", ".jsonl"}:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                if suffix == ".jsonl":
                    payload = json.loads(next((line for line in stream if line.strip()), "{}"))
                else:
                    payload = json.load(stream)
            if isinstance(payload, dict):
                return set(payload)
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                return set(payload[0])
            return set()
        if suffix == ".parquet":
            import pyarrow.parquet as pq  # type: ignore

            return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return set()
    return set()


def relative_or_full(path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            return f"{root.name}/{path.relative_to(root).as_posix()}"
        except ValueError:
            pass
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--search-root", action="append", type=Path, default=[])
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--official-revision", required=True)
    parser.add_argument("--cnn-checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    requested_roots = [args.dataset_root, *args.search_root]
    roots: list[Path] = []
    # Avoid scanning a dataset twice when a wider runtime root is also supplied.
    for candidate in requested_roots:
        resolved = candidate.resolve()
        if any(resolved == root or resolved.is_relative_to(root) for root in roots):
            continue
        roots = [root for root in roots if not root.is_relative_to(resolved)]
        roots.append(resolved)
    checkpoint_files: list[Path] = []
    event_checkpoint_candidates: list[Path] = []
    output_hits: list[dict[str, object]] = []
    scanned_outputs = 0

    excluded_outputs = {args.output_json.resolve(), args.output_md.resolve()}
    for root in roots:
        for path in walk_files(root):
            if path.resolve() in excluded_outputs:
                continue
            suffix = path.suffix.lower()
            lowered = str(path).lower()
            if suffix in CHECKPOINT_SUFFIXES:
                checkpoint_files.append(path)
                if any(hint in lowered for hint in EVENT_HINTS):
                    event_checkpoint_candidates.append(path)
            if suffix in OUTPUT_SUFFIXES:
                scanned_outputs += 1
                fields = read_text_fields(path)
                matched = sorted(fields & OUTPUT_FIELDS)
                if matched:
                    output_hits.append(
                        {
                            "path": relative_or_full(path, roots),
                            "matched_fields": matched,
                            "bytes": path.stat().st_size,
                        }
                    )

    hfo_root = args.dataset_root / "derivatives" / "hfo"
    annotation_root = args.dataset_root / "derivatives" / "hfo_annotation"
    hfo_files = sorted(hfo_root.rglob("*.csv")) if hfo_root.exists() else []
    annotation_files = sorted(annotation_root.rglob("*.parquet")) if annotation_root.exists() else []
    hfo_columns = sorted(read_text_fields(hfo_files[0])) if hfo_files else []
    annotation_schemas = sorted({tuple(sorted(read_text_fields(path))) for path in annotation_files})

    official_source_exists = args.official_source.is_file()
    official_source_sha = sha256(args.official_source) if official_source_exists else None
    cnn_checkpoint_sha = sha256(args.cnn_checkpoint) if args.cnn_checkpoint.is_file() else None
    frozen_predictions_sha = sha256(args.frozen_predictions) if args.frozen_predictions.is_file() else None

    event_checkpoint_available = bool(event_checkpoint_candidates)
    hard_or_soft_output_available = bool(output_hits)
    event_evidence_available = event_checkpoint_available or hard_or_soft_output_available
    terminal = "EVENT_EVIDENCE_AVAILABLE" if event_evidence_available else "EVENT_EVIDENCE_UNAVAILABLE"

    audit = {
        "status": terminal,
        "official_revision": args.official_revision,
        "official_inference_source": str(args.official_source),
        "official_inference_source_exists": official_source_exists,
        "official_inference_source_sha256": official_source_sha,
        "official_inference_requires_model_path": True,
        "official_repository_bundles_event_checkpoint": False,
        "cnn_checkpoint_sha256": cnn_checkpoint_sha,
        "frozen_segment_predictions_sha256": frozen_predictions_sha,
        "searched_roots": [str(root) for root in roots],
        "checkpoint_files_scanned": len(checkpoint_files),
        "event_checkpoint_candidates": [relative_or_full(path, roots) for path in event_checkpoint_candidates],
        "event_output_files_scanned": scanned_outputs,
        "event_output_hits": output_hits,
        "hfo_candidate_files": len(hfo_files),
        "hfo_candidate_columns": hfo_columns,
        "hfo_candidates_are_three_class_predictions": bool(set(hfo_columns) & OUTPUT_FIELDS),
        "annotation_parquet_files": len(annotation_files),
        "annotation_schemas": [list(schema) for schema in annotation_schemas],
        "annotations_are_task2_inference_outputs": False,
        "soft_event_evidence_available": any(
            "pathological_probability" in hit["matched_fields"] for hit in output_hits
        ),
        "hard_event_evidence_available": any("3label_pred" in hit["matched_fields"] for hit in output_hits),
        "new_event_model_trained": False,
        "task2_test_accessed": False,
        "reason": (
            "No official event checkpoint and no existing per-event 3label_pred or pathological_probability outputs "
            "were found. HFO candidate timestamps and limited artifact/spike annotation tables do not satisfy the "
            "protocol fallback."
            if not event_evidence_available
            else "An allowed existing event-evidence source was found."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checkpoint_lines = "\n".join(f"- `{item}`" for item in audit["event_checkpoint_candidates"]) or "- None found."
    output_lines = "\n".join(
        f"- `{hit['path']}`: {', '.join(hit['matched_fields'])}" for hit in output_hits
    ) or "- None found."
    markdown = f"""# Event source audit

Terminal: **`{terminal}`**

The official inference implementation at revision `{args.official_revision}` requires an external
`--model_path`; the repository does not bundle that event checkpoint. The server audit found the
following allowed checkpoint candidates:

{checkpoint_lines}

Existing result files containing `3label_pred` or `pathological_probability`:

{output_lines}

The dataset does contain {len(hfo_files)} HFO candidate CSVs with columns
`{', '.join(hfo_columns)}` and {len(annotation_files)} expert-annotation parquet files. These are
candidate timestamps and a limited event-classification training/evaluation corpus, respectively;
they are not frozen three-class inference outputs covering official Task 2 TRAIN/TEST records.

Under section 6 of the locked ECEA protocol, training a replacement Task-2 event detector is
forbidden. Therefore adapter CV, parameter fitting, freeze, and official TEST evaluation were not
run. This is an availability terminal, not evidence that ECEA helps or fails scientifically.
"""
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": terminal, "json": str(args.output_json), "markdown": str(args.output_md)}))
    return 0 if event_evidence_available else 3


if __name__ == "__main__":
    raise SystemExit(main())
