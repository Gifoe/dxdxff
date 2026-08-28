"""Deterministic provenance binding for GSM8K rows and Prophet prompts.

The published trajectory files do not contain the source question or prompt
token IDs.  The strongest available row-level binding is therefore the exact
``question_{row:04d}`` filename convention plus two independent stored checks:
the frozen prompt's token length and the canonical GSM8K ground truth.  This
module verifies both checks for every row and records cryptographic hashes of
the source question, rendered prompt, raw GSM8K answer, and token IDs.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .parser import parse_gsm8k_answer
from .prompt_state import (
    FROZEN_QUERY_TEMPLATE_SHA256,
    build_prompt_text,
    load_gsm8k_test,
)


PROMPT_INDEX_COLUMNS: Final[tuple[str, ...]] = (
    "case_id",
    "gsm8k_row_index",
    "question_sha256",
    "prompt_sha256",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "prompt_token_len",
    "gsm8k_gt_answer_sha256",
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


class PromptProvenanceError(AssertionError):
    """Raised when a frozen row/prompt/trajectory binding does not validate."""


def sha256_text(text: str) -> str:
    """Hash the exact UTF-8 bytes of a source string."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_prompt_token_ids(token_ids: Sequence[int]) -> str:
    """Hash token IDs as contiguous little-endian signed int64 bytes."""

    normalized = _normalize_token_ids(token_ids)
    payload = np.asarray(normalized, dtype="<i8").tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise PromptProvenanceError(
                f"prompt_token_ids tensor must be 1-D, got {tuple(value.shape)}"
            )
        values = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise PromptProvenanceError(
                f"prompt_token_ids array must be 1-D, got {tuple(value.shape)}"
            )
        values = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise PromptProvenanceError("prompt_token_ids must be a one-dimensional sequence")

    normalized: list[int] = []
    int64 = np.iinfo(np.int64)
    for token_id in values:
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise PromptProvenanceError(
                f"prompt token ID is not an integer: {token_id!r}"
            )
        integer = int(token_id)
        if integer < int64.min or integer > int64.max:
            raise PromptProvenanceError(f"prompt token ID is outside int64: {integer}")
        normalized.append(integer)
    if not normalized:
        raise PromptProvenanceError("prompt_token_ids must not be empty")
    return normalized


def validate_local_tokenizer_snapshot(
    tokenizer_dir: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify every allowlisted local tokenizer file against its manifest."""

    root = Path(tokenizer_dir).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not root.is_dir():
        raise PromptProvenanceError(f"tokenizer directory does not exist: {root}")
    if not manifest_file.is_file():
        raise PromptProvenanceError(f"tokenizer manifest does not exist: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise PromptProvenanceError("tokenizer manifest must be a JSON object")
    if manifest.get("weights_downloaded") is not False:
        raise PromptProvenanceError("tokenizer manifest must state weights_downloaded=false")

    entries = manifest.get("files")
    allowlist = manifest.get("allowlist")
    if not isinstance(entries, list) or not isinstance(allowlist, list):
        raise PromptProvenanceError("tokenizer manifest lacks files/allowlist arrays")
    expected_paths = [str(path) for path in allowlist]
    entry_by_path: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or "relative_path" not in entry:
            raise PromptProvenanceError("invalid tokenizer manifest file entry")
        relative = str(entry["relative_path"])
        if relative in entry_by_path:
            raise PromptProvenanceError(f"duplicate tokenizer manifest path: {relative}")
        entry_by_path[relative] = entry
    if set(entry_by_path) != set(expected_paths):
        raise PromptProvenanceError("tokenizer manifest files and allowlist differ")

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != set(expected_paths):
        missing = sorted(set(expected_paths) - actual_files)
        extra = sorted(actual_files - set(expected_paths))
        raise PromptProvenanceError(
            f"tokenizer snapshot file set differs; missing={missing}, extra={extra}"
        )

    total_bytes = 0
    for relative in sorted(expected_paths):
        path = root / Path(relative)
        entry = entry_by_path[relative]
        payload = path.read_bytes()
        actual_size = len(payload)
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_size != int(entry.get("size_bytes", -1)):
            raise PromptProvenanceError(f"tokenizer size mismatch: {relative}")
        if actual_sha256 != entry.get("sha256"):
            raise PromptProvenanceError(f"tokenizer SHA256 mismatch: {relative}")
        total_bytes += actual_size

    return {
        "n_files": len(expected_paths),
        "total_bytes": total_bytes,
        "model_repo": manifest.get("model_repo"),
        "model_revision": manifest.get("model_revision"),
        "weights_downloaded": False,
    }


def load_frozen_tokenizer(tokenizer_dir: str | Path) -> Any:
    """Load LLaDA's tokenizer strictly from local files, never model weights."""

    root = Path(tokenizer_dir).resolve()
    if not root.is_dir():
        raise PromptProvenanceError(f"tokenizer directory does not exist: {root}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency failure is explicit
        raise RuntimeError("transformers is required to load the frozen tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise PromptProvenanceError("the frozen LLaDA tokenizer must load as a fast tokenizer")
    return tokenizer


def _tokenize_exact(tokenizer: Any, prompt_text: str) -> list[int]:
    # This is deliberately the exact upstream call shape: tokenizer(q) followed
    # by ["input_ids"].  Supplying a different add_special_tokens value would
    # no longer reproduce Prophet's collection script.
    encoded = tokenizer(prompt_text)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise PromptProvenanceError("tokenizer output lacks input_ids")
    return _normalize_token_ids(encoded["input_ids"])


def _load_metadata(path: Path) -> dict[str, Any]:
    # mmap avoids eagerly copying tensor storage, while map_location and
    # weights_only make the CPU/trusted-source contract explicit.
    record = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(record, dict):
        raise PromptProvenanceError(f"trajectory is not a dict: {path.name}")
    required = {"prompt_token_len", "gt_text", "x0_history"}
    missing = sorted(required - set(record))
    if missing:
        raise PromptProvenanceError(
            f"trajectory {path.name} lacks prompt provenance keys: {missing}"
        )
    return record


def build_prompt_index(
    gsm8k_test_path: str | Path,
    trajectory_dir: str | Path,
    tokenizer: Any,
    *,
    expected_count: int = 1319,
    trajectory_steps: int = 256,
    generation_length: int = 256,
) -> pd.DataFrame:
    """Build and strictly validate the frozen row-to-prompt index.

    A dataframe is returned only if every expected trajectory exists, no extra
    trajectory exists, every exact prompt token length matches the stored
    metadata, every x0 width matches prompt+generation length, and every raw
    GSM8K ground truth canonically equals the trajectory's ``gt_text``.
    """

    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise TypeError("expected_count must be an integer")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if isinstance(tokenizer, (str, Path)):
        tokenizer = load_frozen_tokenizer(tokenizer)

    dataset = load_gsm8k_test(gsm8k_test_path, expected_count=expected_count)
    source_dir = Path(trajectory_dir).resolve()
    if not source_dir.is_dir():
        raise PromptProvenanceError(f"trajectory directory does not exist: {source_dir}")

    expected_paths = {
        source_dir / f"question_{index:04d}_steps_{trajectory_steps:03d}.pt"
        for index in range(expected_count)
    }
    actual_paths = set(source_dir.glob("question_*_steps_*.pt"))
    mismatches: list[str] = []
    if actual_paths != expected_paths:
        missing = sorted(path.name for path in expected_paths - actual_paths)
        extra = sorted(path.name for path in actual_paths - expected_paths)
        mismatches.append(f"trajectory file set: missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for row_index, source_row in dataset.iterrows():
        case_id = f"question_{row_index:04d}"
        question = source_row["question"]
        raw_gt = source_row["answer"]
        prompt_text = build_prompt_text(question)
        prompt_token_ids = _tokenize_exact(tokenizer, prompt_text)
        path = source_dir / f"{case_id}_steps_{trajectory_steps:03d}.pt"

        if path.is_file():
            record = _load_metadata(path)
            try:
                stored_prompt_len = int(record["prompt_token_len"])
            except (TypeError, ValueError) as exc:
                raise PromptProvenanceError(
                    f"{case_id}: invalid stored prompt_token_len"
                ) from exc
            if len(prompt_token_ids) != stored_prompt_len:
                mismatches.append(
                    f"{case_id}: prompt_token_len {len(prompt_token_ids)} != "
                    f"stored {stored_prompt_len}"
                )

            x0_history = record["x0_history"]
            try:
                first_block = x0_history[0]
                stored_width = int(first_block.shape[-1])
            except (IndexError, TypeError, AttributeError) as exc:
                raise PromptProvenanceError(
                    f"{case_id}: invalid x0_history for width validation"
                ) from exc
            expected_width = len(prompt_token_ids) + generation_length
            if stored_width != expected_width:
                mismatches.append(
                    f"{case_id}: x0 width {stored_width} != prompt+generation "
                    f"{expected_width}"
                )

            gsm8k_gt = parse_gsm8k_answer(raw_gt)
            trajectory_gt = parse_gsm8k_answer(record["gt_text"])
            if gsm8k_gt is None:
                mismatches.append(f"{case_id}: GSM8K ground truth is not parseable")
            if trajectory_gt is None:
                mismatches.append(f"{case_id}: trajectory gt_text is not parseable")
            if gsm8k_gt is not None and trajectory_gt is not None and gsm8k_gt != trajectory_gt:
                mismatches.append(
                    f"{case_id}: canonical GT {gsm8k_gt} != trajectory {trajectory_gt}"
                )

        rows.append(
            {
                "case_id": case_id,
                "gsm8k_row_index": int(row_index),
                "question_sha256": sha256_text(question),
                "prompt_sha256": sha256_text(prompt_text),
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_ids_sha256": sha256_prompt_token_ids(prompt_token_ids),
                "prompt_token_len": len(prompt_token_ids),
                # This hashes the complete raw GSM8K rationale/#### answer, not
                # merely its canonical numeric suffix.
                "gsm8k_gt_answer_sha256": sha256_text(raw_gt),
            }
        )

    if mismatches:
        preview = "\n".join(mismatches[:50])
        omitted = len(mismatches) - min(len(mismatches), 50)
        suffix = f"\n... {omitted} additional mismatch(es)" if omitted else ""
        raise PromptProvenanceError(
            f"prompt provenance validation found {len(mismatches)} mismatch(es):\n"
            f"{preview}{suffix}"
        )

    frame = pd.DataFrame.from_records(rows, columns=PROMPT_INDEX_COLUMNS)
    validate_prompt_index(frame, expected_count=expected_count)
    return frame


def validate_prompt_index(
    prompt_index: pd.DataFrame,
    *,
    expected_count: int = 1319,
) -> dict[str, Any]:
    """Validate schema, ordering, lengths, hashes, and uniqueness in an index."""

    if not isinstance(prompt_index, pd.DataFrame):
        raise TypeError("prompt_index must be a pandas DataFrame")
    if list(prompt_index.columns) != list(PROMPT_INDEX_COLUMNS):
        raise PromptProvenanceError(
            f"prompt_index columns differ: {list(prompt_index.columns)}"
        )
    if len(prompt_index) != expected_count:
        raise PromptProvenanceError(
            f"prompt_index has {len(prompt_index)} rows, expected {expected_count}"
        )

    errors: list[str] = []
    for position, (_, row) in enumerate(prompt_index.iterrows()):
        case_id = f"question_{position:04d}"
        if row["case_id"] != case_id:
            errors.append(f"row {position}: case_id {row['case_id']!r} != {case_id!r}")
        row_index = row["gsm8k_row_index"]
        if isinstance(row_index, bool) or not isinstance(row_index, Integral):
            errors.append(f"{case_id}: gsm8k_row_index is not integral")
        elif int(row_index) != position:
            errors.append(f"{case_id}: gsm8k_row_index {row_index} != {position}")

        for name in (
            "question_sha256",
            "prompt_sha256",
            "prompt_token_ids_sha256",
            "gsm8k_gt_answer_sha256",
        ):
            value = row[name]
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                errors.append(f"{case_id}: {name} is not lowercase SHA256")

        try:
            token_ids = _normalize_token_ids(row["prompt_token_ids"])
        except PromptProvenanceError as exc:
            errors.append(f"{case_id}: {exc}")
            continue
        stored_length = row["prompt_token_len"]
        if isinstance(stored_length, bool) or not isinstance(stored_length, Integral):
            errors.append(f"{case_id}: prompt_token_len is not integral")
        elif int(stored_length) != len(token_ids):
            errors.append(
                f"{case_id}: prompt_token_len {stored_length} != {len(token_ids)}"
            )
        if row["prompt_token_ids_sha256"] != sha256_prompt_token_ids(token_ids):
            errors.append(f"{case_id}: prompt_token_ids_sha256 mismatch")

    for name in ("case_id", "gsm8k_row_index", "question_sha256", "prompt_sha256"):
        if prompt_index[name].duplicated().any():
            errors.append(f"prompt_index column is not unique: {name}")

    if errors:
        preview = "\n".join(errors[:50])
        omitted = len(errors) - min(len(errors), 50)
        suffix = f"\n... {omitted} additional error(s)" if omitted else ""
        raise PromptProvenanceError(
            f"prompt_index validation found {len(errors)} error(s):\n{preview}{suffix}"
        )

    return {
        "n_rows": len(prompt_index),
        "mismatch_count": 0,
        "first_case_id": prompt_index.iloc[0]["case_id"],
        "last_case_id": prompt_index.iloc[-1]["case_id"],
        "query_template_sha256": FROZEN_QUERY_TEMPLATE_SHA256,
        "token_id_hash_encoding": "little-endian signed int64 bytes",
    }


def validate_prompt_index_against_sources(
    prompt_index: pd.DataFrame,
    gsm8k_test_path: str | Path,
    trajectory_dir: str | Path,
    tokenizer: Any,
    *,
    expected_count: int = 1319,
    trajectory_steps: int = 256,
    generation_length: int = 256,
) -> dict[str, Any]:
    """Rebuild from frozen sources and require exact field-by-field identity."""

    validate_prompt_index(prompt_index, expected_count=expected_count)
    expected = build_prompt_index(
        gsm8k_test_path,
        trajectory_dir,
        tokenizer,
        expected_count=expected_count,
        trajectory_steps=trajectory_steps,
        generation_length=generation_length,
    )
    mismatches: list[str] = []
    for position in range(expected_count):
        observed_row = prompt_index.iloc[position]
        expected_row = expected.iloc[position]
        for column in PROMPT_INDEX_COLUMNS:
            observed = observed_row[column]
            wanted = expected_row[column]
            if column == "prompt_token_ids":
                equal = _normalize_token_ids(observed) == _normalize_token_ids(wanted)
            else:
                equal = observed == wanted
            if not equal:
                mismatches.append(f"row {position} column {column}")
    if mismatches:
        raise PromptProvenanceError(
            f"prompt_index differs from frozen sources at {len(mismatches)} field(s): "
            + ", ".join(mismatches[:20])
        )
    summary = validate_prompt_index(prompt_index, expected_count=expected_count)
    summary["source_rebuild_match"] = True
    return summary


__all__ = [
    "PROMPT_INDEX_COLUMNS",
    "PromptProvenanceError",
    "build_prompt_index",
    "load_frozen_tokenizer",
    "sha256_prompt_token_ids",
    "sha256_text",
    "validate_local_tokenizer_snapshot",
    "validate_prompt_index",
    "validate_prompt_index_against_sources",
]
