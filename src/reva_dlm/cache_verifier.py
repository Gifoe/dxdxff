"""Strict verifier for a frozen READY_FOR_G2 handoff cache."""

from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
from numbers import Integral
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    AUDIT_CHECKPOINTS,
    CACHE_VERSION,
    GEN_LENGTH,
    G0_GATE_PROTOCOL,
    G1_GATE_PROTOCOL,
    GSM8K_REPO,
    GSM8K_REVISION,
    PREREGISTERED_PRIMARY_CHECKPOINT,
    TOTAL_STEPS,
)
from .parser import PARSER_NAME, PARSER_VERSION
from .progress import PROGRESS_MAPPING_RULE
from .prompt_state import FROZEN_QUERY_TEMPLATE_SHA256, MASK_TOKEN_ID
from .utils import read_json, sha256_file


REQUIRED_FILES = {
    "README.md",
    "manifest.json",
    "status.json",
    "source_manifest.json",
    "source_index.parquet",
    "dev_cases.parquet",
    "cpu_features.parquet",
    "g0_labels.parquet",
    "splits.json",
    "g2_train_ids.json",
    "g2_val_ids.json",
    "g3_holdout_ids.json",
    "feature_schema.json",
    "parser_config.json",
    "progress_mapping.json",
    "prompt_index.parquet",
    "sha256sums.txt",
    "state_semantics.json",
    "provenance/protocol.json",
    "provenance/source_code_manifest.json",
}

PROMPT_INDEX_COLUMNS = {
    "case_id",
    "gsm8k_row_index",
    "question_sha256",
    "prompt_sha256",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "prompt_token_len",
    "gsm8k_gt_answer_sha256",
}
STATE_SEMANTICS_FIELDS = {
    "schema_version",
    "state_name",
    "mask_token_id",
    "generation_length",
    "total_steps",
    "history_index_base",
    "step_number_base",
    "checkpoint_locator",
    "prompt_source",
    "pre_step_commits_rule",
    "pre_step_uncommitted_rule",
    "current_stop_now_rule",
    "current_commit_rule",
    "hidden_state_timing",
    "forbidden_future_inputs",
}
EXPECTED_STATE_NAME = "pre_forward_input_producing_raw_x0"
EXPECTED_PROMPT_SOURCE = "prompt_index.parquet.prompt_token_ids"
EXPECTED_MASK_TOKEN_ID = 126336
EXPECTED_CASE_COUNT = 1319

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_TRANSIENT_DIR_NAMES = {"__pycache__", ".pytest_cache", "logs"}
_TRANSIENT_FILE_NAMES = {".DS_Store", "Thumbs.db"}
_TRANSIENT_SUFFIXES = {
    ".bak",
    ".lock",
    ".log",
    ".part",
    ".pyc",
    ".swp",
    ".temp",
    ".tmp",
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"{name} missing columns: {sorted(missing)}")


def _read_id_list(path: Path) -> list[str]:
    value = read_json(path)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AssertionError(f"{path.name} must contain a JSON string list")
    if len(value) != len(set(value)):
        raise AssertionError(f"duplicate IDs in {path.name}")
    return value


def _verify_base_cache(
    cache_dir: Path, *, verify_source_hashes: bool = True
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    missing = sorted(name for name in REQUIRED_FILES if not (cache_dir / name).is_file())
    if missing:
        raise AssertionError(f"cache missing required files: {missing}")

    manifest = read_json(cache_dir / "manifest.json")
    status = read_json(cache_dir / "status.json")
    source_manifest = read_json(cache_dir / "source_manifest.json")
    if manifest.get("cache_version") != CACHE_VERSION:
        raise AssertionError("cache version mismatch")
    if status.get("cache_status") != "READY_FOR_G2":
        raise AssertionError("cache status is not READY_FOR_G2")
    if manifest.get("G0_gate_result") not in {"G0_PASS", "G0_STRONG_PASS"}:
        raise AssertionError(
            "ordinary READY requires G0_PASS/G0_STRONG_PASS; "
            "G0_EXPLORATORY_PASS is not eligible"
        )
    if manifest.get("G1_gate_result") not in {"G1_PASS", "G1_STRONG_PASS"}:
        raise AssertionError("G1 gate is not a pass")

    source_index = pd.read_parquet(cache_dir / "source_index.parquet")
    dev_cases = pd.read_parquet(cache_dir / "dev_cases.parquet")
    features = pd.read_parquet(cache_dir / "cpu_features.parquet")
    labels = pd.read_parquet(cache_dir / "g0_labels.parquet")
    _require_columns(
        source_index,
        {
            "case_id",
            "source_relative_path",
            "source_file_sha256",
            "source_size_bytes",
            "split",
            "prompt_token_len",
            "gt_answer",
        },
        "source_index",
    )
    _require_columns(
        labels,
        {
            "case_id",
            "checkpoint",
            "current_answer",
            "current_correct",
            "final_answer",
            "final_correct",
            "trajectory_type",
            "V",
            "continue_beneficial",
        },
        "g0_labels",
    )
    _require_columns(
        dev_cases,
        {"case_id", "checkpoint", "trajectory_type", "continue_beneficial"},
        "dev_cases",
    )
    if source_index["case_id"].duplicated().any():
        raise AssertionError("source case IDs are not unique")
    if labels.duplicated(["case_id", "checkpoint"]).any():
        raise AssertionError("duplicate G0 case/checkpoint rows")

    metadata_columns = {"case_id", "checkpoint", "split"}
    feature_columns = set(features.columns) - metadata_columns
    if not feature_columns or any(not column.startswith("feature_") for column in feature_columns):
        raise AssertionError("all feature columns must use the feature_ prefix")
    forbidden_fragments = (
        "final_correct",
        "trajectory_type",
        "continue_beneficial",
        "future_",
    )
    if any(any(fragment in column for fragment in forbidden_fragments) for column in features.columns):
        raise AssertionError("label/future column leaked into cpu_features")
    if "V" in features.columns:
        raise AssertionError("V leaked into cpu_features")

    split_info = read_json(cache_dir / "splits.json")
    dev_ids = set(split_info["DEV"])
    holdout_ids = set(split_info["G3_HOLDOUT"])
    train_ids = set(_read_id_list(cache_dir / "g2_train_ids.json"))
    val_ids = set(_read_id_list(cache_dir / "g2_val_ids.json"))
    frozen_holdout = set(_read_id_list(cache_dir / "g3_holdout_ids.json"))
    if dev_ids & holdout_ids or train_ids & val_ids or train_ids & holdout_ids or val_ids & holdout_ids:
        raise AssertionError("split overlap")
    if frozen_holdout != holdout_ids:
        raise AssertionError("g3_holdout_ids differs from frozen split")
    if train_ids | val_ids != dev_ids:
        raise AssertionError("G2 train/val do not exactly partition DEV")
    if set(labels["case_id"]) != dev_ids:
        raise AssertionError("G0 labels do not cover exactly DEV")
    if set(features["case_id"]) != dev_ids:
        raise AssertionError("CPU features do not cover exactly DEV")
    if set(dev_cases["case_id"]) != dev_ids:
        raise AssertionError("dev_cases does not cover exactly DEV")

    primary = float(manifest["primary_checkpoint"])
    if set(float(value) for value in features["checkpoint"].unique()) != {primary}:
        raise AssertionError("feature checkpoint differs from primary")
    if set(float(value) for value in dev_cases["checkpoint"].unique()) != {primary}:
        raise AssertionError("dev_cases checkpoint differs from primary")

    sum_lines = (cache_dir / "sha256sums.txt").read_text(encoding="utf-8").splitlines()
    if not sum_lines:
        raise AssertionError("empty sha256sums.txt")
    for line in sum_lines:
        expected, relative = line.split("  ", 1)
        artifact = cache_dir / relative
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise AssertionError(f"cache artifact hash mismatch: {relative}")

    if verify_source_hashes:
        project_root = cache_dir.parents[1]
        source_root = project_root / source_manifest["source_root_relative"]
        for row in source_index.itertuples(index=False):
            source_path = source_root / row.source_relative_path
            if not source_path.is_file():
                raise AssertionError(f"missing source file: {source_path}")
            if source_path.stat().st_size != int(row.source_size_bytes):
                raise AssertionError(f"source size mismatch: {source_path}")
            if sha256_file(source_path) != row.source_file_sha256:
                raise AssertionError(f"source hash mismatch: {source_path}")

    return {
        "cache_version": manifest["cache_version"],
        "n_source": int(len(source_index)),
        "n_dev": len(dev_ids),
        "n_g3_holdout": len(holdout_ids),
        "primary_checkpoint": primary,
    }


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"{name} must be a JSON object")
    return value


def _canonical_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise AssertionError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AssertionError(f"unsafe {label}: {value!r}")
    if path.as_posix() != value:
        raise AssertionError(f"non-canonical {label}: {value!r}")
    return value


def _is_transient(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        any(part in _TRANSIENT_DIR_NAMES for part in path.parts[:-1])
        or path.name in _TRANSIENT_FILE_NAMES
        or path.suffix.lower() in _TRANSIENT_SUFFIXES
        or path.name.endswith("~")
    )


def _regular_cache_files(cache_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in cache_dir.rglob("*"):
        relative = path.relative_to(cache_dir).as_posix()
        if path.is_symlink():
            raise AssertionError(f"symlinks are forbidden in immutable cache: {relative}")
        if path.is_file():
            if _is_transient(relative):
                raise AssertionError(
                    "temporary/log/editor artifacts are forbidden in immutable cache: "
                    f"{relative}"
                )
            files[relative] = path
    return files


def _parse_hash_manifest(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AssertionError("sha256sums.txt must be readable UTF-8") from exc
    if not lines:
        raise AssertionError("sha256sums.txt is empty")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _HASH_LINE_RE.fullmatch(line)
        if match is None:
            raise AssertionError(
                f"invalid sha256sums.txt line {line_number}; expected "
                "'<lowercase sha256>  <POSIX path>'"
            )
        digest, relative_raw = match.groups()
        relative = _canonical_relative_path(
            relative_raw, label=f"sha256sums path on line {line_number}"
        )
        if relative == "sha256sums.txt":
            raise AssertionError("sha256sums.txt must not hash itself")
        if relative in entries:
            raise AssertionError(f"duplicate hash entry: {relative}")
        entries[relative] = digest
    return entries


def _verify_hash_coverage(cache_dir: Path) -> int:
    """Require every non-transient artifact except the hash list itself."""

    manifest_path = cache_dir / "sha256sums.txt"
    if not manifest_path.is_file():
        raise AssertionError("cache missing required file: sha256sums.txt")
    all_files = _regular_cache_files(cache_dir)
    expected = set(all_files) - {"sha256sums.txt"}
    entries = _parse_hash_manifest(manifest_path)
    if set(entries) != expected:
        raise AssertionError(
            "sha256sums.txt must cover every cache artifact exactly once "
            f"(self excluded); missing={sorted(expected - set(entries))}, "
            f"extra={sorted(set(entries) - expected)}"
        )
    for relative in sorted(expected):
        if sha256_file(all_files[relative]) != entries[relative]:
            raise AssertionError(f"cache artifact hash mismatch: {relative}")
    return len(expected)


def _valid_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AssertionError(f"{label} must be a lowercase 64-character SHA256")
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_parser_and_status(cache_dir: Path, manifest: dict[str, Any]) -> None:
    parser = _require_object(read_json(cache_dir / "parser_config.json"), "parser_config")
    if parser.get("name") != PARSER_NAME or parser.get("version") != PARSER_VERSION:
        raise AssertionError(
            "parser_config name/version differs from the frozen canonical parser"
        )
    expected_parser_version = f"{PARSER_NAME}:{PARSER_VERSION}"
    if manifest.get("answer_parser_version") != expected_parser_version:
        raise AssertionError("manifest answer_parser_version differs from parser_config")
    marker_rule = parser.get("marker_rule")
    fallback_rule = parser.get("fallback_rule")
    if not isinstance(marker_rule, str) or not marker_rule.strip():
        raise AssertionError("parser_config marker_rule must be non-empty")
    if not isinstance(fallback_rule, str) or not fallback_rule.strip():
        raise AssertionError("parser_config fallback_rule must be non-empty")
    normalized_marker = " ".join(marker_rule.lower().split())
    if "answer:" not in normalized_marker or "####" not in normalized_marker:
        raise AssertionError("parser marker_rule must name both Answer: and ####")
    normalized_fallback = " ".join(fallback_rule.lower().split())
    if not all(fragment in normalized_fallback for fragment in ("empty", "whitespace")):
        raise AssertionError(
            "parser v1.1 fallback_rule must explicitly cover empty/whitespace suffixes"
        )
    if parser.get("same_parser_for_gt_mid_final") is not True:
        raise AssertionError("parser_config must use one parser for GT/mid/final")

    status = _require_object(read_json(cache_dir / "status.json"), "status.json")
    if status.get("G0_status") != manifest.get("G0_gate_result"):
        raise AssertionError("status G0_status differs from manifest G0_gate_result")
    if status.get("G1_status") != manifest.get("G1_gate_result"):
        raise AssertionError("status G1_status differs from manifest G1_gate_result")
    if status.get("verified") is not True:
        raise AssertionError("READY status must set verified=true")


def _prompt_token_array(value: Any, *, case_id: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        raw = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise AssertionError(f"prompt_token_ids must be a list for {case_id}")
    if any(
        isinstance(token, (bool, np.bool_)) or not isinstance(token, Integral)
        for token in raw
    ):
        raise AssertionError(f"prompt_token_ids contains a non-integer for {case_id}")
    try:
        return np.asarray(raw, dtype="<i8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise AssertionError(f"prompt token is outside int64 range for {case_id}") from exc


def _verify_prompt_index(cache_dir: Path, source_index: pd.DataFrame) -> int:
    try:
        prompts = pd.read_parquet(cache_dir / "prompt_index.parquet")
    except Exception as exc:  # pyarrow exception types vary by version
        raise AssertionError("prompt_index.parquet is unreadable") from exc
    _require_columns(prompts, PROMPT_INDEX_COLUMNS, "prompt_index")
    _require_columns(
        source_index,
        {"case_id", "prompt_token_len", "source_relative_path"},
        "source_index",
    )
    if len(prompts) != EXPECTED_CASE_COUNT or len(source_index) != EXPECTED_CASE_COUNT:
        raise AssertionError("prompt_index and source_index must each contain 1319 rows")
    if (
        prompts["case_id"].isna().any()
        or not prompts["case_id"].map(lambda value: isinstance(value, str)).all()
        or prompts["case_id"].duplicated().any()
    ):
        raise AssertionError("prompt_index case IDs must be unique strings")
    if source_index["case_id"].duplicated().any():
        raise AssertionError("source_index case IDs are not unique")
    if prompts["gsm8k_row_index"].isna().any() or not prompts[
        "gsm8k_row_index"
    ].map(
        lambda value: not isinstance(value, (bool, np.bool_))
        and isinstance(value, Integral)
    ).all():
        raise AssertionError("gsm8k_row_index must contain non-null integers")
    row_indices = prompts["gsm8k_row_index"].astype(np.int64)
    if row_indices.duplicated().any() or set(row_indices) != set(range(1319)):
        raise AssertionError("gsm8k_row_index must be unique and contiguous 0..1318")
    expected_ids = row_indices.map(lambda index: f"question_{int(index):04d}")
    if not np.array_equal(prompts["case_id"].to_numpy(), expected_ids.to_numpy()):
        raise AssertionError("prompt case_id does not match gsm8k_row_index")
    if set(prompts["case_id"]) != set(source_index["case_id"]):
        raise AssertionError("prompt_index IDs do not exactly match source_index IDs")

    for column in (
        "question_sha256",
        "prompt_sha256",
        "prompt_token_ids_sha256",
        "gsm8k_gt_answer_sha256",
    ):
        for row_number, value in enumerate(prompts[column]):
            _valid_sha256(value, label=f"prompt_index.{column}[{row_number}]")
    computed_lengths: list[int] = []
    for row in prompts.itertuples(index=False):
        tokens = _prompt_token_array(row.prompt_token_ids, case_id=str(row.case_id))
        computed_lengths.append(int(tokens.size))
        digest = hashlib.sha256(tokens.tobytes(order="C")).hexdigest()
        if digest != row.prompt_token_ids_sha256:
            raise AssertionError(f"prompt_token_ids_sha256 mismatch for {row.case_id}")
    prompt_lengths = pd.to_numeric(prompts["prompt_token_len"], errors="coerce")
    if prompt_lengths.isna().any() or not np.array_equal(
        prompt_lengths.to_numpy(dtype=np.int64), np.asarray(computed_lengths)
    ):
        raise AssertionError("prompt_token_len differs from prompt_token_ids length")
    prompt_by_id = prompts.set_index("case_id")["prompt_token_len"].astype(np.int64)
    source_by_id = (
        source_index.set_index("case_id")["prompt_token_len"]
        .reindex(prompt_by_id.index)
        .astype(np.int64)
    )
    if not np.array_equal(prompt_by_id.to_numpy(), source_by_id.to_numpy()):
        raise AssertionError("prompt lengths do not align with source_index")
    if "source_relative_path" in prompts.columns:
        prompt_paths = prompts.set_index("case_id")["source_relative_path"]
        source_paths = source_index.set_index("case_id")["source_relative_path"].reindex(
            prompt_paths.index
        )
        if not prompt_paths.equals(source_paths):
            raise AssertionError("prompt source paths do not align with source_index")
    return len(prompts)


def _read_string_id_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AssertionError(f"{label} must be a JSON string list")
    if len(value) != len(set(value)):
        raise AssertionError(f"duplicate IDs in {label}")
    return value


def _verify_source_splits(
    cache_dir: Path, source_index: pd.DataFrame, manifest: dict[str, Any]
) -> tuple[set[str], set[str]]:
    _require_columns(source_index, {"case_id", "split"}, "source_index")
    splits = _require_object(read_json(cache_dir / "splits.json"), "splits.json")
    if set(splits) != {"DEV", "G3_HOLDOUT"}:
        raise AssertionError("splits.json must contain exactly DEV and G3_HOLDOUT")
    dev_ids = set(_read_string_id_list(splits["DEV"], label="splits.json.DEV"))
    holdout_ids = set(
        _read_string_id_list(splits["G3_HOLDOUT"], label="splits.json.G3_HOLDOUT")
    )
    if dev_ids & holdout_ids or dev_ids | holdout_ids != set(source_index["case_id"]):
        raise AssertionError("splits.json does not disjointly partition source_index")
    if set(source_index["split"].dropna().astype(str)) != {"DEV", "G3_HOLDOUT"}:
        raise AssertionError("source_index split values are not exactly DEV/G3_HOLDOUT")
    source_dev = set(source_index.loc[source_index["split"] == "DEV", "case_id"])
    source_holdout = set(
        source_index.loc[source_index["split"] == "G3_HOLDOUT", "case_id"]
    )
    if source_dev != dev_ids or source_holdout != holdout_ids:
        raise AssertionError("source_index split assignments differ from splits.json")
    if int(manifest.get("n_dev", -1)) != len(dev_ids):
        raise AssertionError("manifest n_dev differs from source_index/splits")
    if int(manifest.get("n_g3_holdout", -1)) != len(holdout_ids):
        raise AssertionError("manifest n_g3_holdout differs from source_index/splits")

    features = pd.read_parquet(cache_dir / "cpu_features.parquet")
    _require_columns(features, {"case_id", "split"}, "cpu_features")
    if set(features["split"].dropna().astype(str)) != {"DEV"}:
        raise AssertionError("cpu_features split column must be DEV only")
    if set(features.loc[features["split"] == "DEV", "case_id"]) != dev_ids:
        raise AssertionError("cpu_features split assignments differ from source_index")
    return dev_ids, holdout_ids


def _nearest_half_up(progress: float, total_steps: int) -> int:
    rounded = int(
        (Decimal(str(progress)) * Decimal(total_steps) + Decimal("0.5"))
        .to_integral_value(rounding=ROUND_FLOOR)
    )
    return min(total_steps, max(1, rounded))


def _same_number(left: Any, right: Any, *, label: str) -> None:
    try:
        matches = np.isclose(float(left), float(right), rtol=0.0, atol=1e-12)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{label} is not numeric") from exc
    if not bool(matches):
        raise AssertionError(f"{label} mismatch: {left!r} != {right!r}")


def _verify_primary_progress(
    cache_dir: Path, manifest: dict[str, Any]
) -> dict[str, int | float]:
    mapping = _require_object(
        read_json(cache_dir / "progress_mapping.json"), "progress_mapping.json"
    )
    if manifest.get("progress_mapping_rule") != PROGRESS_MAPPING_RULE:
        raise AssertionError("manifest progress_mapping_rule differs from frozen protocol")
    if mapping.get("rule") != PROGRESS_MAPPING_RULE:
        raise AssertionError("progress_mapping rule differs from manifest/protocol")
    if manifest.get("steps") != TOTAL_STEPS or mapping.get("total_steps") != TOTAL_STEPS:
        raise AssertionError("total steps must be frozen at 256")
    if manifest.get("gen_length") != GEN_LENGTH:
        raise AssertionError("generation length must be frozen at 256")
    if mapping.get("block_major") is not True:
        raise AssertionError("progress mapping must use block-major order")
    if mapping.get("blocks") != 8 or mapping.get("steps_per_block") != 32:
        raise AssertionError("progress mapping must contain 8 blocks x 32 steps")
    _same_number(
        manifest.get("preregistered_primary_checkpoint"),
        PREREGISTERED_PRIMARY_CHECKPOINT,
        label="manifest preregistered primary checkpoint",
    )
    _same_number(
        mapping.get("preregistered_primary_checkpoint"),
        PREREGISTERED_PRIMARY_CHECKPOINT,
        label="mapping preregistered primary checkpoint",
    )
    try:
        primary = float(manifest["primary_checkpoint"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError("manifest primary_checkpoint is invalid") from exc
    _same_number(
        mapping.get("frozen_primary_checkpoint"), primary, label="frozen primary checkpoint"
    )

    audit = mapping.get("audit_checkpoints")
    if not isinstance(audit, list) or len(audit) != len(AUDIT_CHECKPOINTS):
        raise AssertionError("progress mapping must contain all five audit checkpoints")
    primary_entry: dict[str, Any] | None = None
    for expected_checkpoint, raw_entry in zip(AUDIT_CHECKPOINTS, audit, strict=True):
        entry = _require_object(raw_entry, "progress_mapping audit entry")
        _same_number(entry.get("checkpoint"), expected_checkpoint, label="audit checkpoint")
        step = _nearest_half_up(float(expected_checkpoint), TOTAL_STEPS)
        if entry.get("step_number") != step or entry.get("history_index") != step - 1:
            raise AssertionError("progress mapping step/history index is inconsistent")
        _same_number(
            entry.get("normalized_progress_actual"),
            step / TOTAL_STEPS,
            label="normalized_progress_actual",
        )
        if np.isclose(primary, float(expected_checkpoint), rtol=0.0, atol=1e-12):
            primary_entry = entry
    if primary_entry is None:
        raise AssertionError("primary checkpoint is absent from frozen audit mapping")

    features = pd.read_parquet(cache_dir / "cpu_features.parquet")
    dev_cases = pd.read_parquet(cache_dir / "dev_cases.parquet")
    for name, frame in (("cpu_features", features), ("dev_cases", dev_cases)):
        if "checkpoint" not in frame or not np.isclose(
            frame["checkpoint"].astype(float), primary, rtol=0.0, atol=1e-12
        ).all():
            raise AssertionError(f"{name} checkpoint differs from frozen primary")
    actual_progress = float(primary_entry["normalized_progress_actual"])
    if "feature_progress" in features and not np.isclose(
        features["feature_progress"].astype(float),
        actual_progress,
        rtol=0.0,
        atol=1e-12,
    ).all():
        raise AssertionError("feature_progress differs from primary mapping")
    if "feature_prefix_length" in features and not np.isclose(
        features["feature_prefix_length"].astype(float),
        float(primary_entry["step_number"]),
        rtol=0.0,
        atol=1e-12,
    ).all():
        raise AssertionError("feature_prefix_length differs from primary mapping")
    return {
        "checkpoint": primary,
        "history_index": int(primary_entry["history_index"]),
        "step_number": int(primary_entry["step_number"]),
        "normalized_progress_actual": actual_progress,
    }


def _require_semantic_fragments(
    value: Any, *, field: str, groups: tuple[tuple[str, ...], ...]
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise AssertionError(f"state_semantics.{field} must be a non-empty string")
    normalized = " ".join(value.lower().split())
    for alternatives in groups:
        if not any(fragment in normalized for fragment in alternatives):
            raise AssertionError(
                f"state_semantics.{field} must contain one of {alternatives}"
            )


def _verify_state_semantics(
    cache_dir: Path, primary_locator: dict[str, int | float]
) -> None:
    state = _require_object(
        read_json(cache_dir / "state_semantics.json"), "state_semantics.json"
    )
    missing = STATE_SEMANTICS_FIELDS - set(state)
    if missing:
        raise AssertionError(f"state_semantics missing fields: {sorted(missing)}")
    if state.get("schema_version") != "1.0.0":
        raise AssertionError("state_semantics schema_version must be 1.0.0")
    if state.get("state_name") != EXPECTED_STATE_NAME:
        raise AssertionError(f"state_name must be {EXPECTED_STATE_NAME!r}")
    fixed = {
        "mask_token_id": EXPECTED_MASK_TOKEN_ID,
        "generation_length": GEN_LENGTH,
        "total_steps": TOTAL_STEPS,
        "history_index_base": 0,
        "step_number_base": 1,
    }
    for field, expected in fixed.items():
        if state.get(field) != expected:
            raise AssertionError(f"state_semantics.{field} must be {expected!r}")
    if state.get("prompt_source") != EXPECTED_PROMPT_SOURCE:
        raise AssertionError(
            f"state_semantics.prompt_source must be {EXPECTED_PROMPT_SOURCE!r}"
        )
    locator = _require_object(state["checkpoint_locator"], "checkpoint_locator")
    for field in ("checkpoint", "history_index", "step_number"):
        if field not in locator:
            raise AssertionError(f"checkpoint_locator missing {field}")
    _same_number(locator["checkpoint"], primary_locator["checkpoint"], label="state checkpoint")
    if locator["history_index"] != primary_locator["history_index"]:
        raise AssertionError("state history_index differs from primary progress mapping")
    if locator["step_number"] != primary_locator["step_number"]:
        raise AssertionError("state step_number differs from primary progress mapping")
    if locator.get("progress_mapping_file", "progress_mapping.json") != (
        "progress_mapping.json"
    ):
        raise AssertionError("checkpoint_locator progress_mapping_file is invalid")

    _require_semantic_fragments(
        state["pre_step_commits_rule"],
        field="pre_step_commits_rule",
        groups=(("commit",), ("< t", "before t"), ("restore", "replay")),
    )
    _require_semantic_fragments(
        state["pre_step_uncommitted_rule"],
        field="pre_step_uncommitted_rule",
        groups=(("mask",), ("uncommitted", "not committed")),
    )
    _require_semantic_fragments(
        state["current_stop_now_rule"],
        field="current_stop_now_rule",
        groups=(("x0[t]", "x0_history[t]"), ("<= t", "through t"), ("commit",)),
    )
    _require_semantic_fragments(
        state["current_commit_rule"],
        field="current_commit_rule",
        groups=(("commit",), ("after",), ("forward",), ("exclude", "not include")),
    )
    _require_semantic_fragments(
        state["hidden_state_timing"],
        field="hidden_state_timing",
        groups=(("same",), ("forward",), ("x0[t]", "x0_history[t]")),
    )

    forbidden = state["forbidden_future_inputs"]
    if (
        not isinstance(forbidden, list)
        or not forbidden
        or not all(isinstance(item, str) and item.strip() for item in forbidden)
        or len(forbidden) != len(set(forbidden))
    ):
        raise AssertionError("forbidden_future_inputs must be a unique non-empty string list")
    items = [" ".join(item.lower().split()) for item in forbidden]
    future_markers = ("future", "> t", "after t")
    has_x0 = any("x0" in item and any(mark in item for mark in future_markers) for item in items)
    has_commits = any(
        ("commit" in item or "true_indices" in item)
        and any(mark in item for mark in future_markers)
        for item in items
    )
    joined = " | ".join(items)
    has_final = (
        "final answer" in joined
        or "final_answer" in joined
        or ("ans_posidx" in joined and "pred_token_id" in joined)
    )
    has_gt = "ground truth" in joined or "ground_truth" in joined or "gt_text" in joined
    if not (has_x0 and has_commits and has_final and "correct" in joined and has_gt):
        raise AssertionError(
            "forbidden_future_inputs must explicitly ban future x0, future commits, "
            "final-answer metadata, correctness, and ground truth"
        )


def _workspace_source_inventory(project_root: Path) -> dict[str, Path]:
    """Frozen provenance scope: pyproject plus all Python source and tests."""

    inventory: dict[str, Path] = {}
    pyproject = project_root / "pyproject.toml"
    if pyproject.is_file():
        inventory["pyproject.toml"] = pyproject
    for directory in ("src", "scripts", "tests"):
        root = project_root / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if path.is_file() and "__pycache__" not in path.parts:
                relative = path.relative_to(project_root).as_posix()
                if not _is_transient(relative):
                    inventory[relative] = path
    return inventory


def _source_manifest_entries(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict):
        if value.get("schema_version") not in {None, "1.0.0"}:
            raise AssertionError("source_code_manifest schema_version is unsupported")
        entries = value.get("files")
    else:
        entries = None
    if not isinstance(entries, list) or not entries:
        raise AssertionError("source_code_manifest must contain a non-empty files list")
    if not all(isinstance(entry, dict) for entry in entries):
        raise AssertionError("source_code_manifest entries must be JSON objects")
    return entries


def _verify_source_code_provenance(
    cache_dir: Path,
    project_root: Path,
    *,
    compare_workspace: bool,
) -> int:
    manifest_path = cache_dir / "provenance" / "source_code_manifest.json"
    entries = _source_manifest_entries(read_json(manifest_path))
    required = {
        "workspace_relative_path",
        "snapshot_relative_path",
        "sha256",
        "size_bytes",
    }
    by_workspace: dict[str, dict[str, Any]] = {}
    listed_snapshots: set[str] = set()
    for index, entry in enumerate(entries):
        missing = required - set(entry)
        if missing:
            raise AssertionError(
                f"source_code_manifest entry {index} missing fields: {sorted(missing)}"
            )
        workspace_relative = _canonical_relative_path(
            entry["workspace_relative_path"], label=f"source workspace path {index}"
        )
        snapshot_relative = _canonical_relative_path(
            entry["snapshot_relative_path"], label=f"source snapshot path {index}"
        )
        expected_snapshot = f"provenance/source_snapshot/{workspace_relative}"
        if snapshot_relative != expected_snapshot:
            raise AssertionError(
                "source snapshot path must mirror workspace path exactly: "
                f"{snapshot_relative!r} != {expected_snapshot!r}"
            )
        if workspace_relative in by_workspace or snapshot_relative in listed_snapshots:
            raise AssertionError("duplicate source snapshot manifest entry")
        digest = _valid_sha256(entry["sha256"], label=f"source entry {index} sha256")
        size = entry["size_bytes"]
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 0:
            raise AssertionError(f"source entry {index} size_bytes must be non-negative int")
        snapshot = cache_dir / PurePosixPath(snapshot_relative)
        if not snapshot.is_file() or snapshot.is_symlink():
            raise AssertionError(f"missing regular source snapshot: {snapshot_relative}")
        if snapshot.stat().st_size != int(size):
            raise AssertionError(f"source snapshot size mismatch: {snapshot_relative}")
        if sha256_file(snapshot) != digest:
            raise AssertionError(f"source snapshot hash mismatch: {snapshot_relative}")
        by_workspace[workspace_relative] = entry
        listed_snapshots.add(snapshot_relative)

    snapshot_root = cache_dir / "provenance" / "source_snapshot"
    actual_snapshots = {
        path.relative_to(cache_dir).as_posix()
        for path in snapshot_root.rglob("*")
        if path.is_file()
    }
    if actual_snapshots != listed_snapshots:
        raise AssertionError(
            "source_code_manifest does not exactly cover source_snapshot; "
            f"missing={sorted(actual_snapshots - listed_snapshots)}, "
            f"extra={sorted(listed_snapshots - actual_snapshots)}"
        )

    if compare_workspace:
        inventory = _workspace_source_inventory(project_root)
        if set(inventory) != set(by_workspace):
            raise AssertionError(
                "source snapshot does not exactly match workspace source inventory; "
                f"missing={sorted(set(inventory) - set(by_workspace))}, "
                f"extra={sorted(set(by_workspace) - set(inventory))}"
            )
        for relative, path in inventory.items():
            entry = by_workspace[relative]
            if path.stat().st_size != int(entry["size_bytes"]):
                raise AssertionError(f"workspace source size differs from snapshot: {relative}")
            if sha256_file(path) != entry["sha256"]:
                raise AssertionError(f"workspace source hash differs from snapshot: {relative}")
    return len(entries)


def _verify_source_manifest_metadata(
    cache_dir: Path,
    project_root: Path,
    source_index: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    compare_workspace: bool,
) -> None:
    source_manifest = _require_object(
        read_json(cache_dir / "source_manifest.json"), "source_manifest.json"
    )
    if source_manifest.get("source_file_count") != len(source_index):
        raise AssertionError("source_manifest source_file_count differs from source_index")
    total_bytes = int(pd.to_numeric(source_index["source_size_bytes"]).sum())
    if source_manifest.get("source_total_bytes") != total_bytes:
        raise AssertionError("source_manifest source_total_bytes differs from source_index")
    if source_manifest.get("dataset_revision") != manifest.get("dataset_revision_SHA"):
        raise AssertionError("dataset revision differs between source and cache manifests")
    if source_manifest.get("exact_trajectory_folder") != manifest.get(
        "exact_trajectory_folder"
    ):
        raise AssertionError("trajectory folder differs between source and cache manifests")
    if source_manifest.get("tokenizer_weights_downloaded") is not False:
        raise AssertionError("tokenizer_weights_downloaded must be false")
    for field in (
        "download_manifest_sha256",
        "tokenizer_manifest_sha256",
        "gsm8k_manifest_sha256",
        "gsm8k_test_sha256",
        "query_template_sha256",
    ):
        _valid_sha256(source_manifest.get(field), label=f"source_manifest.{field}")

    if source_manifest.get("gsm8k_repo") != GSM8K_REPO:
        raise AssertionError("source_manifest GSM8K repo differs from frozen protocol")
    if source_manifest.get("gsm8k_revision") != GSM8K_REVISION:
        raise AssertionError("source_manifest GSM8K revision differs from frozen protocol")
    expected_gsm8k_relative = "data/raw/GSM8K/main/test-00000-of-00001.parquet"
    if source_manifest.get("gsm8k_test_relative_path") != expected_gsm8k_relative:
        raise AssertionError("source_manifest GSM8K test path is not frozen main/test")
    test_size = source_manifest.get("gsm8k_test_size_bytes")
    if isinstance(test_size, (bool, np.bool_)) or not isinstance(test_size, Integral):
        raise AssertionError("source_manifest gsm8k_test_size_bytes must be an integer")
    if int(test_size) <= 0:
        raise AssertionError("source_manifest gsm8k_test_size_bytes must be positive")
    if source_manifest.get("query_template_sha256") != FROZEN_QUERY_TEMPLATE_SHA256:
        raise AssertionError("query template SHA differs from frozen Prophet template")
    if source_manifest.get("mask_token_id") != MASK_TOKEN_ID:
        raise AssertionError("source_manifest mask token ID differs from frozen state")
    if source_manifest.get("prompt_index_rows") != EXPECTED_CASE_COUNT:
        raise AssertionError("source_manifest prompt_index_rows must equal 1319")
    if source_manifest.get("state_contract") != EXPECTED_STATE_NAME:
        raise AssertionError("source_manifest state_contract differs from state_semantics")
    cutoff = source_manifest.get("pre_step_commit_cutoff")
    if not isinstance(cutoff, str) or "< t" not in " ".join(cutoff.lower().split()):
        raise AssertionError("source_manifest pre_step_commit_cutoff must be strict < t")

    frozen_gsm8k_path = (
        cache_dir
        / "provenance"
        / "frozen_context"
        / "data"
        / "metadata"
        / "gsm8k_manifest.json"
    )
    if not frozen_gsm8k_path.is_file():
        raise AssertionError("cache is missing frozen GSM8K manifest context")
    if sha256_file(frozen_gsm8k_path) != source_manifest["gsm8k_manifest_sha256"]:
        raise AssertionError("frozen GSM8K manifest hash differs from source_manifest")
    frozen_gsm8k = _require_object(read_json(frozen_gsm8k_path), "gsm8k_manifest")
    if (
        frozen_gsm8k.get("dataset_repo") != GSM8K_REPO
        or frozen_gsm8k.get("dataset_revision") != GSM8K_REVISION
        or frozen_gsm8k.get("configuration") != "main"
        or frozen_gsm8k.get("split") != "test"
    ):
        raise AssertionError("frozen GSM8K manifest repo/revision/config/split mismatch")
    files = frozen_gsm8k.get("files")
    if not isinstance(files, list):
        raise AssertionError("frozen GSM8K manifest files must be a list")
    target_entries = [
        entry
        for entry in files
        if isinstance(entry, dict)
        and entry.get("relative_path") == "main/test-00000-of-00001.parquet"
    ]
    if len(target_entries) != 1:
        raise AssertionError("frozen GSM8K manifest must contain exactly one test parquet")
    target = target_entries[0]
    if target.get("sha256") != source_manifest["gsm8k_test_sha256"]:
        raise AssertionError("GSM8K test SHA differs between frozen/source manifests")
    if target.get("size_bytes") != source_manifest["gsm8k_test_size_bytes"]:
        raise AssertionError("GSM8K test size differs between frozen/source manifests")

    if compare_workspace:
        metadata = {
            "download_manifest_sha256": project_root
            / "data"
            / "metadata"
            / "download_manifest.json",
            "tokenizer_manifest_sha256": project_root
            / "data"
            / "metadata"
            / "tokenizer_manifest.json",
            "gsm8k_manifest_sha256": project_root
            / "data"
            / "metadata"
            / "gsm8k_manifest.json",
        }
        for field, path in metadata.items():
            if not path.is_file() or sha256_file(path) != source_manifest[field]:
                raise AssertionError(f"workspace metadata hash mismatch: {path}")
        workspace_test = project_root / PurePosixPath(expected_gsm8k_relative)
        if not workspace_test.is_file():
            raise AssertionError(f"workspace GSM8K test file is missing: {workspace_test}")
        if workspace_test.stat().st_size != int(source_manifest["gsm8k_test_size_bytes"]):
            raise AssertionError("workspace GSM8K test size differs from source_manifest")
        if sha256_file(workspace_test) != source_manifest["gsm8k_test_sha256"]:
            raise AssertionError("workspace GSM8K test hash differs from source_manifest")


def _verify_code_bundle_and_protocol(
    cache_dir: Path, manifest: dict[str, Any]
) -> str:
    source_manifest = _require_object(
        read_json(cache_dir / "source_manifest.json"), "source_manifest.json"
    )
    code_manifest = read_json(cache_dir / "provenance" / "source_code_manifest.json")
    code_digest = _canonical_json_sha256(code_manifest)
    for name, value in (
        ("source_manifest", source_manifest.get("analysis_code_bundle_sha256")),
        ("manifest", manifest.get("analysis_code_bundle_sha256")),
    ):
        _valid_sha256(value, label=f"{name}.analysis_code_bundle_sha256")
        if value != code_digest:
            raise AssertionError(
                f"{name} analysis_code_bundle_sha256 differs from canonical source manifest"
            )
    if source_manifest["analysis_code_bundle_sha256"] != manifest[
        "analysis_code_bundle_sha256"
    ]:
        raise AssertionError("analysis code bundle hash differs across manifests")

    protocol = _require_object(
        read_json(cache_dir / "provenance" / "protocol.json"), "protocol.json"
    )
    if protocol.get("schema_version") != "1.0.0":
        raise AssertionError("protocol.json schema_version must be 1.0.0")
    if protocol.get("G0") != G0_GATE_PROTOCOL or protocol.get("G1") != G1_GATE_PROTOCOL:
        raise AssertionError("protocol.json differs from current frozen G0/G1 protocols")
    expected_g0 = _canonical_json_sha256(G0_GATE_PROTOCOL)
    expected_g1 = _canonical_json_sha256(G1_GATE_PROTOCOL)
    if protocol.get("G0_sha256") != expected_g0:
        raise AssertionError("protocol.json G0 hash differs from frozen G0 protocol")
    if protocol.get("G1_sha256") != expected_g1:
        raise AssertionError("protocol.json G1 hash differs from frozen G1 protocol")
    if manifest.get("G0_gate_protocol_sha256") != expected_g0:
        raise AssertionError("manifest G0 gate protocol hash mismatch")
    if manifest.get("G1_gate_protocol_sha256") != expected_g1:
        raise AssertionError("manifest G1 gate protocol hash mismatch")
    return code_digest


def verify_cache(
    cache_dir: Path, *, verify_source_hashes: bool = True
) -> dict[str, Any]:
    """Verify a frozen ordinary ``READY_FOR_G2`` handoff cache.

    The cache must include exact prompt tokens, fixed pre-forward state timing,
    complete artifact hashes, and a complete source snapshot.  Setting
    ``verify_source_hashes=False`` skips raw trajectory hashing and comparison
    of the snapshot against the live workspace, but never skips internal cache
    hashes or semantic checks. ``G0_EXPLORATORY_PASS`` is deliberately not an
    ordinary READY-eligible status.
    """

    cache_dir = Path(cache_dir).resolve()
    result = _verify_base_cache(
        cache_dir, verify_source_hashes=bool(verify_source_hashes)
    )
    n_artifacts = _verify_hash_coverage(cache_dir)
    manifest = _require_object(read_json(cache_dir / "manifest.json"), "manifest.json")
    _verify_parser_and_status(cache_dir, manifest)
    source_index = pd.read_parquet(cache_dir / "source_index.parquet")
    n_prompts = _verify_prompt_index(cache_dir, source_index)
    _verify_source_splits(cache_dir, source_index, manifest)
    primary_locator = _verify_primary_progress(cache_dir, manifest)
    _verify_state_semantics(cache_dir, primary_locator)
    project_root = cache_dir.parents[1]
    n_snapshots = _verify_source_code_provenance(
        cache_dir,
        project_root,
        compare_workspace=bool(verify_source_hashes),
    )
    code_bundle_sha256 = _verify_code_bundle_and_protocol(cache_dir, manifest)
    _verify_source_manifest_metadata(
        cache_dir,
        project_root,
        source_index,
        manifest,
        compare_workspace=bool(verify_source_hashes),
    )
    return {
        **result,
        "n_prompts": n_prompts,
        "n_cache_artifacts": n_artifacts,
        "n_source_snapshots": n_snapshots,
        "state_name": EXPECTED_STATE_NAME,
        "analysis_code_bundle_sha256": code_bundle_sha256,
    }


__all__ = ["REQUIRED_FILES", "verify_cache"]
