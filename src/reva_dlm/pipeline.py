"""End-to-end CPU-only Mini G0/G1 execution and cache freezing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from transformers import AutoTokenizer

from .cache_verifier import verify_cache
from .config import (
    AUDIT_CHECKPOINTS,
    BLOCK_LENGTH,
    CACHE_DIR,
    CACHE_VERSION,
    DATASET_REPO,
    DATASET_REVISION,
    FEATURE_WINDOW,
    FROZEN_EXPLORATORY_CHECKPOINT,
    GEN_LENGTH,
    G0_GATE_PROTOCOL,
    G1_GATE_PROTOCOL,
    GSM8K_DIR,
    GSM8K_REPO,
    GSM8K_REVISION,
    GSM8K_TEST_FILE,
    PREREGISTERED_PRIMARY_CHECKPOINT,
    PROJECT_ROOT,
    PROPHET_REVISION,
    RANDOM_SEED,
    RAW_REPO_DIR,
    TOKENIZER_DIR,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    TOTAL_STEPS,
    TRAJECTORY_DIR,
    TRAJECTORY_FOLDER,
)
from .evaluation import compute_binary_metrics, evaluate_models
from .features import extract_causal_features
from .parser import (
    PARSER_NAME,
    PARSER_VERSION,
    answers_equal,
    extract_gsm8k_answer,
)
from .prompt_provenance import build_prompt_index, validate_prompt_index
from .prompt_state import (
    FROZEN_QUERY_TEMPLATE_SHA256,
    MASK_TOKEN_ID,
    reconstruct_pre_step_input,
)
from .progress import PROGRESS_MAPPING_RULE, build_progress_mapping
from .splits import assign_dev_holdout, make_g2_train_val
from .trajectory import (
    flatten_true_indices_history,
    flatten_x0_history,
    load_trajectory,
    reconstruct_states,
)
from .utils import git_revision, read_json, sha256_file, utc_now, write_json


LOGGER = logging.getLogger("reva_dlm")
EXPECTED_KEYS = {
    "x0_history",
    "true_indices_history",
    "correct",
    "pred_text",
    "ans_posidx",
    "pred_ans",
    "gt_text",
    "pred_token_id",
    "gt_token_id",
    "prompt_token_len",
}
CASE_RE = re.compile(r"^(question_\d{4})_steps_256\.pt$")
ANSWER_MARKER_RE = re.compile(r"(?i)(?:####|answer\s*:)")


@dataclass(frozen=True)
class G0Decision:
    status: str
    selected_checkpoint: float | None
    reason: str


def _setup_logging() -> Path:
    log_path = PROJECT_ROOT / "logs" / "reva_mini_g0_g1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)
    return log_path


def _emit_terminal_summary(lines: list[str]) -> None:
    """Print and persist the exact final terminal contract."""

    path = PROJECT_ROOT / "reports" / "final_terminal_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _case_id(path: Path) -> str:
    match = CASE_RE.match(path.name)
    if not match:
        raise AssertionError(f"unexpected trajectory filename: {path.name}")
    return match.group(1)


def _last_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    found = -1
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            found = index
    return found


def _assert_cpu_only() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if torch.version.cuda is not None:
        raise RuntimeError(f"CPU-only torch required, got CUDA build {torch.version.cuda}")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA is available inside the CPU-only Mini G0/G1 process")


def _verify_frozen_downloads() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = PROJECT_ROOT / "data" / "metadata" / "download_manifest.json"
    tokenizer_manifest_path = (
        PROJECT_ROOT / "data" / "metadata" / "tokenizer_manifest.json"
    )
    gsm8k_manifest_path = PROJECT_ROOT / "data" / "metadata" / "gsm8k_manifest.json"
    source_manifest = read_json(manifest_path)
    tokenizer_manifest = read_json(tokenizer_manifest_path)
    gsm8k_manifest = read_json(gsm8k_manifest_path)
    expected = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "target_folder": TRAJECTORY_FOLDER,
        "expected_cases": 1319,
        "endpoint": "https://hf-mirror.com",
    }
    for key, value in expected.items():
        if source_manifest.get(key) != value:
            raise AssertionError(f"download manifest {key} mismatch")
    if tokenizer_manifest.get("model_repo") != TOKENIZER_REPO:
        raise AssertionError("tokenizer repo mismatch")
    if tokenizer_manifest.get("model_revision") != TOKENIZER_REVISION:
        raise AssertionError("tokenizer revision mismatch")
    if tokenizer_manifest.get("weights_downloaded") is not False:
        raise AssertionError("tokenizer manifest does not rule out model weights")
    gsm8k_expected = {
        "dataset_repo": GSM8K_REPO,
        "dataset_revision": GSM8K_REVISION,
        "configuration": "main",
        "split": "test",
        "endpoint": "https://hf-mirror.com",
    }
    for key, value in gsm8k_expected.items():
        if gsm8k_manifest.get(key) != value:
            raise AssertionError(f"GSM8K manifest {key} mismatch")
    if set(gsm8k_manifest.get("allowlist", [])) != {
        "README.md",
        "main/test-00000-of-00001.parquet",
    }:
        raise AssertionError("GSM8K manifest allowlist mismatch")

    entries = source_manifest["files"]
    if len(entries) != 1320:
        raise AssertionError(f"expected README + 1319 trajectories, got {len(entries)}")
    for index, entry in enumerate(entries, 1):
        path = RAW_REPO_DIR / entry["relative_path"]
        if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]):
            raise AssertionError(f"source file missing/size mismatch: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise AssertionError(f"source SHA256 mismatch: {path}")
        if index % 250 == 0:
            LOGGER.info("Verified source hashes %d/%d", index, len(entries))

    forbidden = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    for entry in tokenizer_manifest["files"]:
        path = TOKENIZER_DIR / entry["relative_path"]
        if path.suffix.lower() in forbidden:
            raise AssertionError(f"weight-like tokenizer artifact: {path.name}")
        if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]):
            raise AssertionError(f"tokenizer file missing/size mismatch: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise AssertionError(f"tokenizer SHA256 mismatch: {path}")
    if any(path.suffix.lower() in forbidden for path in TOKENIZER_DIR.rglob("*")):
        raise AssertionError("model weight found in tokenizer-only directory")
    for entry in gsm8k_manifest["files"]:
        path = GSM8K_DIR / entry["relative_path"]
        if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]):
            raise AssertionError(f"GSM8K file missing/size mismatch: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise AssertionError(f"GSM8K SHA256 mismatch: {path}")
    prophet_checkout = PROJECT_ROOT / "external" / "Prophet"
    if git_revision(prophet_checkout) != PROPHET_REVISION:
        raise AssertionError("Prophet checkout is not at the audited commit")
    return source_manifest, tokenizer_manifest, gsm8k_manifest


def _write_environment_report() -> Path:
    report_path = PROJECT_ROOT / "reports" / "environment.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [
        f"created_at_utc={utc_now()}",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"processor={platform.processor()}",
        f"torch={torch.__version__}",
        f"torch_cuda_build={torch.version.cuda}",
        f"torch_cuda_available={torch.cuda.is_available()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"scikit_learn={sklearn.__version__}",
        f"transformers={transformers.__version__}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '<unset>')}",
        f"dataset_revision={DATASET_REVISION}",
        f"tokenizer_revision={TOKENIZER_REVISION}",
        f"prophet_revision={PROPHET_REVISION}",
        "",
        "--- pip freeze --all ---",
        freeze.rstrip(),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _load_local_tokenizer() -> Any:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)


def _validate_schema(record: dict[str, Any], path: Path) -> dict[str, Any]:
    if set(record) != EXPECTED_KEYS:
        raise AssertionError(
            f"{path.name} keys differ from frozen actual schema: {sorted(record)}"
        )
    prompt_length = int(record["prompt_token_len"])
    x0 = record["x0_history"]
    commits = record["true_indices_history"]
    if len(x0) != 8 or len(commits) != 8:
        raise AssertionError(f"{path.name}: expected 8 blocks")
    expected_width = prompt_length + GEN_LENGTH
    for block_index, (x0_block, commit_block) in enumerate(zip(x0, commits, strict=True)):
        if not isinstance(x0_block, torch.Tensor):
            raise AssertionError(f"{path.name}: x0 block is not a tensor")
        if x0_block.dtype != torch.int64 or tuple(x0_block.shape) != (32, expected_width):
            raise AssertionError(
                f"{path.name}: bad x0 block {block_index} {x0_block.dtype}/{tuple(x0_block.shape)}"
            )
        if len(commit_block) != 32:
            raise AssertionError(f"{path.name}: commit block does not contain 32 steps")
        if any(tuple(torch.as_tensor(value).shape) != (1, 2) for value in commit_block):
            raise AssertionError(f"{path.name}: expected one [1,2] commit per step")
    flat_commits = flatten_true_indices_history(commits)
    absolute_positions = [int(step[0, 1]) for step in flat_commits]
    relative_positions = [position - prompt_length for position in absolute_positions]
    if sorted(relative_positions) != list(range(GEN_LENGTH)):
        raise AssertionError(f"{path.name}: commits do not cover generation exactly once")
    return {
        "prompt_token_len": prompt_length,
        "sequence_length": expected_width,
        "blocks": len(x0),
        "steps_per_block": 32,
        "total_steps": len(flat_commits),
        "generation_length": GEN_LENGTH,
    }


def _trajectory_type(current_correct: bool, final_correct: bool) -> str:
    if current_correct and final_correct:
        return "STABLE"
    if not current_correct and final_correct:
        return "RESCUE"
    if current_correct and not final_correct:
        return "CORRUPTION"
    return "UNRECOVERED"


def _audit_sources_and_build_g0(
    tokenizer: Any,
    source_manifest: dict[str, Any],
    prompt_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    files = sorted(TRAJECTORY_DIR.glob("*.pt"))
    if len(files) != 1319:
        raise AssertionError(f"expected 1319 source files, got {len(files)}")
    expected_ids = {f"question_{index:04d}" for index in range(1319)}
    if {_case_id(path) for path in files} != expected_ids:
        raise AssertionError("case IDs are not the complete question_0000..1318 sequence")

    hash_entries = {
        entry["relative_path"]: entry
        for entry in source_manifest["files"]
        if entry["relative_path"].endswith(".pt")
    }
    mapping = build_progress_mapping(AUDIT_CHECKPOINTS, TOTAL_STEPS)
    mapping_by_checkpoint = {float(item["checkpoint"]): item for item in mapping}
    validate_prompt_index(prompt_index)
    prompt_by_case = prompt_index.set_index("case_id", verify_integrity=True)
    sample_rng = random.Random(RANDOM_SEED)
    sampled_ids = set(sample_rng.sample(sorted(expected_ids), 20))

    source_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    corruption_evidence: list[dict[str, Any]] = []
    parser_disagreements: list[dict[str, Any]] = []
    sample_inspections: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    raw_final_differences: list[int] = []
    reconstruction_exact = 0
    raw_last_exact = 0
    invalid_answer_positions = 0
    dev_stored_agreement = 0
    stored_false_canonical_true = 0
    stored_true_canonical_false = 0
    dev_final_empty_marker_fallback = 0
    dev_mid_empty_marker_fallback = 0
    pre_step_state_validated = 0
    n_dev = 0
    n_holdout = 0

    for file_index, path in enumerate(files, 1):
        case_id = _case_id(path)
        split = assign_dev_holdout(case_id, RANDOM_SEED)
        if split == "DEV":
            n_dev += 1
        else:
            n_holdout += 1
        record = load_trajectory(path)
        structural = _validate_schema(record, path)
        prompt_length = structural["prompt_token_len"]
        prompt_row = prompt_by_case.loc[case_id]
        if int(prompt_row["gsm8k_row_index"]) != int(case_id.split("_")[1]):
            raise AssertionError(f"{case_id}: GSM8K row binding mismatch")
        if int(prompt_row["prompt_token_len"]) != prompt_length:
            raise AssertionError(f"{case_id}: prompt index/source length mismatch")
        prompt_lengths.append(prompt_length)
        prompt_token_ids = [int(value) for value in prompt_row["prompt_token_ids"]]
        audit_history_index = int(
            mapping_by_checkpoint[PREREGISTERED_PRIMARY_CHECKPOINT]["history_index"]
        )
        pre_step_state = reconstruct_pre_step_input(
            record,
            prompt_token_ids,
            audit_history_index,
            generation_length=GEN_LENGTH,
            mask_token_id=MASK_TOKEN_ID,
        )
        if pre_step_state.committed_count_before_step != audit_history_index:
            raise AssertionError(f"{case_id}: pre-step commit cutoff mismatch")
        pre_step_state_validated += 1
        states = reconstruct_states(record, generation_length=GEN_LENGTH)
        if len(states) != TOTAL_STEPS or states[-1].committed_count != GEN_LENGTH:
            raise AssertionError(f"{path.name}: incomplete reconstructed state history")
        final_ids = states[-1].committed_token_ids.tolist()
        final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
        exact = final_text == record["pred_text"]
        reconstruction_exact += int(exact)

        raw_last = flatten_x0_history(record["x0_history"])[-1, prompt_length:]
        final_tensor = states[-1].committed_token_ids
        difference = int((raw_last != final_tensor).sum().item())
        raw_final_differences.append(difference)
        raw_last_exact += int(difference == 0)

        pred_token_ids = [int(value) for value in record["pred_token_id"]]
        found_index = _last_subsequence(final_ids, pred_token_ids)
        invalid_answer_positions += int(found_index < 0)

        relative_path = path.relative_to(RAW_REPO_DIR).as_posix()
        source_entry = hash_entries[relative_path]
        gt_answer = extract_gsm8k_answer(record["gt_text"])
        source_rows.append(
            {
                "case_id": case_id,
                "source_relative_path": relative_path,
                "source_file_sha256": source_entry["sha256"],
                "source_size_bytes": int(source_entry["size_bytes"]),
                "split": split,
                "gsm8k_row_index": int(prompt_row["gsm8k_row_index"]),
                "prompt_token_len": prompt_length,
                "gt_answer": gt_answer,
                "dataset_revision": DATASET_REVISION,
                "decode_policy": "low_confidence",
                "constraint_policy": "none",
                "steps": TOTAL_STEPS,
                "gen_length": GEN_LENGTH,
                "block_length": BLOCK_LENGTH,
                "temperature": 0.0,
                "cfg_scale": 0.0,
            }
        )

        if case_id in sampled_ids:
            sample_inspections.append(
                {
                    "case_id": case_id,
                    "keys": sorted(record),
                    "x0_shapes": [list(value.shape) for value in record["x0_history"]],
                    "commit_steps": sum(len(value) for value in record["true_indices_history"]),
                    "prompt_token_len": prompt_length,
                    "final_reconstruction_exact": exact,
                    "raw_last_token_difference": difference,
                    "gen_ids_present": "gen_ids" in record,
                    "pred_token_subsequence_found": found_index >= 0,
                }
            )

        # G3 outcomes are deliberately not parsed, summarized, or placed in a
        # label table.  Structural reconstruction equality above is not an
        # outcome statistic.
        if split == "DEV":
            final_answer = extract_gsm8k_answer(final_text)
            final_correct = answers_equal(final_text, record["gt_text"])
            stored_correct = bool(record["correct"])
            dev_stored_agreement += int(stored_correct == final_correct)
            stored_false_canonical_true += int(not stored_correct and final_correct)
            stored_true_canonical_false += int(stored_correct and not final_correct)
            final_markers = list(ANSWER_MARKER_RE.finditer(final_text))
            dev_final_empty_marker_fallback += int(
                bool(final_markers)
                and not final_text[final_markers[-1].end() :].strip()
                and final_answer is not None
            )
            if stored_correct != final_correct:
                parser_disagreements.append(
                    {
                        "case_id": case_id,
                        "gt_answer": gt_answer,
                        "stored_pred_ans": record["pred_ans"],
                        "canonical_final_answer": final_answer,
                        "stored_correct": stored_correct,
                        "canonical_correct": final_correct,
                    }
                )

            selected_ids = [
                states[int(mapping_by_checkpoint[float(checkpoint)]["history_index"])]
                .candidate_token_ids.tolist()
                for checkpoint in AUDIT_CHECKPOINTS
            ]
            checkpoint_texts = tokenizer.batch_decode(
                selected_ids, skip_special_tokens=True
            )
            for checkpoint, current_text in zip(
                AUDIT_CHECKPOINTS, checkpoint_texts, strict=True
            ):
                current_answer = extract_gsm8k_answer(current_text)
                current_markers = list(ANSWER_MARKER_RE.finditer(current_text))
                dev_mid_empty_marker_fallback += int(
                    bool(current_markers)
                    and not current_text[current_markers[-1].end() :].strip()
                    and current_answer is not None
                )
                current_correct = answers_equal(current_text, record["gt_text"])
                kind = _trajectory_type(current_correct, final_correct)
                value = int(final_correct) - int(current_correct)
                label_rows.append(
                    {
                        "case_id": case_id,
                        "checkpoint": float(checkpoint),
                        "current_answer": current_answer,
                        "current_correct": bool(current_correct),
                        "final_answer": final_answer,
                        "final_correct": bool(final_correct),
                        "trajectory_type": kind,
                        "V": int(value),
                        "continue_beneficial": bool(value > 0),
                    }
                )
                if kind == "CORRUPTION":
                    corruption_evidence.append(
                        {
                            "case_id": case_id,
                            "checkpoint": float(checkpoint),
                            "current_answer": current_answer,
                            "final_answer": final_answer,
                            "gt_answer": gt_answer,
                            "current_has_answer_marker": bool(
                                ANSWER_MARKER_RE.search(current_text)
                            ),
                            "final_has_answer_marker": bool(
                                ANSWER_MARKER_RE.search(final_text)
                            ),
                            "current_text_tail": current_text[-500:].replace("\n", " "),
                            "final_text_tail": final_text[-500:].replace("\n", " "),
                        }
                    )

        if file_index % 100 == 0 or file_index == len(files):
            LOGGER.info("Schema/G0 processing %d/%d", file_index, len(files))

    consistency = reconstruction_exact / len(files)
    if consistency < 0.99:
        raise AssertionError(f"reconstruction consistency {consistency:.4f} is below 0.99")
    source_index = pd.DataFrame(source_rows).sort_values("case_id").reset_index(drop=True)
    labels = pd.DataFrame(label_rows).sort_values(
        ["case_id", "checkpoint"]
    ).reset_index(drop=True)
    evidence = pd.DataFrame(corruption_evidence)
    disagreements = pd.DataFrame(parser_disagreements)
    if set(labels["case_id"]) != set(source_index.query("split == 'DEV'")["case_id"]):
        raise AssertionError("DEV label coverage mismatch")

    schema_summary = {
        "n_source_files": len(files),
        "n_dev": n_dev,
        "n_g3_holdout": n_holdout,
        "actual_keys": sorted(EXPECTED_KEYS),
        "advertised_but_absent_keys": ["gen_ids"],
        "x0_organization": "8 block-major tensors x 32 local steps",
        "x0_shape_template": "[32, prompt_token_len + 256]",
        "x0_dtype": "torch.int64",
        "true_indices_organization": "8 blocks x 32 steps x Tensor[1,2]",
        "prompt_token_len_min": min(prompt_lengths),
        "prompt_token_len_max": max(prompt_lengths),
        "reconstruction_exact_count": reconstruction_exact,
        "reconstruction_consistency": consistency,
        "raw_last_exact_count": raw_last_exact,
        "raw_last_exact_rate": raw_last_exact / len(files),
        "raw_last_mean_token_difference": float(np.mean(raw_final_differences)),
        "raw_last_min_token_difference": min(raw_final_differences),
        "raw_last_max_token_difference": max(raw_final_differences),
        "invalid_final_answer_anchor_count": invalid_answer_positions,
        "dev_stored_correct_agreement_count": dev_stored_agreement,
        "dev_stored_correct_agreement_rate": dev_stored_agreement / n_dev,
        "dev_parser_disagreement_count": len(parser_disagreements),
        "dev_stored_false_canonical_true_count": stored_false_canonical_true,
        "dev_stored_true_canonical_false_count": stored_true_canonical_false,
        "dev_final_empty_marker_fallback_count": dev_final_empty_marker_fallback,
        "dev_mid_empty_marker_fallback_count": dev_mid_empty_marker_fallback,
        "prompt_index_validated_count": int(len(prompt_index)),
        "prompt_index_mismatch_count": 0,
        "query_template_sha256": FROZEN_QUERY_TEMPLATE_SHA256,
        "mask_token_id": MASK_TOKEN_ID,
        "pre_step_state_validated_count": pre_step_state_validated,
        "sample_inspections": sample_inspections,
        "g3_correctness_evaluated": False,
        "g3_used_for_selection": False,
        "g3_structural_validation_performed": True,
        "raw_source_contains_outcome_fields": True,
    }
    return source_index, labels, schema_summary, evidence, disagreements


def _summarize_g0(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint in AUDIT_CHECKPOINTS:
        subset = labels[np.isclose(labels["checkpoint"], checkpoint)]
        counts = subset["trajectory_type"].value_counts()
        n_total = len(subset)
        n_stable = int(counts.get("STABLE", 0))
        n_rescue = int(counts.get("RESCUE", 0))
        n_corruption = int(counts.get("CORRUPTION", 0))
        n_unrecovered = int(counts.get("UNRECOVERED", 0))
        mid_acc = float(subset["current_correct"].mean())
        final_acc = float(subset["final_correct"].mean())
        rcr = n_corruption / n_total
        rsr = n_rescue / n_total
        current_correct_n = n_stable + n_corruption
        current_wrong_n = n_rescue + n_unrecovered
        rows.append(
            {
                "checkpoint": float(checkpoint),
                "n_total": n_total,
                "mid_acc": mid_acc,
                "final_acc": final_acc,
                "n_stable": n_stable,
                "n_rescue": n_rescue,
                "n_corruption": n_corruption,
                "n_unrecovered": n_unrecovered,
                "rcr_joint": rcr,
                "rsr_joint": rsr,
                "corruption_given_current_correct": (
                    n_corruption / current_correct_n if current_correct_n else float("nan")
                ),
                "rescue_given_current_wrong": (
                    n_rescue / current_wrong_n if current_wrong_n else float("nan")
                ),
                "net_refinement_gain": final_acc - mid_acc,
            }
        )
    summary = pd.DataFrame(rows)
    difference = np.abs(
        summary["net_refinement_gain"]
        - (summary["rsr_joint"] - summary["rcr_joint"])
    )
    if float(difference.max()) > 1e-12:
        raise AssertionError("net gain != RSR - RCR")
    return summary


def _decide_g0(summary: pd.DataFrame) -> G0Decision:
    primary = summary[np.isclose(summary["checkpoint"], PREREGISTERED_PRIMARY_CHECKPOINT)].iloc[0]
    if (
        float(primary["rcr_joint"]) >= 0.03
        and int(primary["n_corruption"]) >= 30
        and int(primary["n_rescue"]) >= 30
    ):
        return G0Decision(
            "G0_STRONG_PASS",
            PREREGISTERED_PRIMARY_CHECKPOINT,
            "Preregistered 50% checkpoint met RCR>=0.03 and both count thresholds.",
        )
    if float(primary["rcr_joint"]) < 0.01 and float(summary["rcr_joint"].max()) < 0.01:
        return G0Decision(
            "G0_FAIL",
            None,
            "RCR at 50% and every predefined checkpoint was below 0.01.",
        )

    # Do not select a checkpoint again on parser-v1.1 results.  The 0.60 point
    # and its adjacent 0.50 support rule were selected on the parser-v1.0 DEV
    # diagnostic and frozen before this rerun.  Passing this branch permits an
    # explicitly exploratory G1 run, not an ordinary READY cache.
    exploratory = summary[
        np.isclose(summary["checkpoint"], FROZEN_EXPLORATORY_CHECKPOINT)
    ].iloc[0]
    adjacent = summary[
        np.isclose(
            summary["checkpoint"],
            float(G0_GATE_PROTOCOL["exploratory_required_adjacent_checkpoint"]),
        )
    ].iloc[0]
    minimum_rcr = float(G0_GATE_PROTOCOL["exploratory_min_rcr_joint"])
    minimum_corruption = int(G0_GATE_PROTOCOL["exploratory_min_corruption"])
    if (
        float(exploratory["rcr_joint"]) >= minimum_rcr
        and int(exploratory["n_corruption"]) >= minimum_corruption
        and int(exploratory["n_rescue"])
        >= int(G0_GATE_PROTOCOL["exploratory_min_rescue"])
        and float(adjacent["rcr_joint"]) >= minimum_rcr
        and int(adjacent["n_corruption"]) >= minimum_corruption
    ):
        return G0Decision(
            "G0_EXPLORATORY_PASS",
            FROZEN_EXPLORATORY_CHECKPOINT,
            "The formal 50% strong gate failed. The already-selected 60% DEV "
            "checkpoint retained adjacent 50% support under parser v1.1; this is "
            "same-DEV, post-diagnostic, selection-biased exploratory evidence only.",
        )
    return G0Decision(
        "BORDERLINE_REVIEW_REQUIRED",
        None,
        "Corruption existed but did not satisfy either the strong or explicit contiguous-region criterion.",
    )


def _write_schema_report(schema: dict[str, Any]) -> Path:
    path = PROJECT_ROOT / "reports" / "schema_audit.md"
    sample_rows = pd.DataFrame(
        [
            {
                "case_id": item["case_id"],
                "prompt_len": item["prompt_token_len"],
                "commit_steps": item["commit_steps"],
                "final_exact": item["final_reconstruction_exact"],
                "raw_last_diff": item["raw_last_token_difference"],
                "gen_ids_present": item["gen_ids_present"],
                "pred_anchor_found": item["pred_token_subsequence_found"],
            }
            for item in schema["sample_inspections"]
        ]
    )
    text = f"""# ReVA Mini schema audit

## Frozen source

- Dataset: `{DATASET_REPO}`
- Revision: `{DATASET_REVISION}`
- Exact folder: `{TRAJECTORY_FOLDER}`
- Files: {schema['n_source_files']} (`DEV={schema['n_dev']}`, frozen `G3_HOLDOUT={schema['n_g3_holdout']}`)
- G3 correctness/trajectory labels evaluated: **{schema['g3_correctness_evaluated']}**
- G3 rows used for feature/model/gate selection: **{schema['g3_used_for_selection']}**
- Structural validation ran on all source files: **{schema['g3_structural_validation_performed']}**
- Raw holdout sources contain outcome-bearing fields: **{schema['raw_source_contains_outcome_fields']}**

## Actual schema and organization

Every file has the same ten keys: `{', '.join(schema['actual_keys'])}`. The dataset card advertises `gen_ids`, but it is absent from every target file. `x0_history` is {schema['x0_organization']} with shape template `{schema['x0_shape_template']}` and dtype `{schema['x0_dtype']}`. `true_indices_history` is {schema['true_indices_organization']}. Prompt length ranges from {schema['prompt_token_len_min']} to {schema['prompt_token_len_max']} tokens.

All {schema['prompt_index_validated_count']} exact prompts were rebuilt from frozen `openai/gsm8k` rows using the Prophet template SHA256 `{schema['query_template_sha256']}` and the local tokenizer snapshot. Row index, canonical GT, prompt length, and x0 width had {schema['prompt_index_mismatch_count']} mismatches. The pre-forward input contract (exact prompt + commits at indices `<t` + masks elsewhere) was reconstructed for {schema['pre_step_state_validated_count']} trajectories. Prompt token IDs and hashes are retained for exact future state reconstruction; mask token ID is {schema['mask_token_id']}.

The official generator stores raw argmax `x0` before restoring previously committed tokens. Accordingly, raw `x0_history[-1]` is not the final output: only {schema['raw_last_exact_count']}/{schema['n_source_files']} files are token-identical to the final replay, with mean {schema['raw_last_mean_token_difference']:.3f} differing tokens (range {schema['raw_last_min_token_difference']}–{schema['raw_last_max_token_difference']}).

## Reconstruction validation

Commit replay followed by local tokenizer decoding exactly matches stored `pred_text` for {schema['reconstruction_exact_count']}/{schema['n_source_files']} files ({schema['reconstruction_consistency']:.3%}). This exceeds the 99% prerequisite. The stop-now state at step `t` is defined as all `<=t` committed tokens fixed to their replayed values and all remaining positions filled from that step's raw `x0`; this matches Prophet's early-exit fill semantics. It does not use final `ans_posidx`, final `pred_token_id`, or any future commit.

The final answer anchor is semantically invalid in {schema['invalid_final_answer_anchor_count']} files because the collector converts subsequence-search `-1` into `prompt_token_len - 1`. Neither `ans_posidx` nor `pred_token_id` is used for labels or features.

## Parser audit on DEV only

Parser v1.1 is a protocol revision made after the v1.0 audit exposed an empty-terminal-marker edge case and was frozen before this rerun. It agrees with the stored collector flag for {schema['dev_stored_correct_agreement_count']}/{schema['n_dev']} DEV cases ({schema['dev_stored_correct_agreement_rate']:.3%}). Direction-specific disagreements are: stored false → canonical true = {schema['dev_stored_false_canonical_true_count']}; stored true → canonical false = {schema['dev_stored_true_canonical_false_count']}. Empty-marker fallback applied to {schema['dev_final_empty_marker_fallback_count']} DEV final text(s) and {schema['dev_mid_empty_marker_fallback_count']} DEV checkpoint text(s). Stored `correct` is never used as a G0/G1 label; the same v1.1 parser is used for GT, current, and final text.

## Deterministic 20-case structural inspection

{_markdown_table(sample_rows)}
"""
    path.write_text(text, encoding="utf-8")
    return path


def _plot_g0(summary: pd.DataFrame) -> None:
    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(summary["checkpoint"], summary["rcr_joint"], marker="o", label="RCR (corruption)")
    plt.plot(summary["checkpoint"], summary["rsr_joint"], marker="o", label="RSR (rescue)")
    plt.xlabel("Normalized decoding progress")
    plt.ylabel("Joint probability on DEV")
    plt.xticks(summary["checkpoint"])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "g0_rcr_rsr_vs_progress.png", dpi=180)
    plt.close()

    categories = ["n_stable", "n_rescue", "n_corruption", "n_unrecovered"]
    labels = ["STABLE", "RESCUE", "CORRUPTION", "UNRECOVERED"]
    x = np.arange(len(summary))
    width = 0.19
    plt.figure(figsize=(8.0, 4.8))
    for offset, (column, label) in enumerate(zip(categories, labels, strict=True)):
        plt.bar(x + (offset - 1.5) * width, summary[column], width, label=label)
    plt.xticks(x, [f"{value:.0%}" for value in summary["checkpoint"]])
    plt.xlabel("Normalized decoding progress")
    plt.ylabel("DEV trajectory count")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(figure_dir / "g0_transition_counts.png", dpi=180)
    plt.close()


def _write_g0_report(
    summary: pd.DataFrame,
    decision: G0Decision,
    schema: dict[str, Any],
    evidence: pd.DataFrame,
) -> Path:
    path = PROJECT_ROOT / "reports" / "G0_REPORT.md"
    selected_evidence = (
        evidence[np.isclose(evidence["checkpoint"], decision.selected_checkpoint)]
        if decision.selected_checkpoint is not None and not evidence.empty
        else evidence.iloc[0:0]
    )
    marker_ok = (
        int(
            (
                selected_evidence["current_has_answer_marker"]
                & selected_evidence["final_has_answer_marker"]
            ).sum()
        )
        if not selected_evidence.empty
        else 0
    )
    text = f"""# Mini G0 trajectory audit

## Protocol

Only the outcome-blind SHA256 split's DEV partition ({schema['n_dev']} cases) is summarized. The {schema['n_g3_holdout']} G3 holdout IDs were frozen before label analysis and no holdout correctness or trajectory-type statistic was computed or used. Their raw `.pt` files physically contain outcome-bearing fields, so the information is accessible rather than cryptographically sealed. Progress uses `{PROGRESS_MAPPING_RULE}` over the official block-major 256-step history.

## Results

{_markdown_table(summary)}

For every row, `net_refinement_gain == RSR_joint - RCR_joint` to numerical tolerance. The preregistered checkpoint remains 0.50 for the strong gate.

## Gate decision

- Status: **{decision.status}**
- Frozen G1 checkpoint: **{decision.selected_checkpoint if decision.selected_checkpoint is not None else 'none'}**
- Reason: {decision.reason}

This is not a strong pass unless the status explicitly says `G0_STRONG_PASS`. `G0_EXPLORATORY_PASS` permits the preexperiment to complete G1, but it is not eligible for an ordinary `READY_FOR_G2` cache. At the frozen exploratory checkpoint there are {len(selected_evidence)} reconstructed corruption cases; {marker_ok}/{len(selected_evidence)} have an explicit answer marker in both current and final decoded text. All evidence rows are retained in `reports/g0_corruption_evidence.csv`; no final-answer anchor was used.

The prompt authorized qualitative DEV-side selection for a continuous 40%–60%-area corruption region. The 60% point and operational support rule (joint RCR >=1% and >=10 corruptions at 50% and 60%) came from the parser-v1.0 DEV diagnostic and were frozen before this parser-v1.1 rerun. The checkpoint is not reselected on the new labels. This remains same-DEV, post-diagnostic and selection-biased, not independent validation or preregistration.

## Scientific limitation

These trajectories come from the public GSM8K test set. This is feasibility evidence, not a clean confirmatory benchmark protocol. A later paper must regenerate development trajectories outside its final evaluation set.
"""
    path.write_text(text, encoding="utf-8")
    return path


def _feature_descriptions() -> dict[str, str]:
    return {
        "progress": "Completed iterations / frozen total of 256 (constant at a fixed checkpoint).",
        "prefix_length": "Number of observed iterations through t (constant at a fixed checkpoint).",
        "current_masked_ratio": "Fraction not yet committed after iteration t.",
        "current_committed_ratio": "Fraction committed after iteration t.",
        "current_candidate_token_length": "Token count of the complete stop-now candidate.",
        "current_text_length": "Decoded character count at t.",
        "current_text_token_length": "Whitespace-delimited decoded text length at t.",
        "current_answer_present": "Whether the canonical parser finds an answer at t.",
        "current_answer_length": "Canonical answer character length.",
        "current_answer_token_length": "Whitespace-delimited canonical answer length.",
        "current_answer_persistence": "Consecutive prefix steps with the current parsed answer.",
        "current_answer_persistence_normalized": "Persistence divided by observed prefix length.",
        "current_answer_first_seen_time": "First prefix index containing the current answer.",
        "current_answer_first_seen_normalized": "First-seen index normalized by the observed prefix.",
        "recent_answer_change_count": "Parsed-answer changes in the last 16 causal transitions.",
        "recent_answer_change_rate": "Fraction of last 16 transitions changing parsed answer.",
        "recent_answer_edit_distance": "Mean normalized character edit distance over recent answers.",
        "current_answer_stability": "One minus most recent parsed-answer edit distance.",
        "recent_answer_stability": "Mean recent parsed-answer stability.",
        "recent_answer_stability_slope": "Linear slope of recent parsed-answer stability.",
        "current_token_change_rate": "Normalized token edit distance from t-1 to t.",
        "recent_token_change_rate": "Mean normalized token edit distance in the recent window.",
        "current_token_stability": "One minus current token change rate.",
        "recent_stability": "Mean recent full-candidate token stability.",
        "recent_stability_slope": "Linear slope of recent full-candidate token stability.",
        "current_commit_speed": "Latest causal increase in committed ratio.",
        "recent_commit_speed": "Mean causal increase in committed ratio over the recent window.",
        "past_oscillation_count": "Returns to a previously left non-empty parsed answer through t.",
        "distinct_answers_seen": "Number of distinct non-empty parsed answers observed through t.",
    }


def _extract_primary_features(
    tokenizer: Any,
    source_index: pd.DataFrame,
    primary_checkpoint: float,
) -> pd.DataFrame:
    mapping = {
        float(item["checkpoint"]): item
        for item in build_progress_mapping(AUDIT_CHECKPOINTS, TOTAL_STEPS)
    }
    checkpoint_index = int(mapping[float(primary_checkpoint)]["history_index"])
    dev_index = source_index[source_index["split"] == "DEV"].sort_values("case_id")
    rows: list[dict[str, Any]] = []
    for processed, source_row in enumerate(dev_index.itertuples(index=False), 1):
        path = RAW_REPO_DIR / source_row.source_relative_path
        record = load_trajectory(path)
        states = reconstruct_states(record, generation_length=GEN_LENGTH)
        prefix_states = states[: checkpoint_index + 1]
        candidates = [state.candidate_token_ids for state in prefix_states]
        committed_masks = [state.committed_mask for state in prefix_states]
        texts = tokenizer.batch_decode(
            [state.candidate_token_ids.tolist() for state in prefix_states],
            skip_special_tokens=True,
        )
        parsed_answers = [extract_gsm8k_answer(text) for text in texts]
        values = extract_causal_features(
            candidates,
            committed_masks,
            parsed_answers,
            texts,
            checkpoint_index,
            window=FEATURE_WINDOW,
            total_steps=TOTAL_STEPS,
        )
        rows.append(
            {
                "case_id": source_row.case_id,
                "checkpoint": float(primary_checkpoint),
                "split": "DEV",
                **{f"feature_{name}": float(value) for name, value in values.items()},
            }
        )
        if processed % 50 == 0 or processed == len(dev_index):
            LOGGER.info("Causal feature extraction %d/%d", processed, len(dev_index))
    features = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    if features.drop(columns=["case_id", "checkpoint", "split"]).isna().any().any():
        raise AssertionError("NaN in causal feature matrix")
    return features


def _best_model(metrics: pd.DataFrame) -> pd.Series:
    return metrics.sort_values(
        ["auroc", "auprc", "model"], ascending=[False, False, True]
    ).iloc[0]


def _auroc_ci(result: dict[str, Any], model: str) -> tuple[float, float]:
    row = result["bootstrap_ci"][
        (result["bootstrap_ci"]["model"] == model)
        & (result["bootstrap_ci"]["metric"] == "auroc")
    ].iloc[0]
    return float(row["ci_lower"]), float(row["ci_upper"])


def _decide_g1_gate(
    overall_auc: float,
    overall_ci_lower: float,
    overall_ci_upper: float,
    matched_best: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Apply the frozen, machine-readable G1 decision rule."""

    key_names = tuple(G1_GATE_PROTOCOL["key_matched_tasks"])
    key_stably_high = all(
        task in matched_best
        and matched_best[task]["auroc"]
        >= float(G1_GATE_PROTOCOL["matched_fail_auroc_min"])
        and matched_best[task]["auroc_ci_lower"]
        >= float(G1_GATE_PROTOCOL["matched_fail_ci_lower_min"])
        for task in key_names
    )
    minimum_count_evaluable_matched = [
        value
        for value in matched_best.values()
        if value["n_positive"]
        >= int(G1_GATE_PROTOCOL["minimum_count_evaluable_matched_positive"])
    ]
    overall_stably_high = (
        overall_auc >= float(G1_GATE_PROTOCOL["primary_fail_auroc_min"])
        and overall_ci_lower >= float(G1_GATE_PROTOCOL["primary_fail_ci_lower_min"])
    )
    if overall_ci_upper <= float(G1_GATE_PROTOCOL["strong_primary_ci_upper_max"]) and all(
        value["auroc"] < float(G1_GATE_PROTOCOL["matched_fail_auroc_min"])
        for value in minimum_count_evaluable_matched
    ):
        return (
            "G1_STRONG_PASS",
            "The primary cheap-feature AUROC upper confidence limit was <=0.70 "
            "and no minimum-count matched task reached 0.80.",
        )
    if overall_stably_high and key_stably_high:
        return (
            "G1_FAIL",
            "Cheap features were stably >=0.80 overall and both frozen key "
            "matched controls were stably >=0.80.",
        )
    return (
        "G1_PASS",
        "Cheap features did not stably solve both frozen key matched controls; "
        "the simple proxy set is incomplete.",
    )


def _run_g1(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    primary_checkpoint: float,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_labels = labels[np.isclose(labels["checkpoint"], primary_checkpoint)].copy()
    merged = primary_labels.merge(features, on=["case_id", "checkpoint"], validate="one_to_one")
    feature_columns = [column for column in features.columns if column.startswith("feature_")]
    X = merged[feature_columns]
    y = merged["continue_beneficial"].astype(int)
    case_ids = merged["case_id"]
    primary_result = evaluate_models(
        X,
        y,
        case_ids,
        random_state=RANDOM_SEED,
        max_splits=5,
        n_repeats=3,
        n_bootstrap=1000,
        include_single_features=True,
    )
    primary_result["metrics"].to_csv(
        PROJECT_ROOT / "reports" / "g1_primary_metrics.csv", index=False
    )
    primary_result["bootstrap_ci"].to_csv(
        PROJECT_ROOT / "reports" / "g1_primary_bootstrap_ci.csv", index=False
    )
    primary_result["oof_predictions"].to_parquet(
        PROJECT_ROOT / "reports" / "g1_primary_oof_predictions.parquet", index=False
    )
    primary_result["fold_predictions"].to_parquet(
        PROJECT_ROOT / "reports" / "g1_primary_fold_predictions.parquet", index=False
    )

    task_specs = [
        (
            "A_same_current_wrong",
            ~merged["current_correct"],
            merged["trajectory_type"].eq("RESCUE"),
            "RESCUE vs UNRECOVERED within current-wrong",
        ),
        (
            "B_same_current_correct",
            merged["current_correct"],
            merged["trajectory_type"].eq("CORRUPTION"),
            "CORRUPTION vs STABLE within current-correct",
        ),
        (
            "C_same_final_correct",
            merged["final_correct"],
            merged["trajectory_type"].eq("RESCUE"),
            "RESCUE vs STABLE within final-correct",
        ),
        (
            "D_same_final_wrong",
            ~merged["final_correct"],
            merged["trajectory_type"].eq("CORRUPTION"),
            "CORRUPTION vs UNRECOVERED within final-wrong",
        ),
    ]
    matched_metrics: list[pd.DataFrame] = []
    matched_cis: list[pd.DataFrame] = []
    matched_oof: list[pd.DataFrame] = []
    matched_configs: dict[str, Any] = {}
    matched_results: dict[str, dict[str, Any]] = {}
    for task_index, (task, mask, target, description) in enumerate(task_specs):
        subset = merged.loc[mask].copy()
        subset_target = target.loc[mask].astype(int)
        counts = subset_target.value_counts()
        if len(counts) < 2 or int(counts.min()) < 2:
            matched_configs[task] = {
                "status": "NOT_EVALUABLE",
                "description": description,
                "class_counts": {str(key): int(value) for key, value in counts.items()},
            }
            continue
        result = evaluate_models(
            subset[feature_columns],
            subset_target,
            subset["case_id"],
            random_state=RANDOM_SEED + task_index + 1,
            max_splits=5,
            n_repeats=3,
            n_bootstrap=1000,
            include_single_features=False,
        )
        metric_frame = result["metrics"].copy()
        metric_frame.insert(0, "description", description)
        metric_frame.insert(0, "task", task)
        ci_frame = result["bootstrap_ci"].copy()
        ci_frame.insert(0, "task", task)
        oof_frame = result["oof_predictions"].copy()
        oof_frame.insert(0, "task", task)
        matched_metrics.append(metric_frame)
        matched_cis.append(ci_frame)
        matched_oof.append(oof_frame)
        matched_configs[task] = {"description": description, **result["cv_config"]}
        matched_results[task] = result

    matched_metric_frame = pd.concat(matched_metrics, ignore_index=True)
    matched_ci_frame = pd.concat(matched_cis, ignore_index=True)
    matched_oof_frame = pd.concat(matched_oof, ignore_index=True)
    matched_metric_frame.to_csv(
        PROJECT_ROOT / "reports" / "g1_matched_metrics.csv", index=False
    )
    matched_ci_frame.to_csv(
        PROJECT_ROOT / "reports" / "g1_matched_bootstrap_ci.csv", index=False
    )
    matched_oof_frame.to_parquet(
        PROJECT_ROOT / "reports" / "g1_matched_oof_predictions.parquet", index=False
    )
    write_json(
        PROJECT_ROOT / "reports" / "g1_cv_protocol.json",
        {
            "primary": primary_result["cv_config"],
            "matched": matched_configs,
            "threshold": 0.5,
            "preprocessing": "fit inside each training fold",
            "G3_holdout_used": False,
            "gate_protocol": G1_GATE_PROTOCOL,
            "gate_protocol_sha256": _canonical_json_sha256(G1_GATE_PROTOCOL),
        },
    )

    best = _best_model(primary_result["metrics"])
    best_lower, best_upper = _auroc_ci(primary_result, str(best["model"]))
    matched_best: dict[str, Any] = {}
    for task, result in matched_results.items():
        row = _best_model(result["metrics"])
        lower, upper = _auroc_ci(result, str(row["model"]))
        matched_best[task] = {
            "model": str(row["model"]),
            "auroc": float(row["auroc"]),
            "auroc_ci_lower": lower,
            "auroc_ci_upper": upper,
            "auprc": float(row["auprc"]),
            "n_samples": int(row["n_samples"]),
            "n_positive": int(row["n_positive"]),
            "n_negative": int(row["n_negative"]),
        }

    overall_auc = float(best["auroc"])
    g1_status, gate_reason = _decide_g1_gate(
        overall_auc, best_lower, best_upper, matched_best
    )
    best_fold_rows: list[dict[str, Any]] = []
    best_folds = primary_result["fold_predictions"][
        primary_result["fold_predictions"]["model"].eq(str(best["model"]))
    ]
    for (repeat, fold), fold_frame in best_folds.groupby(["repeat", "fold"], sort=True):
        fold_metrics = compute_binary_metrics(
            fold_frame["y_true"], fold_frame["y_probability"]
        )
        best_fold_rows.append(
            {
                "model": str(best["model"]),
                "repeat": int(repeat),
                "fold": int(fold),
                "n_samples": int(len(fold_frame)),
                "n_positive": int(fold_frame["y_true"].sum()),
                **fold_metrics,
            }
        )
    best_fold_metrics = pd.DataFrame(best_fold_rows)
    best_fold_metrics.to_csv(
        PROJECT_ROOT / "reports" / "g1_best_model_fold_metrics.csv", index=False
    )
    gate = {
        "status": g1_status,
        "reason": gate_reason,
        "best_model": str(best["model"]),
        "best_cheap_auroc": overall_auc,
        "best_cheap_auroc_ci_lower": best_lower,
        "best_cheap_auroc_ci_upper": best_upper,
        "best_cheap_auprc": float(best["auprc"]),
        "matched_best": matched_best,
        "primary_prevalence": float(y.mean()),
        "n_primary": int(len(y)),
        "gate_protocol_version": G1_GATE_PROTOCOL["version"],
        "gate_protocol_sha256": _canonical_json_sha256(G1_GATE_PROTOCOL),
        "best_fold_auroc_min": float(best_fold_metrics["auroc"].min()),
        "best_fold_auroc_max": float(best_fold_metrics["auroc"].max()),
        "best_fold_auroc_mean": float(best_fold_metrics["auroc"].mean()),
        "best_fold_auroc_std": float(best_fold_metrics["auroc"].std(ddof=1)),
    }
    return gate, primary_result, matched_metric_frame, matched_ci_frame, primary_labels


def _write_g1_report(
    gate: dict[str, Any],
    primary_result: dict[str, Any],
    matched_metrics: pd.DataFrame,
    features: pd.DataFrame,
    primary_checkpoint: float,
) -> Path:
    path = PROJECT_ROOT / "reports" / "G1_REPORT.md"
    metrics = primary_result["metrics"].copy()
    full_models = metrics[metrics["feature"] == "__all__"]
    top_singles = metrics[metrics["feature"] != "__all__"].nlargest(5, "auroc")
    display = pd.concat([full_models, top_singles], ignore_index=True)[
        [
            "model",
            "n_samples",
            "n_positive",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "f1",
            "brier",
        ]
    ]
    matched_display = matched_metrics[
        [
            "task",
            "model",
            "n_samples",
            "n_positive",
            "n_negative",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "f1",
            "brier",
        ]
    ]
    feature_columns = [column for column in features if column.startswith("feature_")]
    constant_features = [
        column for column in feature_columns if features[column].nunique(dropna=False) <= 1
    ]
    descriptions = _feature_descriptions()
    feature_rows = pd.DataFrame(
        [
            {
                "feature": f"feature_{name}",
                "definition": description,
                "constant_at_primary": f"feature_{name}" in constant_features,
            }
            for name, description in descriptions.items()
        ]
    )
    text = f"""# Mini G1 cheap causal-feature audit

## Target and checkpoint

The primary action target is `CONTINUE_BENEFICIAL = 1 iff V_t > 0` (RESCUE); STABLE, CORRUPTION, and UNRECOVERED are negative. G1 uses the DEV-selected and now frozen checkpoint **{primary_checkpoint:.2f}**. G3 holdout rows are absent from every fit, feature selection, threshold decision, and performance report.

## Causal features

{_markdown_table(feature_rows)}

Because this public trajectory configuration commits exactly one token per iteration, nominal progress, prefix length, masked ratio, committed ratio, and commit speed are constant at a fixed checkpoint. They are retained as schema/audit features but cannot provide discrimination. No confidence proxy is present in the `.pt` schema, and no model weights were downloaded.

## Evaluation protocol

- RepeatedStratifiedKFold: {primary_result['cv_config']['effective_splits']} folds × {primary_result['cv_config']['effective_repeats']} repeats.
- Every imputer/scaler/classifier is fit inside its training fold.
- Class imbalance: balanced class weights; no negative downsampling.
- Hard threshold: fixed at 0.5; no threshold tuning.
- Confidence intervals: 1,000 deterministic case-level bootstrap resamples of averaged repeated OOF predictions.
- Models: every single-feature balanced logistic baseline, all-feature balanced logistic regression, and balanced HistGradientBoosting.
- Gate aggregation: maximum OOF AUROC over this frozen model universe. This is conservative against advancing to G2, but its bootstrap CI is conditional on the selected model and does not correct for model-selection multiplicity.

## Primary results

The table shows both prespecified full models and the five strongest single-feature baselines. `reports/g1_primary_metrics.csv` contains every baseline.

{_markdown_table(display)}

Best cheap model: `{gate['best_model']}`, AUROC={gate['best_cheap_auroc']:.4f} (95% CI {gate['best_cheap_auroc_ci_lower']:.4f}–{gate['best_cheap_auroc_ci_upper']:.4f}), AUPRC={gate['best_cheap_auprc']:.4f}; positive prevalence={gate['primary_prevalence']:.4f}.

Across the 15 individual validation folds for that model, AUROC ranged from {gate['best_fold_auroc_min']:.4f} to {gate['best_fold_auroc_max']:.4f} (mean {gate['best_fold_auroc_mean']:.4f}, sample SD {gate['best_fold_auroc_std']:.4f}). This variability is reported explicitly rather than treating the aggregate OOF point estimate as uniformly stable.

## Matched controls

{_markdown_table(matched_display)}

The corruption-conditioned tasks have limited positive counts and must not be interpreted as high-power null results. The protocol's ten-positive cutoff means only "minimum-count evaluable"; it is not a power analysis. Full bootstrap intervals are in `reports/g1_matched_bootstrap_ci.csv`.

## Gate decision

- Status: **{gate['status']}**
- Reason: {gate['reason']}

This gate asks whether cheap temporal proxies already make a hidden-state audit unnecessary. It does not claim that a weak cheap baseline proves hidden states will work.

The machine-readable rule was frozen before this parser-v1.1 rerun in `reports/g1_cv_protocol.json` (SHA256 `{gate['gate_protocol_sha256']}`). Like G0's exploratory rule, its operational details were finalized after the initial v1.0 DEV audit and are not confirmatory preregistration.
"""
    path.write_text(text, encoding="utf-8")
    return path


def _snapshot_analysis_code(staging: Path) -> dict[str, Any]:
    """Freeze the complete local analysis-code inventory for a non-Git project."""

    inventory = [PROJECT_ROOT / "pyproject.toml"]
    for directory in ("src", "scripts", "tests"):
        inventory.extend(
            path
            for path in (PROJECT_ROOT / directory).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    inventory = sorted(set(inventory), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
    entries: list[dict[str, Any]] = []
    for source in inventory:
        if not source.is_file():
            raise AssertionError(f"source provenance file missing: {source}")
        workspace_relative = source.relative_to(PROJECT_ROOT).as_posix()
        snapshot_relative = f"provenance/source_snapshot/{workspace_relative}"
        destination = staging / snapshot_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append(
            {
                "workspace_relative_path": workspace_relative,
                "snapshot_relative_path": snapshot_relative,
                "sha256": sha256_file(source),
                "size_bytes": int(source.stat().st_size),
            }
        )
    manifest = {"schema_version": "1.0.0", "files": entries}
    write_json(staging / "provenance" / "source_code_manifest.json", manifest)
    return manifest


def _state_semantics(primary: float) -> dict[str, Any]:
    primary_mapping = next(
        item
        for item in build_progress_mapping(AUDIT_CHECKPOINTS, TOTAL_STEPS)
        if np.isclose(float(item["checkpoint"]), float(primary))
    )
    return {
        "schema_version": "1.0.0",
        "state_name": "pre_forward_input_producing_raw_x0",
        "mask_token_id": MASK_TOKEN_ID,
        "generation_length": GEN_LENGTH,
        "total_steps": TOTAL_STEPS,
        "history_index_base": 0,
        "step_number_base": 1,
        "prompt_source": "prompt_index.parquet.prompt_token_ids",
        "checkpoint_locator": {
            "checkpoint": float(primary),
            "history_index": int(primary_mapping["history_index"]),
            "step_number": int(primary_mapping["step_number"]),
            "progress_mapping_file": "progress_mapping.json",
        },
        "pre_step_commits_rule": "restore only tokens committed at history indices < t",
        "pre_step_uncommitted_rule": "all positions not committed before t remain mask_token_id",
        "current_stop_now_rule": "raw x0_history[t] with replayed committed tokens from history indices <= t",
        "current_commit_rule": "the commit selected at t occurs after forward t and is excluded from the pre-forward input at t",
        "hidden_state_timing": "the same forward pass that produces raw x0_history[t]",
        "forbidden_future_inputs": [
            "x0_history rows with history index > t",
            "true_indices_history commits with history index > t",
            "final ans_posidx pred_token_id pred_text and other final-answer metadata",
            "final correctness and trajectory labels",
            "ground truth answer tokens as model input",
        ],
    }


def _snapshot_context_and_reports(staging: Path) -> None:
    context_paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "external" / "Prophet" / "analysis" / "generate.py",
        PROJECT_ROOT
        / "external"
        / "Prophet"
        / "analysis"
        / "collect_decoding_traj_gsm8k.py",
        PROJECT_ROOT / "data" / "metadata" / "download_manifest.json",
        PROJECT_ROOT / "data" / "metadata" / "tokenizer_manifest.json",
        PROJECT_ROOT / "data" / "metadata" / "gsm8k_manifest.json",
    ]
    context_entries: list[dict[str, Any]] = []
    for source in context_paths:
        if not source.is_file():
            raise AssertionError(f"context provenance file missing: {source}")
        relative = source.relative_to(PROJECT_ROOT).as_posix()
        destination_relative = f"provenance/frozen_context/{relative}"
        destination = staging / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        context_entries.append(
            {
                "workspace_relative_path": relative,
                "snapshot_relative_path": destination_relative,
                "sha256": sha256_file(source),
                "size_bytes": int(source.stat().st_size),
            }
        )
    write_json(
        staging / "provenance" / "frozen_context_manifest.json",
        {"schema_version": "1.0.0", "files": context_entries},
    )

    report_names = [
        "environment.txt",
        "schema_audit.md",
        "schema_audit.json",
        "G0_REPORT.md",
        "G1_REPORT.md",
        "g0_checkpoint_summary.csv",
        "g0_corruption_evidence.csv",
        "parser_disagreements_dev.csv",
        "g1_cv_protocol.json",
        "g1_primary_metrics.csv",
        "g1_primary_bootstrap_ci.csv",
        "g1_matched_metrics.csv",
        "g1_matched_bootstrap_ci.csv",
        "g1_best_model_fold_metrics.csv",
    ]
    for report_name in report_names:
        source = PROJECT_ROOT / "reports" / report_name
        if not source.is_file():
            raise AssertionError(f"required report provenance missing: {source}")
        destination = staging / "provenance" / "reports" / report_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _build_cache(
    source_manifest: dict[str, Any],
    tokenizer_manifest: dict[str, Any],
    gsm8k_manifest: dict[str, Any],
    source_index: pd.DataFrame,
    prompt_index: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    g0_decision: G0Decision,
    g1_gate: dict[str, Any],
    environment_report: Path,
) -> dict[str, Any]:
    if g0_decision.selected_checkpoint is None:
        raise AssertionError("cannot build cache without a frozen checkpoint")
    if g0_decision.status not in set(G0_GATE_PROTOCOL["ordinary_ready_eligible_statuses"]):
        raise AssertionError("exploratory G0 is not eligible for an ordinary READY cache")
    if g1_gate["status"] not in {"G1_PASS", "G1_STRONG_PASS"}:
        raise AssertionError("G1 gate is not eligible for a READY cache")
    if CACHE_DIR.exists():
        raise FileExistsError(
            f"immutable cache already exists: {CACHE_DIR}; bump CACHE_VERSION rather than reuse"
        )

    staging = CACHE_DIR.with_name(CACHE_DIR.name + ".staging")
    if staging.exists():
        raise RuntimeError(f"staging cache already exists and will not be overwritten: {staging}")
    staging.mkdir(parents=True)
    (staging / "provenance").mkdir()

    primary = float(g0_decision.selected_checkpoint)
    dev_cases = labels[np.isclose(labels["checkpoint"], primary)].copy()
    dev_ids = sorted(source_index.loc[source_index["split"] == "DEV", "case_id"])
    holdout_ids = sorted(
        source_index.loc[source_index["split"] == "G3_HOLDOUT", "case_id"]
    )
    g2_train, g2_val = make_g2_train_val(
        dev_cases["case_id"],
        dev_cases["trajectory_type"],
        RANDOM_SEED,
        train_fraction=0.75,
    )
    if set(g2_train) & set(g2_val) or (set(g2_train) | set(g2_val)) != set(dev_ids):
        raise AssertionError("invalid G2 split")
    if (set(g2_train) | set(g2_val)) & set(holdout_ids):
        raise AssertionError("G3 holdout entered G2 split")

    source_index.to_parquet(staging / "source_index.parquet", index=False)
    prompt_index.to_parquet(staging / "prompt_index.parquet", index=False)
    dev_cases.to_parquet(staging / "dev_cases.parquet", index=False)
    features.to_parquet(staging / "cpu_features.parquet", index=False)
    labels.to_parquet(staging / "g0_labels.parquet", index=False)
    write_json(staging / "splits.json", {"DEV": dev_ids, "G3_HOLDOUT": holdout_ids})
    write_json(staging / "g2_train_ids.json", g2_train)
    write_json(staging / "g2_val_ids.json", g2_val)
    write_json(staging / "g3_holdout_ids.json", holdout_ids)

    descriptions = _feature_descriptions()
    feature_schema = {
        "schema_version": "1.0.0",
        "metadata_columns": ["case_id", "checkpoint", "split"],
        "causal_cutoff": "All values use global history indices <= frozen checkpoint t.",
        "forbidden_columns": [
            "final_correct",
            "trajectory_type",
            "V",
            "continue_beneficial",
            "future_*",
        ],
        "features": [
            {
                "name": column,
                "dtype": str(features[column].dtype),
                "description": descriptions[column.removeprefix("feature_")],
                "constant_at_primary": bool(features[column].nunique(dropna=False) <= 1),
            }
            for column in features.columns
            if column.startswith("feature_")
        ],
    }
    write_json(staging / "feature_schema.json", feature_schema)
    write_json(
        staging / "parser_config.json",
        {
            "name": PARSER_NAME,
            "version": PARSER_VERSION,
            "protocol_revision": "parser-v1.1 rerun after v1.0 empty-marker defect",
            "numeric_type": "decimal.Decimal",
            "marker_rule": (
                "after the last Answer: or #### marker, take the first valid number; "
                "a nonempty nonnumeric suffix is a parse failure"
            ),
            "fallback_rule": (
                "use the last valid rationale number when there is no marker or when "
                "the terminal marker suffix is empty/whitespace only"
            ),
            "supports": ["comma", "sign", "decimal", "currency symbol", "trailing units"],
            "missing_equals_missing": False,
            "same_parser_for_gt_mid_final": True,
        },
    )
    write_json(
        staging / "progress_mapping.json",
        {
            "rule": PROGRESS_MAPPING_RULE,
            "total_steps": TOTAL_STEPS,
            "block_major": True,
            "blocks": 8,
            "steps_per_block": 32,
            "audit_checkpoints": build_progress_mapping(AUDIT_CHECKPOINTS, TOTAL_STEPS),
            "preregistered_primary_checkpoint": PREREGISTERED_PRIMARY_CHECKPOINT,
            "frozen_primary_checkpoint": primary,
            "selection_scope": (
                "DEV only; numeric exploratory rule was operationalized after the initial "
                "G0 DEV diagnostic and frozen before G1"
            ),
        },
    )
    write_json(staging / "state_semantics.json", _state_semantics(primary))
    source_code_manifest = _snapshot_analysis_code(staging)
    _snapshot_context_and_reports(staging)
    protocol = {
        "schema_version": "1.0.0",
        "G0": G0_GATE_PROTOCOL,
        "G1": G1_GATE_PROTOCOL,
        "G0_sha256": _canonical_json_sha256(G0_GATE_PROTOCOL),
        "G1_sha256": _canonical_json_sha256(G1_GATE_PROTOCOL),
    }
    write_json(staging / "provenance" / "protocol.json", protocol)
    code_bundle_sha256 = _canonical_json_sha256(source_code_manifest)
    gsm8k_test_entry = next(
        entry
        for entry in gsm8k_manifest["files"]
        if entry["relative_path"] == "main/test-00000-of-00001.parquet"
    )
    cache_source_manifest = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "exact_trajectory_folder": TRAJECTORY_FOLDER,
        "source_root_relative": "data/raw/DLM-Decoding-Analysis",
        "source_file_count": 1319,
        "source_total_bytes": int(source_index["source_size_bytes"].sum()),
        "download_endpoint": source_manifest["endpoint"],
        "download_method": source_manifest["download_method"],
        "download_manifest_sha256": sha256_file(
            PROJECT_ROOT / "data" / "metadata" / "download_manifest.json"
        ),
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_allowlist": tokenizer_manifest["allowlist"],
        "tokenizer_weights_downloaded": False,
        "tokenizer_manifest_sha256": sha256_file(
            PROJECT_ROOT / "data" / "metadata" / "tokenizer_manifest.json"
        ),
        "gsm8k_repo": GSM8K_REPO,
        "gsm8k_revision": GSM8K_REVISION,
        "gsm8k_test_relative_path": "data/raw/GSM8K/main/test-00000-of-00001.parquet",
        "gsm8k_test_sha256": gsm8k_test_entry["sha256"],
        "gsm8k_test_size_bytes": int(gsm8k_test_entry["size_bytes"]),
        "gsm8k_manifest_sha256": sha256_file(
            PROJECT_ROOT / "data" / "metadata" / "gsm8k_manifest.json"
        ),
        "query_template_sha256": FROZEN_QUERY_TEMPLATE_SHA256,
        "mask_token_id": MASK_TOKEN_ID,
        "prompt_index_rows": int(len(prompt_index)),
        "state_contract": "pre_forward_input_producing_raw_x0",
        "pre_step_commit_cutoff": "history indices < t",
        "prophet_repo": "https://github.com/pixeli99/Prophet",
        "prophet_revision": PROPHET_REVISION,
        "analysis_code_bundle_sha256": code_bundle_sha256,
    }
    write_json(staging / "source_manifest.json", cache_source_manifest)
    manifest = {
        "cache_version": CACHE_VERSION,
        "created_at": utc_now(),
        "git_commit": git_revision(PROJECT_ROOT),
        "random_seed": RANDOM_SEED,
        "python_version": platform.python_version(),
        "dataset_repo": DATASET_REPO,
        "dataset_revision_SHA": DATASET_REVISION,
        "exact_trajectory_folder": TRAJECTORY_FOLDER,
        "benchmark": "GSM8K official test (preexperiment only)",
        "model": "LLaDA-8B-Instruct",
        "decoding_policy": "low_confidence",
        "constraint_policy": "none",
        "steps": TOTAL_STEPS,
        "gen_length": GEN_LENGTH,
        "block_length": BLOCK_LENGTH,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "preregistered_primary_checkpoint": PREREGISTERED_PRIMARY_CHECKPOINT,
        "primary_checkpoint": primary,
        "progress_mapping_rule": PROGRESS_MAPPING_RULE,
        "answer_parser_version": f"{PARSER_NAME}:{PARSER_VERSION}",
        "parser_protocol_revision": "v1.1 empty-terminal-marker semantic revision",
        "G0_gate_protocol_sha256": protocol["G0_sha256"],
        "G1_gate_protocol_sha256": protocol["G1_sha256"],
        "G0_gate_result": g0_decision.status,
        "G1_gate_result": g1_gate["status"],
        "G0_gate_reason": g0_decision.reason,
        "G1_gate_reason": g1_gate["reason"],
        "G1_best_cheap_model": g1_gate["best_model"],
        "G1_best_cheap_AUROC": g1_gate["best_cheap_auroc"],
        "G1_best_cheap_AUPRC": g1_gate["best_cheap_auprc"],
        "n_dev": len(dev_ids),
        "n_g3_holdout": len(holdout_ids),
        "G3_outcomes_used": False,
        "raw_G3_source_contains_outcome_fields": True,
        "scientific_scope": "preexperiment; ordinary READY requires formal G0 eligibility",
        "analysis_code_bundle_sha256": code_bundle_sha256,
    }
    write_json(staging / "manifest.json", manifest)
    write_json(
        staging / "status.json",
        {
            "cache_status": "READY_FOR_G2",
            "G0_status": g0_decision.status,
            "G1_status": g1_gate["status"],
            "verified": True,
            "next_stage": "G2_GPU_HIDDEN_STATE_AUDIT",
        },
    )

    readme = f"""# {CACHE_VERSION}

Immutable handoff cache for a CPU-only ReVA-DLM feasibility preexperiment.

## Frozen state locator

For every `case_id`, obtain the exact original prompt IDs from `prompt_index.parquet`; hashes bind them to the frozen GSM8K row, Prophet query template, and tokenizer snapshot. Load the trusted `.pt` located by `source_index.parquet`. To reconstruct the model input for history index `t`, concatenate the exact prompt with generated positions committed strictly before `t`, leaving every other generated position at mask token {MASK_TOKEN_ID}. Hidden states/logits for `t` must come from this same forward pass, the one that produced raw `x0_history[t]`.

The separate stop-now candidate `y_t` is post-iteration: restore commits at indices `<=t` into raw `x0_history[t]`. Do not confuse this candidate with the pre-forward input. Do not use future x0/commits, `ans_posidx`, `pred_token_id`, `pred_text`, final correctness, or ground truth as model inputs. The complete machine-readable contract is in `state_semantics.json`.

The frozen checkpoint is {primary:.2f}. The exact mapping contains both 1-based `step_number` and 0-based `history_index`. G2 must use only `g2_train_ids.json` and `g2_val_ids.json` for layer/probe/threshold selection. G3 IDs are frozen in `g3_holdout_ids.json` and were not used for G0/G1 model selection; their raw source files do contain accessible outcome fields, so they are not described as cryptographically sealed.

## Scope

No LLaDA weights are contained here. Original `.pt` tensors remain the source of truth rather than being duplicated into parquet. This public GSM8K-test cache supports scientific feasibility only, not a claim of untouched confirmatory evaluation.
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")
    hash_lines = []
    for artifact in sorted(path for path in staging.rglob("*") if path.is_file()):
        if artifact.name == "sha256sums.txt":
            continue
        hash_lines.append(
            f"{sha256_file(artifact)}  {artifact.relative_to(staging).as_posix()}"
        )
    (staging / "sha256sums.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    result = verify_cache(staging, verify_source_hashes=True)
    staging.replace(CACHE_DIR)
    verify_cache(CACHE_DIR, verify_source_hashes=False)
    return result


def _write_final_report(
    source_manifest: dict[str, Any],
    schema: dict[str, Any],
    g0_summary: pd.DataFrame,
    g0_decision: G0Decision,
    g1_gate: dict[str, Any] | None,
    primary_result: dict[str, Any] | None,
    matched_metrics: pd.DataFrame | None,
    cache_result: dict[str, Any] | None,
) -> Path:
    path = PROJECT_ROOT / "reports" / "REVA_MINI_G0_G1_REPORT.md"
    if g1_gate is None:
        g1_section = "G1 was not run because G0 did not pass its gate."
        recommendation = "STOP"
    else:
        full_metrics = primary_result["metrics"]
        best = _best_model(full_metrics)
        matched_table = _markdown_table(
            matched_metrics[
                ["task", "model", "n_samples", "n_positive", "auroc", "auprc"]
            ]
        )
        g1_section = f"""### G1 features and protocol

All 29 CPU features are prefix-causal and are stored separately from labels. Evaluation used {primary_result['cv_config']['effective_splits']}×{primary_result['cv_config']['effective_repeats']} repeated stratified OOF predictions, fold-local transforms, balanced weights, fixed threshold 0.5, and 1,000 case-level bootstrap samples. No G3 holdout row was used.

Best cheap baseline: `{best['model']}`; AUROC={best['auroc']:.6f}, AUPRC={best['auprc']:.6f}, balanced accuracy={best['balanced_accuracy']:.6f}, F1={best['f1']:.6f}, Brier={best['brier']:.6f}. Bootstrap AUROC 95% CI is {g1_gate['best_cheap_auroc_ci_lower']:.6f}–{g1_gate['best_cheap_auroc_ci_upper']:.6f}.

Individual-fold AUROC range for that model: {g1_gate['best_fold_auroc_min']:.6f}–{g1_gate['best_fold_auroc_max']:.6f} (mean {g1_gate['best_fold_auroc_mean']:.6f}, sample SD {g1_gate['best_fold_auroc_std']:.6f}).

Matched controls:

{matched_table}

G1 decision: **{g1_gate['status']}** — {g1_gate['reason']}"""
        if cache_result is not None:
            recommendation = "READY_FOR_G2"
        elif (
            g0_decision.status == "G0_EXPLORATORY_PASS"
            and g1_gate["status"] in {"G1_PASS", "G1_STRONG_PASS"}
        ):
            recommendation = "BORDERLINE_REVIEW_REQUIRED"
        else:
            recommendation = "STOP"

    cache_section = (
        f"The immutable cache was built at `cache/{CACHE_VERSION}/` and independently re-verified: {cache_result}. `sha256sums.txt` covers every cache artifact, while `source_index.parquet` pins every original source SHA256."
        if cache_result is not None
        else "No ordinary READY cache was created because the scientific eligibility gates did not both pass."
    )
    text = f"""# ReVA CPU-only Mini G0 + G1 report

## 1. Data provenance and mirror download

- Dataset: `{DATASET_REPO}` at revision `{DATASET_REVISION}`.
- Exact configuration: `{TRAJECTORY_FOLDER}` (GSM8K, LLaDA-8B-Instruct, low-confidence remasking, no constraint, gen/steps 256, block length 32, temperature 0, CFG 0).
- Download: `{source_manifest['download_method']}` starting exclusively from `https://hf-mirror.com`, with resumable `.part` files, skip-on-success, sizes, and SHA256 manifest. The higher-level `hf` client was attempted first but the host proxy caused TLS EOF, so the documented curl mirror fallback was used.
- Selected payload: README plus exactly 1,319 target `.pt` files, {source_manifest['total_bytes']} bytes. No MMLU or alternate GSM8K configuration was downloaded.
- Tokenizer: five allowlisted small files from `{TOKENIZER_REPO}` at `{TOKENIZER_REVISION}`; no `.safetensors`, `.bin`, or `.pt` model weight was downloaded.
- Prompt source: `openai/gsm8k` main/test at revision `{GSM8K_REVISION}`, downloaded only through `https://hf-mirror.com`; all 1,319 row/question/prompt/token-length/GT bindings match. Exact prompt IDs are retained in the report-side prompt index. Prophet query-template SHA256 is `{FROZEN_QUERY_TEMPLATE_SHA256}` and mask token ID is `{MASK_TOKEN_ID}`.
- Prophet source of truth: commit `{PROPHET_REVISION}`.

## 2. Schema and reconstruction

Actual target files contain ten keys and omit the advertised `gen_ids`. Every trajectory is 8 blocks × 32 steps; each step commits one unique generated position. Prompt lengths are {schema['prompt_token_len_min']}–{schema['prompt_token_len_max']}.

Commit replay reconstructs stored `pred_text` exactly for {schema['reconstruction_exact_count']}/{schema['n_source_files']} cases. Direct raw-last `x0` is exact for only {schema['raw_last_exact_count']}/{schema['n_source_files']}, proving that raw history cannot be decoded without replay. Fourteen final answer anchors are invalid and are not used. The model input that produces raw `x0[t]` is separately fixed as exact prompt + commits at indices `<t` + masks elsewhere; the stop-now candidate is raw `x0[t]` with commits at indices `<=t` restored. Full details are in `reports/schema_audit.md`.

Parser v1.1 is a protocol revision made after the initial v1.0 audit found an empty-terminal-marker defect and was frozen before this rerun. It is shared by GT/current/final. On DEV, stored false → canonical true occurs {schema['dev_stored_false_canonical_true_count']} times and stored true → canonical false occurs {schema['dev_stored_true_canonical_false_count']} times; the report preserves this direction rather than collapsing all disagreements into an undifferentiated count. Empty-marker fallback applies to {schema['dev_final_empty_marker_fallback_count']} final and {schema['dev_mid_empty_marker_fallback_count']} checkpoint text(s).

## 3. Split and leakage controls

SHA256(case ID + seed {RANDOM_SEED}) froze {schema['n_dev']} DEV and {schema['n_g3_holdout']} G3 holdout cases before label analysis. G0/G1 do not compute or report G3 correctness/trajectory labels and never use G3 rows for features, fitting, or gates. Structural validation did inspect all source files; raw holdout `.pt` files contain accessible outcome fields, so they are not described as information-sealed. Mid-step reconstruction, answer parsing, and every G1 feature use only states at or before `t`. The explicit future-mutation test passes. Labels/final correctness are stored outside `cpu_features.parquet`.

## 4. G0 results

{_markdown_table(g0_summary)}

G0 decision: **{g0_decision.status}**. {g0_decision.reason}

The strong 50% criterion was not silently weakened. If an exploratory checkpoint is used, its DEV selection is explicit and frozen before G1; it is not presented as strong evidence.

The 60% exploratory checkpoint and its numeric adjacent-support rule were selected on the parser-v1.0 DEV diagnostic, then frozen before this parser-v1.1 rerun. The point is not reselected here. This is same-DEV, post-diagnostic and selection-biased; `G0_EXPLORATORY_PASS` permits completing G1 but not an ordinary READY cache.

## 5. G1 results

{g1_section}

## 6. Cache and integrity

{cache_section}

## 7. Recommendation

**{recommendation}**

This remains a preexperiment on public GSM8K test trajectories. Even a READY result does not justify a paper-level untouched-test claim; confirmatory work requires newly generated development trajectories and independent benchmarks.
"""
    path.write_text(text, encoding="utf-8")
    return path


def run_pipeline() -> dict[str, Any]:
    _setup_logging()
    _assert_cpu_only()
    LOGGER.info("Starting CPU-only ReVA Mini G0/G1")
    source_manifest, tokenizer_manifest, gsm8k_manifest = _verify_frozen_downloads()
    environment_report = _write_environment_report()
    tokenizer = _load_local_tokenizer()
    LOGGER.info("Validating exact GSM8K row/prompt/token bindings for all 1,319 cases")
    prompt_index = build_prompt_index(GSM8K_TEST_FILE, TRAJECTORY_DIR, tokenizer)
    source_index, labels, schema, evidence, disagreements = _audit_sources_and_build_g0(
        tokenizer, source_manifest, prompt_index
    )
    reports_dir = PROJECT_ROOT / "reports"
    source_index.to_parquet(reports_dir / "source_index_preexperiment.parquet", index=False)
    prompt_index.to_parquet(reports_dir / "prompt_index_preexperiment.parquet", index=False)
    prompt_index.to_parquet(reports_dir / "prompt_index.parquet", index=False)
    labels.to_parquet(reports_dir / "g0_labels_dev.parquet", index=False)
    evidence.to_csv(reports_dir / "g0_corruption_evidence.csv", index=False)
    disagreements.to_csv(reports_dir / "parser_disagreements_dev.csv", index=False)
    write_json(reports_dir / "schema_audit.json", schema)
    write_json(
        reports_dir / "progress_mapping.json",
        {
            "rule": PROGRESS_MAPPING_RULE,
            "mapping": build_progress_mapping(AUDIT_CHECKPOINTS, TOTAL_STEPS),
        },
    )
    write_json(
        reports_dir / "splits_preexperiment.json",
        {
            "seed": RANDOM_SEED,
            "method": "SHA256(seed:case_id) first 64 bits < 0.8",
            "DEV": sorted(source_index.query("split == 'DEV'")["case_id"]),
            "G3_HOLDOUT": sorted(source_index.query("split == 'G3_HOLDOUT'")["case_id"]),
        },
    )
    write_json(
        reports_dir / "gate_protocol_frozen_before_v1_1_rerun.json",
        {
            "schema_version": "1.0.0",
            "G0": G0_GATE_PROTOCOL,
            "G1": G1_GATE_PROTOCOL,
            "G0_sha256": _canonical_json_sha256(G0_GATE_PROTOCOL),
            "G1_sha256": _canonical_json_sha256(G1_GATE_PROTOCOL),
        },
    )
    _write_schema_report(schema)

    g0_summary = _summarize_g0(labels)
    g0_summary.to_csv(reports_dir / "g0_checkpoint_summary.csv", index=False)
    _plot_g0(g0_summary)
    g0_decision = _decide_g0(g0_summary)
    if g0_decision.selected_checkpoint is not None:
        write_json(
            reports_dir / "state_semantics.json",
            _state_semantics(float(g0_decision.selected_checkpoint)),
        )
    _write_g0_report(g0_summary, g0_decision, schema, evidence)
    LOGGER.info("G0 decision: %s (%s)", g0_decision.status, g0_decision.reason)

    g1_eligible_statuses = set(G0_GATE_PROTOCOL["g1_run_eligible_statuses"])
    if (
        g0_decision.status not in g1_eligible_statuses
        or g0_decision.selected_checkpoint is None
    ):
        write_json(
            PROJECT_ROOT / "cache" / "status.json",
            {
                "cache_status": (
                    "STOP_G0"
                    if g0_decision.status == "G0_FAIL"
                    else "BORDERLINE_REVIEW_REQUIRED"
                ),
                "G0_status": g0_decision.status,
                "reason": g0_decision.reason,
            },
        )
        _write_final_report(
            source_manifest, schema, g0_summary, g0_decision, None, None, None, None
        )
        print("REVA_MINI_G0_G1_STOP")
        print("FAILED_GATE=G0")
        print(f"REASON={g0_decision.reason}")
        return {"G0_status": g0_decision.status, "cache_status": "STOP_G0"}

    primary = float(g0_decision.selected_checkpoint)
    features = _extract_primary_features(tokenizer, source_index, primary)
    features.to_parquet(reports_dir / "cpu_features_dev.parquet", index=False)
    g1_gate, primary_result, matched_metrics, matched_cis, primary_labels = _run_g1(
        features, labels, primary
    )
    _write_g1_report(g1_gate, primary_result, matched_metrics, features, primary)
    LOGGER.info("G1 decision: %s (%s)", g1_gate["status"], g1_gate["reason"])

    if g1_gate["status"] == "G1_FAIL":
        write_json(
            PROJECT_ROOT / "cache" / "status.json",
            {
                "cache_status": "STOP_G1",
                "G0_status": g0_decision.status,
                "G1_status": g1_gate["status"],
                "reason": g1_gate["reason"],
            },
        )
        _write_final_report(
            source_manifest,
            schema,
            g0_summary,
            g0_decision,
            g1_gate,
            primary_result,
            matched_metrics,
            None,
        )
        print("REVA_MINI_G0_G1_STOP")
        print("FAILED_GATE=G1")
        print(f"REASON={g1_gate['reason']}")
        return {
            "G0_status": g0_decision.status,
            "G1_status": g1_gate["status"],
            "cache_status": "STOP_G1",
        }

    ordinary_ready_eligible = (
        g0_decision.status
        in set(G0_GATE_PROTOCOL["ordinary_ready_eligible_statuses"])
        and g1_gate["status"] in {"G1_PASS", "G1_STRONG_PASS"}
    )
    if not ordinary_ready_eligible:
        write_json(
            PROJECT_ROOT / "cache" / "status.json",
            {
                "cache_status": "BORDERLINE_REVIEW_REQUIRED",
                "G0_status": g0_decision.status,
                "G1_status": g1_gate["status"],
                "primary_checkpoint": primary,
                "candidate_cache_version": CACHE_VERSION,
                "reason": (
                    "G1 completed, but the same-DEV post-diagnostic G0 exploratory "
                    "checkpoint is not eligible for an ordinary READY cache."
                ),
            },
        )
        _write_final_report(
            source_manifest,
            schema,
            g0_summary,
            g0_decision,
            g1_gate,
            primary_result,
            matched_metrics,
            None,
        )
        selected = primary_labels[primary_labels["checkpoint"].eq(primary)]
        counts = selected["trajectory_type"].value_counts()
        primary_g0 = g0_summary[np.isclose(g0_summary["checkpoint"], primary)].iloc[0]
        _emit_terminal_summary(
            [
                "REVA_MINI_G0_G1_STOP",
                "FAILED_GATE=G0_ORDINARY_READY_ELIGIBILITY",
                f"G0_STATUS={g0_decision.status}",
                f"G1_STATUS={g1_gate['status']}",
                "CACHE_STATUS=BORDERLINE_REVIEW_REQUIRED",
                f"PRIMARY_CHECKPOINT={primary}",
                f"N_DEV={schema['n_dev']}",
                f"N_G3_HOLDOUT={schema['n_g3_holdout']}",
                f"N_STABLE={int(counts.get('STABLE', 0))}",
                f"N_RESCUE={int(counts.get('RESCUE', 0))}",
                f"N_CORRUPTION={int(counts.get('CORRUPTION', 0))}",
                f"N_UNRECOVERED={int(counts.get('UNRECOVERED', 0))}",
                f"RCR_PRIMARY={float(primary_g0['rcr_joint']):.8f}",
                f"RSR_PRIMARY={float(primary_g0['rsr_joint']):.8f}",
                f"G1_BEST_CHEAP_AUROC={g1_gate['best_cheap_auroc']:.8f}",
                f"G1_BEST_CHEAP_AUPRC={g1_gate['best_cheap_auprc']:.8f}",
                "NEXT_STAGE=STOP_PENDING_INDEPENDENT_G0_VALIDATION",
            ]
        )
        return {
            "G0_status": g0_decision.status,
            "G1_status": g1_gate["status"],
            "cache_status": "BORDERLINE_REVIEW_REQUIRED",
        }

    cache_result = _build_cache(
        source_manifest,
        tokenizer_manifest,
        gsm8k_manifest,
        source_index,
        prompt_index,
        labels,
        features,
        g0_decision,
        g1_gate,
        environment_report,
    )
    write_json(
        PROJECT_ROOT / "cache" / "status.json",
        {
            "cache_status": "READY_FOR_G2",
            "cache_path": str(CACHE_DIR),
            "G0_status": g0_decision.status,
            "G1_status": g1_gate["status"],
            "primary_checkpoint": primary,
        },
    )
    _write_final_report(
        source_manifest,
        schema,
        g0_summary,
        g0_decision,
        g1_gate,
        primary_result,
        matched_metrics,
        cache_result,
    )

    selected = primary_labels[primary_labels["checkpoint"].eq(primary)]
    counts = selected["trajectory_type"].value_counts()
    primary_g0 = g0_summary[np.isclose(g0_summary["checkpoint"], primary)].iloc[0]
    print("REVA_MINI_G0_G1_COMPLETE")
    print(f"G0_STATUS={g0_decision.status}")
    print(f"G1_STATUS={g1_gate['status']}")
    print("CACHE_STATUS=READY_FOR_G2")
    print(f"PRIMARY_CHECKPOINT={primary}")
    print(f"N_DEV={schema['n_dev']}")
    print(f"N_G3_HOLDOUT={schema['n_g3_holdout']}")
    print(f"N_STABLE={int(counts.get('STABLE', 0))}")
    print(f"N_RESCUE={int(counts.get('RESCUE', 0))}")
    print(f"N_CORRUPTION={int(counts.get('CORRUPTION', 0))}")
    print(f"N_UNRECOVERED={int(counts.get('UNRECOVERED', 0))}")
    print(f"RCR_PRIMARY={float(primary_g0['rcr_joint']):.8f}")
    print(f"RSR_PRIMARY={float(primary_g0['rsr_joint']):.8f}")
    print(f"G1_BEST_CHEAP_AUROC={g1_gate['best_cheap_auroc']:.8f}")
    print(f"G1_BEST_CHEAP_AUPRC={g1_gate['best_cheap_auprc']:.8f}")
    print(f"CACHE_PATH={CACHE_DIR}")
    print("CACHE_VERIFY_OK")
    print("NEXT_STAGE=G2_GPU_HIDDEN_STATE_AUDIT")
    return {
        "G0_status": g0_decision.status,
        "G1_status": g1_gate["status"],
        "cache_status": "READY_FOR_G2",
        "cache": cache_result,
    }


__all__ = ["run_pipeline"]
