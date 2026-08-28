from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import struct

import pandas as pd
import pytest
import torch

from reva_dlm.config import GSM8K_TEST_FILE, PROJECT_ROOT, TOKENIZER_DIR, TRAJECTORY_DIR
from reva_dlm.prompt_provenance import (
    PROMPT_INDEX_COLUMNS,
    PromptProvenanceError,
    build_prompt_index,
    load_frozen_tokenizer,
    sha256_prompt_token_ids,
    sha256_text,
    validate_local_tokenizer_snapshot,
    validate_prompt_index,
)
from reva_dlm.prompt_state import (
    FROZEN_QUERY_TEMPLATE_SHA256,
    QUERY_TEMPLATE,
    build_prompt_text,
)


class _FakeTokenizer:
    def __call__(self, text):
        # Stable, nonempty, signed-int64-compatible IDs; equal-length synthetic
        # questions make the canonical GT check carry the row binding test.
        return {"input_ids": [ord(character) for character in text]}


def _write_synthetic_sources(tmp_path: Path) -> tuple[Path, Path, _FakeTokenizer]:
    dataset_path = tmp_path / "test.parquet"
    trajectory_dir = tmp_path / "trajectories"
    trajectory_dir.mkdir()
    rows = pd.DataFrame(
        {
            "question": ["AA?", "BB?"],
            "answer": ["work\n#### 12", "work\n#### -3.5"],
        }
    )
    rows.to_parquet(dataset_path, index=False)
    tokenizer = _FakeTokenizer()
    for index, row in rows.iterrows():
        prompt_ids = tokenizer(build_prompt_text(row["question"]))["input_ids"]
        gt_text = "12" if index == 0 else "-3.5"
        torch.save(
            {
                "prompt_token_len": len(prompt_ids),
                "gt_text": gt_text,
                "x0_history": [torch.zeros((1, len(prompt_ids) + 256), dtype=torch.long)],
            },
            trajectory_dir / f"question_{index:04d}_steps_256.pt",
        )
    return dataset_path, trajectory_dir, tokenizer


def test_query_template_is_exact_frozen_prophet_template():
    assert hashlib.sha256(QUERY_TEMPLATE.encode("utf-8")).hexdigest() == (
        "231d269274b6a04711d192c935b7a4785da99e6823afbc817972a1d51f48425e"
    )
    assert FROZEN_QUERY_TEMPLATE_SHA256 == (
        "231d269274b6a04711d192c935b7a4785da99e6823afbc817972a1d51f48425e"
    )
    assert "\n\nA question?\n\nRemember to put" in build_prompt_text("A question?")

    upstream_path = (
        PROJECT_ROOT
        / "external"
        / "Prophet"
        / "analysis"
        / "collect_decoding_traj_gsm8k.py"
    )
    tree = ast.parse(upstream_path.read_text(encoding="utf-8"))
    upstream_template = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "QUERY_TEMPLATE"
            for target in node.targets
        ):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "strip"
                and not value.args
                and not value.keywords
            ):
                upstream_template = ast.literal_eval(value.func.value).strip()
            else:
                upstream_template = ast.literal_eval(value)
            break
    assert upstream_template == QUERY_TEMPLATE


def test_token_id_hash_is_little_endian_int64_bytes():
    token_ids = [1, 256, 126336]
    expected = hashlib.sha256(struct.pack("<qqq", *token_ids)).hexdigest()
    assert sha256_prompt_token_ids(token_ids) == expected


def test_builder_emits_fixed_deterministic_schema(tmp_path):
    dataset_path, trajectory_dir, tokenizer = _write_synthetic_sources(tmp_path)
    first = build_prompt_index(
        dataset_path,
        trajectory_dir,
        tokenizer,
        expected_count=2,
    )
    second = build_prompt_index(
        dataset_path,
        trajectory_dir,
        tokenizer,
        expected_count=2,
    )
    assert list(first.columns) == list(PROMPT_INDEX_COLUMNS)
    assert first.to_dict(orient="records") == second.to_dict(orient="records")
    assert first["case_id"].tolist() == ["question_0000", "question_0001"]
    assert first["gsm8k_row_index"].tolist() == [0, 1]
    assert first.loc[0, "question_sha256"] == sha256_text("AA?")
    assert first.loc[0, "prompt_sha256"] == sha256_text(build_prompt_text("AA?"))
    assert first.loc[0, "gsm8k_gt_answer_sha256"] == sha256_text("work\n#### 12")
    assert validate_prompt_index(first, expected_count=2)["mismatch_count"] == 0


def test_builder_rejects_row_to_trajectory_gt_swap(tmp_path):
    dataset_path, trajectory_dir, tokenizer = _write_synthetic_sources(tmp_path)
    path = trajectory_dir / "question_0000_steps_256.pt"
    record = torch.load(path, map_location="cpu", weights_only=False)
    record["gt_text"] = "-3.5"
    torch.save(record, path)
    with pytest.raises(PromptProvenanceError, match="question_0000: canonical GT"):
        build_prompt_index(
            dataset_path,
            trajectory_dir,
            tokenizer,
            expected_count=2,
        )


def test_builder_rejects_prompt_length_mismatch(tmp_path):
    dataset_path, trajectory_dir, tokenizer = _write_synthetic_sources(tmp_path)
    path = trajectory_dir / "question_0001_steps_256.pt"
    record = torch.load(path, map_location="cpu", weights_only=False)
    record["prompt_token_len"] += 1
    torch.save(record, path)
    with pytest.raises(PromptProvenanceError, match="prompt_token_len"):
        build_prompt_index(
            dataset_path,
            trajectory_dir,
            tokenizer,
            expected_count=2,
        )


def test_frame_validator_rejects_token_hash_tampering(tmp_path):
    dataset_path, trajectory_dir, tokenizer = _write_synthetic_sources(tmp_path)
    frame = build_prompt_index(
        dataset_path,
        trajectory_dir,
        tokenizer,
        expected_count=2,
    )
    frame.at[1, "prompt_token_ids"] = frame.at[1, "prompt_token_ids"] + [999]
    with pytest.raises(PromptProvenanceError, match="prompt_token_ids_sha256 mismatch"):
        validate_prompt_index(frame, expected_count=2)


def test_all_1319_frozen_rows_bind_to_prophet_trajectories():
    tokenizer_manifest = (
        GSM8K_TEST_FILE.resolve().parents[3] / "metadata" / "tokenizer_manifest.json"
    )
    required = [
        GSM8K_TEST_FILE,
        TRAJECTORY_DIR / "question_0000_steps_256.pt",
        TRAJECTORY_DIR / "question_1318_steps_256.pt",
        TOKENIZER_DIR / "tokenizer.json",
        tokenizer_manifest,
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("frozen GSM8K, trajectories, or tokenizer snapshot is absent")
    pytest.importorskip("transformers")

    snapshot = validate_local_tokenizer_snapshot(TOKENIZER_DIR, tokenizer_manifest)
    assert snapshot["weights_downloaded"] is False
    tokenizer = load_frozen_tokenizer(TOKENIZER_DIR)
    prompt_index = build_prompt_index(
        GSM8K_TEST_FILE,
        TRAJECTORY_DIR,
        tokenizer,
    )
    summary = validate_prompt_index(prompt_index)
    assert summary["n_rows"] == 1319
    assert summary["mismatch_count"] == 0
    assert summary["first_case_id"] == "question_0000"
    assert summary["last_case_id"] == "question_1318"
    assert prompt_index["question_sha256"].is_unique
    assert prompt_index["prompt_sha256"].is_unique
    assert (prompt_index["prompt_token_len"] > 0).all()
