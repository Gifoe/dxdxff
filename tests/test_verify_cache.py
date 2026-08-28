from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reva_dlm.cache_verifier import (
    EXPECTED_CASE_COUNT,
    _canonical_json_sha256,
    _verify_code_bundle_and_protocol,
    _verify_hash_coverage,
    _verify_parser_and_status,
    _verify_prompt_index,
    _verify_source_code_provenance,
    _verify_source_manifest_metadata,
    _verify_state_semantics,
)
from reva_dlm.config import G0_GATE_PROTOCOL, G1_GATE_PROTOCOL, GSM8K_REPO, GSM8K_REVISION
from reva_dlm.parser import PARSER_NAME, PARSER_VERSION
from reva_dlm.prompt_state import FROZEN_QUERY_TEMPLATE_SHA256, MASK_TOKEN_ID
from reva_dlm.utils import sha256_file


def _write_hash_list(cache: Path) -> None:
    lines = []
    for path in sorted(item for item in cache.rglob("*") if item.is_file()):
        if path.name == "sha256sums.txt":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(cache).as_posix()}")
    (cache / "sha256sums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_hash_list_must_cover_every_artifact_exactly(tmp_path: Path):
    cache = tmp_path / "cache"
    (cache / "provenance").mkdir(parents=True)
    (cache / "manifest.json").write_text("{}", encoding="utf-8")
    (cache / "provenance" / "note.md").write_text("frozen", encoding="utf-8")
    _write_hash_list(cache)
    assert _verify_hash_coverage(cache) == 2

    (cache / "unlisted.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="cover every cache artifact exactly once"):
        _verify_hash_coverage(cache)


def test_temporary_or_log_artifacts_are_rejected(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}", encoding="utf-8")
    (cache / "run.log").write_text("not immutable", encoding="utf-8")
    _write_hash_list(cache)
    with pytest.raises(AssertionError, match="temporary/log/editor"):
        _verify_hash_coverage(cache)


def _prompt_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_rows = []
    source_rows = []
    for index in range(EXPECTED_CASE_COUNT):
        case_id = f"question_{index:04d}"
        tokens = np.asarray([index, -index], dtype="<i8")
        prompt_rows.append(
            {
                "case_id": case_id,
                "gsm8k_row_index": index,
                "question_sha256": hashlib.sha256(f"q{index}".encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256(f"p{index}".encode()).hexdigest(),
                "prompt_token_ids": tokens.tolist(),
                "prompt_token_ids_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                "prompt_token_len": 2,
                "gsm8k_gt_answer_sha256": hashlib.sha256(
                    f"a{index}".encode()
                ).hexdigest(),
            }
        )
        source_rows.append(
            {
                "case_id": case_id,
                "prompt_token_len": 2,
                "source_relative_path": f"folder/{case_id}.pt",
            }
        )
    return pd.DataFrame(prompt_rows), pd.DataFrame(source_rows)


def test_prompt_index_checks_ids_lengths_and_little_endian_hash(tmp_path: Path):
    prompts, sources = _prompt_tables()
    prompts.to_parquet(tmp_path / "prompt_index.parquet", index=False)
    assert _verify_prompt_index(tmp_path, sources) == EXPECTED_CASE_COUNT

    prompts.at[10, "prompt_token_ids_sha256"] = "0" * 64
    prompts.to_parquet(tmp_path / "prompt_index.parquet", index=False)
    with pytest.raises(AssertionError, match="prompt_token_ids_sha256 mismatch"):
        _verify_prompt_index(tmp_path, sources)


def _valid_state() -> dict:
    return {
        "schema_version": "1.0.0",
        "state_name": "pre_forward_input_producing_raw_x0",
        "mask_token_id": 126336,
        "generation_length": 256,
        "total_steps": 256,
        "history_index_base": 0,
        "step_number_base": 1,
        "checkpoint_locator": {
            "checkpoint": 0.6,
            "history_index": 153,
            "step_number": 154,
            "progress_mapping_file": "progress_mapping.json",
        },
        "prompt_source": "prompt_index.parquet.prompt_token_ids",
        "pre_step_commits_rule": "restore only tokens committed at history indices < t",
        "pre_step_uncommitted_rule": (
            "all positions not committed before t remain mask_token_id"
        ),
        "current_stop_now_rule": (
            "raw x0_history[t] with replayed committed tokens from history indices <= t"
        ),
        "current_commit_rule": (
            "the commit selected at t occurs after forward t and is excluded from "
            "the pre-forward input at t"
        ),
        "hidden_state_timing": "the same forward pass that produces raw x0_history[t]",
        "forbidden_future_inputs": [
            "future x0_history indices > t",
            "future true_indices commit history > t",
            "final answer metadata: ans_posidx and pred_token_id",
            "correctness field: correct",
            "ground truth field: gt_text",
        ],
    }


def test_state_semantics_freezes_pre_forward_timing(tmp_path: Path):
    path = tmp_path / "state_semantics.json"
    path.write_text(json.dumps(_valid_state()), encoding="utf-8")
    locator = {"checkpoint": 0.6, "history_index": 153, "step_number": 154}
    _verify_state_semantics(tmp_path, locator)

    invalid = _valid_state()
    invalid["forbidden_future_inputs"] = ["final answer metadata"]
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(AssertionError, match="explicitly ban future x0"):
        _verify_state_semantics(tmp_path, locator)


def _write_source_snapshot(cache: Path, workspace: Path) -> None:
    inventory = [
        workspace / "pyproject.toml",
        workspace / "src" / "module.py",
        workspace / "scripts" / "run.py",
        workspace / "tests" / "test_run.py",
    ]
    entries = []
    for source in inventory:
        relative = source.relative_to(workspace).as_posix()
        snapshot_relative = f"provenance/source_snapshot/{relative}"
        snapshot = cache / snapshot_relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        entries.append(
            {
                "workspace_relative_path": relative,
                "snapshot_relative_path": snapshot_relative,
                "sha256": sha256_file(snapshot),
                "size_bytes": snapshot.stat().st_size,
            }
        )
    manifest = cache / "provenance" / "source_code_manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": "1.0.0", "files": entries}), encoding="utf-8"
    )


def test_source_snapshot_is_complete_and_matches_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"
    for path, content in (
        (workspace / "pyproject.toml", "[project]\nname='x'\n"),
        (workspace / "src" / "module.py", "VALUE = 1\n"),
        (workspace / "scripts" / "run.py", "print('x')\n"),
        (workspace / "tests" / "test_run.py", "def test_x(): pass\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _write_source_snapshot(cache, workspace)
    assert (
        _verify_source_code_provenance(
            cache, workspace, compare_workspace=True
        )
        == 4
    )

    (workspace / "src" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="workspace source hash differs"):
        _verify_source_code_provenance(cache, workspace, compare_workspace=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_parser_config_and_status_cross_checks(tmp_path: Path):
    parser = {
        "name": PARSER_NAME,
        "version": PARSER_VERSION,
        "marker_rule": "after Answer: or #### take the first valid number",
        "fallback_rule": "use last rationale number for empty/whitespace marker suffix",
        "same_parser_for_gt_mid_final": True,
    }
    status = {
        "cache_status": "READY_FOR_G2",
        "G0_status": "G0_PASS",
        "G1_status": "G1_PASS",
        "verified": True,
    }
    manifest = {
        "answer_parser_version": f"{PARSER_NAME}:{PARSER_VERSION}",
        "G0_gate_result": "G0_PASS",
        "G1_gate_result": "G1_PASS",
    }
    _write_json(tmp_path / "parser_config.json", parser)
    _write_json(tmp_path / "status.json", status)
    _verify_parser_and_status(tmp_path, manifest)

    parser["fallback_rule"] = "use the last number"
    _write_json(tmp_path / "parser_config.json", parser)
    with pytest.raises(AssertionError, match="empty/whitespace"):
        _verify_parser_and_status(tmp_path, manifest)


def _source_metadata_fixture(tmp_path: Path):
    workspace = tmp_path / "workspace"
    cache = workspace / "cache" / "version"
    metadata_dir = workspace / "data" / "metadata"
    gsm_test = workspace / "data" / "raw" / "GSM8K" / "main" / "test-00000-of-00001.parquet"
    gsm_test.parent.mkdir(parents=True)
    gsm_test.write_bytes(b"frozen-gsm8k-test")
    test_hash = sha256_file(gsm_test)
    gsm_manifest = {
        "dataset_repo": GSM8K_REPO,
        "dataset_revision": GSM8K_REVISION,
        "configuration": "main",
        "split": "test",
        "files": [
            {
                "relative_path": "main/test-00000-of-00001.parquet",
                "sha256": test_hash,
                "size_bytes": gsm_test.stat().st_size,
            }
        ],
    }
    download_manifest = {"source": "mirror"}
    tokenizer_manifest = {"weights_downloaded": False}
    _write_json(metadata_dir / "gsm8k_manifest.json", gsm_manifest)
    _write_json(metadata_dir / "download_manifest.json", download_manifest)
    _write_json(metadata_dir / "tokenizer_manifest.json", tokenizer_manifest)
    frozen_gsm = (
        cache
        / "provenance"
        / "frozen_context"
        / "data"
        / "metadata"
        / "gsm8k_manifest.json"
    )
    frozen_gsm.parent.mkdir(parents=True)
    frozen_gsm.write_bytes((metadata_dir / "gsm8k_manifest.json").read_bytes())
    source_index = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "source_size_bytes": [10, 20],
        }
    )
    manifest = {
        "dataset_revision_SHA": "d" * 40,
        "exact_trajectory_folder": "folder",
    }
    source_manifest = {
        "source_file_count": 2,
        "source_total_bytes": 30,
        "dataset_revision": "d" * 40,
        "exact_trajectory_folder": "folder",
        "tokenizer_weights_downloaded": False,
        "download_manifest_sha256": sha256_file(metadata_dir / "download_manifest.json"),
        "tokenizer_manifest_sha256": sha256_file(metadata_dir / "tokenizer_manifest.json"),
        "gsm8k_repo": GSM8K_REPO,
        "gsm8k_revision": GSM8K_REVISION,
        "gsm8k_test_relative_path": "data/raw/GSM8K/main/test-00000-of-00001.parquet",
        "gsm8k_test_sha256": test_hash,
        "gsm8k_test_size_bytes": gsm_test.stat().st_size,
        "gsm8k_manifest_sha256": sha256_file(metadata_dir / "gsm8k_manifest.json"),
        "query_template_sha256": FROZEN_QUERY_TEMPLATE_SHA256,
        "mask_token_id": MASK_TOKEN_ID,
        "prompt_index_rows": EXPECTED_CASE_COUNT,
        "state_contract": "pre_forward_input_producing_raw_x0",
        "pre_step_commit_cutoff": "history indices < t",
    }
    _write_json(cache / "source_manifest.json", source_manifest)
    return workspace, cache, source_index, manifest, gsm_test


def test_gsm8k_and_state_source_manifest_cross_checks(tmp_path: Path):
    workspace, cache, source_index, manifest, gsm_test = _source_metadata_fixture(tmp_path)
    _verify_source_manifest_metadata(
        cache,
        workspace,
        source_index,
        manifest,
        compare_workspace=True,
    )
    gsm_test.write_bytes(b"tampered")
    with pytest.raises(AssertionError, match="GSM8K test size|GSM8K test hash"):
        _verify_source_manifest_metadata(
            cache,
            workspace,
            source_index,
            manifest,
            compare_workspace=True,
        )


def test_code_bundle_and_frozen_protocol_hashes(tmp_path: Path):
    code_manifest = {
        "schema_version": "1.0.0",
        "files": [
            {
                "workspace_relative_path": "src/x.py",
                "snapshot_relative_path": "provenance/source_snapshot/src/x.py",
                "sha256": "a" * 64,
                "size_bytes": 1,
            }
        ],
    }
    code_hash = _canonical_json_sha256(code_manifest)
    source_manifest = {"analysis_code_bundle_sha256": code_hash}
    protocol = {
        "schema_version": "1.0.0",
        "G0": G0_GATE_PROTOCOL,
        "G1": G1_GATE_PROTOCOL,
        "G0_sha256": _canonical_json_sha256(G0_GATE_PROTOCOL),
        "G1_sha256": _canonical_json_sha256(G1_GATE_PROTOCOL),
    }
    manifest = {
        "analysis_code_bundle_sha256": code_hash,
        "G0_gate_protocol_sha256": protocol["G0_sha256"],
        "G1_gate_protocol_sha256": protocol["G1_sha256"],
    }
    _write_json(tmp_path / "provenance" / "source_code_manifest.json", code_manifest)
    _write_json(tmp_path / "provenance" / "protocol.json", protocol)
    _write_json(tmp_path / "source_manifest.json", source_manifest)
    assert _verify_code_bundle_and_protocol(tmp_path, manifest) == code_hash

    manifest["G1_gate_protocol_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="G1 gate protocol hash"):
        _verify_code_bundle_and_protocol(tmp_path, manifest)
