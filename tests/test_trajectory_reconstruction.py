from pathlib import Path

import pytest
import torch

from reva_dlm.trajectory import (
    flatten_x0_history,
    load_trajectory,
    reconstruct_final_token_ids,
    reconstruct_states,
)


def _synthetic_record():
    # Two blocks, two local steps per block.  The two leading columns are the
    # prompt.  Raw x0 deliberately changes already committed token guesses.
    x0_history = [
        torch.tensor(
            [
                [10, 11, 100, 101, 102, 103],
                [10, 11, 200, 201, 202, 203],
            ]
        ),
        torch.tensor(
            [
                [10, 11, 300, 301, 302, 303],
                [10, 11, 400, 401, 402, 403],
            ]
        ),
    ]
    # Commit relative generation positions 1, 0, 3, and 2 respectively.
    true_indices_history = [
        [torch.tensor([[0, 3]]), torch.tensor([[0, 2]])],
        [torch.tensor([[0, 5]]), torch.tensor([[0, 4]])],
    ]
    return {
        "x0_history": x0_history,
        "true_indices_history": true_indices_history,
        "prompt_token_len": 2,
        # This sentinel proves reconstruction does not depend on the field.
        "gen_ids": [999, 999, 999, 999],
    }


def test_x0_flattening_is_block_major():
    flattened = flatten_x0_history(_synthetic_record()["x0_history"])
    assert flattened[:, 2].tolist() == [100, 200, 300, 400]


def test_replay_overwrites_raw_changes_to_past_commits():
    states = reconstruct_states(_synthetic_record())
    assert [state.candidate_token_ids.tolist() for state in states] == [
        [100, 101, 102, 103],
        [200, 101, 202, 203],
        [200, 101, 302, 303],
        [200, 101, 402, 303],
    ]
    assert [state.committed_count for state in states] == [1, 2, 3, 4]
    assert [state.committed_mask.tolist() for state in states] == [
        [False, True, False, False],
        [True, True, False, False],
        [True, True, False, True],
        [True, True, True, True],
    ]
    assert [(state.block_index, state.step_in_block) for state in states] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_final_ids_come_only_from_commit_replay_not_gen_ids_or_last_raw_x0():
    record = _synthetic_record()
    assert reconstruct_final_token_ids(record).tolist() == [200, 101, 402, 303]
    assert reconstruct_final_token_ids(record).tolist() != record["gen_ids"]
    assert reconstruct_final_token_ids(record).tolist() != flatten_x0_history(
        record["x0_history"]
    )[-1, 2:].tolist()


def test_cpu_load_is_explicit_about_trusted_pickle(tmp_path, monkeypatch):
    expected = _synthetic_record()
    checkpoint = tmp_path / "sample.pt"
    torch.save(expected, checkpoint)

    real_load = torch.load
    observed = {}

    def recording_load(*args, **kwargs):
        observed.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr("reva_dlm.trajectory.torch.load", recording_load)
    loaded = load_trajectory(checkpoint)
    assert observed["map_location"] == "cpu"
    assert observed["weights_only"] is False
    assert all(tensor.device.type == "cpu" for tensor in loaded["x0_history"])


def _published_sample_paths() -> tuple[Path, Path] | None:
    root = Path(__file__).resolve().parents[1]
    trajectory_dir = (
        root
        / "data"
        / "raw"
        / "DLM-Decoding-Analysis"
        / "question_histories_low_conf_none_index_genlen_step256_blocklen32"
    )
    tokenizer_dir = root / "data" / "raw" / "LLaDA-8B-Instruct-tokenizer"
    files = sorted(trajectory_dir.glob("question_*_steps_256.pt"))
    if not files or not (tokenizer_dir / "tokenizer.json").is_file():
        return None
    return files[0], tokenizer_dir


def test_published_sample_replay_decodes_exact_stored_final_text():
    paths = _published_sample_paths()
    if paths is None:
        pytest.skip("published trajectory/tokenizer is not present on disk")
    transformers = pytest.importorskip("transformers")
    sample_path, tokenizer_dir = paths

    record = load_trajectory(sample_path)
    # The actual published target files omit gen_ids; this test must remain
    # valid for that real schema.
    assert "gen_ids" not in record
    final_ids = reconstruct_final_token_ids(record)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    replayed_text = tokenizer.decode(final_ids.tolist(), skip_special_tokens=True)
    assert replayed_text == record["pred_text"]

    raw_last_ids = flatten_x0_history(record["x0_history"])[
        -1, int(record["prompt_token_len"]) :
    ]
    raw_last_text = tokenizer.decode(raw_last_ids.tolist(), skip_special_tokens=True)
    assert raw_last_text != record["pred_text"]
