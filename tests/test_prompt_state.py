from pathlib import Path

import pytest
import torch

from reva_dlm.config import GSM8K_TEST_FILE, TOKENIZER_DIR, TRAJECTORY_DIR
from reva_dlm.prompt_state import (
    MASK_TOKEN_ID,
    build_prompt_text,
    load_gsm8k_test,
    reconstruct_pre_step_input,
)
from reva_dlm.trajectory import load_trajectory


def test_frozen_gsm8k_prompt_and_pre_step_state():
    transformers = pytest.importorskip("transformers")
    if not GSM8K_TEST_FILE.is_file():
        pytest.skip("frozen GSM8K test source is not downloaded")
    frame = load_gsm8k_test(GSM8K_TEST_FILE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_DIR, local_files_only=True
    )
    record = load_trajectory(TRAJECTORY_DIR / "question_0000_steps_256.pt")
    prompt_ids = tokenizer(build_prompt_text(frame.iloc[0]["question"]))["input_ids"]
    state = reconstruct_pre_step_input(record, prompt_ids, history_index=153)
    assert len(prompt_ids) == record["prompt_token_len"]
    assert state.input_token_ids.shape[0] == record["prompt_token_len"] + 256
    assert state.committed_count_before_step == 153
    assert int((state.generation_token_ids == MASK_TOKEN_ID).sum()) == 103
    assert state.committed_mask_before_step.dtype == torch.bool
