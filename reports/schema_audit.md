# ReVA Mini schema audit

## Frozen source

- Dataset: `YefanZhou98/DLM-Decoding-Analysis`
- Revision: `91beb881acaa0b6edfccd88e8d19c08ec5e1225b`
- Exact folder: `question_histories_low_conf_none_index_genlen_step256_blocklen32`
- Files: 1319 (`DEV=1055`, frozen `G3_HOLDOUT=264`)
- G3 correctness/trajectory labels evaluated: **False**
- G3 rows used for feature/model/gate selection: **False**
- Structural validation ran on all source files: **True**
- Raw holdout sources contain outcome-bearing fields: **True**

## Actual schema and organization

Every file has the same ten keys: `ans_posidx, correct, gt_text, gt_token_id, pred_ans, pred_text, pred_token_id, prompt_token_len, true_indices_history, x0_history`. The dataset card advertises `gen_ids`, but it is absent from every target file. `x0_history` is 8 block-major tensors x 32 local steps with shape template `[32, prompt_token_len + 256]` and dtype `torch.int64`. `true_indices_history` is 8 blocks x 32 steps x Tensor[1,2]. Prompt length ranges from 94 to 259 tokens.

All 1319 exact prompts were rebuilt from frozen `openai/gsm8k` rows using the Prophet template SHA256 `231d269274b6a04711d192c935b7a4785da99e6823afbc817972a1d51f48425e` and the local tokenizer snapshot. Row index, canonical GT, prompt length, and x0 width had 0 mismatches. The pre-forward input contract (exact prompt + commits at indices `<t` + masks elsewhere) was reconstructed for 1319 trajectories. Prompt token IDs and hashes are retained for exact future state reconstruction; mask token ID is 126336.

The official generator stores raw argmax `x0` before restoring previously committed tokens. Accordingly, raw `x0_history[-1]` is not the final output: only 14/1319 files are token-identical to the final replay, with mean 7.842 differing tokens (range 0–37).

## Reconstruction validation

Commit replay followed by local tokenizer decoding exactly matches stored `pred_text` for 1319/1319 files (100.000%). This exceeds the 99% prerequisite. The stop-now state at step `t` is defined as all `<=t` committed tokens fixed to their replayed values and all remaining positions filled from that step's raw `x0`; this matches Prophet's early-exit fill semantics. It does not use final `ans_posidx`, final `pred_token_id`, or any future commit.

The final answer anchor is semantically invalid in 14 files because the collector converts subsequence-search `-1` into `prompt_token_len - 1`. Neither `ans_posidx` nor `pred_token_id` is used for labels or features.

## Parser audit on DEV only

Parser v1.1 is a protocol revision made after the v1.0 audit exposed an empty-terminal-marker edge case and was frozen before this rerun. It agrees with the stored collector flag for 964/1055 DEV cases (91.374%). Direction-specific disagreements are: stored false → canonical true = 91; stored true → canonical false = 0. Empty-marker fallback applied to 3 DEV final text(s) and 229 DEV checkpoint text(s). Stored `correct` is never used as a G0/G1 label; the same v1.1 parser is used for GT, current, and final text.

## Deterministic 20-case structural inspection

| case_id | prompt_len | commit_steps | final_exact | raw_last_diff | gen_ids_present | pred_anchor_found |
| --- | --- | --- | --- | --- | --- | --- |
| question_0001 | 97 | 256 | True | 4 | False | True |
| question_0066 | 129 | 256 | True | 10 | False | True |
| question_0084 | 95 | 256 | True | 6 | False | True |
| question_0117 | 101 | 256 | True | 3 | False | True |
| question_0309 | 134 | 256 | True | 11 | False | True |
| question_0354 | 156 | 256 | True | 10 | False | True |
| question_0376 | 110 | 256 | True | 12 | False | True |
| question_0452 | 113 | 256 | True | 9 | False | True |
| question_0492 | 104 | 256 | True | 4 | False | True |
| question_0500 | 109 | 256 | True | 5 | False | True |
| question_0530 | 144 | 256 | True | 16 | False | True |
| question_0778 | 151 | 256 | True | 14 | False | True |
| question_0831 | 157 | 256 | True | 4 | False | True |
| question_0832 | 124 | 256 | True | 11 | False | True |
| question_1039 | 127 | 256 | True | 12 | False | True |
| question_1063 | 119 | 256 | True | 6 | False | True |
| question_1076 | 119 | 256 | True | 0 | False | True |
| question_1163 | 107 | 256 | True | 13 | False | True |
| question_1271 | 123 | 256 | True | 12 | False | True |
| question_1294 | 158 | 256 | True | 9 | False | True |
