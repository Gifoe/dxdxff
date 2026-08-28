# ReVA CPU-only Mini G0 + G1 report

## 1. Data provenance and mirror download

- Dataset: `YefanZhou98/DLM-Decoding-Analysis` at revision `91beb881acaa0b6edfccd88e8d19c08ec5e1225b`.
- Exact configuration: `question_histories_low_conf_none_index_genlen_step256_blocklen32` (GSM8K, LLaDA-8B-Instruct, low-confidence remasking, no constraint, gen/steps 256, block length 32, temperature 0, CFG 0).
- Download: `curl mirror resolver fallback (resumable .part files)` starting exclusively from `https://hf-mirror.com`, with resumable `.part` files, skip-on-success, sizes, and SHA256 manifest. The higher-level `hf` client was attempted first but the host proxy caused TLS EOF, so the documented curl mirror fallback was used.
- Selected payload: README plus exactly 1,319 target `.pt` files, 1148351444 bytes. No MMLU or alternate GSM8K configuration was downloaded.
- Tokenizer: five allowlisted small files from `GSAI-ML/LLaDA-8B-Instruct` at `08b83a6feb34df1a6011b80c3c00c7563e963b07`; no `.safetensors`, `.bin`, or `.pt` model weight was downloaded.
- Prompt source: `openai/gsm8k` main/test at revision `740312add88f781978c0658806c59bc2815b9866`, downloaded only through `https://hf-mirror.com`; all 1,319 row/question/prompt/token-length/GT bindings match. Exact prompt IDs are retained in the report-side prompt index. Prophet query-template SHA256 is `231d269274b6a04711d192c935b7a4785da99e6823afbc817972a1d51f48425e` and mask token ID is `126336`.
- Prophet source of truth: commit `460afe41c7063a29a9893675aca07b985997bb83`.

## 2. Schema and reconstruction

Actual target files contain ten keys and omit the advertised `gen_ids`. Every trajectory is 8 blocks × 32 steps; each step commits one unique generated position. Prompt lengths are 94–259.

Commit replay reconstructs stored `pred_text` exactly for 1319/1319 cases. Direct raw-last `x0` is exact for only 14/1319, proving that raw history cannot be decoded without replay. Fourteen final answer anchors are invalid and are not used. The model input that produces raw `x0[t]` is separately fixed as exact prompt + commits at indices `<t` + masks elsewhere; the stop-now candidate is raw `x0[t]` with commits at indices `<=t` restored. Full details are in `reports/schema_audit.md`.

Parser v1.1 is a protocol revision made after the initial v1.0 audit found an empty-terminal-marker defect and was frozen before this rerun. It is shared by GT/current/final. On DEV, stored false → canonical true occurs 91 times and stored true → canonical false occurs 0 times; the report preserves this direction rather than collapsing all disagreements into an undifferentiated count. Empty-marker fallback applies to 3 final and 229 checkpoint text(s).

## 3. Split and leakage controls

SHA256(case ID + seed 20260827) froze 1055 DEV and 264 G3 holdout cases before label analysis. G0/G1 do not compute or report G3 correctness/trajectory labels and never use G3 rows for features, fitting, or gates. Structural validation did inspect all source files; raw holdout `.pt` files contain accessible outcome fields, so they are not described as information-sealed. Mid-step reconstruction, answer parsing, and every G1 feature use only states at or before `t`. The explicit future-mutation test passes. Labels/final correctness are stored outside `cpu_features.parquet`.

## 4. G0 results

| checkpoint | n_total | mid_acc | final_acc | n_stable | n_rescue | n_corruption | n_unrecovered | rcr_joint | rsr_joint | corruption_given_current_correct | rescue_given_current_wrong | net_refinement_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.250000 | 1055 | 0.124171 | 0.814218 | 122 | 737 | 9 | 187 | 0.008531 | 0.698578 | 0.068702 | 0.797619 | 0.690047 |
| 0.400000 | 1055 | 0.176303 | 0.814218 | 177 | 682 | 9 | 187 | 0.008531 | 0.646445 | 0.048387 | 0.784810 | 0.637915 |
| 0.500000 | 1055 | 0.289100 | 0.814218 | 293 | 566 | 12 | 184 | 0.011374 | 0.536493 | 0.039344 | 0.754667 | 0.525118 |
| 0.600000 | 1055 | 0.423697 | 0.814218 | 431 | 428 | 16 | 180 | 0.015166 | 0.405687 | 0.035794 | 0.703947 | 0.390521 |
| 0.750000 | 1055 | 0.592417 | 0.814218 | 612 | 247 | 13 | 183 | 0.012322 | 0.234123 | 0.020800 | 0.574419 | 0.221801 |

G0 decision: **G0_EXPLORATORY_PASS**. The formal 50% strong gate failed. The already-selected 60% DEV checkpoint retained adjacent 50% support under parser v1.1; this is same-DEV, post-diagnostic, selection-biased exploratory evidence only.

The strong 50% criterion was not silently weakened. If an exploratory checkpoint is used, its DEV selection is explicit and frozen before G1; it is not presented as strong evidence.

The 60% exploratory checkpoint and its numeric adjacent-support rule were selected on the parser-v1.0 DEV diagnostic, then frozen before this parser-v1.1 rerun. The point is not reselected here. This is same-DEV, post-diagnostic and selection-biased; `G0_EXPLORATORY_PASS` permits completing G1 but not an ordinary READY cache.

## 5. G1 results

### G1 features and protocol

All 29 CPU features are prefix-causal and are stored separately from labels. Evaluation used 5×3 repeated stratified OOF predictions, fold-local transforms, balanced weights, fixed threshold 0.5, and 1,000 case-level bootstrap samples. No G3 holdout row was used.

Best cheap baseline: `hist_gradient_boosting_balanced`; AUROC=0.777847, AUPRC=0.657636, balanced accuracy=0.696189, F1=0.650707, Brier=0.192503. Bootstrap AUROC 95% CI is 0.749622–0.804846.

Individual-fold AUROC range for that model: 0.733053–0.823721 (mean 0.771034, sample SD 0.030712).

Matched controls:

| task | model | n_samples | n_positive | auroc | auprc |
| --- | --- | --- | --- | --- | --- |
| A_same_current_wrong | logistic_all_features | 608 | 428 | 0.582671 | 0.757011 |
| A_same_current_wrong | hist_gradient_boosting_balanced | 608 | 428 | 0.604569 | 0.774000 |
| B_same_current_correct | logistic_all_features | 447 | 16 | 0.599188 | 0.052058 |
| B_same_current_correct | hist_gradient_boosting_balanced | 447 | 16 | 0.534948 | 0.037803 |
| C_same_final_correct | logistic_all_features | 859 | 428 | 0.803093 | 0.793615 |
| C_same_final_correct | hist_gradient_boosting_balanced | 859 | 428 | 0.864584 | 0.838992 |
| D_same_final_wrong | logistic_all_features | 196 | 16 | 0.698611 | 0.137733 |
| D_same_final_wrong | hist_gradient_boosting_balanced | 196 | 16 | 0.542014 | 0.090684 |

G1 decision: **G1_PASS** — Cheap features did not stably solve both frozen key matched controls; the simple proxy set is incomplete.

## 6. Cache and integrity

No ordinary READY cache was created because the scientific eligibility gates did not both pass.

## 7. Recommendation

**BORDERLINE_REVIEW_REQUIRED**

This remains a preexperiment on public GSM8K test trajectories. Even a READY result does not justify a paper-level untouched-test claim; confirmatory work requires newly generated development trajectories and independent benchmarks.
