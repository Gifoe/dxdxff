# Historical sources and immutable inputs

The two historical implementations were read before writing the controls, including the original patient-level TimeConv train/evaluation code and the Omni-D2 model, bank, trainer, evaluator, protocol and aggregate results. This experiment does not reconstruct the head optimizer from its report: `source.py` imports the original `Head('C1')`, `train_heads.train`, split constructor, threshold selector and metric functions after verifying all executable hashes in the historical pre-outer freeze.

Historical public parent: `ded7036eed988e8f52eb96c30b5b422e66ca61d5` (Omni-D2). Original Experiment A source: `caaa11b864716039153cfe25c7ccbd6fa033f4e6`. Protocol and artifact hashes are recorded in `PROTOCOL_LOCK.json` and the public verification audits, not inferred from filenames.

## Runtime lineage

The existing historical source resolves its private bank to `C:\omni_d2_patient_source_seed42_runtime`, its original recordings/checkpoints to `C:\omni_patient5fold_a1plugin_seed42_runtime_r2`, and its code to `E:\DRE-nips\new-pipeline\7-11\omni_d2_patient_source_seed42_v1\code`. This is the original Windows server; no patient data are copied into this repository.

The new runtime is isolated at `C:\omni_pr_mechanism_seed42_runtime`. The existing bank consists of 620 patient/fold cells, including unknown-label observable good channels. Preparation verifies every bank SHA, five CNN checkpoint hashes, original record/cache provenance, 124-patient manifest, historical PR checkpoints, historical prediction ledger and thresholds. It independently replays C0 and C1-PR with exact logit equality and metric tolerance `1e-12`. `audit/HISTORICAL_REPRODUCTION.json` reports the measured errors.

## Clinical and scoring semantics

The immutable ordered operational label rule excludes non-good channels; outcome=1/resection=0 gives normal=0, otherwise SOZ=1 gives pathological=1, otherwise unknown=-1. Original labels and channel ordering are independently matched to the archived original baseline caches. Unknown good channels contribute to normalization only. Primary evaluation is the mean of independently calculated patient metrics across the same 124 patients/10,202 labeled patient-channel units, not official pooled Task-2.

Original 32D embeddings, float64 frozen baseline logits and all EDF/segment aggregation remain unchanged. C0 and historical PR use their archived thresholds without a new fit. No raw signal extraction, CNN inference regeneration, CNN training or PR retraining occurs.

## Leakage and interpretation limits

Only outer FIT labels enter either new head's inner CV/refit. ABS moments are fitted separately on each inner-training population, then recomputed on full outer FIT for the refit. Each patient has equal weight in the global moments. Permutations are label-blind, within-patient derangements, fixed across epochs and inner/refit uses, with independent FIT/HELD seed roles. New held predictions are sealed before their labels are read.

The outer-FIT backbone has seen the inner Head validation patients. Inner head-OOF is therefore not fully out-of-sample end-to-end performance. Outer held patients remain excluded from both training stages. The cohort and its historical results have already been viewed: these comparisons and bootstrap intervals are exploratory repeated-cohort evidence, not independent confirmation or causal identification.
