# Source audit (written before new model-score analyses)

## Scope and historical evidence

The matched A0 reference is TabM commit d0277ae6a9fb888167157dddd3371ae06384bc6e: development Macro-F1 0.6380797828499001, EZ-F1 0.431198131752259, AP 0.518235100110259, AUROC 0.7106669304311658. Frozen TabM FIT/VAL input banks and original A0 checkpoints are reused. No outer test performance is read or computed here.

Required historical FINAL_REPORT files were read in full. Direct patient-shift and optimal-cardinality studies used a different B0 development score (0.628380), not this A0. Their oracle reached 0.702841, but OOF learned decisions fell to 0.591867 (shift) and 0.584579/0.589634/0.623452 (cardinality variants); none passed transfer gates. Thus threshold headroom has already been demonstrated in a different score protocol, and label-blind recovery has negative evidence.

Crosspatient geometry (archived branch `codex/crosspatient-feature-geometry-audit-seed42-v1`) used A1, not A0. Label-using local linear probes substantially improved AP, whereas simple unlabeled alignments did not pass their gate. Seizure-resolved geometry (archived branch `codex/seizure-resolved-geometry-identifiability-seed42-v1`) likewise used A1: supervised oracle directions were stable, but unlabeled context direction prediction and AP failed. Neither is zero-shot direction identifiability for A0.

Active few-shot calibration also used A1: B8 uncertainty AP gain +0.0229, CI [0.0087,0.0394], while half-support AP gain +0.0161 failed its separate +0.03 capacity criterion. Simulated retrospective labels are not evidence that clinicians can supply unbiased support at deployment. TabM W2 improved A0 ranking slightly but not Macro-F1 (delta -0.000313, CI crossing zero). All historical protocols and endpoints differ from this controlled A0 probe.

## Actual executed export path

The original PR-UAS gate used server `pr_uncertainty_aware_supervision_seed42_v1/code/reproduce_b0_v2.py` (SHA256 595aac45e0d6838f898b1bea550743464a52486b77e510a73c6f0922dace7857). It called `outcome_hifos.cache_schema.load_cache_contract`, `task1_baselines.cache_io.task1_feature_records` and `task1_baselines.feature_aggregation.build_channel_feature_table(profile='p2_matched_simple')`. Feature order was manifest-checked. Patient-index canonical labels were authoritative; record-label fallback was only available when index labels were absent. The actual 80-patient cache has complete canonical labels, and export-to-index alignment is verified again.

Original cache: `D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl`, SHA256 9b5bb58a0175aba494ad79e30c5a65c7cee33c7d464273f9bcf62b9b280b0087. Original export SHA256 1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd. Split SHA256 fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278. Original core SHA256 1b1ccb0a361e6245cb2e6089347e147036aac90437c0992194f6db7bebf1503a. All checkpoint and bank hashes are listed in PROTOCOL_LOCK.json.

## Upstream clinical conversion inspected

`build_neuroez_c_four_center_caches.py` loads `bn_pdgs_ranker.data_readers.loader` for LZU/HUP/multicenter and `neuroez_c.data.pediatric` for pediatric; per-center caches are merged without relabeling. `run_baseline_three_centers/build_center_caches_from_patient_records.py::_patient_index_entry` converts reader NEZ labels to EZ-positive index labels. Task1 converts them back to NEZ for export.

`bn_pdgs_ranker/data_readers/schemas.py::build_patient_records` takes the minimum reader NEZ label across seizures for a canonical channel (union of EZ designations), while preserving the first available channel metadata. This can explain index vs first-record metadata differences, but will be explicitly checked against actual run labels rather than assumed.

Inspected adapter rules: HUP `bids_common.read_bids_channel_table` accepts SOZ OR resection OR a SOZ/seizure-onset/resection status-description cue. Multicenter uses SOZ unless explicitly configured `soz_or_resected`; actual metadata source categories determine the cached mode. LZU uses annotated spreadsheet channel numbers after bad-channel exclusion. Pediatric prefers explicit channel-level workbook EZ/NEZ and usability fields, otherwise parsed description fallback; actual `label_mode` and sources are audited, with sparse source categories suppressed. Unknown biological EZ truth is not recoverable from these proxies alone.

The inspected alternative repository `2f1a32aaae_clinical_target.py` / server `neuroez_c/task2/clinical_target.py` is NOT called by this actual Task1 path. Its HUP SOZ OR resection and fallback policies must not be attributed to A0. Direct comparison is diagnostic only and never replaces labels.

Current source hashes are output in audit/CLINICAL_SOURCE_INVENTORY.json. The cache does not embed execution-time adapter hashes; current source plus actual stored rule/label crosschecks establish operational lineage, not a perfect historical code-version attestation. This limitation is not concealed.

## Legal OOF and representation dependency

PR-UAS FIT-OOF artifacts are checked against original raw feature identities/order, query exclusions from BOTH teacher training and internal checkpoint selection, teacher checkpoint/output hashes, teacher TRAIN-only preprocessing, and full once-only FIT coverage. Their frozen scores are ten-pass MC dropout means, not deterministic full-FIT A0. Teacher inputs have their own cross-fitted preprocessing, as required; the direction PCA uses one outer FIT-only prepared 88D bank. This shared raw feature lineage is checked exactly, rather than approximating missing OOF scores by final A0 training scores. MC averaging, smaller teacher membership and preprocessing differences can undermine regularization transfer and are disclosed. If any provenance gate fails, B is blocked; A and C continue.

No banned data-extraction skill is used. No new EEG, feature export, backbone, labels, or outer-test predictions are generated.

## Reviewed report content hashes

| Experiment | SHA256 of reviewed Git report bytes |
|---|---|
| A0 TabM | be250d2a8fc48a3c0d69d37c9ca524581d288405c242981b1c3f7b10581daeb1 |
| Direct shift | e6fbaa79bb4a3ba90856b4a233dc5cad23a2bd60d402adf45c1b925d386249dd |
| Optimal cardinality | 7f63e07b362e266ff037b66be6925c1cd332c102ab347f78049e73dd00144b6e |
| Crosspatient geometry | 76f3a7f05a68f4e713c5e0db392ae350b3f1ee2075ab008e6c2988a42734f38d |
| Seizure geometry | 930ee1fa548241672d39a309872c8041ad3fff7a58502177c6dc9cdbd3e1e50e |
| Active fewshot | b1864fca9eba5673953cfe8d3144b4ee805cb1826396792706beb91701e93e14 |
