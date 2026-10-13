# Compact A0 → Raw-only → complete matched ablations

Completed 2026-10-14. Terminal: **COMPLETE_A0_COMPACT_RAW_FIRST_ABLATION_OUTER_EXPLORATORY**.

All30 new cells ran: R0 five-fold raw-only first, then F1–F5 twenty-five feature cells. F0/F6/F7 reuse15 admitted historical model-fold cells. Raw results were independently verified and published at commit22425ef38e34cf65d64e5dc761bdf375e3c8d20c **before feature training**. The first Stage III SSH request never connected; no feature process or publication marker existed. The identical frozen command was submitted after read-only checks restored the same server connection.

The fixed compact F0 exceeds this raw-only model. Patient Z removal and ABS-only lose ranking. DELTA/RATIO deletions do not establish ranking loss individually, while RATIO deletion reduces Macro-F1. NoLN improves AP/MRR/Top1 but has lower F1 point estimates. ZDELTA/Source restoration shows no supported primary gain. This does not show every F0 component is necessary or that F0 is the best-scoring variant.

**F0 remains the user-defined complete baseline**, not reselected from this table. The previously viewed80-patient cohort makes all findings exploratory, not blind external clinical confirmation.

## Table A — separate input/model baseline

| Model / input | Parameters | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 Complete A0-compact | 8817 | 0.736977 | 0.530653 | 0.624236 | 0.414671 | 0.694267 | 0.587500 |
| R0 Raw-only waveform baseline | 18153 | 0.633626 | 0.408496 | 0.543958 | 0.321269 | 0.588369 | 0.475000 |

F0 is masked88D with62 ABS/DELTA/RATIO columns and native LayerNorm. R0 is waveform-only: exact RawTiny32 for all valid500-sample windows → mean per available seizure → available-seizure mean/population std64 → learned Linear64→88 → differentiable full-patient latent-Z → original **initial** A0 shared classifier; all jointly trained. It uses no engineered coordinates/scaler, fitted F0 weights, A1 task head or old RawTiny scores.

This is not a one-factor feature ablation: architecture/pooling/normalization differ, raw250Hz Nyquist125Hz cannot provide full80–150Hz high gamma, and some original engineered features may use higher native-rate data. Poor R0 performance is not a universal verdict against raw EEG. Clock/acquisition/filter limitations remain.

## Table B — complete compact ablation

✓ retained; ✗ absent. All engineered interfaces remain88D. F6/F7 are restoration controls, not deletions. F3/F4 keep ZDELTA absent.

| Configuration | Patient Z | ABS | DELTA | ZDELTA | RATIO | Source | LN | Columns /88 | Params | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| F0 Complete A0-compact | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | 62 | 8817 | 0.736977 | 0.530653 | 0.624236 | 0.414671 | 0.694267 |
| F1 w/o Patient Z | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | 62 | 8817 | 0.659539 | 0.461353 | 0.505065 | 0.254935 | 0.630580 |
| F2 ABS Only | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | 18 | 8817 | 0.699011 | 0.482819 | 0.605307 | 0.390291 | 0.652969 |
| F3 w/o DELTA | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | 36 | 8817 | 0.738373 | 0.534921 | 0.617512 | 0.406897 | 0.724262 |
| F4 w/o RATIO | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | 44 | 8817 | 0.737767 | 0.530031 | 0.610257 | 0.393977 | 0.716469 |
| F5 w/o LayerNorm | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | 62 | 8641 | 0.748867 | 0.560798 | 0.609797 | 0.406446 | 0.790162 |
| F6 + ZDELTA restored | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | 88 | 8817 | 0.739768 | 0.537887 | 0.624919 | 0.419405 | 0.728620 |
| F7 + Source restored | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | 62 | 9221 | 0.733994 | 0.526393 | 0.618839 | 0.409059 | 0.697692 |

F0/F6/F7 are privately hash-verified historical score reuse. R0/F1–F5 are new actual training and one frozen outer pass. Source-conditioned old deletion rows, old all88D noLN/noPatientZ and A1 fixed-query RawTiny outcomes are not substituted.

## Remaining patient-equal metrics

All11 original metrics plus predicted EZ fraction and mean confusion counts are in OUTER_ALL_MODELS_SUMMARY.csv. Primary unit is equal mean of80 unique patients, not pooled channels or equal-five-fold means.

| Model | NEZ-F1 | Balanced accuracy | Top1 | Sensitivity | Specificity | Accuracy | Predicted EZ fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 | 0.833800 | 0.663990 | 0.587500 | 0.460046 | 0.867935 | 0.763172 | 0.203485 |
| R0 | 0.766647 | 0.574269 | 0.475000 | 0.368038 | 0.780500 | 0.699405 | 0.241122 |
| F1 | 0.755195 | 0.571712 | 0.525000 | 0.330311 | 0.813112 | 0.713204 | 0.214922 |
| F2 | 0.820322 | 0.639921 | 0.537500 | 0.433662 | 0.846180 | 0.739020 | 0.214719 |
| F3 | 0.828126 | 0.658724 | 0.625000 | 0.457212 | 0.860237 | 0.752898 | 0.203860 |
| F4 | 0.826538 | 0.656095 | 0.612500 | 0.454472 | 0.857718 | 0.754130 | 0.210493 |
| F5 | 0.813149 | 0.664533 | 0.712500 | 0.489102 | 0.839965 | 0.737777 | 0.224720 |
| F6 | 0.830434 | 0.666405 | 0.637500 | 0.471033 | 0.861776 | 0.758773 | 0.207411 |
| F7 | 0.828618 | 0.653030 | 0.600000 | 0.446865 | 0.859195 | 0.760591 | 0.206251 |

## Paired patient bootstrap

10,000 common80-ID draws, seed42; all patient channels remain together, thresholds/checkpoints stay fixed. Direction **F0 minus other**, positive means the other is lower. Effect[unadjusted percentile95% interval]; conditional descriptive repeated-test evidence, not multiplicity/selection/seed corrected.

| Comparison | ΔAUROC [CI] | ΔAUPRC [CI] | ΔMacro-F1 [CI] | ΔEZ-F1 [CI] | ΔMRR [CI] |
|---|---|---|---|---|---|
| F0−R0 | 0.103351 [0.054102, 0.152366] | 0.122156 [0.065015, 0.179153] | 0.080277 [0.043348, 0.116817] | 0.093401 [0.028778, 0.155591] | 0.105898 [-0.015117, 0.220413] |
| F0−F1 | 0.077439 [0.037241, 0.117843] | 0.069300 [0.027273, 0.110860] | 0.119170 [0.083534, 0.155542] | 0.159736 [0.104346, 0.214571] | 0.063687 [-0.020871, 0.146890] |
| F0−F2 | 0.037966 [0.006793, 0.069595] | 0.047834 [0.007922, 0.090040] | 0.018929 [-0.003698, 0.041012] | 0.024379 [-0.013189, 0.062029] | 0.041297 [-0.053819, 0.137377] |
| F0−F3 | -0.001396 [-0.022693, 0.016779] | -0.004269 [-0.018736, 0.009553] | 0.006724 [-0.003664, 0.017020] | 0.007774 [-0.010016, 0.024955] | -0.029995 [-0.084996, 0.024917] |
| F0−F4 | -0.000790 [-0.008668, 0.007522] | 0.000621 [-0.008287, 0.009392] | 0.013978 [0.003478, 0.024599] | 0.020694 [0.003646, 0.038626] | -0.022202 [-0.069347, 0.020480] |
| F0−F5 | -0.011890 [-0.027684, 0.003964] | -0.030145 [-0.054703, -0.007622] | 0.014438 [-0.000084, 0.029568] | 0.008225 [-0.016156, 0.032263] | -0.095895 [-0.162019, -0.032225] |
| F0−F6 | -0.002791 [-0.013290, 0.006292] | -0.007235 [-0.020354, 0.006002] | -0.000684 [-0.009662, 0.008299] | -0.004734 [-0.019539, 0.009966] | -0.034353 [-0.089027, 0.022209] |
| F0−F7 | 0.002984 [-0.006058, 0.012279] | 0.004260 [-0.004258, 0.012537] | 0.005397 [-0.006669, 0.018734] | 0.005611 [-0.015061, 0.029188] | -0.003425 [-0.041466, 0.032129] |

## Interpretation

- **Patient Z:** F0−F1 AUROC0.077439[0.037241,0.117843], AP0.069300[0.027273,0.110860], Macro-F10.119170[0.083534,0.155542]. The strongest detrimental engineered deletion under this protocol; not a universal normalization or physiological causal law.
- **ABS Only:** AUROC/AP losses have intervals above zero; F1/top-ranking losses are not all resolved. The relative views collectively add ranking evidence versus ABS-only, without identifying their individual causal effects.
- **DELTA:** F3 slightly exceeds F0 in AUROC/AP/MRR point estimates; ranking and F1 difference intervals cross zero. Individual DELTA necessity or harm is not established.
- **RATIO:** F4 has near-unchanged AUROC/AP but loses Macro-F10.013978[0.003478,0.024599]. Operating-point dependence under own validation-selected thresholds, not a demonstrated ranking increment.
- **LayerNorm:** F5−F0 AP0.030145[0.007622,0.054703], MRR0.095895[0.032225,0.162019], Top10.125000[0.037500,0.212500] favor noLN. AUROC0.011890[-0.003964,0.027684] is unresolved. Macro-F1 declines0.014438 but its interval still narrowly crosses zero; specificity/accuracy point estimates fall. **F0 is not uniformly superior to noLN.** No new model or threshold rescue is selected after this finding.
- **Restorations:** F6/F7 primary AUROC/AP/Macro-F1/EZ-F1 intervals cross zero. No supported benefit or harm of restoring ZDELTA/Source. F6 is all88D source-free, not complete compact; F7 adds404 source parameters and must not be called w/o Source.
- **Mask meaning:** post-preprocessing zero masks change input-dependent LayerNorm statistics as well as information; auxiliary duplicates and constant-zero positions remain. Retraining differences are not causal physiological feature importance.

## Confusion counts — distinguish units

Mean patient TP/FP/TN/FN are in the primary summary. The separate pooled counts below always total7635 channels,1743 EZ,5892 NEZ.

| Model | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| F0 | 729 | 799 | 5093 | 1014 |
| R0 | 643 | 1235 | 4657 | 1100 |
| F1 | 630 | 1089 | 4803 | 1113 |
| F2 | 691 | 945 | 4947 | 1052 |
| F3 | 702 | 857 | 5035 | 1041 |
| F4 | 736 | 868 | 5024 | 1007 |
| F5 | 723 | 1010 | 4882 | 1020 |
| F6 | 742 | 834 | 5058 | 1001 |
| F7 | 748 | 855 | 5037 | 995 |

F5 has211 additional pooled FP and6 fewer TP than F0, despite higher patient-mean sensitivity. Do not turn that difference in weighting into a pooled sensitivity improvement. Ranking gains do not guarantee threshold transport.

## Development and subgroup artifacts

65 legal validation appearances from47 IDs; original equal-appearance mean primary, separate47-ID-equal summary secondary. These are not65 independent patients.

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|
| F0 | 0.716569 | 0.508559 | 0.642067 | 0.433516 | 0.676591 |
| R0 | 0.630896 | 0.389998 | 0.567631 | 0.347431 | 0.580004 |
| F1 | 0.683886 | 0.491006 | 0.566073 | 0.293143 | 0.711814 |
| F2 | 0.687139 | 0.456533 | 0.617246 | 0.397969 | 0.612188 |
| F3 | 0.714565 | 0.532494 | 0.641413 | 0.434639 | 0.712062 |
| F4 | 0.709740 | 0.510629 | 0.642203 | 0.440167 | 0.646653 |
| F5 | 0.736567 | 0.552577 | 0.645573 | 0.438483 | 0.762103 |
| F6 | 0.710667 | 0.518235 | 0.638080 | 0.431198 | 0.686550 |
| F7 | 0.721787 | 0.520316 | 0.648520 | 0.445185 | 0.676749 |

Full-precision by-fold/by-center tables and fixed-threshold bootstrap are provided for both stages/estimands. The five outer folds contain16/16/17/15/16 patients and1631/1498/1746/1425/1335 channels. Known-center subgroup tables are descriptive, not unseen-center tests. No center-specific choice or patient exclusion.

## Executed identity and integrity

- Exact80-patient/256-seizure/7635-channel contract,1743 EZ/5892 NEZ; original feature/order/partition/source hashes and FIT/VAL/test roles verify. Raw coverage1471965 valid channel-windows and15074 record-windows; every waveform window matches its original slice within valid support. No extra filter, resampling, spectral feature or EEG acquisition.
- Authoritative F0 is d2_no_source_no_zdelta_outer_seed42_v1, not a similar newest checkpoint or source-conditioned rank-zero head. F0 epochs3/3/2/3/4 and thresholds0.465/0.240/0.320/0.335/0.410 replay. All15 reusable checkpoint/bank/selection/score bindings verify; baseline outer is never regenerated.
- Name-derived duplicate-consistent masks after original preprocessing; F3/F4 retain no ZDELTA. F1 uses separately FIT-fitted original absolute imputer/scaler with no Patient Z. Other retained inputs are bitwise original values. All new models scratch initialized from the original constructor; F5 only removes176 LN parameters. No fitted F0 warm start.
- Original unchanged trainer: seed42+1009×fold, FIT-only NEZ-positive class-weighted BCE, one update per patient, AdamW0.001/wd0.0001, clip1, max30/min6/patience6, source penalty0, own validation Macro-F1/EZ-F1/earlier epoch and original0.005 threshold-grid/ties.
- Raw51 epochs in627.914 measured trainer/collection seconds; feature282 epochs in22.664 seconds. **333 actual competitive epochs**, excluding synthetic tests/two one-step FIT diagnostics, implementation/admission, inference/statistics and transfer. No completed clinical epoch discarded or native model crash. Transport launch/pump issues and the before-start Stage III connect failure are disclosed; identical commands continued only after process/status inspection.
- All stage models, preprocessors, masks, own thresholds and source bindings freeze before new outer inference; each stage's entire new score set seals before that process loads global clinical NPZ y. Stage I historical-label provenance verification and legacy pickle field materialization are separately disclosed.
- Two preclinical fixture repairs: explicit construction seeding, and mean-loss-unit gradient probe with deterministic CPU resume qualification. Initial bindings/diagnostics remain preserved. No matrix/data/model/clinical loss/hyperparameter change; fixed tolerances not widened. Bitwise GPU floating-point trajectory/resume is not claimed. Stored model/optimizer/RNG and tested CPU/CUDA RNG restore are preserved.
- Independent score-only checks PASS: raw290 + feature1160 =1450 checked patient/model cells (1305 unique9-arm appearances; repeated F0 union checked), **256 paired intervals**, every primary/fold/center/unique-ID aggregate and all9 pooled confusion matrices. Maximum metric drift3.33e-16; no new verifier inference or threshold reoptimization.
- Raw publication22425ef preceded25 feature cells; exactly30 new cells and15 historical model-fold cells reused. No extra arm/seed/grid/normalization/loss/ensemble/rescue.

## Limits and stop

Tables are exploratory conditional evidence with this provenance, **not newly blind JBHI confirmation**. Prior outcomes informed the user-defined F0 identity. Resampling fixed patient scores does not remove selection/adaptive viewing, multiple testing, center dependence, seed/split uncertainty or acquisition/EDF+D clock limits. Raw/engineered inputs are not information-identical. No causal onset propagation, universal raw inadequacy, or supported Source/ZDELTA benefit claim follows.

Every row is actual new execution or verified historical reuse; no missing result filled with zero. Only source, hashes and anonymous aggregates are published. Signals, feature banks, clinical IDs/labels, individual predictions, weights, optimizer/RNG states and logs remain trusted-server private. **Stop after seed42; no v2 or test-driven redesign.**
