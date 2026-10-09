# A0 Patient Decision Identifiability — seed42

**Terminal: `PATIENT_ORACLE_HEADROOM_NOT_RECOVERABLE` under the tested probes.** There is substantial retrospective threshold headroom, but neither eight-label intercept adaptation nor fixed PCA4 direction adaptation reliably recovers it on held-out query channels. Half-channel supervision improves AP without a reliable Macro-F1 gain. Operational clinical targets differ across centers; their causal contribution is unproven.

This is an exploratory mechanism audit, not an improved model or zero-shot deployment result. All conclusions use the repeatedly studied development cohort. No outer TEST evaluation, retraining, checkpoint modification, label change or new raw EEG extraction occurred.

## Frozen A0 and oracle capacity

80 original patients, 7,635 canonical channels, 88 original features; 65 development patient-fold appearances/47 unique IDs. Fresh replay passed all five checkpoint, bank, split, feature-order, preprocessing, epoch and threshold checks. Per-channel probability drift stayed below 2e-7; patient metrics matched the previous A0 within 1e-12. Selected fold thresholds remain 0.465/0.275/0.340/0.355/0.445; epochs 3/1/2/2/2.

| Full-channel decision rule | Macro-F1 | EZ-F1 | EZ-AP | EZ-AUROC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Original frozen A0 | 0.638080 | 0.431198 | 0.518235 | 0.710667 | 0.465904 | 0.864968 |
| Patient oracle, historical .005 grid | 0.704175 | 0.527702 | 0.518235 | 0.710667 | 0.538686 | 0.896108 |
| Patient oracle, all achievable thresholds | **0.706285** | 0.527385 | 0.518235 | 0.710667 | 0.532185 | 0.905117 |
| True-EZ-count TopK, nondeployable | 0.652410 | 0.459325 | 0.518235 | 0.710667 | 0.459325 | 0.845494 |

Exhaustive oracle headroom is **+0.068205**, unique-patient-cluster 95% CI **[0.051667, 0.085513]**, exceeding the +0.030 gate. Both threshold oracles and true-count TopK use patient labels and are not deployment baselines. TopK's +0.014330 CI [-0.005811, 0.033791] is much smaller than threshold oracle headroom; knowing the true EZ count is not equivalent to choosing the F1-optimal operating point.

The grid loses 0.002110 mean Macro-F1 relative to exact boundaries (maximum cell loss 0.023852). Exact score ties move together; no true-count TopK tie-cut occurred in these scores, but synthetic tie-cut tests cover that distinct control. Every cell has a continuous optimal numeric threshold interval; none has multiple distinct optimal decision sets or disconnected optimal regions. Mean distance from the original threshold to the optimal decision-equivalent region is 0.115697. Individual intervals/thresholds remain private.

Confusion terms are averaged within patients, not pooled channel metrics: mean TP 8.108→10.277, FP 11.769→5.938, TN 66.215→72.046, FN 10.415→8.246. Both FP and FN reduction contribute on average; some cells trade errors differently. The oracle preserves AP/AUROC/MRR/Top1 exactly, so this is operating-point capacity, not improved ranking. It does not uniquely or causally decompose all classification error into calibration versus ranking.

| Descriptive headroom group | Cells | Unique IDs* | Mean headroom | A0 F1 | Oracle F1 | A0 AP |
|---|---:|---:|---:|---:|---:|---:|
| Little, <0.01 | 12 | 11 | 0.004993 | 0.751299 | 0.756292 | 0.638304 |
| Moderate, 0.01–<0.05 | 17 | 17 | 0.028758 | 0.641057 | 0.669815 | 0.440996 |
| Large, ≥0.05 | 36 | 27 | 0.107904 | 0.598934 | 0.706838 | 0.514686 |

*IDs can occur in different groups across folds; group counts are not independent cohorts. Median headroom 0.055826; 10th/90th percentiles 0.006233/0.160611. Eighteen cells have large headroom and AP≥0.5. Twenty of 65 cells remain below oracle F1 0.65. These are retrospective descriptions, not target-patient selection rules. Headroom associations with AP/AUROC are weak (Spearman 0.0293/0.0245), and with MRR/Top1 0.1656/0.1739; they do not identify an unlabeled calibration policy.

| Center | Development cells / IDs | A0 Macro-F1 | Oracle Macro-F1 | Gain |
|---|---:|---:|---:|---:|
| HUP | 28 / 21 | 0.669631 | 0.750485 | +0.080855 |
| LZU | 14 / 11 | 0.590312 | 0.639903 | +0.049591 |
| Multicenter | 11 / 9 | 0.740741 | 0.784341 | +0.043601 |
| Pediatric | 12 / 6 | 0.526085 | 0.609045 | +0.082960 |

Headroom is positive in all centers and all five folds, but baseline ranking/ceilings differ substantially. Center-specific paired oracle CIs are included in PATIENT_CLUSTER_BOOTSTRAP.csv. No center-specific adaptation or threshold was fitted.

## B8 bias versus direction identifiability

Support is exactly eight label-blind uncertainty-selected channels nearest the frozen fold threshold, stable canonical tie resolution. The same remaining query channels are used by P0/P1/P2 and the label-permutation control. Labels are retrospective simulated support, not claimed to be clinically obtainable. PCA4 is FIT-only and fixed; all backbones stay frozen. Original numerical thresholds are used unchanged.

The existing PR-UAS teachers are legal FIT-patient OOF: query patients were excluded from both training and internal checkpoint selection. Raw feature identities, teacher TRAIN-only preprocessors and all twenty checkpoint/output hashes matched. Their scores are frozen ten-pass MC means, not deterministic full-FIT A0 scores. Per-fold P1/P2 lambdas, selected only on that fold's FIT-OOF B8 query episodes and sealed before new VAL probes, are (1,1), (1,0.1), (1,10), (1,10), (1,10). There was no new search after seeing results.

| B8 uncertainty, matched query | Macro-F1 | EZ-F1 | EZ-AP | EZ-AUROC | MRR | Top1 | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 frozen A0 | 0.633955 | 0.411322 | 0.522299 | 0.713784 | 0.689879 | 0.584615 | 0.442064 | 0.882163 |
| P1 intercept | 0.635694 | 0.402696 | 0.522299 | 0.713784 | 0.689879 | 0.584615 | 0.400687 | 0.909207 |
| P2 intercept + PCA4 direction | 0.628533 | 0.400183 | 0.502185 | 0.706767 | 0.616460 | 0.507692 | 0.416111 | 0.881973 |
| P2 support-label permutation | 0.607165 | 0.374001 | 0.477514 | 0.675629 | 0.640396 | 0.553846 | 0.393072 | 0.860120 |

| Primary paired contrast | Macro-F1 delta [95% CI] | AP delta | Positive F1 folds | Gate |
|---|---:|---:|---:|---|
| P1−P0 | +0.001739 [-0.006096, 0.009447] | 0 exactly | 2/5 | Fail |
| P2−P0 | −0.005423 [-0.030296, 0.022254] | −0.020113 | 3/5 | No reliable gain |
| P2−P1 | −0.007161 [-0.033703, 0.021030] | −0.020113 | 3/5 | Fail |
| True P2−permuted P2 | +0.021368 [0.001053, 0.046578] | +0.024671 | 5/5 | Negative control positive; not a baseline improvement |

P1 fails magnitude, interval, fold and EZ-F1 nondecline criteria: EZ-F1 delta −0.008626. The small F1 point improvement is associated with greater specificity but lower sensitivity. P2 fails direction-identifiability criteria. True labels outperform permutation on Macro-F1, but this does not rescue P2's deterioration relative to the unadapted query baseline; AP superiority over permutation has a CI crossing zero. The direction moves rank positions frequently (mean changed-position fraction 0.8706), not necessarily usefully.

P1 cannot change ranking because adding one constant preserves logit order; numerical ranking metric identity passed in every episode. P2 changes ranking, so its AP losses are real under the tested query protocol rather than a threshold artifact.

## Random B8 and higher-label-budget capacity

All repetitions are label-blind, unstratified and averaged **within cell** before clustering. Query sets differ by policy/budget: compare methods only within a row block, not absolute AP/F1 between budgets.

| Budget/policy | Method | Macro-F1 | EZ-F1 | EZ-AP | EZ-AUROC | MRR | Top1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Random B8, 20 repetitions | P0 | 0.637811 | 0.430725 | 0.521616 | 0.711753 | 0.685762 | 0.580769 |
| Random B8 | P1 | 0.625411 | 0.393907 | 0.521616 | 0.711753 | 0.685762 | 0.580769 |
| Random B8 | P2 | 0.628284 | 0.408699 | 0.517487 | 0.716259 | 0.669919 | 0.575385 |
| Half-channel support, 10 repetitions | P0 | 0.632710 | 0.422646 | 0.534580 | 0.709506 | 0.685742 | 0.569231 |
| Half-channel support | P1 | 0.621261 | 0.386425 | 0.534580 | 0.709506 | 0.685742 | 0.569231 |
| Half-channel support | P2 | 0.631747 | 0.408656 | 0.554116 | 0.731280 | 0.696676 | 0.583077 |

Random B8 P1−P0 F1 −0.012400, CI [-0.026296, 0.000859]; P2−P0 −0.009527, CI [-0.026893, 0.006070]. Neither supports adaptation.

Half-support P2−P1 F1 **+0.010487**, CI **[-0.008660, 0.030839]**, 4/5 positive folds; AP **+0.019536**, CI **[0.002013, 0.037768]**, 5/5 positive folds. True P2−permuted P2 F1 +0.060333, CI [0.035975, 0.087351]. This is evidence of label-dependent **ranking capacity**, but the primary F1 interval still includes zero. P2−P0 F1 is −0.000962, CI [-0.017145, 0.014860]; EZ-F1 −0.013991. Thus the prespecified higher-budget success gate also fails and `HIGH_LABEL_BUDGET_REQUIRED` is not justified as the terminal.

One-class support occurs in 15/65 uncertainty cells (23.08%), 27.0% of random-B8 episodes and 1.08% of half-support episodes. No classes were fabricated; all solves converged with zero fallback. Under uncertainty B8, one-class cells have mean full-patient EZ prevalence 0.12327 versus 0.24237 for two-class cells and substantially worse baseline ranking. These source/label-scarcity strata are descriptive and cannot be used to select a new policy after inspecting labels. All 373 one-class permutation episodes are explicitly degenerate. Chance identical permutations among mixed-class episodes are reported separately in integrity audits.

## FIT-OOF transport limitations

Membership, normalization fitting sets and score-generating stochasticity differ between legal teachers and final A0. In fold1, patient-mean OOF score SD is 0.20317 versus VAL 0.15553; predicted EZ fraction at the original threshold is 0.34350 versus 0.29271. In fold5, SD is 0.16290 versus 0.14201. Other folds differ less or in the opposite direction. FIT/VAL label prevalence also differs. These facts make regularization-transfer mismatch plausible, but cannot causally establish why the probe failed. No in-sample score substitution, post-hoc lambda adjustment, PCA change, query-label fitting or retuning was performed.

## Clinical target consistency

Actual A0 uses complete canonical **Task1 patient-index labels**, not Task2 `center_clinical_target()`. Current adapter code and actual cached metadata establish heterogeneous operational rules: HUP composite SOZ/resection/status cue; multicenter SOZ; LZU spreadsheet-number clinical EZ; pediatric workbook-backed target with sparse categories pooled. Cohort counts total 80/7,635 exactly.

Every exported label matches the patient index and cached cross-run EZ union. The 22 LZU index-vs-first-record differences are explained by the source builder's minimum-NEZ/union-EZ aggregation, not annotation errors. Independent HUP SOZ/resection fields are unavailable, so their disagreement cannot be inferred. Current code hashes are not execution-time attestations; historical command flags and biological target validity need source-document review.

A0 development source-group Macro-F1: HUP 0.669631, LZU 0.590312, multicenter 0.740741, pediatric 0.526085. Corresponding AP 0.555377/0.515798/0.631022/0.331026. These groups are confounded with center and composition; no causal label-noise inference or outcome-based relabeling is warranted. Component C is `HETEROGENEOUS_SOURCE_DEFINITIONS_CONFIRMED`, with the provenance limits above—not proof that heterogeneity caused model failure.

## Statistics, engineering and scientific decision

Every new paired comparison uses 10,000 seed42 unique-ID cluster draws retaining all appearances; support repeats are first averaged within patient-fold cells. Equal appearance means preserve the original A0 estimand. Metric tables report valid-cell denominators; bootstrap tables also report contributing unique IDs. Undefined AUROC is missing, never 0.5. All inference is exploratory after repeated cohort use.

23 synthetic tests and real FIT-only smoke gates passed; all five real A0 replays passed. All 2,015 completed support episodes were independently replayed with maximum metric drift **0.0** in isolated fold processes. Engineering failures and process-isolation remedies are documented in IMPLEMENTATION_AUDIT, not silently omitted. No new performance-driven model decision was made after the protocol freeze.

The evidence supports **retrospective operating-point headroom** and some **high-budget label-dependent ranking information**, not reliable few-shot decision improvement or zero-shot identifiability. It does not identify a single causal bottleneck, nor prove the entire representation insufficient: only these fixed 4D probes failed to recover the F1 oracle headroom.

**One prioritized next experiment:** an independently sourced, clinician-adjudicated **target-definition harmonization/provenance validation audit** with frozen A0 scores and prespecified matched-query analyses. First verify whether center-specific proxies represent comparable clinical targets, including cross-seizure union policy, before investing in another adaptation model. This is a recommendation only; no new experiment was started and no improvement magnitude is promised.

## Explicit answers to the 16 requested questions

1. **Exact A0 reproduction?** Yes: all five hashes/inputs/states/epochs/thresholds pass; score tolerance <2e-7, metric tolerance 1e-12.
2. **Actual matched oracle ceiling?** 0.7062849901 exhaustive; 0.7041745181 on the .005 grid.
3. **Headroom?** +0.0682052073, CI [0.0516668828, 0.0855126197].
4. **Threshold-limited patient profiles?** 36 cells have ≥0.05 headroom, including 18 with AP≥0.5; identities remain private and profiles are nondeployable.
5. **Difficult even at oracle?** 20/65 cells have oracle F1<0.65; LZU/pediatric contribute 14 of these cells.
6. **Consistency across centers?** Positive oracle gain everywhere, magnitude 0.04360–0.08296; distinct baseline ranking/ceilings remain.
7. **B8 bias gain?** +0.001739 with CI crossing zero, 2/5 positive folds and declining EZ-F1: gate fails.
8. **B8 direction beats bias?** No: F1 −0.007161, AP −0.020113, intervals cross zero.
9. **Beats label permutation?** True P2 F1 beats permutation by +0.021368 with positive CI, but does not beat P0 or P1. This is not useful net adaptation.
10. **High-budget capacity?** AP gain +0.019536 is positive with CI; decision F1 versus P0 is −0.000962, so full capacity-success gate fails.
11. **Scarcity/prevalence?** One-class B8 support is common and corresponds descriptively to lower EZ prevalence/poorer ranking; not a causal or prospective acquisition conclusion.
12. **OOF-to-VAL transfer?** Legal provenance passes, but MC averaging, training size, preprocessing and prevalence shifts limit transfer interpretation. They do not authorize in-sample substitution.
13. **Actual target definitions?** Task1 canonical indices built from center-specific clinical proxies and union across seizures; not alternative Task2 mapping.
14. **Source differences confirmed?** Operational heterogeneity confirmed; perfect historical code-version/clinical truth provenance remains unavailable.
15. **Zero-shot adaptation established?** No. Oracle and support-label probes cannot establish it; existing unsuccessful label-blind threshold/cardinality/geometry studies remain relevant negative evidence.
16. **Single next experiment?** Independent clinician-adjudicated target-definition/provenance audit with frozen scores; not another automatic zero-shot head.
