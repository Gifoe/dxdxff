# Source-conditioned A0 and patient posterior decoder — completed seed42 audit

**Terminal: `POSTERIOR_IDENTIFIABILITY_FAILED`.** The frozen Gaussian posterior assumption fails its prespecified global FIT-OOF class-ordering requirement. No label flip, forced mean separation, alternative density, hyperparameter search or threshold rescue was performed. Five formal D2 scorers and all20 D2 OOF teachers completed;20 original A0 teachers were freshly replayed. D0/D2 full development metrics are available. D1/D3 are **not estimable as complete65-cell arms**, not assigned zero or partial means.

## All full-arm metrics

| Metric | D0 frozen A0 | D1 posterior | D2 source A0 | D3 source+posterior |
|---|---:|---|---:|---|
| macro_f1 | 0.638080 | not_estimable | 0.647926 | not_estimable |
| ez_f1 | 0.431198 | not_estimable | 0.443199 | not_estimable |
| nez_f1 | 0.844961 | not_estimable | 0.852653 | not_estimable |
| balanced_accuracy | 0.665436 | not_estimable | 0.669549 | not_estimable |
| ez_auprc | 0.518235 | not_estimable | 0.519936 | not_estimable |
| ez_auroc | 0.710667 | not_estimable | 0.718221 | not_estimable |
| ez_mrr | 0.686550 | not_estimable | 0.719610 | not_estimable |
| top1_is_ez | 0.584615 | not_estimable | 0.615385 | not_estimable |
| sensitivity | 0.465904 | not_estimable | 0.463999 | not_estimable |
| specificity | 0.864968 | not_estimable | 0.875099 | not_estimable |
| accuracy | 0.773174 | not_estimable | 0.790571 | not_estimable |


Same65 patient-fold appearances,47 unique IDs,6,273 channel appearances; original80/7,635/88D cohort/frozen5fold. A0 replay within2e-7 and all original metrics within1e-12 passed. Exact frozen A0 probabilities are reused only after that gate. D2 checkpoints/probabilities were independently replayed; D3 was bound to those exact checkpoints, never separately trained.

| Metric | D2-D0 | 95% cluster CI | Positive folds |
|---|---:|---|---:|
| macro_f1 | +0.009846 | [-0.004723,+0.026640] | 3/5 |
| ez_f1 | +0.012001 | [-0.008008,+0.037035] | 3/5 |
| nez_f1 | +0.007691 | [-0.007192,+0.021219] | 3/5 |
| balanced_accuracy | +0.004113 | [-0.010760,+0.020807] | 3/5 |
| ez_auprc | +0.001701 | [-0.013134,+0.018090] | 3/5 |
| ez_auroc | +0.007554 | [-0.009004,+0.025187] | 5/5 |
| ez_mrr | +0.033060 | [-0.008210,+0.081282] | 4/5 |
| top1_is_ez | +0.030769 | [-0.029412,+0.096774] | 2/5 |
| sensitivity | -0.001905 | [-0.032115,+0.031340] | 2/5 |
| specificity | +0.010130 | [-0.014087,+0.031144] | 3/5 |
| accuracy | +0.017397 | [+0.000260,+0.034507] | 4/5 |


10,000 seed42 paired unique-ID cluster draws preserve repeated appearances, fixed checkpoints and thresholds. All required posterior contrasts are explicitly not_estimable, because their complete arms failed FIT density validity. Intervals do not correct selection optimism, repeated development use or multiplicity. This is exploratory development, not outer test confirmation.

## The decisive FIT-only falsification

| Fold | Posterior family | Global EZ mean | Global NEZ mean | NEZ−EZ | Valid |
|---|---|---:|---:|---:|---|
| 1 | D1 | -0.665380 | 0.274675 | +0.940055 | True |
| 1 | D3 | -0.694915 | 0.231608 | +0.926523 | True |
| 2 | D1 | -0.656856 | 0.542873 | +1.199729 | True |
| 2 | D3 | -0.690676 | 4.291824 | +4.982501 | True |
| 3 | D1 | -1.086508 | -182.467953 | -181.381445 | False |
| 3 | D3 | -1.013217 | -177.340159 | -176.326943 | False |
| 4 | D1 | 43.038604 | 14.335927 | -28.702676 | False |
| 4 | D3 | 44.048190 | 14.669166 | -29.379024 | False |
| 5 | D1 | 53.003906 | 6.376425 | -46.627482 | False |
| 5 | D3 | 52.808544 | 6.398845 | -46.409699 | False |


D1 blocked folds: [3, 4, 5]; D3 blocked folds: [3, 4, 5]. These means use the frozen patient median/IQR normalized logits, equal patient weight within each observed class, and legal deterministic teacher predictions. They are not electrode-weighted estimates. Positive NEZ-minus-EZ is required. Common variance gives monotone EZ posterior only when this orientation holds. A nonpositive global gap prevents the assumed model from supplying a legal posterior: **source fallback cannot repair an invalid global density**.

Insufficient data and ordering are distinguished in POSTERIOR_IDENTIFIABILITY_AUDIT.csv; low-IQR fractions are also reported. Fixed-prior/MAP calibration diagnostics use only valid FIT densities and are explicitly optimistic/incomplete, not substitute validation results. Initial formal posterior inference completed only folds1–2 before the fold3 gate; no partial aggregate performance is used for selection or to claim D1/D3 success. Remaining folds' FIT gates were then audited separately. No invalid density generated target predictions.

The read-only NORMALIZATION_TAIL_AUDIT.csv identifies1–2 low-IQR FIT patients per failed fold (about1.9–3.9%). Normalized absolute logits reach 58185.5; fold3 D1 NEZ global mean -182.467953 receives -182.394158 from the low-IQR group, versus -0.073795 from all other patients' contribution. Folds4–5 similarly show EZ mean domination by low-IQR tails. Thus floor1e-5 limits division but does not bound tail influence on Gaussian first/second moments. This directly explains the computed pooled-order reversal in this implementation; it does not establish a biological cause. No patient was excluded, score clipped, density refitted with a different rule or validation result rescued.

Patient centering/IQR normalization also removes location/scale information and can erase class-mixture information. Heterogeneous operational targets and OOF-to-full-FIT scorer transport remain possible additional limitations, not proven causes or evidence that all score posterior methods are impossible. The unchanged assumption was falsified here; it was not replaced after seeing outcomes.

## Source-conditioned scorer and errors

| Center | D0 Macro-F1 | D2 Macro-F1 | Delta |
|---|---:|---:|---:|
| hup | 0.669631 | 0.680818 | +0.011187 |
| lzu | 0.590312 | 0.615956 | +0.025645 |
| multicenter | 0.740741 | 0.724433 | -0.016308 |
| pediatric | 0.526085 | 0.538345 | +0.012260 |


D2 changes within-patient rankings and shares a jointly trained8817-parameter backbone plus404 source parameters. Exact zero initial correction and original shared initialization were verified; unknown source uses shared-head fallback. Source correction magnitudes, hidden low-rank energy ratios and effective ranks are in SOURCE_CORRECTION_AUDIT.csv. Source identity is an operational acquisition category, not biology, patient embedding or clinical-label predictor. No unseen-center transfer was evaluated.

D2 gains 96 / loses 57 EZ true positives and adds 110 / removes 208 false positives. It corrects 304 original errors and spoils 167 correct decisions over repeated channel appearances. Those pooled counts do not replace patient-equal F1. D2 historical oracle-headroom recovery is 14.4363%; D1/D3 recovery is undefined. The historical label-using0.7062849901 oracle never enters inference.

## Explicit answers to19 questions

1. A0 exactly reproduced? Yes, all5 original checkpoint/preprocessor/identity/epoch/tau gates; metrics within1e-12, probabilities within2e-7.
2. Original cohort preserved? Yes,80 patients/7,635 canonical pairs/88 original features and frozen5fold. No clinical relabeling, alternative80-patient cohort or outer evaluation.
3. D1 outperform D0? Cannot establish: complete D1 is blocked on FIT global ordering. Do not average only successful folds.
4. D2 outperform D0? Macro-F1 delta +0.009846, CI [-0.004723,+0.026640]; fixed source continuation gate=False.
5. D3 outperform all others? Not estimable. No fabricated full D3 result or complementarity claim.
6. Source conditioning improves ranking? AP delta +0.001701, AUROC delta +0.007554, with full paired intervals above.
7. Posterior improves patient Macro-F1? Not established: FIT validity blocks complete evaluation.
8. MAP improves fixed source prior? Only valid FIT-fitted diagnostics in FIXED_PRIOR_COMPARISON.csv are available; optimistic and incomplete, no full VAL benefit is claimed.
9. Conditional distributions distinguishable/stable? No under the required global monotone ordering in the failed folds. Adequate observations alone do not make this normalized Gaussian family valid.
10. D1 preserves A0 ordering? Synthetic tests and completed valid-fold assertions pass. Invalid global densities were not applied or flipped.
11. D3 improves EZ detection or specificity? Not estimable. D2 sensitivity/spec deltas -0.001905/+0.010130; these are not D3 metrics.
12. D3 improves all centers? Not estimable. Source-only D2 center effects are reported separately above.
13. A0 oracle headroom recovered? D2 14.4363%; D1/D3 undefined. No biological oracle claim or individual unstable recovery ratio.
14. D3 reaches.658? Not evaluable, not a numeric failure substituted for missing predictions.
15. D3 reaches.700? Not evaluable; no independent performance claim.
16. Main failure mechanism? Prespecified class-ordering failure in normalized FIT-OOF score moments. The separate arithmetic audit shows near-zero-IQR patients dominating first/second moments; the frozen epsilon floor does not prevent extreme tails. No solver, leakage or checkpoint failure caused this gate. Biological/source causes and OOF transport are not causally established.
17. Further posterior modeling justified? This particular Gaussian/common-variance/median-IQR mechanism is unsupported. Do not enlarge or tune it automatically.
18. Further source scoring justified? Fixed standalone source gate=False; point gains and CIs above qualify its strength. Combined posterior success is not established.
19. Single next direction? A separately locked FIT-only score-collapse/normalization-identifiability audit on frozen A0/D2 outputs, before proposing any replacement posterior. First determine why within-patient IQR collapses while a few score tails remain. Do not tune this run or start another scoring model. This is a recommendation only; no new experiment was initiated.

## Engineering and integrity

32 named checks and real FIT smoke passed before real training, including original optimizer/selection parity,9221 parameters, source/shared gradients, exact zero initialization, legalOOF exclusions, decoder all-k/brute parity, constant/empty/unknown safety and deterministic interrupted resume. Source protocol has not changed.

Windows startup engineering deviations: initial hidden launcher produced no work; foreground startup revealed a null child ExitCode reporting problem after completed A0 replay. The PowerShell wrapper was repaired without changing Python/science, retaining all old logs. During fold3 teacher startup, Windows Application Error1000 confirmed python312.dll0xc0000005 before any teacher epoch. Exact-step resume completed remaining teachers, discarding no completed epoch. Later failures were the intended scientific ordering gate, not restarted/tuned away. Private logs/checkpoints remain preserved.

All-cohort NPZ clinical `y` member was never materialized in this run; development labels come from SHA-sealed original FIT/VAL banks. Actual OOF teacher checkpoints and final selected models are hash-verified. The final reporting-only script audits each failed density separately and independently replays completed scorers, without training, rescoring to improve outcomes, changing checkpoint/threshold, or adding a fifth model. Public files omit patient IDs, channel records, individual proportions, predictions, checkpoint tensors, caches and runtime logs.
