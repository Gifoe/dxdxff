# D2 Source-Rank Compression — completed seed42 development experiment

| Model | Params | Macro-F1 | EZ-F1 | NEZ-F1 | BA | EZ-AP | EZ-AUROC | MRR | Top1 | Sensitivity | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R4 | 9221 | 0.647926 | 0.443199 | 0.852653 | 0.669549 | 0.519936 | 0.718221 | 0.719610 | 0.615385 | 0.463999 | 0.875099 | 0.790571 |
| R0 | 8821 | 0.636982 | 0.424982 | 0.848981 | 0.663367 | 0.517063 | 0.707296 | 0.702515 | 0.600000 | 0.450575 | 0.876159 | 0.776475 |
| R1 | 8921 | 0.643679 | 0.435388 | 0.851971 | 0.661554 | 0.518756 | 0.711650 | 0.720252 | 0.630769 | 0.448790 | 0.874317 | 0.786271 |
| R2 | 9021 | 0.644188 | 0.439254 | 0.849122 | 0.666625 | 0.514952 | 0.717052 | 0.729208 | 0.630769 | 0.462777 | 0.870472 | 0.783328 |

Terminal: `RANK4_REMAINS_BEST`. Registered primary R2 gate: False. Qualifying exploratory secondary candidates: [].

Same65 patient-fold appearances,47 unique IDs,6273 channel appearances; original80/7635/88D/five folds. R4 is frozen original D2, not newly trained. No outer TEST. This is repeatedly used development, not independent testing or unseen-center transfer.

## Paired unique-ID cluster bootstrap

| Contrast | Macro-F1 delta | 95% percentile CI | Positive folds | Positive unique IDs |
|---|---:|---|---:|---:|
| R0-R4 | -0.010944 | [-0.024746, +0.001952] | 0/5 | 16/47 |
| R1-R4 | -0.004247 | [-0.012860, +0.004565] | 0/5 | 17/47 |
| R2-R4 | -0.003738 | [-0.009900, +0.002256] | 1/5 | 13/47 |
| R2-R1 | +0.000509 | [-0.007872, +0.008349] | 3/5 | 14/47 |
| R1-R0 | +0.006698 | [-0.005274, +0.020643] | 4/5 | 14/47 |
| R2-R0 | +0.007207 | [-0.003859, +0.018602] | 5/5 | 21/47 |

10,000 paired seed42 unique-ID draws retain repeated appearances. Fixed checkpoints/thresholds; no reoptimization. All11 metric intervals/all6 contrasts are in PAIRED_BOOTSTRAP.csv. Nominal intervals do not adjust prior searches, selection optimism or multiplicity.

## Source dependence

| Source | R4 Macro-F1 | R0-R4 | R1-R4 | R2-R4 |
|---|---:|---:|---:|---:|
| hup | 0.680818 | -0.015361 | -0.005357 | -0.002418 |
| lzu | 0.615956 | -0.018435 | +0.003046 | -0.008999 |
| multicenter | 0.724433 | +0.010759 | -0.006722 | +0.005523 |
| pediatric | 0.538345 | -0.011796 | -0.007896 | -0.009166 |

## Explicit answers to24 required questions

1. R4 reproduced: yes; strict state mapping and bitwise generalized-versus-legacy CPU forward parity; fresh archived logit tolerance<2e-6, then exact frozen score/metric reuse. No R4 retraining.
2. Original splits/labels/88D preprocessing retained bitwise. All101 archived artifacts are hash-verified. NPZ all-cohort y is never loaded; only sealed FIT/VAL labels are used.
3. R0-R4 Macro-F1 -0.010944; full secondary criterion=False.
4. R1-R4 Macro-F1 -0.004247; full secondary criterion=False.
5. R2-R4 Macro-F1 -0.003738; registered full primary gate=False.
6. Descriptively highest Macro-F1 is R4=0.647926. This post-outcome descriptive maximum is not an independently selected/confirmed rank.
7. Monotonic improvement while reducing rank4->2->1->0: False. Point ordering alone does not establish a rank mechanism.
8. R2 positive Macro-F1 folds versus R4: 1/5.
9. EZ-AP deltas versus R4: R0 -0.002873, R1 -0.001180, R2 -0.004984.
10. EZ-AUROC deltas versus R4: R0 -0.010925, R1 -0.006571, R2 -0.001169.
11. EZ-F1 deltas versus R4: R0 -0.018217, R1 -0.007812, R2 -0.003945.
12. Sensitivity/specificity deltas: R0 -0.013424/+0.001060, R1 -0.015209/-0.000782, R2 -0.001222/-0.004626.
13. Recovered/spoiled errors over repeated channel appearances: R0 133/234; TP gained/lost 35/96, R1 112/134; TP gained/lost 22/41, R2 69/106; TP gained/lost 17/31. These are not independent channel observations.
14. Bias-versus-direction is isolated only under the same fixed trained scorer; R0 changes the jointly trained shared backbone too. Inference Macro-F1 changes (intervention minus full): R1/DIRECTION_OFF -0.006402, R1/BIAS_OFF -0.002419, R2/DIRECTION_OFF -0.009201, R2/BIAS_OFF -0.006066. This is not proof that bias causally accounts for cross-model gains.
15. Same-model source-constant bias ranking invariance is verified mathematically and by fixed-ranking intervention. Direction-off patient-equal pair reversals: R1 0.039732, R2 0.040996. Directional terms can alter ranking; shared-backbone retraining can also alter cross-model rankings.
16. Effective ranks by folds1-5: R4: [4, 4, 4, 4, 4]; R0: [0, 0, 0, 0, 0]; R1: [1, 1, 1, 1, 1]; R2: [2, 2, 2, 2, 2]. SVD analyzes U@V, not separately unidentified factors.
17. SOURCE_DIRECTION_AUDIT.csv reports all source vector norms and defined pairwise cosines. Rank1 vectors are collinear with possibly opposing signs. Zero-vector cosine is missing/not defined, never assigned a fabricated zero similarity.
18. Low-count source findings are in the source table above and GENERALIZATION_DIAGNOSTICS.csv. Original source patients: HUP36/LZU21/Multicenter15/Pediatric8; development unique counts21/11/9/6. No small-source or unseen-center generalization is inferred from architecture alone.
19. Negative source deltas are reported without exclusion. Severe collapse is prelocked as>.050 absolute source Macro-F1 decline and blocks supported status, without posthoc explanation waiver.
20. Paired intervals above quantify development differences; if zero is included, gains are not clearly distinguished from patient-level variation. Even positive nominal intervals are not independent confirmation or a causal isolation of overfit.
21. Variants meeting +.010 magnitude alone: []. Full-gate candidates: [].
22. Any compressed point Macro-F1>=.700: False. Full-gate target: False.
23. Stop the source-rank compression line under this fixed family. No rank3/lambda/seed/threshold rescue is authorized.
24. Source heterogeneity and unreliable patient-level decisions on repeatedly reused development remain unresolved. This experiment cannot establish biological source mechanisms, clinical label truth, prospective transfer or causal overfitting. A rank decrease alone is not sufficient evidence of better generalization.

## Mechanism and engineering caveats

Generalization diagnostics use in-sample FIT predictions at the selected checkpoint and selected VAL threshold; FIT-minus-VAL gaps are optimistic descriptions, not OOF or causal overfit estimates. Interventions are inference-only and never retrained. Source-bias scalars are acquisition-category corrections, not disease prevalence estimates.
R0 removes400/9221 parameters (4.34%), R1 removes300 (3.25%), R2 removes200 (2.17%). Most capacity remains in the shared backbone; compression is not a substantial compute reduction. Original mean-square shrinkage normalizes different-rank parameter spaces; lambda was not retuned.
Primary metrics use original float32 sigmoid/ranking semantics. Bias-off ranking reuses full frozen ordering after verifying float64 source-constant translation, avoiding artificial rank changes from finite precision; direction-off ranks by the shared base. Classification always uses fixed selected thresholds.
All34 engineering checks and15 FIT smokes pass before training. Exactly15 new runs; no R4 retraining or outer TEST. Private IDs, labels, scores, matrices, checkpoints, optimizer/RNG states and runtime logs stay on the original server.

## Execution verification

Protocol SHA256 `d552acea82b037ec42dceda277ef00babb06dba901346896a803cc17094b6bbe`. Exactly15 selected sealed cells; 169 completed epochs; epoch-body time 24.90s, excluding preparation/tests/checkpoint/report I/O.
