# FullPool privileged Student completion — seed 42

## Decision

**Terminal: `STRONG_PRIVILEGED_SIGNAL_NOT_ZEROSHOT_DISTILLABLE`.** Under this frozen exact-A1 Student protocol, no privileged-KD variant materially beat its matched continued-hard-label control. The strongest Student, D1 PairKD, reached AP **0.561999**, versus D0+ **0.561943** and historical original A1 **0.576743**. Its paired gain over D0+ was only **+0.000056 AP** (47-patient 95% bootstrap CI **[-0.000005, +0.000176]**, positive in **1/5** folds). This is not evidence of transfer.

This is a development/VLOO result, not an external or previously untouched outer test. The v1 terminal `PRIVILEGED_TEACHER_SIGNAL_GATE_FAILED` remains unchanged.

## Provenance and study sequence

The v2 lock (SHA-256 `acfcbd600fbbd666ce5ccc8d480b50b6e131a7f7358da697860bb9d6db81fcb2`) was committed before new Student target outcomes. Frozen v1 Teacher targets were numerically replayed over 869 OOF patient-context files and 4,345 cross-fit heads with maximum score error **0**. However, v1 did not save a historical per-file SHA manifest: byte-for-byte historical hash identity is **not verifiable**. A fresh input hash manifest was locked before v2 training.

The new prospective Teacher AP eligibility passed: FIT OOF Teacher minus A1 AP **+0.077982**, patient-ID CI **[+0.048096, +0.108110]**, **5/5** folds positive, with own scored channel labels excluded from the Teacher fit. This does **not** reverse the v1 stop: v1's MRR and Top1 deltas were **−0.007661** and **−0.039100**, respectively, failing its original early-rank point-estimate gate.

Thirty models (six variants × five folds) were FIT-selected before target labels were used. D0+ was trained first. FIT-only D1/D2 means exceeded D0+ slightly, triggering predeclared D4. All target scores/checkpoints were then hash-frozen before the single 65-cell × 20-query B=0 evaluation. No target Teacher, support labels, adaptation, calibration or external dataset was used.

## Target results

| Model | EZ-AP | ΔAP vs D0+ | Paired 95% CI | Positive folds |
| --- | ---: | ---: | ---: | ---: |
| D0 original A1 | 0.576743 | — | — | — |
| D0+ continued hard-label | 0.561943 | 0 | — | — |
| D1 PairKD | **0.561999** | +0.000056 | [−0.000005, +0.000176] | 1/5 |
| D2 ListKD | 0.560675 | −0.001268 | [−0.008464, +0.005160] | 1/5 |
| D3a correction, all FIT | 0.561938 | −0.000005 | [−0.000016, +0.000002] | 0/5 |
| D3b correction, Teacher-weighted | 0.561951 | +0.000008 | [0, +0.000022] | 1/5 |
| D4 pair+list | 0.560545 | −0.001398 | [−0.005815, +0.001154] | 0/5 |

D0+ was **−0.014800 AP** relative to historical D0 (CI **[−0.031798, +0.004617]**, 1/5 folds positive). This cannot be attributed solely to continued training: the historical D0 is a target-specific VLOO-selected A1 reference, whereas all new Students start from a fold-shared epoch-30 A1 state. KD attribution therefore uses the matched D0+ control.

The best Student D1 had **MRR 0.735908**, **Top1 0.644686**, **macro-F1 0.624504**, **EZ-F1 0.414496**, **BA 0.670941**, **AUROC 0.732622**, **NDCG 0.753483**, and predicted EZ fraction **0.227243**. Its delta versus original A1 was **−0.014744 AP** (paired CI **[−0.031719, +0.004664]**). By the predeclared FullPool/A1 development references, `teacher_added_headroom_transfer_percent` was **−12.74%**; this is a negative headroom difference, not a measure of Teacher ability learned.

No Student reached AP **0.590**, current B8 **0.599632**, or best B8 **0.605291**. The FullPool AP **0.692464** is a privileged, nondeployable development reference only.

## Mechanism and patient strata

On FIT-meta, the Teacher–A1 pair agreement was **0.7800**. D0+ Teacher agreement was **0.7885**; D1 was **0.7886**, and D3a/D3b remained **0.7885**. On Teacher–A1 disagreement pairs, D0+, D1 and both D3 variants all had **0.1805** agreement. In the highest correction-magnitude stratum, D1 and D3 improved **0** over D0+; D2 gained only **0.00243** agreement there while reducing low-correction agreement and target AP. The claimed high-correction preference transfer was therefore **not demonstrated**.

Target strata were fixed solely from original A1 AP: 16 poor, 16 medium and 15 strong unique patients. D1's AP changes versus original A1 were approximately **+0.0120**, **−0.0280**, **−0.0323** across these strata. But its deltas versus matched D0+ were essentially zero: **+0.000009**, **0**, **+0.000155**. The apparent poor-stratum improvement is not attributable to privileged KD.

## Direct answers to the 17 requested questions

1. Teacher target numerical replay: **yes, exactly** (error 0); historical per-file hash-exact replay: **not verifiable**, because v1 lacked a hash manifest.
2. New Teacher AP eligibility: **yes**; v1's original terminal remains failed.
3. D0+ versus historical A1: **no improvement** (−0.014800 AP); this difference also includes a changed fold-shared versus target-specific checkpoint-selection arrangement.
4. D1 versus D0+: **no meaningful/significant improvement** (+0.000056; CI crosses 0).
5. D2 versus D0+: **no** (−0.001268).
6. D3a versus D0+: **no** (−0.000005).
7. D3b improves D3a by only ~0.000013 AP and exceeds D0+ by +0.000008 with CI lower 0; **no meaningful improvement**.
8. Highest-AP Student: **D1 PairKD**.
9. D1 AP/MRR/Top1/macro-F1/EZ-F1/BA: **0.561999 / 0.735908 / 0.644686 / 0.624504 / 0.414496 / 0.670941**.
10. D1 minus D0+: **+0.000056 AP**, CI **[−0.000005, +0.000176]**.
11. D1 minus historical A1: **−0.014744 AP**, CI **[−0.031719, +0.004664]**.
12. Headroom transfer η: **−12.74%**.
13. AP ≥0.590: **no**.
14. AP ≥current B8 0.599632: **no**.
15. AP ≥best B8 0.605291: **no**.
16. Preferential learning of the largest Teacher–A1 disagreement pairs: **no** for D1/D3; only a tiny FIT-meta D2 high-correction agreement increase, without target gain.
17. The observed negative result is **failure of transfer beyond matched D0+** under this frozen Student/protocol. It is not evidence that all future zero-shot architectures must fail. D0+ versus historical D0 does not isolate a pure continued-training effect.

The strongest supported claim is narrow: a strong patient-internal, cross-fitted privileged Teacher AP signal existed on FIT, but this predeclared exact-A1 Student distillation did not yield a useful B=0 cross-patient gain. Stop this R4 Teacher-distillation route under the current design; no post-outcome model or grid change was made.
