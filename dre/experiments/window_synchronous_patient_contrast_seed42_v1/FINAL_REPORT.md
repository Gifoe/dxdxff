# Window-Synchronous Patient Contrast — Seed42

## Completed Outer summary (requested after the inner audit)

Patient-equal metrics,80 original Outer identities once each /7,635 channels. Each original fold uses a new fixed FIT-only TRAIN/threshold-selection split;15 final LR probes. Same182D C0/C1/C2 features and classifier, no original Development labels for model/threshold selection. All15 models/thresholds froze before Outer inference, all predictions before clinical y access.

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | NEZ-F1 | MRR | Top1-EZ |
|---|---:|---:|---:|---:|---:|---:|
| C0 Own temporal |0.672639|0.480328|0.507376|0.294877|0.719875|0.660688|0.537500|
| C1 True synchronous context |0.711834|0.507647|0.574056|0.345849|0.802264|0.685648|0.587500|
| C2 Stage-local shifted context |0.698389|0.506131|0.557412|0.329349|0.785474|0.726676|0.650000|

| Model | Balanced accuracy | Sensitivity | Specificity | Accuracy | Predicted EZ fraction | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 |0.585768|0.416197|0.755340|0.674841|0.281771|699|1393|4499|1044|
| C1 |0.635603|0.422034|0.849173|0.730233|0.198274|609|869|5023|1134|
| C2 |0.619983|0.415266|0.824701|0.715811|0.219672|607|988|4904|1136|

Metric columns are equal-patient means; integer confusion terms are **total unique-channel counts**, not the mean confusion terms in OUTER_PATIENT_EQUAL_METRICS. Do not reconstruct patient-equal sensitivity from pooled counts.

### Outer paired contrasts

10,000 common patient-ID draws,seed42,fixed numeric thresholds. Percentile95% intervals, conditional/descriptive, unadjusted for historical experimentation and multiple tests.

| Contrast | AUROC delta [95% CI] | AUPRC delta [95% CI] | Macro-F1 delta [95% CI] | EZ-F1 delta [95% CI] |
|---|---|---|---|---|
| C1-C0 |+0.039195 [0.002149,0.077865]|+0.027319 [0.001728,0.054562]|+0.066680 [0.036374,0.097638]|+0.050972 [0.005068,0.095866]|
| C1-C2 |+0.013444 [-0.006018,0.034423]|+0.001517 [-0.013195,0.016250]|+0.016644 [0.000828,0.033772]|+0.016499 [-0.005760,0.039053]|
| C2-C0 |+0.025750 [-0.000103,0.051625]|+0.025803 [0.002509,0.049775]|+0.050036 [0.027039,0.073235]|+0.034473 [-0.007324,0.074259]|

C1 improves AUROC againstC0 for51/80 people, worsens29. AgainstC2 it improves39,worsens39,ties2. C1-C0 is positive in4/5 original folds. The critical C1-C2 AUROC interval crosses zero: **Outer replication gate fails**, even though both AUROC point margins pass.

Outer MRR/Top1 are not improved againstC2: differences -0.041028 [-0.087256,0.003169] and -0.062500 [-0.150000,0.012500]. Shifted C2 has the highest point estimates. These contrary retrieval results must not be described as synchronization-specific ranking improvement.

## Original strict FIT-only inner audit

All80 patients were included in label-free reference construction. Exact original FIT union is79 people with255 patient-fold appearances (one cohort person never FIT).60 probes,182D per arm. Neither this estimator nor Outer is the historical65-target Development or47-ID estimator.

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1-EZ |
|---|---:|---:|---:|---:|---:|---:|
| C0 |0.646775|0.460826|0.476677|0.278854|0.664036|0.545098|
| C1 |0.714885|0.518632|0.564222|0.354661|0.681501|0.560784|
| C2 |0.688805|0.491432|0.545395|0.325626|0.669148|0.545098|

The secondary unique-patient-equal AUROC estimates (first average repeats per person) areC0 0.657411,C1 0.719404,C2 0.693936. Other metrics are in UNIQUE_PATIENT_SUMMARY. Primary inner bootstrap clusters79 identities while retaining all appearances; never bootstrap255 independent people or channels.

Inner C1-C0 AUROC +0.068110 [0.043811,0.093032], AP+0.057806 [0.033766,0.084832], Macro-F1+0.087546 [0.064069,0.113832], EZ-F1+0.075807 [0.048132,0.104375]. Inner C1-C2 AUROC +0.026080 [0.010009,0.044364], AP+0.027200 [0.005379,0.054001], Macro-F1+0.018828 [0.003303,0.036653], EZ-F1+0.029035 [0.008117,0.051931]. All5 fold AUROC C1-C0 point differences are positive.

The achieved-disruption audit passed before labels:349,930/349,930 eligible sequences nonzero and materially shifted; absolute zero-lag correlation reduction0.403042; identical masks/multisets; cyclic autocorrelation drift7.77e-16;56/84 context summary coordinates changed. The28 mean-related coordinates are invariant by construction. No shuffle rescue occurred.

## Every continuation condition

The initial gate is left unchanged. Outer applies the same predictive margins as a separately locked exploratory replication check.

| Condition | Required | Inner actual / pass | Outer actual / pass |
|---|---|---|---|
| C1-C0 AUROC |>=0.010|0.068110 /PASS|0.039195 /PASS|
| C1-C2 AUROC |>=0.005|0.026080 /PASS|0.013444 /PASS|
| C1 AP >=C0 |nondecrease|+0.057806 /PASS|+0.027319 /PASS|
| C1 AP >=C2 |nondecrease|+0.027200 /PASS|+0.001517 /PASS|
| Macro-F1 vsC0 |loss<=0.005|+0.087546 /PASS|+0.066680 /PASS|
| EZ-F1 vsC0 |loss<=0.005|+0.075807 /PASS|+0.050972 /PASS|
| Specificity vsC0 |loss<=0.010|+0.088331 /PASS|+0.093833 /PASS|
| Positive original folds vsC0 |>=4|5 /PASS|4 /PASS|
| C1-C2 AUROC CI lower |strictly>0|0.010009 /PASS|-0.006018 /FAIL|
| Disruption |PASS|PASS|same frozen control PASS|
| Alignment/preprocessing/leakage |PASS|independent verification PASS|independent verification PASS|

Inner terminal: `SYNCHRONIZATION_MECHANISM_PROMISING_NOT_INDEPENDENTLY_CONFIRMED`.

**Combined conclusion after requested Outer: `PATIENT_CONTEXT_POSSIBLY_USEFUL_SYNCHRONIZATION_NOT_IDENTIFIED`.** Do not relabel the inner gate as failed or report overall synchronization confirmation. No TIDE or other deep model is designed/trained automatically.

## Answers to the twelve requested questions

1. **Does the cache support analysis?** Yes within its annotation-relative representation:80/256/7635 contract, variable window lengths, named-column mapping, genuine masks and reference identities pass. No independent EDF/onset/clock validation is added.
2. **Simultaneous coverage?** All80 have genuine simultaneous windows;100% valid baseline-qualified self windows have enough other channels in each stage/center. Observed record-channel incidences24995/25267;272 absent. Full early support across at least one seizure covers7635/7635. Conditional coverage is not full observation of every channel in every seizure.
3. **Own temporal change predictive?** The baseline+self C0 representation yields inner AUROC0.646775 and Outer0.672639. This is evidence for that composite representation, not an isolated causal test of change alone.
4. **True context improves AUROC?** C1-C0 is positive with descriptive positive intervals in both estimators. C1-C2 is positive only as an Outer point contrast; its Outer interval includes zero.
5. **AP/F1 improvements?** C1-C0 AP/Macro-F1/EZ-F1 intervals are positive both inner and Outer. Outer C1-C2 AP/EZ-F1 intervals cross zero; Macro-F1 interval is positive. Absolute F1 remains modest; not a new strong-A0 benchmark claim.
6. **Synchronization beats matched shift?** Inner yes under the predeclared gate. Outer does not satisfy the mandatory positive AUROC interval. Generic/contextual summary effects remain plausible.
7. **Fold/center consistency?** Inner AUROC improves all5 folds/all4 known centers; Outer4/5 folds and3/4 centers. Pediatric reverses (C1-C0 -0.057111,C1-C2 -0.069425); fold4 also reverses. Known centers are not unseen-center generalization.
8. **Channel-count robustness?** Inner gains are nonmonotonic across channel strata, and prescribed linear adjustment retains positive C1-C2 interval. It is descriptive, not causal, and was not separately searched on Outer.
9. **Ranking or thresholds?** Both. Inner C1-C0 Spearman0.714708 and ~20.9% mean pair reversals; Outer0.698834 and ~21.7%. But Outer MRR/Top1 are lower thanC2 and their contrast intervals include zero, so altered order is not reliably better EZ retrieval.
10. **TP versus FP?** Inner repeated appearances vsC0 netTP+55/netFP-1373; Outer unique channels netTP-90/netFP-524. Patient-equal sensitivity slightly improves on Outer despite fewer pooled TP. Outer Macro-F1 benefit is chiefly compatible with specificity/FP reduction, not uniformly recovering more EZs.
11. **Justify TIDE-EZ?** Not a confirmed synchronization-specific claim after Outer. The original inner gate permits proposing a future study; the Outer extension weakens that rationale. No automatic deep-model training or rescue is executed.
12. **Which assumption fails?** The claim that correct temporal correspondence reproducibly beats an equally sized shifted-context representation fails its Outer AUROC uncertainty gate. Reference identifiability/coverage do not fail. This is not evidence that all cross-channel biology is irrelevant.

## Interpretation and execution limits

For the user's question about similar AUROC but lower F1, see `AUROC_F1_INTERPRETATION.md`. It separates the65-appearance Development reference from historical80-person A0 Outer, and documents both worsenedTP/FN andFP terms. No threshold-only explanation or A0 reproduction is established; no Outer threshold optimization is performed.

Actual75 fixed LR fits completed:60 inner and15 final Outer. No CNN/TCN/Transformer/GNN/attention, additional seed, feature/stage search or new Raw/EDF extraction. The only engineering repairs were reporting the actual79-person FIT union, NumPy2 batched OLS RHS shape, and explicitly decoding the Outer UTF8 protocol before any final probe fit. Model/feature/threshold seals and exact source hashes are retained.

Original Development labels/scores are never evaluated or used in selection. The newly authorized global clinical y member is first loaded only after all15 Outer scores seal; each score is evaluated only in its original TEST fold. It is not used to train/threshold any cell. Full-cohort totals1743/5892 are now actually rechecked in this separate authorized phase.

Outer is **user-authorized post-inner exploratory repeated-test**, not blind confirmation: prior repeated use of this cohort cannot be undone. Bootstrap holds trained models/thresholds fixed and does not account for model-selection, seed or historical experimentation uncertainty. Existing-cache filtering/EDF+D/onset provenance remains unresolved; no normal-tissue or clinical propagation mechanism is identified. Code, hashes and nonidentifying aggregates only are published. All private labels/features/IDs/models/scores/logs remain on the trusted server.
