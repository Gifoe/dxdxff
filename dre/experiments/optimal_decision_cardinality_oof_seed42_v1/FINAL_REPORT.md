# Optimal decision-cardinality OOF: final seed-42 report

**Terminal: `CURRENT_LABEL_BLIND_PATIENT_CONTEXT_NOT_SUFFICIENT_FOR_0P70`.** The fixed C3 model failed its five-condition outer-test gate, so no C3 outer-test predictions or labels were read. This is an exploratory analysis of the historical 80-patient Task-1 cohort, whose outer outcomes had already been inspected in earlier work—not a fresh multiseed confirmation.

1. **Does optimal Top-K reproduce the ~0.70 threshold oracle?** Yes. On 65 outer-validation patient-fold episodes, Top-K and threshold oracle Macro-F1 were both `0.702841` (zero discrepancy). On 255 outer-fit OOF episodes, Top-K was `0.699772` versus reachable threshold `0.699556`; one exact-score-tied patient accounted for the +`0.000216` mean difference. One validation patient had disconnected optimal K values; the target retained both regions.

2. **How different is K* from the true EZ count?** They correlate but are not interchangeable. Using the diagnostic optimal K nearest to B0's formal K0, `corr(q*, true fraction)` was `0.751` on fit OOF and `0.804` on validation. Only `13.3%` and `12.3%`, respectively, were within one channel of the true count. On validation, the signed `q* − true fraction` median was `−0.0254`, with q10/q90 `−0.1621/+0.1244`; mean was `−0.0160`. Aggregate tertile diagnostics are in `A/KSTAR_VS_TRUEK.csv`. K* maximizes Macro-F1 **for B0's imperfect ranking** and is neither biological EZ extent nor a deployable oracle.

3. **Does predicting a cardinality correction work better than prior raw-threshold prediction?** No. C3 validation Macro-F1 was `0.584579`, versus `0.628380` for formal B0 (−4.380 pp, 0/5 positive folds). The prior raw-threshold A1 was `0.591867`; neither method transferred, and C3 did not repair that failure.

4. **Is a constant correction sufficient?** No. C1 was `0.623452` (−0.493 pp versus B0) and positive on only one fold.

5. **Does score-shape context C2 help?** No. C2 was `0.589634` (−3.875 pp; 0/5 positive folds).

6. **Does absolute pre-patient-z context C3 help?** No under this representation. C3 was `0.584579`, another −0.505 pp below C2. Its fit-only 8D PCA was legal and explained mean `87.86%` variance, but variance retention is not evidence of a predictive decision signal.

7. **Does C3 recover ≥15% of oracle headroom?** No. The validation oracle was `0.702841`, a `7.446 pp` gap above B0. C3 degraded instead, yielding a recovered-headroom fraction of `−0.588` and mean fold correlation with oracle count correction `−0.0060`.

8. **Does it pass the locked outer-test gate?** No: `0/5` gate conditions met. EZ-F1 fell `7.140 pp`, in addition to the Macro-F1 loss. Outer-test arrays were not read for this experiment. No post-hoc PCA, loss, architecture or ranking changes were made.

9. **Next decision.** The previous raw-threshold method and this predeclared cardinality head both failed validation. Stop rearranging the same current label-blind patient and B0-score information into further decision-head variants. A new attempt would need genuinely new patient-level information (for example, clinically relevant anatomy/spatial context), a separately locked protocol, and independent confirmation. This result does **not** establish that such information will succeed or that 0.70 is deployably reachable.

`A/B0_REUSE_AUDIT.json`, `A/K_ORACLE_PARITY.md`, `A/K_ORACLE_SUMMARY.json`, `A/PATIENT_CONTEXT_AUDIT.json`, `A/VALIDATION_RESULTS.csv`, `A/VALIDATION_GATE.json` and `A/VALIDATION_SUMMARY.md` provide the audit trail. No private patient/channel records, raw descriptors, logits, labels, models or checkpoints are published.
