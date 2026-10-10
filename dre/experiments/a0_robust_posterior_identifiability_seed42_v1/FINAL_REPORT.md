# A0 robust posterior identifiability v2 — seed42

Terminal: **`NUMERICAL_REPAIR_WITHOUT_DECISION_GAIN`**. New independent protocol; previous experiment unmodified. No new scorer/teacher training, raw EEG, label change, outer TEST, rescue tuning or validation-based representation selection.

## Complete matched development results

| Method | Status | Macro-F1 | EZ-F1 | EZ-AP | EZ-AUROC |
|---|---|---:|---:|---:|---:|
| D0 | COMPLETE | 0.638080 | 0.431198 | 0.518235 | 0.710667 |
| D2 | COMPLETE | 0.647926 | 0.443199 | 0.519936 | 0.718221 |
| D1_R0 | NOT_ESTIMABLE_FIT_GATE_FAILED | NOT_ESTIMABLE | NOT_ESTIMABLE | NOT_ESTIMABLE | NOT_ESTIMABLE |
| D1_R1 | NOT_ESTIMABLE_FIT_GATE_FAILED | NOT_ESTIMABLE | NOT_ESTIMABLE | NOT_ESTIMABLE | NOT_ESTIMABLE |
| D3_R0 | NOT_ESTIMABLE_FIT_GATE_FAILED | NOT_ESTIMABLE | NOT_ESTIMABLE | NOT_ESTIMABLE | NOT_ESTIMABLE |
| D3_R1 | NOT_ESTIMABLE_FIT_GATE_FAILED | NOT_ESTIMABLE | NOT_ESTIMABLE | NOT_ESTIMABLE | NOT_ESTIMABLE |


Original80 patients/7635 canonical pairs/88D/frozen5fold; development65 appearances/47 IDs/6273 channel appearances. All11 metrics and per-fold/source tables are delivered. Fixed-prior rows, where admitted, are diagnostics rather than selectable scorer models. No partial posterior folds are substituted for complete arms.

## FIT-only admission, not independent validation

| Arm | Numeric 5-fold gate | MAP−fixed F1 | Prior MAE−MAP MAE | Positive folds | Admitted |
|---|---|---:|---:|---|---|
| D1_R0 | True | -0.014705 | -0.024076 | 0/5 | False |
| D1_R1 | True | -0.006941 | -0.014829 | 0/5 | False |
| D3_R0 | True | -0.022125 | -0.042514 | 0/5 | False |
| D3_R1 | True | -0.008693 | -0.012631 | 0/5 | False |


Each pseudo-target is excluded from density labels, source prior and R1 floor. Its own OOF teacher excluded it from TRAIN and selection. However other density-input OOF teachers can have trained or selected on its labels; actual exposure is in OOF_TEACHER_EXPOSURE_AUDIT.json. This is a screening diagnostic, not fully nested prospective validation. No teacher retraining conceals this limitation.

Near-zero IQR no longer creates unbounded R1 moments. H1 numerical repair is supported only where full Gate A passes. That does not establish predictive H2: the narrow Gaussian/MAP utility gate must pass separately. If no arm is admitted, no conclusion about its VAL performance is available. A negative FIT gate says patient-mixture signal was not established under this fixed family, not that all unlabeled distributions are information-free.

The Gaussian two-means/common-variance family, equal-patient moments, nu10 hierarchy, Beta20 prior, MAP solver, bounds and all-k ratio-of-expected-counts Macro-F1 approximation remain unchanged. Only R0/R1 preprocessing differs. Ranking ties from clipping/saturation preserve original logit order then canonical identity. Posteriors/mixture fractions refer to operational benchmark labels, not biological EZ truth.

No label-guided floor, source rescaling, new density or posterior rescue was used. Severe-collapse screening was frozen before outcomes: any invalid LOO/numerical/MAP result or >20% near-boundary/all-or-none decisions in any FIT group prevents admission. These screening cutoffs are not performance-optimized constants.

## Statistics and transport

10,000 seed42 paired unique-patient cluster draws retain all repeated development appearances. Missing metrics remain missing. Required posterior contrasts are NOT_ESTIMABLE when blocked. The controls are unchanged and CIs do not remove repeated development use, prior scorer checkpoint/threshold selection optimism or multiplicity.

OOF_TO_VAL_TRANSPORT_AUDIT.csv uses label-blind score summaries for every scorer/representation. VAL scores neither fit parameters nor select candidates. Source/group/channel shifts are descriptive; no independent-channel significance is claimed. Labels are revealed only after any admitted fixed inference; no post-outcome family choice.

## Explicit answers to20 requested questions

1. Frozen A0/D2 reproduced? Yes: current immutable-artifact and exact prediction-metric replay passed; predecessor independent model-forward replay remains hash-verified. No new scorer forward output replaced the frozen predictions.

2. Legal OOF logits recovered? Yes: all101 locked private artifacts,40 teacher checkpoints/provenance, once-per-FIT-channel coverage, identity and TRAIN-only preprocessor hashes passed. No teacher scores were regenerated.

3. R0 global ordering? A0 5/5; D2 5/5. Full numerical admission includes counts, finite results and R1 clipping rules, not ordering alone.

4. R1 global ordering? A0 5/5; D2 5/5. Full numerical admission includes counts, finite results and R1 clipping rules, not ordering alone.

5. Tail domination removed? R1 transformed |z|<=8 and every patient class second moment<=64, verified. The floor is FIT-only and LOO-excludes query. Boundedness does not verify Gaussian likelihood correctness or biological separation.

6. R1 clipping activation? FIT full-fold all-channel fractions range 0.000000%–0.000000%; source-specific maxima and patient-fraction quantiles are in ROBUST_NORMALIZATION_AUDIT.csv. Frozen IQR floors range 0.416712–0.701351.

7. Within-patient ranking preserved? Yes: transformation monotonicity, ordered Gaussian posterior and decoder original-logit/canonical ties are asserted. MAP-vs-fixed can change calibration/cardinality, not ranking. Full ranking metrics use the unchanged scorer order.

8. FIT distribution stability? Finite bounded R1 moments are established where gates pass; patient heterogeneity, small class gaps and LOO invalidity are separately reported. Numerical validity is not proof of stable transferable class densities.

9. Source hierarchy improves stability? Shrinkage/fallback behavior is measured in SOURCE_GAUSSIAN_PARAMETERS.json. A known-source inadequate/reversed fit uses global density with the original source prior. No ablation or unseen-center comparison was authorized, so no causal hierarchy improvement is claimed.

10. MAP improves prevalence accuracy? FIT LOO prior-minus-MAP MAEs are in the table above; required improvement is +0.005. No target labels enter the estimate. These are operational-label mixture proportions, not clinical prevalence.

11. MAP improves FIT F1? Table above compares identical densities/scores/decoder with fixed prior; +0.005,3 positive groups and EZ-F1 nondecline are required jointly. Exposure prevents fully nested independent interpretation.

12. Admitted combinations? None; no posterior VAL inference is authorized. All frozen condition booleans are published; they were not weakened.

13. Full five-fold development results? Table above and VALIDATION_SUMMARY.csv; blocked arms are NOT_ESTIMABLE, not zero and not averages of successful folds.

14. Posterior outperforms A0? Not estimable; no admitted complete arm. Paired intervals remain exploratory.

15. Posterior outperforms D2? Not estimable; no admitted complete arm.

16. MAP versus fixed prior attribution? FIT screening isolates MAP; admitted VAL fixed-prior contrasts are in PAIRED_BOOTSTRAP.csv. No gain can be attributed to MAP if that contrast fails; absent VAL comparisons are explicitly unavailable.

17. OOF-to-VAL transport? Patient-median Wasserstein range 0.130279–0.441466; transformed-score channel KS range 0.031025–0.258423. Full per-source shifts/floor/clip/spread are descriptive, not channel-independent tests or fitting inputs. No causal transport failure is inferred merely from a shift.

18. Any complete model reaches .658? False. Missing posterior results are not numerical failures or successes.

19. Any complete model reaches .700? False. These are repeatedly reused development patients, not new independent testing.

20. Continue or stop? Stop this locked run after the terminal; no automatic replacement likelihood, backbone or tuning. Numerical repair alone is not evidence for useful patient mixture information. A narrow negative result does not prove that physiological signals or all posterior mechanisms lack information.

## Integrity and delivery

C0–C9 synthetic controls, exact old-family parity, masked target-label isolation, deterministic repeats and real FIT-only normalization smoke are required before analysis. Inputs/protocol/code are hash-bound. Density/admission artifacts are sealed before VAL; independent deterministic rerun verifies aggregate outputs without changing choices. Checkpoints, individual patient/channel scores, labels, mixture proportions, OOF ledgers and logs remain on the trusted server. Only code, hashes and non-identifying aggregates are published.
