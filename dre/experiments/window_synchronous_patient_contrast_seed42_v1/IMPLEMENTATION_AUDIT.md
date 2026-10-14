# Implementation and leakage audit

## Execution

Trusted server execution completed.13 synthetic tests passed locally and on the server. The data/control stage covered80 patients and passed before loading FIT-only labels.60 logistic cells were fitted exactly once (5 original folds x4 nested groups x3 arms). Summed actual per-cell fit/preprocess/threshold/predict wall time was14.9076 seconds; maximum lbfgs iterations34, no convergence cap or rescue. This is not the end-to-end elapsed time including cache load, data construction, bootstrap and verification.

All60 model/scaler/median/threshold/prediction artifacts sealed before held-out aggregate scoring. No LR refit was performed during evaluation recovery or verification. Independent verification reconstructed TRAIN-only preprocessors and selection thresholds, checked all frozen artifact hashes, independently recalculated765 patient metric cells with sklearn and reproduced48 paired metric intervals. See `INDEPENDENT_VERIFICATION.json`:PASS.

## Protocol

Original five-fold manifest unchanged. Each original FIT set is deterministically partitioned into4 held-out groups. Remaining patients are deterministically split20% threshold selection (ceil) and80% TRAIN. All roles are disjoint per cell; preprocessing and LR only use TRAIN, selection only uses selection patients, never held-out labels. No center or identity predictor. All three arms receive182D and identical patient assignments, self inputs, masks, imputation/scaling rules, objective and solver.

Patient-equal sample weight1/canonical-channel-count gives each TRAIN patient total1. No class weighting, search, second normalization or Patient Z. Finite weighted-median imputation and weighted StandardScaler parameters are estimated only on TRAIN. C0's intentional84 zero coordinates remain zero; shared first98 imputer/scaler parameters are identical across arms. EZ-positive mapping: original y_nez0 ->positive1; predict p(EZ)>=tau. Threshold grid0:.005:1, fixed patient-mean Macro-F1/EZ-F1/balanced-accuracy/nearest0.5/smaller ties. No threshold optimization in bootstrap or interventions.

255 evaluated original-FIT appearances belong to79 unique identities, not80 independent test patients. Unique-ID clusters keep all appearances together in10,000 common seed42 draws. Primary estimator weights appearances equally; secondary estimator first averages each identity's appearances then weights identities equally. Neither estimator is the historical65-target Development,47-ID or80-ID Outer estimator.

## Engineering recovery ledger

An initial detached launch produced no freeze/status/log content and was no longer active. The checked foreground execution then ran the data stage and all60 probes. No duplicate active process was started and no result-bearing epoch/model was discarded.

First aggregation stopped at a hardcoded unique-count80 assertion: actual original FIT union is79. The original split was independently counted before outcome inspection. Exact source-hash allowlist repairs only the aggregate count/validation, not any model, feature, label admission or threshold. A subsequent NumPy2 batch-RHS shape error occurred in the prescribed OLS diagnostic; adding an explicit singleton RHS column implements the same frozen equations and passed an identity-matrix synthetic test. Source before/after hashes are in ENGINEERING_REPAIR_LOCK and SOURCE_HASHES. Original input/table/probe seals and model artifact bytes were preserved. Final evaluation-only replay succeeded; no test-driven scientific iteration occurred.

## Limits

Installed skills were not used. No raw or EDF extraction, CNN/temporal/graph training, TIDE design/training, additional seed, Development classification or Outer inference/evaluation was run. Monolithic feature-cache deserialization may incidentally materialize clinical metadata; it is never indexed or used. Original-fold FIT labels are legal only relative to their corresponding fold. Extensive historical reuse of this cohort and conditional fixed-model bootstrap prevent treating the result as independent confirmation.

The preceding no-Outer statement applies to the **original inner phase** only. The later authorized extension adds15 final LR fits and one80-person Outer pass; total75 fits. Outer source explicitly decodes the authorization lock asUTF8 after a GBK decode failure before any final fit. Two additional tests pass (final role split/reproducibility and NumPy2 vector solve). All15 models/thresholds freeze first; all15 scores seal before clinical y access. Independent sklearn parity checks240 Outer metric cells and TRAIN-only preprocessors/thresholds/model hashes are reconstructed/checked. No Development evaluation, new extraction, scientific tuning or deep model follows. Outer is exploratory repeated-test and its synchronization-specific CI gate fails.
