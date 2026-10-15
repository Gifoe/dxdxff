# DELTA14: linear SVM and AdaBoost, with and without PR (seed42)

This is a feature-classifier experiment, not raw-encoder training. A = original D1 physiological DELTA14 without patient normalization; B = the same 14 coordinates after the original full-patient population Z. No concatenation, new features, extraction, or GPU computation.

## Fixed comparison

- Original 80 patients, 7,635 canonical channels, 256 seizures and original five FIT/validation/Outer patient partitions.
- Linear SVM uses `SVC(kernel='linear', probability=False)`, not `LinearSVC`. Both PR arms share the original paired-validation-selected RBF C: folds 1–5 = 1, 1, 10, 1, 1. No new C search. Mean imputation and StandardScaler fit only on FIT.
- SVM probabilities use FIT-only three-fold patient-held-out OOF Platt calibration. Each subfold refits preprocessing without its held-out patients. The final SVC sees FIT only; calibration uses patient weights without class balancing.
- AdaBoost in sklearn 1.7.2 uses discrete SAMME with depth-one decision trees. Four predeclared configurations: 100/300 estimators crossed with learning rates 0.05/0.1. Mean imputation fits FIT only; no standard scaling or extra probability calibration. Early perfect-fit termination is permitted by the installed estimator.
- Initial training sample weights equalize patient contribution then apply inverse pooled FIT class frequency. Subsequent boosting weight updates do **not** preserve equal patient totals.
- AdaBoost shares one configuration between PR arms in each fold, chosen by the mean validation patient Macro-F1, AP, AUROC, then earlier grid index, matching the parent paired-selection policy. Each arm selects its own validation threshold on the fixed 0.005 grid. No FIT+validation refit.
- All 20 selected model/preprocessing/calibration/threshold bundles must be hash-frozen before new Outer inference. All 20 private score files must be sealed before reading the full evaluation labels.
- Original RBF SVM scores in both PR arms are hash-verified, reused without new inference, and reported as a historical reference. Primary metrics must reproduce exactly. The kernel contrast has the same C and calibration recipe, but newly fitted linear margins and calibrators.

## Execution and outputs

`PROTOCOL_LOCK.json` was fixed before this run, including the user's subsequent request for the no-PR arms. `smoke.py` exercises both estimators on synthetic data only. `run.py` uses the existing trusted feature-baseline environment and hard-gated artifacts; it never downloads data or installs dependencies. Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 1 before execution. Server filesystem locations in the code are implementation dependencies, not distributable data.

The initial run completed and sealed all 20 new model-fold score files, then stopped during aggregation: the parent `means` function retained an early-bound default metric list without the expanded NEZ-F1/EZ-fraction fields. `reporting/finalize.py` replaces only that reporting default and reconstructs tables from the already sealed scores. It verifies all 20 score hashes and model hashes and does not call training or inference. The original training source/input binding remains byte-unchanged; `REPORTING_REPAIR_AUDIT.json` records the reporting-helper hash. The NumPy degrees-of-freedom warning reflects undefined fold statistics, not failed model training; coverage tables retain undefined values rather than replacing them with favorable zeros.

Resume is permitted only when the frozen source/input/config binding matches; candidates and Outer scores are reused only with their hashes/bindings. No completed scientific evaluation is repeated to select a better outcome.

Public results distinguish patient-macro, pooled-channel, fold and center metrics, undefined-metric coverage, ranking, NEZ-F1, accuracy and signed EZ-fraction bias (predicted minus true fraction). A positive bias means overprediction of EZ. Paired 10,000-patient-bootstrap intervals use seed42 and fixed models/thresholds; these conditional intervals do not include training-seed uncertainty or multiplicity correction.

The cohort has been evaluated before: all comparisons are **exploratory repeated-test**, not blind confirmation. Native AdaBoost probabilities are not asserted to be calibrated. Patient-local PR uses all unlabeled channels of that patient, without labels or cross-patient normalization.

Only code, hashes and anonymous aggregate artifacts belong in the repository. Private features, fitted classifiers (including SVM support vectors), scalers, labels, IDs, predictions and logs remain on the trusted server.
