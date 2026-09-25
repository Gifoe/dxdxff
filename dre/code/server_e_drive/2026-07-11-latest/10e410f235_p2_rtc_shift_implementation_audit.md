# P2_RTC_SHIFT Implementation Audit

Baseline path: `P2_TEMPORAL_Q10` uses the P23 temporal/q10 backbone with NEZ=1 and `P(NEZ)` scores.  Its nested P23 trainer already produces four patient-wise inner-OOF folds for each outer training set, so RTC_SHIFT reuses that protocol rather than adding another split layer.

The implementation adds a frozen two-cohort contract: `primary90` is the exact `reference/all90_subjects.csv` allow-list; `sensitivity80` is the same ledger minus the exact ten IDs in `configs/task1_sensitivity80_exclude_suspected_10.csv`.  Both write a cohort audit and a hashed outer-fold ledger.

RTC additions are incremental: A1 changes only the tail evidence to logit-space soft-min shrinkage; A2 adds a label-free count/agreement gate; A3 adds a hard-label tail rank loss with detached direct/anchor high-confidence EZ selection; A4 changes only validation checkpoint tie-breaking; A5 adds a bounded patient-constant shift fitted exclusively from outer-train inner-OOF outputs.  No center feature, test label, true count, raw input, EMA, soft relabeling, or V3 input enters any RTC layer.
