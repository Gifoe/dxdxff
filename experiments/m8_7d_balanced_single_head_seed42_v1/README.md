# Original7D, balanced single-head fusion

Target: an internally selected single-head model with **full80 equal-patient
Macro-F1 >0.64**, keeping the original seven physiological measurements.
No new performance result exists at registration. The paused Raw-only
optimization is a different task and is not resumed here.

The existing S3 single-head7D+Gate achieved0.633492 Macro-F1 on the historical
exploratory Outer; simple7D concat achieved0.631264. Those outcomes motivate a
new prospective trial, not independent testing. The6D result is ineligible:
this task keeps Variance and every other original coordinate.

The three arms keep the original CNN, Gate, Raw masks, pre-reference, phase
and seizure aggregation, all14 preprocessed physiological coordinates and
one110→64→1 head. C0 is the exact existing S3 control. C1 balances the two
modalities' input dimensions at the classifier, using fixed square-root
constants; C2 adds a bounded low-rank hidden interaction to C1. There is no D1
predictor, scalar prediction residual, ensemble or second classification head.
C0/C1 have10,377parameters; C2 has10,378. New variants start from random
matched initialization, not trained S3/D1 weights. Neither change is claimed
novel or guaranteed to improve performance.

Only two new candidates×five original FIT/Development folds are trained.
The five identical S3 controls are reused only after actual replay/identity
and all201 threshold-decision checks. Inputs, objective, budget, optimizer
and validation rules match the original experiment. Architecture selection
uses Development only, with its65appearances of47people, and is explicitly
selection-conditioned. If neither candidate passes the locked eligibility
gate, no new Outer is run. If one qualifies, all checkpoints, thresholds,
scalers, candidate and evaluator must be separately frozen before a complete
fivefold80-patient pass. All results of that pass remain exploratory repeated-test.

The goal cannot be met by a Development score, pooled-channel F1, EZ-F1,
favorable fold, interval upper bound or post-hoc best Outer secondary.
All nine metrics, coverage, source/fold stability and paired uncertainty are
reported. Old source/seals/results remain unchanged. Only code/hashes and
anonymous actual aggregate artifacts are uploaded to GitHub.
