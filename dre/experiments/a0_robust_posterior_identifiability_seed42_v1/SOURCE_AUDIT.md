# Historical source review, before new analysis

Reviewed the previous posterior FINAL_REPORT at GitHub commit 394dbff, its
posterior/model/prepare/train/finalize source and protocol; normalization tails,
identifiability, fixed-prior diagnostics, OOF distribution, fold metrics and
actual OOF provenance. Also reviewed the decision-identifiability report at
cc22d019. Frozen local predecessor contents match the published tree.

The previous global class-order reversals occurred after median/IQR floor1e-5
normalization, not evidence of reversed raw A0 ranking. A few nearly constant
patients dominated moments; the largest aggregate reported magnitude was
58185.5 (the prompt's approximate fold examples are not the maximum across both
scorer families). Previous posterior results cover only successful folds1–2;
there is no complete D1/D3 result to reuse or claim.

D0/D2 full development Macro-F1 is 0.6380797828499001/0.6479260773776448;
D2's +0.009846 interval crosses zero and only 3/5 folds improve. Historical
label-using oracle 0.7062849901 is retrospective capacity, not an inference
input. Earlier eight-label and cardinality probes did not recover this capacity.

Only previously sealed banks, selected scorer outputs and legal deterministic
OOF scores are used. This run neither imports a training entrypoint nor writes
the prior runtime. The current hash snapshot has 101 frozen private artifacts;
original teacher/output/checkpoint hashes are additionally verified against
the previous public provenance. No MC means, in-sample FIT outputs or raw EEG.

Each OOF queried patient was absent from that teacher's TRAIN and selection,
but other patients' OOF teachers can have trained/selected on the pseudo-target.
The LOO posterior refit is therefore a FIT screening diagnostic, not fully
nested independent validation. Actual exposure will be counted, not repaired
by unauthorized scorer retraining.

New analysis is limited to raw logits R0 and the specified FIT-only floor and
bounded R1. Gaussian moments/shrinkage/fallback/prior/MAP/decoder are unchanged.
Source means retain heterogeneous operational clinical targets. No biological
prevalence, unseen-center performance or clinical benefit is inferred. A failed
narrow likelihood does not prove all physiological signal uninformative.
