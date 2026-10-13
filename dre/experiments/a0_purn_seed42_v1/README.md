# A0 + PURN, seed42

Completed: `PURN_DEVELOPMENT_GATE_FAILED`. Subsequently user-authorized frozen
outer is also complete: PURN AUROC0.728407 versus original A00.739768,
paired delta−0.011361[−0.023761,−0.001255]. No supported replacement gain.
See [FINAL_REPORT.md](FINAL_REPORT.md) for the development study and all14
questions; [OUTER_REPORT.md](OUTER_REPORT.md) for complete outer results.

Fixed 2x2 development experiment: untouched original LayerNorm88, RMS88,
unique56 LayerNorm and proposed unique56 RMS. All four use the original88D
Patient-Z/FIT-preprocessed interface, 176 input affine parameters and8817
total parameters, identical original linear initialization, trainer/objective/
optimizer and legal validation selection. B0 is replayed, not retrained.

Unique coordinates affect only statistic estimation. No deduplication, masking,
new feature extraction, hidden branch, capacity increase or patient-label-based
adaptation is allowed. Replacement arms are not forced to equal initial B0
logits; independent NumPy reference tests validate their different formulas.

Trusted-server execution order:

```text
python code/baseline_admission.py
python code/prepare.py
python tests/engineering.py
python code/train.py
python code/evaluate.py
python code/verify.py
```

Hash-bound original artifacts are required. Training and model evaluation use
the original scientific runtime; independent verification reads frozen private
scores in the existing analysis runtime. All source/tests and the protocol are
bound before formal training. Completed evaluation rejects scientific rerun.

Primary estimator:65 equally weighted validation-fold appearances from47
unique IDs. Secondary estimator: average appearances within ID, then47-ID mean.
10,000 seed42 bootstrap draws retain all appearances of each sampled identity,
with thresholds fixed; intervals remain descriptive for a historically reused
development cohort. The original development lock prohibited new outer.
The later explicit user request is recorded separately in
OUTER_PROTOCOL_AMENDMENT.json; the failed development gate is not changed.

Frozen outer extension (not training):

```text
python outer_tests/contract.py
python outer_code/run.py prepare
python outer_code/run.py evaluate
python outer_code/run.py verify
```

The separate outer runtime reuses old B0 scores, seals all15 new B1/B2/B3 score
files before full y access, and compares80 unique patients once each. It is
exploratory repeated-outer, not a new blind or external confirmation.

Public files are source, protocols/hashes and anonymous aggregates only.
Banks, checkpoints, IDs, clinical/channel records, individual predictions and
logs remain private. No installed skill is used. Final measurements and the
continuation decision are actual executed results, not planned placeholders.
