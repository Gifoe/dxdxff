# Implementation and execution audit

Protocol frozen before new training/outcomes. Shared network initialization
matches original seeded PRMLP; V uses a dedicated RNG, and u/b start zero.
Joint gradients reach shared layers and source vectors. Unknown category uses
the shared classifier.9221 total parameters are asserted.

Original training means one optimizer update per patient, channel-mean hard BCE
with FIT EZ/NEZ pos_weight. It is NOT one pooled all-patient update. Source L2
uses locked normalized parameter means and inverse FIT source-patient weights.
Selection/seed/dropout/patient order/optimizer/patience/min-stop semantics are
unchanged. Checkpoint files atomically retain model, optimizer, RNG, best state
and history; completed-artifact SHA checks gate reuse.

Each of 32 named pretraining tests plus real FIT-only smoke has a public compact
audit. Synthetic original-optimizer equivalence disables only the new source
correction/penalty for a parity test, not a new formal model. Interrupted-resume
model/history equality is tested. Planned OOF exclusion is checked before
training; actual teacher artifacts are independently checked afterward.

Posterior fitting is outer FIT OOF only, uses patient-equal class first/second
moments and fixed source shrinkage. Insufficient/reversed source ordering uses
global density; invalid global ordering blocks, never reverses labels. MAP
has a strictly concave objective (Beta alpha/beta>1) and deterministic derivative
bisection. Inference APIs accept neither labels nor true count. All k are checked,
zero-denominator F1=0, smaller-k exact ties. Constant/empty/unknown cases are tested.

FIT falsification/density artifacts are sealed before VAL posterior inference.
Its FIT calibration is in-sample density fitting and explicitly optimistic.
Inference probability ties caused by saturation preserve mathematically
monotonic original-logit ordering; matching rank metrics are asserted.
The full posterior replay path is available in `code/verify_fold.py`, but was
not executed for a complete D1/D3 arm: folds 3–5 failed the FIT ordering gate.
No invalid density produced target predictions. Valid-fold monotonicity and
synthetic decoder checks passed; partial folds are not reported as full arms.
No outer TEST loader or evaluator exists.

## Completed execution and independent checks

Five formal D2 models and 20 new D2 inner teachers completed. Independent
forward replay verified all five selected scorers and all 40 A0/D2 teacher
query outputs, including exact TRAIN preprocessor hashes and zero query overlap
with teacher training/selection/outer partitions. Teacher logit drift was zero.
The code/*.py content and scientific protocol remain bound to their pretraining
hashes in DATA_AND_SPLIT_AUDIT.json; reporting-only scripts under tests/ were
added after the scientific blockage without changing these bindings.

The fixed normalized FIT Gaussian ordering fails for D1 and D3 in folds 3–5.
Reporting-only completion audits those gates, emits full D0/D2 metrics and
10,000 paired patient-cluster draws, and marks complete D1/D3 metrics unavailable.
An arithmetic tail-contribution audit identifies 1–2 low-IQR FIT patients per
failed fold dominating normalized moments. It does not change normalization,
drop patients, clip logits or refit an alternative density.

Engineering deviations were limited to the PowerShell launcher/exit-code
wrapper and an exact-step resume following a Windows python312.dll access
violation before any fold3 teacher epoch. Old logs and completed checkpoints
were preserved; no completed epoch was discarded. The posterior ordering gate
was treated as a scientific failure, not repaired or retried with new settings.

Final terminal is POSTERIOR_IDENTIFIABILITY_FAILED. D2's primary gain is below
the frozen continuation threshold and its 95% interval includes zero. Full
D1/D3 performance, combined complementarity and the 0.658/0.700 targets are
not established. FINAL_REPORT.md documents all 19 requested answers.

Private clinical identities, scores, proportions, checkpoints and logs remain
on the trusted server. Public outputs contain only source, aggregate metrics,
non-identifying diagnostics and cryptographic provenance. Any engineering
failure must preserve artifacts and be documented here, not outcome-tuned away.
