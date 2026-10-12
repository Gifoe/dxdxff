# A0 + Patient-relative Magnitude Adapter — seed42

Exactly four configurations: the verified 8,817-parameter source-free A0,
removing input LayerNorm, an equal-capacity hidden-only residual, and PMA.
No new EEG extraction, feature subset, model search, or additional objective.
The backbone and adapter train jointly from the shared original initialization.

The primary development unit is the original **65 validation patient-fold
appearances from 47 unique patients**, not the 80-patient outer population.
Intervals resample 47 patient identities, retaining every sampled identity's
fold appearances. Development checkpoint and threshold selection uses these
validation observations; these numbers are exploratory, not held-out test
confirmation. The seven frozen continuation criteria determine whether a
new B3-only outer evaluation is permitted.

## Reproduction

The Windows runner requires the original private legal FIT/VAL banks, frozen
source-free checkpoints, feature/split identity archive and original trainer.
Their content hashes and code identity are checked by `code/runtime.py`.
Defaults are intentionally fixed to the existing authorized server artifacts;
private patient records and artifacts are not distributed here.

Run with the original scientific CUDA environment, in this order:

```text
python -u code/prepare.py
python -u tests/engineering.py
python -u code/train.py
python -u code/evaluate.py
```

Then run `code/verify.py` in the separately validated analysis environment.
It independently checks sealed predictions using sklearn and reproduces
identity-cluster bootstrap intervals, without new model inference.
`code/evaluate.py --outer` refuses access unless the complete development
gate and independent verification pass. No B1/B2 outer evaluation is allowed.

Public artifacts are code, protocol, source/implementation reports and compact
aggregate results only. No raw EEG, feature banks, weights, patient-channel
predictions, patient IDs, credentials or runtime logs belong in this directory.

See `FINAL_REPORT.md` for the completed result and stop decision. No installed
skills were used in this experiment.
