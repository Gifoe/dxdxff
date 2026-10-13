# B2 main model: matched single-view Development, seed42

This is **not** the old full42 F0 single-view study. The user-designated main
is the completed pruning study's B2:36 canonical ABS/DELTA/RATIO coordinates,
after deleting beta/low-gamma/high-gamma DELTA mean/std. Main checkpoints and
predictions are reused unchanged. Three single views are derived strictly
inside those36 coordinates and independently scratch trained, not fine-tuned.

| Arm | Coordinates | Parameters | DELTA spectral candidates restored? |
|---|---:|---:|---|
| MAIN_B2 | 36 | 3649 | No |
| ABS_ONLY | 14 | 1537 | No |
| DELTA_ONLY | 8 | 961 | No |
| RATIO_ONLY | 14 | 1537 | No |

The original Patient Z,88D FIT-only preprocessing, multiplicity-aware LN88
semantics, original initialization, loss, AdamW, patient updates and checkpoint/
threshold selection rules remain unchanged. Five folds,65 validation appearances
from47 IDs,15 new cells. No new Outer TEST, waveform extraction, extra seed,
two-view rescue or threshold rule. No installed skill used.

```powershell
python -m pytest PATH_TO_EXPERIMENT/tests -q
python PATH_TO_EXPERIMENT/code/run.py --runtime PRIVATE_NEW_RUNTIME --source TRUSTED_SOURCE_ROOT --prior VERIFIED_PRUNING_RUNTIME
```

The trusted prior banks/checkpoints are required; exact hashes reject a similar
substitute. Existing runtime resumes only with identical source/protocol bindings.
NumPy imports before Torch in the previously verified Python3.13 environment.
The copied folding source is byte-identical to the previous executed model.
`.gitattributes` preserves exact artifact bytes, including server CSV line endings.

Read `FINAL_REPORT.md` for actual metrics, fold heterogeneity, patient-ID bootstrap
and interpretive limits. Prior B2 designation followed viewed Development results;
it is a user-selected reference, not a fresh validated winner. This experiment
is adaptive reused-cohort Development, not blind clinical confirmation.

Only source, hashes and anonymous aggregates are public. Features, identities,
labels, scores, weights, scalers, optimizer/RNG state and logs remain private.
