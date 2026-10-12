# Frozen 88D feature-view audit

The authoritative export is the original D2 feature archive, SHA256
`1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd`.
Its feature-order digest is
`b29b29129d4d4a275d9cd077aa47e46cf428fe0f25913dea75a51ff0c7ea6ca9`.
Feature-name order is recovered from the actual archive, not assumed offsets.
The machine-readable manifest records every zero-based column index, name,
view, base physiological feature, aggregation, auxiliary duplicate relation
and observed constant status.

## Exact generation source

- `task1_baselines/feature_aggregation.py`, SHA256
  `426344bfa6ef0dd4b1dc59fe267d7920cef020fdcb42e834efc9a9ae6b4ca273`.
- `neuroez_c/evidence_views.py`, SHA256
  `201f888411a8f45fa363cfb2ac11c165a04ae2eef5abcd3aad64aedfe3098b03`.

These exact historical source bytes exist on the original server. Matching
LF-normalized archived copies were reviewed locally. The runner hash-checks
the server files and calls their `p2_matched_simple` generator on synthetic
inputs to require exact feature-name/order parity with the frozen archive.
It never re-extracts raw EEG or regenerates the scientific features.

The primary block selects nine features: delta and theta log-bandpower, beta,
low/high gamma log-bandpower, RMS, variance, line length/sec, spectral entropy.
For each feature it constructs ABS, DELTA, ZDELTA, RATIO. Per-seizure window
means are computed, followed by across-seizure mean and standard deviation.
The physics block selects high gamma, line length/sec, RMS and variance, under
ZDELTA and DELTA, with the same two aggregation statistics.

| View | Primary dimensions | Auxiliary dimensions | Total |
| --- | ---: | ---: | ---: |
| ABS | 18 | 0 | 18 |
| DELTA | 18 | 8 | 26 |
| ZDELTA | 18 | 8 | 26 |
| RATIO | 18 | 0 | 18 |
| Total | 72 | 16 | 88 |

Every name matches the explicit grammar
`(b0|physics)_(abs|delta|zdelta|ratio)__feature__seizure_(mean|std)`.
No unmapped or multiply assigned columns are accepted.

## Duplicate columns

All 16 auxiliary columns have a main-block counterpart with the same view,
feature and aggregation. Original source uses the same pre-window mask,
epsilon1e-5, pre-mean/pre-SD and deviation formulas for these selected features.
The runner requires exact elementwise equality, including missingness, over
**all 7635 frozen rows**, and exact parity in the synthetic generation replay.
No numerical tolerance is used to call these 16 columns duplicates.

Both duplicate copies are removed together whenever their view is removed.
Full D2 is not deduplicated, and no extra arm is introduced.

## Additional frozen-data observation

The delta-band and theta-band features, under all four views and both seizure
statistics, are **constant zero** in the frozen export: indices
`0,1,2,3,18,19,20,21,36,37,38,39,54,55,56,57`.
This is distinct from the 16 auxiliary duplicates. Do not confuse the
delta-frequency band with the DELTA reference view. Do not infer the cause of
the zero features, relabel the data or claim that their biological signal is
irrelevant. The frozen cache is preserved exactly.

Cross-view equalities between these zero columns do not expose excluded-view
information. Structural masks still remove every excluded-view name. The
public mask table separately reports retained columns, retained varying
columns and retained auxiliary duplicates; the nominal active feature count
does not mean independent physiological degrees of freedom.

## Masking contract

ABS Only keeps18 columns; w/o DELTA keeps62; w/o ZDELTA keeps62; w/o RATIO
keeps70. Counts are obtained from verified names. Original patient_z and
stored FIT-only preprocessing run on all88 columns before masking. Retained
values are bitwise equal to the original inputs, and removed columns are
exactly zero in actual FIT/validation/TEST model-input tensors. LayerNorm and
model width remain unchanged. No relative-view auxiliary copy survives
ABS Only. Existing w/o Patient Z does not remove any physiological view.
