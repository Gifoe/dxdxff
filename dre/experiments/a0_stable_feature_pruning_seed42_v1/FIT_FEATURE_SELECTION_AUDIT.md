# FIT-only selection and leakage boundary

Every outer fold has three deterministic patient-group screening splits within
its original FIT patients. Sorted IDs are fold-seeded and partitioned in three.
For each untouched screening split, the remaining IDs are independently
permuted using the prelocked seed; ceil20% form checkpoint-selection patients,
the others train. None of the corresponding original fold's held-out
Development IDs enters these splits. Patient roles are defined per original
fold, not by excluding the union of validation IDs from every other fold's FIT.

The original Patient Z operator uses each patient's own unlabeled channels.
The original imputer/scaler are fitted anew on **inner training patients only**,
not checkpoint-selection or screening patients. This stricter nesting avoids
using an outer-FIT-fitted scaler that had already observed screening inputs.
The original 88D preprocessing algorithm remains unchanged; final Development
models use the original outer-FIT-fitted, frozen legal bank values.

Each split independently trains a full42 model and21 one-group-deletion models,
with matched initialization seed/loss/optimizer/update order/selection rules.
Epoch and threshold choice use checkpoint-selection labels only; untouched
screening labels are read only for deletion utility. All330 tasks are executed.
Screening effects are equal means within each patient and then equal means
across the three screening folds, as locked; no pooled-channel shortcut.

Eligible: mean deletion-minus-full AUROC≥.002, AP≥0, Macro-F1≥−.002,
positive AUROC in at least2/3 splits. At most3 eligible groups are ranked by
AUROC gain, AP gain, authoritative group index. No automatic removal of ABS
or RATIO when a spectral DELTA candidate is unreliable. Group combinations
are not assumed additive; their utility is tested by final B3.

B4 removes exactly the same **number of groups** as that fold's B3, by a fixed
prelocked RNG choice without rerolling. Random groups may overlap selected
groups; this is not a reason to reroll. Different original multiplicities can
make their virtual retained-position counts differ, while their physical
two-coordinate-per-group parameter counts match.

All five subset decisions and random controls are frozen before any new
comparative Development results are computed. No global subset is reselected
from Development. B3/B4 alias B1 in zero-deletion folds, avoiding an artificial
different training effect. Feature correlation and numeric-floor diagnostics
are FIT-only descriptions, never eligibility criteria.

Final validation checkpoint/threshold selection intentionally follows the
historical original Development protocol; this is not a fresh blind test. No
new outer inference or outer score/label outcome is opened. The old feature
archive is hash-checked and its metadata used for cohort admission; only legal
FIT/VAL bank feature/label arrays are used for this experiment.
