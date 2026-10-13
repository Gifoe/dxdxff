# Actual implementation, admission and execution

The original source-free constructor instantiates the original SourceA0.network,
not a reimplemented MLP. B0 retains native PyTorch LayerNorm. B1/B2/B3 replace
only that module with UniqueFeatureNorm. All88 coordinates continue to the
original Linear88to96, GELU, Dropout.15 and classifier. Original Source V/u/b
are not registered. All arms have8,817 parameters,176 affine parameters, and
identical original initial linear tensors and one/zero gamma/beta per fold.
The original source-free initial-state hashes, seed, objective and training
protocol match the historical checkpoint binding. No trained B0 warm-start.

UniqueFeatureNorm selects a nontrainable integer statistic buffer, accumulates
population moments or uncentered second moments in float64, adds epsilon1e-5
inside sqrt, normalizes all88 coordinates, applies88 gamma/beta, then returns
the original dtype. No detach, clipping, extra scaling/gate, new head, center
normalizer or input reordering. New-arm initial logits intentionally need not
equal B0. The independently written NumPy formulas, not an artificial identity
correction, validate their normalization.

The immutable manifest determines56 representatives by names, constant flags
and duplicate_of relationships. Actual post-Patient-Z and post-FIT tensors
verify all16 duplicate pairs exactly, all16 zero columns, nonconstant FIT
representatives and unchanged input values. Original imputer/scaler replay is
bitwise. Feature/archive/order/split hashes are the exact requested values;
all five legal banks exclude their outer patients. The global NPZ x/y members
are not materialized during development; only permitted identity/order/split
metadata are read there. Label counts in the initial cohort contract are
historical hash-bound metadata, not a new outer-label audit at that stage.

## Pretraining tests and actual scientific training

476 server checks PASS before any new clinical training, including NumPy
float64/float32 agreement on random, zero, identical, tiny, huge signed, −40/+68,
duplicate and actual legal FIT/VAL tensors; shape/epsilon/population variance/
affine broadcasting; finite CPU/CUDA forward/backward; no input mutation;
all four statistical distinctions; exact parameter and shared linear identity.
All three constructors also pass exact synthetic uninterrupted-versus-resumed
model/buffer/Adam/RNG/history/selection equivalence. Synthetic resume runs are
engineering tests, not extra clinical arms.

B0's five frozen development checkpoints replay logits/probabilities bitwise,
selection/thresholds exactly, and all historical requested metrics. B1/B2/B3
complete all15 prescribed cells,138 epochs, in one healthy formal attempt.
Recorded trainer-call time sums12.968455 seconds, including training/validation
and checkpoint writes but excluding preparation, tests, startup, bootstrap,
verification, artifact transfer and reporting. Existing compact features are
already cached; this is not raw EEG training. No epoch was discarded, manual
early-stop or scientific retry occurred. All six development phase stderr
files are empty; no active PURN process remains.

The unchanged original trainer is invoked via a temporary constructor adapter,
source penalty0. Original post-constructor RNG reseeding, patient update order,
NEZ-positive FIT-weighted BCE, AdamW lr.001/wd.0001, clip1, max30/min6/patience6,
Macro-F1/EZ-F1/earlier checkpoint selection and numeric .005 threshold ties stay
fixed. Each arm's threshold is selected on its legal validation patients only.
Selected eval-mode FIT/VAL losses in TRAINING_EFFICIENCY.csv are diagnostics,
not selection inputs. Code/tests/protocol did not change after admission.

Development freeze SHA256:
`8f71ea545914f119eec8f71fee25e6530ce5dfb1505c6ce3b18a8adcc50e18cc`.
Original protocol SHA256:
`2130ce5f6ee4f85679c4dec497d769b2d5646fc8f9f42fbaabe0fe01a8392247`.

Independent verification reads sealed scores, not fresh model inference. It
checks260 patient/appearance cells against sklearn, all96 paired/interaction
intervals using alternate identity multiplicities, input/mapping/parameter/
affine/checkpoint/threshold/source hashes, secondary ID estimator and aggregate
gate parity. Independently expressed NumPy moment/duplicate checks also PASS.
Maximum metric drift3.33e-16. A local results-directory omission and one failed
artifact transfer were corrected without rerunning or changing science.

## Explicit user-authorized outer extension

After development failed, the user explicitly requested completing outer.
OUTER_PROTOCOL_AMENDMENT.json records that additional authority. The original
PROTOCOL_LOCK, development terminal/gate and code/test bytes remain unchanged.
New outer source/test files are separate directories, outside the development
binding glob. No training, reselection, checkpoint modification, epsilon change
or rescue is performed. Existing B0 outer scores are reused;15 new frozen
B1/B2/B3 fold inferences are run on CPU matching the historical outer backend.

All20 selected checkpoint/threshold/preprocessor dependencies are locked before
new outer inference. Stored FIT transforms replay bitwise; CPU validation logits
and probabilities match frozen CUDA development within the predeclared2e-6 /
2e-7 limits with identical threshold decisions. All15 new score files are atomic
and hash-sealed before full-cohort y access. Original bank/export label parity,
80 patients/7,635 channels each once,1,743 EZ/5,892 NEZ are checked. No old score
or threshold is changed. Independent outer verification checks320 patient-model
cells and96 contrast/interaction intervals, maximum drift3.33e-16, PASS.

Outer freeze SHA256:
`a59e065c63cf8ce9a0a7dfbea8889c1e9dbfa706227b8dcfcecf12070b2ecad9`.
Prediction seal SHA256:
`571756163fdc6235a21cf35afa7968383ff9f33595db3c93bddf10e16eaf766e`.

This is user-authorized post-development exploratory repeated-outer evaluation,
not blind or external confirmation. Detailed predictions, IDs, clinical rows,
original banks/checkpoints and logs remain private; only audited code, hashes
and anonymous aggregates are delivered. No installed skills were used.
