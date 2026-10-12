# Implementation and integrity audit

## Fixed computation

`code/model.py` retains the original shared initialization and Dropout/head.
B1 removes only the LayerNorm parameters. B2 adds Linear(96,32), GELU,
Linear(32,96) and a 96D output scale initialized to zero. B3 adds
Linear(96,32), GELU, Linear(2,32), tanh, elementwise product and
Linear(32,96). Both residual output weights/biases initialize to zero;
there is no double-zero scalar gate that prevents upstream learning.

Statistics use population variance and the original epsilon 1e-5 in float64.
Normalizer fitting gives each FIT patient total weight 1/P and each of its
channels weight 1/(P*C). Mean/std are frozen buffers, not learned parameters.
No validation/test observations or labels fit this transform. Original88D
Patient Z and FIT-only imputer/scaler are replayed without changes.

## Training / selection

The unchanged original trainer performs one update per FIT patient, original
patient order and fold seed, AdamW lr0.001/wd0.0001, FIT class-weighted BCE,
gradient clipping1, max30/min6/patience6. All backbone parameters remain
trainable. Every arm selects its own checkpoint/threshold by the original
Macro-F1/EZ-F1/earlier-epoch and 0.005-grid rules. No AUROC-driven replacement
of the selected checkpoint is made.

The original scientific environment is reused for training/inference. An
independent analysis environment runs sklearn verification of already sealed
scores; it never trains, regenerates predictions or refits thresholds.

## Checks and interventions

- Exact parameter counts: 8817/8641/15185/15185.
- Exact original B0 validation metrics and thresholds across five folds.
- FIT-only preprocessing replay and feature/order/partition hashes.
- Duplicate/constant columns retained on legal FIT/VAL inputs.
- Patient-equal statistic fit: held-out perturbation and channel replication tests.
- Random and actual-validation initial B2/B3 predictions equal A0 bitwise.
- Shared layer initialization and adapter hidden initialization parity.
- Output projection receives initial gradient; other adapter components
  receive gradients after it activates.
- Exact resumed weights/buffers/Adam slots/RNG/history/selection in a
  synthetic split, for all three newly trained configurations.
- M1 changes only within-patient statistic correspondence, preserving the
  multiset and the input/hidden tensors; checkpoint and threshold remain fixed.
- M3 disables only the residual on the **trained B3** backbone. It is not B0.
- Per-patient canonical channel ordering fixes ranking ties.
- Model/scaler/threshold/code/prediction hashes lock all 20 development cells.
- Independent sklearn metric parity, aggregate parity and 10,000 unique-ID
  paired bootstrap verification precede any conditional outer access.

Full feature archive during development is used only for its content hash,
unlabeled identities, order and split metadata. The code does not materialize
global x/y; each fold uses its pre-existing legal FIT/VAL bank. Outer scores
or labels cannot enter development through the default evaluator path.

Runtime checkpoints, banks, detailed predictions and patient records stay on
the server. Only source and compact anonymous aggregates are distributed.
