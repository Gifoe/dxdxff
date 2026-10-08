# PR-UAS source audit (before training)

## Baseline recovered

The prescribed CalibRank directory is byte-identical in the working tree and
`codex/omni-a1feature-complementarity-audit-seed42-v1` (c8dc22b).
Its `prepare_b0_embeddings.py`, `BASELINE_AUDIT.md`, feature manifest and diagnostic
code identify the original `task1_baselines.patient_controls` implementation.
The trusted server retains that source, the feature cache, historical ledger,
all five original seed42 checkpoints and the earlier replay audit. A fresh replay
is required; the earlier PASS is not substituted for current verification.

The 88 ordered inputs are the 44 deterministic P2 descriptors (36 B0 ABS/DELTA/
ZDELTA/LOGR-like ratio plus eight physics descriptors), window-mean aggregated
within seizure and mean/std aggregated across seizures. No raw EEG extraction or
E1/E3 export is used. The input manifest excludes labels, center, clinical counts,
resection, outcomes and identities from model inputs. Original complete patient
channel context is retained for label-free within-patient z scoring.

The exact backbone has 8,817 parameters. NEZ=1; sigmoid(logit) is NEZ probability.
Patient-z -> mean SimpleImputer -> StandardScaler are fitted on FIT only. Training
makes one channel-mean weighted-BCE AdamW update per patient, patient order from
`default_rng(seed+epoch)`, gradient norm clipped to 1, lr .001, decay .0001.
Positive weight is binary-FIT EZ count / NEZ count. Early stop is epoch>=6 and
stale>=6, with max30. It is not pooled-channel loss or a new balanced sampler.

Threshold candidates are rounded `arange(0,1.0001,.005)`. Lexicographic maximization:
patient Macro-F1, patient EZ-F1, pooled balanced accuracy, proximity to .5,
smaller threshold. Checkpoint key is validation patient Macro-F1, EZ-F1, earlier
epoch. Comparisons are `p_nez >= threshold`. Historical outer references are not
development-validation baselines.

Initialization caveat: the old wrapper constructs the model before `_fit_torch`
reseeds (42+1009*fold). Its original initial weights were not saved, so historical
training-from-scratch bitwise reproduction cannot be asserted. Frozen checkpoint
prediction reproduction is the mandatory gate. New matched students use one
explicitly seeded initial state per fold, identical across A0/A1/A2, and the exact
unchanged B0 model/loss/preprocessing/optimizer/selection semantics. This makes a
new matched A0 control, not an assertion that its selected epoch must equal history.

## Existing uncertainty methods are not PR-UAS

- P23 `c30e429274_p23_noisy_ez_loss.py`: asymmetric clean-NEZ/noisy-EZ handling,
  EMA/current-model scores, evidence percentile reliability, ranking/core/delta
  penalties and different loss aggregation. None is copied into the primary loss.
- N5 `7dafafd1e5_n5_pu_rankcal.py`: patient OOF exclusion utility is relevant,
  but EZ-only reliability, focal PU, reliable-EZ ranking, center calibration and
  constrained thresholds are incompatible with this task and are not used.
- CalibRank `uncertainty_diagnostic.py`: post-hoc `4p(1-p)` confidence gate and
  corrected/spoiled decision summaries; no OOF MC-dropout supervision. Reuse the
  decision-change idea, not its test-derived tertiles or adapter.
- [BUNDL](https://github.com/deeksha-ms/BUNDL), README inspected 2026-10-09:
  uncertainty-aware Bayesian noisy-label learning with a KL-divergence-based loss
  for seizure detection. PR-UAS is not that loss and no equivalence is claimed.

The new experimental combination is strictly FIT-only four-way patient OOF
teachers with disjoint internal checkpoint-selection patients, frozen ten-pass
dropout mean/entropy, symmetric bounded target correction, and two matched
hard-label/fixed-smoothing controls. Entropy is a deterministic function of model
confidence, not an independently identified clinical label-error probability.

## Runtime and safety

Use the existing isolated NumPy1.26.4 runtime (unchanged Torch2.8 CUDA) to avoid
previous native NumPy2.2 metric crashes; do not alter the shared environment.
Fresh baseline replay and threshold-parity tests must pass there. Private feature
arrays, preprocessing, predictions, teacher identities, checkpoints, RNG and logs
stay on the server. Only code and aggregate artifacts may be published.
The user-prohibited extraction skill is not used. No cohort or seizure exclusions
from the raw E1/E3 experiments are carried over.
