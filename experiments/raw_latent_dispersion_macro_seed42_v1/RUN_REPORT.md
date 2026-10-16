# Learned Raw dispersion residual — completed internal experiment

The actual target remains patient-equal **Macro-F1>0.65**. The full-cohort
historical Raw-only reference is0.630493. This internal experiment did not
identify a candidate to replace that reference, and does not establish the
target. All numbers below were actually run and independently recomputed.

## Actual matched inner results

The first assessment path of each original ownFIT was used:86 patient
appearances from59 unique people, not86 independent people. Each arm used the
same roles, labels, masks, full patient channels and all genuine windows.
Original Development and ownOuter were not used by this internal experiment.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native frozen inner encoder | 0.601974 | 0.718595 | 0.517026 | 0.370762 | 0.833186 | 0.751410 | 0.738937 | 0.604651 |
| Mean fine-tune | 0.602684 | 0.718937 | 0.517541 | 0.371594 | 0.833773 | 0.752276 | 0.739422 | 0.604651 |
| Mean residual control | 0.602684 | 0.718915 | 0.517299 | 0.371594 | 0.833773 | 0.752276 | 0.737484 | 0.604651 |
| Dispersion residual | 0.601974 | 0.718595 | 0.517026 | 0.370762 | 0.833186 | 0.751410 | 0.738937 | 0.604651 |

The mean fine-tune minus Native Macro-F1 difference is+0.000710, with a
10,000-patient-cluster-bootstrap95% interval[-0.000299,0.001834]. The
same-parameter mean residual has the same F1 decisions. Dispersion minus Native
is exactly0 because allfive CAL selections returned to epoch0. Dispersion
minus the mean residual is-0.000710, interval[-0.001834,0.000299]. Repeated
appearances are resampled together by identity; channels/windows are not
independent biological replicates.

These inner means are **not** directly comparable to the0.630493 completeOuter
mean: the assessment patients, overlap pattern, encoder training sample size
and source checkpoints differ. These repeatedly used inner paths are
selection-conditioned, not independent discovery data.

## Selection and interpretation

All15 registered first-path end-to-end fine-tunes completed, using77 epochs.
The mean and mean-residual models selected epoch1 in fold1 and epoch0 in the
otherfour folds; dispersion selected epoch0 in allfive. CAL-only epoch and
threshold selection were fixed before fitting. No worse checkpoint was forced
into the report to make the added block appear different from Native.

The first-path candidate missed the registered+0.01 Macro-F1 advancement rule
in everyfold (best observed fold gain+0.003591). Therefore no extra assessment
paths were run. Allfive ownFIT choices were frozen to Native. No new original
Development fit or Outer evaluation has been run in this experiment. The
prospective final-stage specification in `PROTOCOL_LOCK.json` is not presented
as an executed result; no missing fullFIT/Outer cells are filled in.

The conclusion is narrow: this residual and warm fine-tuning/selection budget
did not produce a retained improvement. It does **not** prove that latent
dispersion contains no information: the selected models never retained a
nonzero dispersion residual. Nor does it prove that all Raw architectures
cannot exceed0.65. Reuse of CAL-selected initial checkpoints favors retaining
an already good calibration state and limits this warm-adaptation experiment.

## What was changed and what was controlled

The original learned24 convolution responses have30 positions in each500-sample
waveform. The dispersion block adds `0.25*tanh(W*SD)` to their mean, while the
capacity control adds `0.25*tanh(W*mean)`. Both W matrices have576 parameters,
no bias and zero initialization. Fine-tune and Native have12,529 parameters;
the two residual variants have13,105. Generic residual pooling is not claimed
novel. No handcrafted physiological attribute, frequency band, source/patient
ID, D1 score or independent classifier was introduced.

Pre-latent reference subtraction, two post-onset stages, available-seizure
mean/SD, complete-patient Z, original Raw normalization and one final classifier
were unchanged. Each complete patient had a full differentiable forward and
backward across all valid windows; no window sampling, detached temporal
pooling or truncated patient-Z gradient was used.

Source inspection confirms that the legacy crossfit constructors copied an
untrained CNN RNG template before their own TRAIN/CAL-only Raw fitting; they
did not load a fullFIT-trained CNN into an internal assessment path. The new
model directly constructs the pure Raw network, loads that path's selected
Raw state, and accepts only waveform, mask and original time inputs. Legacy
mixed metadata objects are physically unpickled to retain roles/identities/
ownFIT labels, but feature measurements never enter this model.

## Verification and scope

Four targeted source tests passed. The largest complete patient in the union
of79 actual training identities (164channels,48,380windows) completed direct
forward/backward for allthree variants, with finite full gradients, identical
initial logits and at most1,153,928,704 allocated GPU bytes. All15 original
CAL replays had maximum logit/probability error0 and unchanged201-grid decisions.
These are engineering admissions, not classification benefit evidence.

The independent verifier reconstructed all15 selected states, recalculated
CAL and HELD predictions, and checked344 patient metric rows across20 logical
cells. Maximum replay error was0, and independent sklearn patient metrics
matched to below1e-12. All original sources/checkpoints/private scores/logs and
the preceding fullOuter seal were preserved.

Terminal: `COMPLETE_VERIFIED_DISPERSION_INTERNAL`. Verification: `PASS`.
NewOuter passes:0. Goal achieved:**false**. The strongest actually verified
full-cohort Raw-only reference remains0.630493. The broader optimization goal
stays active; this negative internal experiment is not substituted for it.

`results/` contains actual anonymous per-cell/fold/source aggregates, training
histories, paired/absolute intervals, racing and selection records and audit
terminals. Sources with fewer thanfive distinct patients are suppressed in
outcome summaries. IDs, individual features/scores/labels, raw waveforms,
checkpoints and logs remain private. Any later formal test remains exploratory
repeated-test because historicalOuter outcomes have been repeatedly viewed.
