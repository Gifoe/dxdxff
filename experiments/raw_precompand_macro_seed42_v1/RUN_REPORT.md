# Raw pre-reference waveform view fusion — actual complete results

## Outcome against the real target

The target is **patient-equal Macro-F1>0.65**, not EZ-positive,pooled,weighted
or best-fold F1. This study did **not** reach it. The fixed precompanded-view
procedure is0.631432, versus the exact historical Raw-only reference0.630493;
its paired Macro-F1 interval crosseszero. The internally selected procedure
is0.629086. Ranking gains must not be presented as achieving the F1 goal.

All values below are from actual server runs and independent verification.
Allfour procedures were named before final evaluation and are reported,
including the negative capacity control and the negative selected policy.

## Complete fivefold results

Same80patients,7,635channels,256seizures and1,471,965valid500-sample Raw250
windows. These are **patient means**; pooled and fold-equal summaries are
separate artifacts and are not substituted for the target.

| Procedure | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ fraction bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native Raw-only | 0.630493 | 0.741296 | 0.550114 | 0.426475 | 0.834511 | 0.757014 | 0.784454 | 0.700000 | -0.041381 |
| Same-view residual control | 0.628211 | 0.742186 | 0.553158 | 0.418317 | 0.838104 | 0.760437 | 0.785964 | 0.700000 | -0.055103 |
| Precompanded waveform residual | 0.631432 | 0.759384 | 0.570215 | 0.420221 | 0.842644 | 0.766018 | 0.802328 | 0.725000 | -0.066138 |
| Own-FIT-selected procedure | 0.629086 | 0.752208 | 0.555576 | 0.417572 | 0.840600 | 0.762843 | 0.795123 | 0.712500 | -0.062857 |

Nominal paired10,000-patient-bootstrap differences:

| Comparison | Metric | Difference | 95% interval |
| --- | --- | ---: | --- |
| Precompand − Native | Macro-F1 | +0.000939 | [-0.009937,0.012318] |
| Precompand − Native | AUROC | +0.018088 | [0.006123,0.031208] |
| Precompand − Native | AP | +0.020102 | [0.007053,0.034144] |
| Precompand − Native | EZ-F1 | -0.006254 | [-0.024802,0.012792] |
| Precompand − same-view control | Macro-F1 | +0.003222 | [-0.006749,0.013313] |
| Precompand − same-view control | AUROC | +0.017198 | [0.005455,0.030441] |
| Precompand − same-view control | AP | +0.017058 | [0.004181,0.030700] |
| Selected − Native | Macro-F1 | -0.001407 | [-0.009943,0.007347] |

Outer outcomes have been viewed repeatedly across the project. These are
**exploratory repeated-test comparisons**, not independent confirmation or
clinical-efficacy evidence. Intervals are descriptive and do not undo that
selection history or adjust for all historical experimentation.

## Internal evidence and frozen selection

All15 original inner paths were evaluated:255 assessment appearances from79
unique patients, not255 independent people. Each ownFIT selected only from
its own three HELD paths; no global best from overlapping outerfolds was used.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | MRR | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 0.609084 | 0.726108 | 0.536426 | 0.397296 | 0.820872 | 0.755360 | 0.647059 |
| Same-view control | 0.608187 | 0.726757 | 0.536841 | 0.392959 | 0.823416 | 0.759474 | 0.658824 |
| Precompanded view | 0.609149 | 0.733456 | 0.551183 | 0.395136 | 0.823163 | 0.770663 | 0.674510 |

The internal Macro difference for Precompand−Native was+0.000065, interval
[-0.008830,0.009480]; AP difference+0.014757, interval[0.001063,0.029497].
Against the capacity control, the AP interval was[0.000103,0.029145]. These
previously used paths remain selection-conditioned. Repeated appearances
were carried together by patient-ID-cluster bootstrap.

The fixed source-stability/complexity rule chose Native in folds1,2,3 and
Precompand in folds4,5. Those choices were frozen before original Development
was used. Both added fixed arms and this selected policy were then evaluated;
the policy was not replaced by a better-looking Outer arm afterward.

## Model and controls

Inputs remain only original waveform,valid mask and original times. The
first view is the historical per-window centered/SD-normalized/clipped signal.
The second centers raw samples, uses one common **pre-only** scale across all
available channels within each seizure, then fixed asinh companding and clip8.
The scale uses the lower median of positive per-channel pre RMS values
(`torch.median`). No RMS,variance,bandpower,ratio or source/patient ID is
appended to the classifier. This is waveform normalization, not a separate
physiological-feature predictor or an acquisition-provenance repair.

Both actual waveform views pass through the **same frozen CNN**. After the
original pre-latent subtraction and two phase means,the second neural summary
is projected into the first by `Linear(other,0.25*tanh(M))`, M24×24,zero
initialized. The control projects the original view instead. Both therefore
have576 fitted parameters and13,105 total model parameters, versus12,529
Native. The original CNN and one classifier remain unchanged. No independent
D1 branch,auxiliary classifier,physiological descriptor or new waveform
extraction was used. Generic multi-view/residual fusion is not claimed novel.

The same available-seizure mean/SD and complete-patient float64 Z follow
fusion. Their derivatives through M are retained. The caches contain frozen
learned neural phase summaries, not sampled windows or detached trainable
encoder outputs; the CNN is intentionally frozen rather than claimed to be
end-to-end updated. The historical64 unused classifier slots are literalzero.

## Actual execution and isolation

- Internal:30 matrix fits/1,227 epochs,all765 ownFIT patient-encoder caches.
  Existing bodies originally used their TRAIN/CAL only. New M used
  TRAIN+CHOICE for loss,CAL for epoch/threshold,HELD for ownFIT selection.
  Original Development/ownOuter were not used in this internal stage.
- Final:10 matrix fits/554 epochs,original FIT loss and Development-only
  epoch/threshold. The actual historical fullFIT Raw-only body/head is the
  warm reference per fold. Reuse of Development for original body selection
  and new matrix selection is not independent validation.
- All20 final logical model/threshold rows,including exact aliases,were
  hash-frozen before one joint final prediction pass. All20 score cells were
  sealed before reading the fullOuter truth archive. No model,threshold,
  normalization or selection rule was changed from Outer results.
- Native was replayed against the original sealed probabilities with unchanged
  decisions,then the exact historical Native scores were reused. No fresh
  baseline was selected to make the comparison easier.

The actual body/head hashes stayed unchanged throughout matrix fitting.
Initial identity,matchedcapacity,gain/pre-only normalization/zero-pre behavior,
channel-order/padding and raw/cache-gradient checks passed. Actual largest
training-patient raw/cache logits,loss and M gradients were identical. All765
ownFIT Native logit replays and all201 CAL-grid decisions met the fixed gate.

Independent final verification checked model/body/matrix/threshold hashes,
selected aliases,archive identities and ownOuter role separation,then
recomputed allnine patient metrics over320 rows. Maximum metric difference
was below1e-12. A separate fresh read verified all80 actual Raw banks/masks:
80patients/7,635channels/256seizures/1,471,965validwindows. No patient/channel
was selected away for band quality or model performance. There were no zero-pre
channel-seizure units or fallback seizures; only28 of735,982,500 normalized
samples clipped. Existing cache/filter and clinical-label limitations remain.

## Engineering incident and deliverables

One initial source-transfer authentication response timeout was recovered by
reconnecting,verifying the already transferred file hashes and sending only
missing files. Final fitting then once failed before any parameter update in
`python313.dll` while reading old metadata. A fresh independent metadata read
and hashes passed; one identical-command recovery with new logs completed.
Root cause is not proven. No code/data/runtime/scientific gate changed and no
completed epoch was discarded. Original logs and allprivate artifacts remain;
see `EXECUTION_AUDIT.json` for the exact scope.

`results/` contains allinternal cells,histories,selection,source/fold summaries
and paired intervals. `results/final_stage_v1/` contains allfour complete
procedures,patient means,pooled results,fold/source/stability/coverage,paired
and absolute intervals,and independent model/metric/Raw coverage verification.
Sources with fewer thanfive distinct patients are suppressed in outcome
summaries. Raw,IDs,labels,weights,individual features/scores and logs stay on
the trusted server; only code,hashes and anonymous summaries are public.

Final terminal: `COMPLETE_VERIFIED_FOUR_PRECOMPAND_PROCEDURES`.
Independent model/metric and fresh Raw-coverage verifications: `PASS`.
**Goal established:false**. The0.65 target is not met and the goal staysactive.

The internal ranking-versus-classification mismatch warrants a separately
registered investigation of patient-level decision calibration,not more
post-hoc threshold changes to this frozen study. That is a next hypothesis,
not an experiment or improvement claimed in this report.
