# Raw pre-state retention and contrast, Macro-F1, seed42

Prospective bounded internal experiment. **No results have yet been obtained.**
The full80 patient-equal Macro-F1 target remains >0.65, not an EZ-F1, pooled,
internal or confidence-upper-bound target. The preceding complete joint-moment
experiment failed it; its report and all negative controls remain available.

The question is whether pre-latent subtraction discards useful learned Raw
baseline state. Subtraction's earlier conditional benefit does not answer
this. Historical long-CNN context experiments conditioned on post-Z patient
vectors; they did not retain the original pre-state and were EZ-oriented.
This experiment claims neither a new physiological measurement nor novelty.

One shared CNN and one head compare:

| Arm | Classifier information before complete-patient Z |
|---|---|
| DELTA_DUP | Two copies of phase-specific post-minus-pre mean/SD |
| POST_DUP | Two copies of unreferenced post-phase mean/SD |
| PRE_DELTA | Post-minus-pre mean/SD, plus two copies of pre-state mean/SD |

All three have 14,577 parameter positions and the same exact random initial
CNN/head state in each fold. Duplicate controls match shape/parameters, not
effective information capacity. No trained old weights, D1 features/logits,
manual physiological predictors or independent branch enters the models.
Pre-only waveform normalization is unchanged even for POST_DUP; that arm
does not remove all use of pre-onset data. All valid Raw windows are encoded;
the fixed pre/0to20 aggregation support is unchanged.

The original ownFIT TRAIN/CAL/CHOICE/HELD roles are reused. Three genuinely
fresh fits run in each of five first-path assessments. A candidate must beat
all three fixed references, including the stronger preceding compact PRE_MEAN,
before its ownFIT receives the two further replication paths. At most45fits.
Only legal original ownFIT labels are used; no Development or new Outer run
is exposed by this program. Strong-reference replication cells that the old
experiment never ran are marked absent, not fabricated or unequally averaged.

Final eligibility is stricter than the preceding experiment: within each
ownFIT, a fresh arm needs >=0.015 replicated Macro gain, nominal paired lower
bound >0 against both complete fixed references, improvement in at least two
paths and the fixed AUROC safety condition. It must individually outperform
the stronger compact PRE first path too. Procedure selection stays within
the same ownFIT; no global winner is chosen from overlapping folds. Exact
rules are locked in PROTOCOL_LOCK.json. If no fresh arm qualifies, this
experiment stops after internal reporting without another Outer evaluation.

Five changed-path tests and actual largest legal full-patient backward
admission are required. `run_internal.ps1` needs a sustained foreground SSH
owner, uses fresh log names and preserves atomic model/optimizer/RNG state.
It reuses the byte-pinned prior fit implementation under current model/role/
runtime modules; legacy JOINT text in that engine's stdout is not model identity.
Nonzero worker exit stops the pipeline; a written PASS is not a normal exit.
Native failures must be diagnosed before skipping a completed stage or resuming.

The independent finalizer replays retained CAL/HELD checkpoints, checks all
201 threshold decisions, all nine patient metrics, public cell summaries,
all162 complete-first-path interval rows and replication gate intervals.
Complete first-path results (86 appearances/59 unique people) stay separate
from the selected replication subset and the uncomputed full80 test.

Only code, hashes and anonymous aggregates will be pushed to GitHub. Raw,
IDs, truth arrays, weights, individual scores/latents and runtime logs remain
private. All eventual Outer results are exploratory repeated-test. This
protocol is technical execution documentation, not a claim of model success.

## Actual prebinding check, 2026-10-17

The deployed public source manifest passed. All five unique changed-path CPU
tests passed (pytest:5 passed in1.14s). No formal fit or classification result
is implied. The real complete-patient GPU admission is still required before
formal training. The protocol/source are pushed before that formal training.

## Actual admission and training snapshot

The protocol/source commit was `aad65d31c5ca682d8e56db81037223e0a12c9fea`,
verified at the GitHub remote before formal fitting. The real complete legal
training-patient admission then passed all three networks:164channels,
48,380validwindows,14,577parameters,finite all-parameter gradients and
nonzero CNN/head gradients. All five fold initial-state matches passed.
Admission made zero optimizer updates; its actual anonymous artifact is
[ADMISSION.json](results/internal_v1/ADMISSION.json).

At09:14Asia/Shanghai on2026-10-17, the first DELTA_DUP internal fit was active
through epoch9, approximately1.8seconds/epoch, empty error log. This is a dated
snapshot, not an end-to-end timing estimate or an internal/final result.
The original Development/Outer were not used. No new performance claim is made.

The older four-component evidence ledger remains
[MECHANISM_EVIDENCE.md](../raw_temporal_context_macro_seed42_v1/MECHANISM_EVIDENCE.md).
Those frozen-encoder/readout deletions are conditional evidence, not proofs
that all four components are universally necessary in retrained CNNs. In
particular, the cross-seizure SD benefit was not established. This new
from-scratch POST_DUP comparison addresses latent-reference subtraction under
the Macro-oriented protocol; it does not silently replace the older evidence.
