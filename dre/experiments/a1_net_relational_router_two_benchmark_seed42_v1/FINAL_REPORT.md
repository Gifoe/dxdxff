| Model | Benchmark | AP | AUROC | Macro-F1 | MRR | Top1 |
|---|---|---:|---:|---:|---:|---:|
| A1 (historical) | Ictal | 0.576743 | 0.746382 | 0.620810 | 0.740038 | 0.654771 |
| A1-NET | Ictal | — | — | — | — | — |
| A1 (historical) | Omni | 0.550466* | 0.688690 | 0.593662 | — | — |
| Official-style Raw CNN | Omni | — | 0.798767 | 0.599267 test-Youden | — | — |
| Raw + Patient Context | Omni | — | — | — | — | — |
| A1-NET | Omni | — | — | — | — | — |

# A1-NET seed42: stopped at the frozen raw-encoder gate

*The historical Omni A1 AP is patient-level, whereas the official CNN
reproduction AUROC/F1 below are labeled EDF-channel-level. They are not
like-for-like comparisons.*

**Exact terminal: `RAW_ENCODER_REPRODUCTION_FAILED`.** The published Omni
TimeConv-CNN Task-2 channel result was AUROC 0.8061 and macro-F1 0.6469.
Our frozen checkpoint obtained AUROC **0.798767** but official-style
test-Youden macro-F1 **0.599267** on 8,104 labeled EDF-channel pairs. The
predeclared near-public gate required both AUROC ≥0.7761 and macro-F1
≥0.6169. It therefore failed on F1. A separate fixed-0.5-threshold
diagnostic was F1 0.659754; substituting that threshold after observing the
test would invalidate the locked gate, so it was **not** used to pass.

The official evaluation code flips both normal/pathological label and score
orientation and selects Youden on the test ROC; our scoring uses the same
rule. The pinned feature extractors default to 300 Hz while the paper/CNN
config specify 1000 Hz; the train CLI defaults to five epochs while the paper
describes ten. We froze the paper/config's 1000-Hz, ten-epoch interpretation
before test access. These source inconsistencies preclude a claim of exact
paper reproduction, but do not license a post-test rerun under a new
frequency or threshold. The native-signal reconstruction, train/test
membership, source/model hashes, complete extraction, and selected
checkpoint are documented in `RAW_ENCODER_AUDIT.json` and
`OMNI_CNN_REPRODUCTION.md`.

The high sample-level internal validation F1 is not independent evidence of
patient generalization. The official test here is an **exploratory
repeated-test benchmark; not a fresh blind confirmation**, because its
outcomes were viewed in prior experiments. It was not used to tune the model.

Under the user-specified hard stop, **no AEC graph, patient-relative head,
OOF router, ictal Raw-Net or A1-NET was trained/evaluated**. Therefore the
requested graph-value, router-gate distribution, center-preference and
two-benchmark success questions have no results; inventing them would be
misleading. The strongest supported claim is only that this frozen
official-style CNN achieved near-published AUROC while failing the locked
Youden-F1 reproduction criterion. Do not continue this A1-NET route under
the current preregistered experiment.
