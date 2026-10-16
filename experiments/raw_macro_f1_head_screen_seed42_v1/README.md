# Macro-F1-aligned Raw readout study

The target is patient-equal **Macro-F1>0.65**, following the user's correction,
not EZ-positive F1>0.65. The actual historical Raw-only reference is0.630493.
This bounded study first reuses genuine patient-isolated Raw encoders, rather
than fitting another grid of full waveform networks.

Completed: all120 internal logical cells,105 head fits, followed by the full
fivefold Native/Refit/Selected evaluation. Patient Macro-F1 is respectively
**0.630493 /0.627736 /0.624314**. None exceeds0.65. See
[the actual run report](RUN_REPORT.md) and anonymous `results/` artifacts.

Stage1 has15 inner assessment paths, eight fixed arms and105 trainable small
heads. Original Development and ownOuter are excluded. Encoders are frozen,
were trained only on their original internal TRAIN patients, and never used
CHOICE/HELD labels to fit or select their weights. The new readout fits
TRAIN+CHOICE; CAL selects epoch/threshold; HELD compares fixed readouts.

| Arm | Change | Interpretation |
| --- | --- | --- |
| NATIVE | Original head/checkpoint/threshold | Actual untouched frozen-encoder control |
| REFIT | Same head refit with Macro-aware loss | Additional readout fitting, not an architectural novelty |
| PATIENT | Learned whole-patient latent context modulates the same hidden classifier | Patient context beyond final featurewiseZ |
| LOCAL | Identical additional parameters without patient averaging | Capacity-matched control |
| NO_REFERENCE | Drop pre-latent subtraction | Frozen-encoder readout ablation |
| NO_PHASES | Single0–20 phase duplicated into the same slots | Frozen-encoder readout ablation |
| NO_SD | Drop between-seizureSD beyond mean | Frozen-encoder readout ablation |
| NO_Z | Drop final patientZ | Also changes numerical conditioning |

Every input is the complete-patient96D learned Raw summary, or its exact
source-bound one-factor pool deletion. No physiological descriptor, D1 output,
late0–30 extension, patient/source ID or artificial feature tensor is a model
input. Every readout has one final classifier. PATIENT/LOCAL share zero-output
conditioner initialization and parameter count; generic DeepSets/FiLM is not
claimed novel. The original160-input head is retained with64 literal zero slots
to preserve actual historical head initialization and numerical fidelity.

`PROTOCOL_LOCK.json` fixes all roles, loss, budget, own-FIT selection and final
fivefold evaluation rules before fitting. Internal reused overlapping paths
remain selection-conditioned. Four deletion rows are **not** end-to-end encoder
retraining. Earlier EZ-directed end-to-end deletions remain separate evidence.

After independent internal verification, each original FIT selects its own arm.
The registered final stage uses the actual fullFIT reference encoder, refits
only the readout on original FIT, and selects its epoch/threshold only on
original Development. All final Native/Refit/Selected states are frozen before
one joint Outer pass. No best model is selected or retrained from Outer.
Historical viewing makes the final result exploratory repeated-test.

## Execution

Only use the existing trusted-server caches; do not regenerate waveforms.
Keep the foreground SSH owner alive. An unused RunTag preserves logs.

```powershell
& D:\raw_macro_f1_head_screen_seed42_v1\run_bridge_v2.ps1 -RunTag 01 -StopAfter All
& D:\raw_macro_f1_head_screen_seed42_v1\run_final.ps1 -RunTag 01 -StopAfter Preparation
& D:\raw_macro_f1_head_screen_seed42_v1\run_final.ps1 -RunTag 02 -StopAfter All
& E:\Anaconda\python.exe D:\raw_macro_f1_head_screen_seed42_v1\verify_final.py
```

The original CPU-versus-CUDA absolute-logit admission failed before any fit.
`CPU_BRIDGE_AMENDMENT.json` records the diagnosis and explicit pre-fit change:
Native replay/verification now uses the original complete-patient CUDA FP32
arithmetic (all15 diagnosed paths were bit-exact), while new trainable heads
retain CPU FP32. The CPU bridge must preserve all201 calibration-grid decisions
and probability error<=1e-6. The failed CPU logit gate is not reported as passed.
The five original critical sources and original runtime freeze/logs remain
unchanged; the versioned wrapper uses a separate hash-bound runtime.

Completion of the internal screen is not completion of the0.65 goal. Only an
actually verified full80-patient frozen Raw procedure can establish that point
target. Code and anonymous summaries are published; raw, IDs, embeddings,
weights, scores, labels and logs remain private. No classification values are
filled before the corresponding run has completed and been verified.
