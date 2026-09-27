# Frozen A1 patient-conditioned low-rank adapter — development result

This is development-only and exploratory on a historically viewed cohort. No outer predictions or metrics were computed. The monolithic legacy cache initializer nevertheless materialized all 80 patient labels; therefore the literal no-outer-label-read rule was not met and this is not a strictly sealed confirmation.

The frozen A1 patient-attention layer builds R4 from all channel features before the adapter's label-blind context/query split; query features can therefore indirectly affect R4 context. Query labels do not enter the context encoder. See `IMPLEMENTATION_AUDIT.md`.

A1 replay: 150/150 checkpoints, maximum grid error 0; exact R4 classifier-input dimension 64 obtained by hook because ordinary A1 forward does not expose the contextual embedding key.

## VLOO patient-equal results

| Variant | EZ-AP | EZ-MRR | Top1-EZ | EZ-AUROC | NDCG | Macro-F1 | EZ-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.5572 | 0.7421 | 0.6769 | 0.7431 | 0.7757 | 0.6260 | 0.4143 |
| P1 | 0.5576 | 0.7583 | 0.6923 | 0.7415 | 0.7792 | 0.6313 | 0.4242 |
| P2 | 0.5497 | 0.7614 | 0.6923 | 0.7368 | 0.7749 | 0.6161 | 0.3981 |
| P3 | 0.5522 | 0.7475 | 0.6769 | 0.7381 | 0.7742 | 0.6184 | 0.4018 |

## Interpretation against the registered questions

- P0 exactly reproduces frozen A1. P1's global low-rank capacity gives only +0.0004 EZ-AP over P0; its Macro-F1 rises by +0.0053, but this is not a meaningful gain on the primary ranking metric.
- Patient conditioning does not beat either control: P2-P0 EZ-AP is -0.0075 (paired patient bootstrap 95% CI [-0.0172, +0.0016]); P2-P1 is -0.0079 ([-0.0174, +0.0010]). P2 is AP-positive in only 1/5 folds against either control. P3-P0 and P3-P1 are also negative on mean EZ-AP.
- P2 improves EZ-MRR by +0.0193 and Top1-EZ by +0.0154 against P0, but loses EZ-AUPRC, Macro-F1 (-0.0099) and EZ-F1 (-0.0162); therefore the improvement is not a coherent ranking-and-decision gain. P3's MRR gain over P0 is only +0.0054, with no Top1 gain and lower Macro-F1/EZ-F1.
- Wrong-patient context does **not** remove an advantage: correct versus shuffled EZ-AP is 0.5497 versus 0.5516 for P2, and 0.5522 versus 0.5548 for P3. The shuffled-context diagnostic was never used for selection.
- Adapter coefficients are numerically patient-varying (maximum per-component standard deviation 0.616 for P2 and 0.624 for P3), not a collapsed constant. That variation does not establish useful conditioning, because correct context loses to wrong context on EZ-AP.
- With 50% or 70% unlabeled channels, generated coefficient vectors remain moderately to strongly aligned with full context, and aggregate ranking metrics remain close; see the per-fold context-robustness CSVs. This is robustness of an ineffective mechanism, not evidence of benefit.
- FIT-only geometry supervision adds +0.0024 EZ-AP to P2 (95% CI [-0.0062, +0.0124]), but lowers EZ-MRR by -0.0139 and Top1-EZ by -0.0154. It fails its separate fixed gate. No validation labels were used to fit the teacher.
- The prior patient-specific oracle demonstrates that patient-specific EZ coordinates can exist; this experiment shows that this particular deployable zero-label context adapter does **not** reliably infer useful coordinates. It neither refutes the oracle nor validates the proposed transfer mechanism.

All five prespecified AP contrasts, fold signs, controls, parameter diagnostics, and 10,000-resample patient-level paired bootstrap intervals are available in the compact CSV/JSON outputs. Neither fixed success gate passes. Do not run an outer test or start another adapter search.

Terminals: `PATIENT_CONDITIONED_COORDINATE_ADAPTATION_NOT_SUPPORTED`; `GEOMETRY_TEACHER_NOT_SUPPORTED`.

STOP_PATIENT_CONDITIONED_ADAPTER_EXPLORATION
