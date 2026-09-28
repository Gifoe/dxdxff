# Unified A1-TF seed 42 — final report

| Model | Benchmark | Regime | AP | AUROC | Macro-F1 | MRR | Top1 |
|---|---|---|---:|---:|---:|---:|---:|
| Original A1 | Original | Ictal | 0.5767 | 0.7464 | 0.6208 | 0.7400 | 0.6548 |
| A1-TF | Original | Ictal | 0.5638 | 0.7341 | 0.6167 | 0.7332 | 0.6393 |
| A1-Interictal v2 | Omni | Interictal | 0.5505 | 0.6887 | 0.5937 | 0.6302 | 0.5682 |
| A1-TF | Omni | Interictal | 0.5636 | 0.7322 | 0.6158 | 0.6465 | 0.5909 |

AP means 65-cell × 20-repetition matched-query AP for Ictal, but patient-equal AP over positive-label patients for Omni. The two AP columns are not directly comparable.

## Verdict

**A1_TF_UNIFIED_VIABILITY_NOT_SUPPORTED.** Module topology is identical (67,403 parameters each), but the predeclared performance gates fail. Ictal A1-TF AP 0.563798 versus original A1 0.576743; paired ΔAP -0.012945, 95% CI [-0.029420, +0.005669]. The required ≥0.5717 AP is not met. MRR changes -0.006796; Top1 changes -0.015516, its CI [-0.063081, +0.026851].

Omni N1 versus N0: Macro-F1 0.615783 versus 0.593662; AUROC 0.732192 versus 0.688690; patient-equal AP 0.563646 versus 0.550466. The paired 10,000-draw CI for patient-equal AP Δ is [+0.002617, +0.025704]. The prespecified Macro-F1 ≥0.65 and AUROC ≥0.78 targets are not met. Macro-F1 ≥0.68/0.70 and AUROC ≥0.80 also fail.

## Mechanism and limitations

Omni trained alpha = -0.021178; alpha=0 gives Macro-F1 0.623855, AUROC 0.732075, patient-equal AP 0.562838. The full model does not consistently outperform its own alpha-zero intervention, so N1-over-N0 is not evidence of an isolated TF benefit. Ictal endpoint alphas range -0.027732 to +0.023749; endpoint validation alpha-zero AP changes have mixed signs. Those endpoint checkpoints are not the VLOO-selected checkpoints and cannot establish the selected I1 TF effect: training retained score snapshots but only the final epoch weights. No post-outcome retraining was done to fill that gap.

Descriptor reconstruction is mixed. Omni inner-validation descriptors mostly show positive correlation, but descriptor 6 has negative R². Several ictal descriptors have nearly constant targets and extremely negative or undefined R²; the TF branch has not reliably reconstructed all nine A1 descriptors. See the per-descriptor CSV.

Omni source strata are heterogeneous: source-sink performs best, OpenIEEG is substantially weaker, and Zurich has zero pathological channels, making ranking and positive-class metrics non-estimable. The main bottleneck is low pathological sensitivity and limited ranking discrimination, not just threshold placement.

The architecture topology, shared A1/TF modules, and zero-init fusion match. However the complete training protocols do **not** differ only in their physiological reference: Ictal starts from historical A1 and uses a teacher anchor; Omni trains from scratch without that anchor, as the supplied instructions also require. Consequently this is a same-topology two-benchmark study, not a controlled reference-operator-only intervention.

Ictal uses previously viewed historical fixed-query development targets; no new outer test was opened. Omni N1 checkpoint and threshold were frozen before its official test was accessed once, and test outcomes were not used to tune N1. Historical I0/N0 outcomes were already visible, so neither comparison is a fresh blind confirmation. There is no detected within-run test tuning, but literal absence of leakage cannot be proved from code alone.

## Audit index

See `outputs/` for primary metrics, paired bootstrap, stage gate and selection, threshold/intervention, reconstruction, label usage and test-access audits. Only compact CSV/JSON and code are published; no checkpoint, raw iEEG, TF cache, or private ictal patient-channel query rows are uploaded.
