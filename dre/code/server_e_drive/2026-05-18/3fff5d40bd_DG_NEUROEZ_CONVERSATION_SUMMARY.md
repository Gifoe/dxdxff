# DG-NeuroEZ Conversation Summary

## Current Task Definition

The project is now framed as patient-level NEZ-positive channel classification:

```text
NEZ = 1
EZ = 0
model score = p_NEZ(channel)
pred_NEZ = p_NEZ >= threshold_nez
pred_EZ = p_NEZ < threshold_nez
```

Patient-macro classification metrics are primary. EZ ranking metrics remain
secondary diagnostic metrics.

## Latest Experimental Conclusion

The previous broad ablation showed:

```text
Baseline is stable across seeds.
Graph-dropout-0.05 is the only plausible light structural candidate.
V4-light did not beat baseline-family on core classification metrics.
Weak CNN variants systematically degraded performance.
Patient adversarial and CORAL are not useful main-line modules.
```

Therefore, the next stage is not module stacking. It is calibrated
patient-relative graph classification.

## NeuroEZ-B Calibrated Pipeline

Retained pipeline:

```text
Patient-wise split
-> 30-second pre-onset + 30-second post-onset peri-onset context
-> loaded or prepared 30-second window cache
-> success_only patients
-> self-comparison window features: abs + delta + zdelta + ratio
-> mixed_abs_delta adjacency
-> graph-spectral encoder with adjacency message passing and channel attention
-> mean temporal pooling
-> cross-seizure MIL attention
-> patient-relative channel ranker
-> p_NEZ
-> threshold classification and EZ ranking evaluation
```

Current rerun scope:

```text
Only NeuroEZ-B baseline is rerun first.
Do not rerun GD005/loss/feature/graph ablations unless the 30+30s baseline needs further follow-up.
```

Scheduled baseline experiments:

```text
1 Baseline_seed42
2 Baseline_seed2024
3 Baseline_seed3407
```

Executable script:

```bash
bash run_neuroez_b_calibrated.sh
```

The script writes and refreshes:

```text
experiment_manifest.csv
combined_heldout_summary.csv
combined_key_metrics.csv
group_mean_std_key_metrics.csv
```

Training schedule:

```text
Default epochs = 80
Early stopping is not allowed until epoch > 50
Default patience = 6 after that minimum epoch gate
```

Cache behavior:

```bash
PREPOST_CONTEXT_SEC="${PREPOST_CONTEXT_SEC:-30.0}"
CACHE_PATH="${CACHE_PATH:-/root/nips-E/outputs_dgneuroez_nez_v3_small/cache/techez_prepost_dynamic_cache_v5_9cb761e87145.pkl}"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-${OUTPUT_ROOT}/_shared_window_cache}"
```

The script defaults to the existing 30-second cache. If `CACHE_PATH` is unset,
it can build or reuse a shared 30-second cache under `SHARED_CACHE_ROOT`.

Primary comparison:

```text
patient_macro_f1
patient_macro_balanced_accuracy
patient_macro_ez_f1
patient_macro_auroc_ez / patient_macro_auprc_ez
pooled_macro_f1
pooled_balanced_accuracy
ez_recall_at_true_count
ez_mrr
```
