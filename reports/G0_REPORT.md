# Mini G0 trajectory audit

## Protocol

Only the outcome-blind SHA256 split's DEV partition (1055 cases) is summarized. The 264 G3 holdout IDs were frozen before label analysis and no holdout correctness or trajectory-type statistic was computed or used. Their raw `.pt` files physically contain outcome-bearing fields, so the information is accessible rather than cryptographically sealed. Progress uses `nearest-half-up over post-iteration states; step_number is 1-based and history_index is step_number - 1` over the official block-major 256-step history.

## Results

| checkpoint | n_total | mid_acc | final_acc | n_stable | n_rescue | n_corruption | n_unrecovered | rcr_joint | rsr_joint | corruption_given_current_correct | rescue_given_current_wrong | net_refinement_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.250000 | 1055 | 0.124171 | 0.814218 | 122 | 737 | 9 | 187 | 0.008531 | 0.698578 | 0.068702 | 0.797619 | 0.690047 |
| 0.400000 | 1055 | 0.176303 | 0.814218 | 177 | 682 | 9 | 187 | 0.008531 | 0.646445 | 0.048387 | 0.784810 | 0.637915 |
| 0.500000 | 1055 | 0.289100 | 0.814218 | 293 | 566 | 12 | 184 | 0.011374 | 0.536493 | 0.039344 | 0.754667 | 0.525118 |
| 0.600000 | 1055 | 0.423697 | 0.814218 | 431 | 428 | 16 | 180 | 0.015166 | 0.405687 | 0.035794 | 0.703947 | 0.390521 |
| 0.750000 | 1055 | 0.592417 | 0.814218 | 612 | 247 | 13 | 183 | 0.012322 | 0.234123 | 0.020800 | 0.574419 | 0.221801 |

For every row, `net_refinement_gain == RSR_joint - RCR_joint` to numerical tolerance. The preregistered checkpoint remains 0.50 for the strong gate.

## Gate decision

- Status: **G0_EXPLORATORY_PASS**
- Frozen G1 checkpoint: **0.6**
- Reason: The formal 50% strong gate failed. The already-selected 60% DEV checkpoint retained adjacent 50% support under parser v1.1; this is same-DEV, post-diagnostic, selection-biased exploratory evidence only.

This is not a strong pass unless the status explicitly says `G0_STRONG_PASS`. `G0_EXPLORATORY_PASS` permits the preexperiment to complete G1, but it is not eligible for an ordinary `READY_FOR_G2` cache. At the frozen exploratory checkpoint there are 16 reconstructed corruption cases; 13/16 have an explicit answer marker in both current and final decoded text. All evidence rows are retained in `reports/g0_corruption_evidence.csv`; no final-answer anchor was used.

The prompt authorized qualitative DEV-side selection for a continuous 40%–60%-area corruption region. The 60% point and operational support rule (joint RCR >=1% and >=10 corruptions at 50% and 60%) came from the parser-v1.0 DEV diagnostic and were frozen before this parser-v1.1 rerun. The checkpoint is not reselected on the new labels. This remains same-DEV, post-diagnostic and selection-biased, not independent validation or preregistration.

## Scientific limitation

These trajectories come from the public GSM8K test set. This is feasibility evidence, not a clean confirmatory benchmark protocol. A later paper must regenerate development trajectories outside its final evaluation set.
