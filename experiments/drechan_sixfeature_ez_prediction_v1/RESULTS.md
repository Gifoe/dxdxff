# Experiment B results: physiological feature predictive informativeness

Completed: 14 configurations, 80 patients, 7,635 channels (1,743 EZ / 5,892 NEZ), 256 seizures. All original five-fold roles were retained. There were 1,410 unique formal LR fits: 840 inner-CV candidates, 70 selected final models and 500 models across 100 label-permutation repetitions. No historical classifier model/score was reused.

## Main eight configurations

Values are equal-patient means [nominal 95% patient-bootstrap CI, 10,000 seed42 draws]. Every row covers all 80 patients. C was selected inside FIT by patient-held-out AUROC; CAL selected the Macro-F1 threshold. Models were not refitted on FIT+CAL.

| Input | AUROC | AP | Macro-F1 | Top-1 |
|---|---:|---:|---:|---:|
| Beta (+PR) | 0.723 [0.683, 0.761] | 0.518 [0.462, 0.573] | 0.585 [0.556, 0.614] | 0.625 [0.512, 0.725] |
| Low Gamma (+PR) | 0.713 [0.668, 0.754] | 0.499 [0.443, 0.557] | 0.581 [0.547, 0.615] | 0.562 [0.450, 0.675] |
| Partial High Gamma (+PR) | 0.719 [0.681, 0.758] | 0.495 [0.438, 0.553] | 0.535 [0.500, 0.569] | 0.500 [0.388, 0.613] |
| RMS (+PR) | 0.663 [0.617, 0.709] | 0.479 [0.427, 0.531] | 0.581 [0.548, 0.614] | 0.588 [0.475, 0.700] |
| Line Length (+PR) | 0.748 [0.707, 0.787] | 0.549 [0.494, 0.604] | 0.631 [0.602, 0.661] | 0.688 [0.588, 0.787] |
| Spectral Entropy (+PR) | 0.720 [0.678, 0.761] | 0.521 [0.464, 0.578] | 0.611 [0.580, 0.642] | 0.588 [0.475, 0.700] |
| All Six (w/o PR) | 0.722 [0.679, 0.764] | 0.520 [0.464, 0.575] | 0.482 [0.439, 0.525] | 0.613 [0.500, 0.725] |
| All Six (+PR) | 0.757 [0.718, 0.793] | 0.554 [0.501, 0.607] | 0.626 [0.596, 0.654] | 0.662 [0.562, 0.762] |

Within the eight prespecified main rows, Full+PR has the highest AUROC/AP point estimates, but Line Length+PR has the highest Macro-F1 and Top-1. All four Full+PR-minus-Line-Length+PR intervals include zero, even before multiplicity correction. These results do not establish that the joint feature set outperforms the strongest single-feature control. Across all 14 configurations, supplementary Line Length NoPR has a numerically higher AUROC (0.760) than Full+PR (0.757); it is not substituted for a prespecified main control.

Secondary endpoints:

| Input | EZ-F1 | NEZ-F1 | Accuracy | MRR |
|---|---:|---:|---:|---:|
| Beta (+PR) | 0.340 [0.294, 0.386] | 0.830 [0.803, 0.855] | 0.747 [0.716, 0.777] | 0.740 [0.661, 0.817] |
| Low Gamma (+PR) | 0.335 [0.281, 0.390] | 0.827 [0.800, 0.852] | 0.741 [0.708, 0.772] | 0.674 [0.589, 0.759] |
| Partial High Gamma (+PR) | 0.292 [0.240, 0.348] | 0.777 [0.734, 0.816] | 0.703 [0.663, 0.739] | 0.628 [0.540, 0.712] |
| RMS (+PR) | 0.372 [0.325, 0.418] | 0.791 [0.755, 0.823] | 0.709 [0.672, 0.743] | 0.689 [0.603, 0.774] |
| Line Length (+PR) | 0.430 [0.384, 0.478] | 0.833 [0.806, 0.857] | 0.757 [0.730, 0.784] | 0.770 [0.690, 0.845] |
| Spectral Entropy (+PR) | 0.404 [0.351, 0.456] | 0.818 [0.789, 0.845] | 0.745 [0.714, 0.776] | 0.692 [0.606, 0.775] |
| All Six (w/o PR) | 0.349 [0.293, 0.406] | 0.615 [0.537, 0.688] | 0.627 [0.571, 0.680] | 0.722 [0.639, 0.802] |
| All Six (+PR) | 0.422 [0.375, 0.469] | 0.830 [0.802, 0.854] | 0.757 [0.728, 0.783] | 0.753 [0.672, 0.830] |

## Supplement: every individual feature with and without PR

| Input | AUROC | AP | Macro-F1 | Top-1 |
|---|---:|---:|---:|---:|
| Beta (+PR) | 0.723 [0.683, 0.761] | 0.518 [0.462, 0.573] | 0.585 [0.556, 0.614] | 0.625 [0.512, 0.725] |
| Low Gamma (+PR) | 0.713 [0.668, 0.754] | 0.499 [0.443, 0.557] | 0.581 [0.547, 0.615] | 0.562 [0.450, 0.675] |
| Partial High Gamma (+PR) | 0.719 [0.681, 0.758] | 0.495 [0.438, 0.553] | 0.535 [0.500, 0.569] | 0.500 [0.388, 0.613] |
| RMS (+PR) | 0.663 [0.617, 0.709] | 0.479 [0.427, 0.531] | 0.581 [0.548, 0.614] | 0.588 [0.475, 0.700] |
| Line Length (+PR) | 0.748 [0.707, 0.787] | 0.549 [0.494, 0.604] | 0.631 [0.602, 0.661] | 0.688 [0.588, 0.787] |
| Spectral Entropy (+PR) | 0.720 [0.678, 0.761] | 0.521 [0.464, 0.578] | 0.611 [0.580, 0.642] | 0.588 [0.475, 0.700] |
| Beta (w/o PR) | 0.714 [0.673, 0.754] | 0.517 [0.461, 0.572] | 0.485 [0.444, 0.526] | 0.625 [0.512, 0.725] |
| Low Gamma (w/o PR) | 0.699 [0.650, 0.745] | 0.507 [0.449, 0.567] | 0.445 [0.410, 0.481] | 0.562 [0.450, 0.675] |
| Partial High Gamma (w/o PR) | 0.615 [0.560, 0.669] | 0.438 [0.379, 0.499] | 0.425 [0.403, 0.446] | 0.487 [0.375, 0.600] |
| RMS (w/o PR) | 0.660 [0.612, 0.706] | 0.473 [0.423, 0.525] | 0.461 [0.425, 0.498] | 0.562 [0.450, 0.675] |
| Line Length (w/o PR) | 0.760 [0.721, 0.796] | 0.550 [0.495, 0.605] | 0.462 [0.418, 0.506] | 0.650 [0.550, 0.750] |
| Spectral Entropy (w/o PR) | 0.718 [0.672, 0.761] | 0.518 [0.459, 0.576] | 0.471 [0.425, 0.519] | 0.588 [0.475, 0.700] |

| Input | EZ-F1 | NEZ-F1 | Accuracy | MRR |
|---|---:|---:|---:|---:|
| Beta (+PR) | 0.340 [0.294, 0.386] | 0.830 [0.803, 0.855] | 0.747 [0.716, 0.777] | 0.740 [0.661, 0.817] |
| Low Gamma (+PR) | 0.335 [0.281, 0.390] | 0.827 [0.800, 0.852] | 0.741 [0.708, 0.772] | 0.674 [0.589, 0.759] |
| Partial High Gamma (+PR) | 0.292 [0.240, 0.348] | 0.777 [0.734, 0.816] | 0.703 [0.663, 0.739] | 0.628 [0.540, 0.712] |
| RMS (+PR) | 0.372 [0.325, 0.418] | 0.791 [0.755, 0.823] | 0.709 [0.672, 0.743] | 0.689 [0.603, 0.774] |
| Line Length (+PR) | 0.430 [0.384, 0.478] | 0.833 [0.806, 0.857] | 0.757 [0.730, 0.784] | 0.770 [0.690, 0.845] |
| Spectral Entropy (+PR) | 0.404 [0.351, 0.456] | 0.818 [0.789, 0.845] | 0.745 [0.714, 0.776] | 0.692 [0.606, 0.775] |
| Beta (w/o PR) | 0.266 [0.210, 0.326] | 0.703 [0.634, 0.765] | 0.668 [0.614, 0.719] | 0.731 [0.648, 0.809] |
| Low Gamma (w/o PR) | 0.170 [0.120, 0.222] | 0.721 [0.653, 0.783] | 0.672 [0.616, 0.724] | 0.675 [0.588, 0.761] |
| Partial High Gamma (w/o PR) | 0.036 [0.014, 0.062] | 0.813 [0.767, 0.853] | 0.729 [0.684, 0.772] | 0.604 [0.514, 0.695] |
| RMS (w/o PR) | 0.264 [0.215, 0.317] | 0.658 [0.588, 0.725] | 0.634 [0.580, 0.684] | 0.676 [0.589, 0.759] |
| Line Length (w/o PR) | 0.302 [0.247, 0.360] | 0.623 [0.541, 0.698] | 0.627 [0.568, 0.683] | 0.753 [0.674, 0.829] |
| Spectral Entropy (w/o PR) | 0.359 [0.305, 0.417] | 0.583 [0.504, 0.661] | 0.602 [0.543, 0.659] | 0.691 [0.603, 0.775] |

## Seven prespecified Full+PR-minus-control comparisons

Differences and interval limits are absolute percentage points. Ordinary intervals are paired 95% percentile intervals. The four primary endpoints across all seven contrasts form one 28-comparison family; additional Bonferroni percentile intervals use confidence level 1−0.05/28. Secondary endpoints have nominal intervals only. The extreme adjusted limits use approximately nine bootstrap order statistics, so their Monte Carlo precision is limited. No comparator was selected after viewing these results.

| Control | Endpoint | Difference (pp) | Nominal 95% CI (pp) | 28-family interval (pp) |
|---|---|---:|---:|---:|
| Beta (+PR) | AUROC | +3.35 | [0.44, 6.59] | [-1.13, 8.75] |
| Beta (+PR) | AP | +3.60 | [0.53, 6.73] | [-1.03, 8.64] |
| Beta (+PR) | Macro_F1 | +4.08 | [1.58, 6.67] | [-0.10, 8.36] |
| Beta (+PR) | EZ_F1 | +8.18 | [4.31, 12.07] | Not in primary family |
| Beta (+PR) | NEZ_F1 | -0.03 | [-1.83, 1.88] | Not in primary family |
| Beta (+PR) | Accuracy | +0.96 | [-1.24, 3.31] | Not in primary family |
| Beta (+PR) | MRR | +1.33 | [-5.02, 7.57] | Not in primary family |
| Beta (+PR) | Top_1 | +3.75 | [-6.25, 13.75] | [-13.75, 20.00] |
| Low Gamma (+PR) | AUROC | +4.43 | [1.02, 8.17] | [-0.93, 10.50] |
| Low Gamma (+PR) | AP | +5.53 | [2.52, 8.74] | [0.85, 10.60] |
| Low Gamma (+PR) | Macro_F1 | +4.47 | [1.67, 7.25] | [0.27, 9.15] |
| Low Gamma (+PR) | EZ_F1 | +8.67 | [4.15, 13.15] | Not in primary family |
| Low Gamma (+PR) | NEZ_F1 | +0.27 | [-1.57, 2.31] | Not in primary family |
| Low Gamma (+PR) | Accuracy | +1.53 | [-0.72, 4.03] | Not in primary family |
| Low Gamma (+PR) | MRR | +7.92 | [1.16, 14.83] | Not in primary family |
| Low Gamma (+PR) | Top_1 | +10.00 | [0.00, 20.00] | [-5.00, 26.25] |
| Partial High Gamma (+PR) | AUROC | +3.78 | [0.85, 6.70] | [-1.17, 8.44] |
| Partial High Gamma (+PR) | AP | +5.91 | [2.28, 9.59] | [0.30, 11.61] |
| Partial High Gamma (+PR) | Macro_F1 | +9.09 | [6.52, 11.76] | [5.02, 13.61] |
| Partial High Gamma (+PR) | EZ_F1 | +12.96 | [9.30, 16.65] | Not in primary family |
| Partial High Gamma (+PR) | NEZ_F1 | +5.23 | [2.06, 8.87] | Not in primary family |
| Partial High Gamma (+PR) | Accuracy | +5.37 | [2.41, 8.79] | Not in primary family |
| Partial High Gamma (+PR) | MRR | +12.53 | [5.53, 19.96] | Not in primary family |
| Partial High Gamma (+PR) | Top_1 | +16.25 | [7.50, 26.25] | [2.41, 31.34] |
| RMS (+PR) | AUROC | +9.36 | [5.60, 13.40] | [3.74, 16.29] |
| RMS (+PR) | AP | +7.54 | [3.91, 11.27] | [2.27, 13.78] |
| RMS (+PR) | Macro_F1 | +4.43 | [1.61, 7.40] | [0.11, 9.46] |
| RMS (+PR) | EZ_F1 | +4.98 | [0.58, 9.42] | Not in primary family |
| RMS (+PR) | NEZ_F1 | +3.89 | [1.69, 6.48] | Not in primary family |
| RMS (+PR) | Accuracy | +4.79 | [2.15, 7.74] | Not in primary family |
| RMS (+PR) | MRR | +6.43 | [-1.27, 14.11] | Not in primary family |
| RMS (+PR) | Top_1 | +7.50 | [-3.75, 18.75] | [-8.75, 25.00] |
| Line Length (+PR) | AUROC | +0.88 | [-0.89, 2.94] | [-1.71, 4.30] |
| Line Length (+PR) | AP | +0.52 | [-1.21, 2.32] | [-2.09, 3.41] |
| Line Length (+PR) | Macro_F1 | -0.58 | [-2.51, 1.35] | [-3.67, 2.32] |
| Line Length (+PR) | EZ_F1 | -0.83 | [-4.13, 2.32] | Not in primary family |
| Line Length (+PR) | NEZ_F1 | -0.33 | [-1.37, 0.73] | Not in primary family |
| Line Length (+PR) | Accuracy | -0.08 | [-1.48, 1.40] | Not in primary family |
| Line Length (+PR) | MRR | -1.65 | [-5.87, 2.51] | Not in primary family |
| Line Length (+PR) | Top_1 | -2.50 | [-10.00, 3.75] | [-13.75, 7.50] |
| Spectral Entropy (+PR) | AUROC | +3.72 | [1.06, 6.64] | [-0.31, 8.47] |
| Spectral Entropy (+PR) | AP | +3.28 | [0.89, 5.64] | [-0.62, 6.88] |
| Spectral Entropy (+PR) | Macro_F1 | +1.45 | [-0.51, 3.41] | [-1.80, 4.68] |
| Spectral Entropy (+PR) | EZ_F1 | +1.78 | [-1.24, 4.78] | Not in primary family |
| Spectral Entropy (+PR) | NEZ_F1 | +1.13 | [-0.54, 2.82] | Not in primary family |
| Spectral Entropy (+PR) | Accuracy | +1.13 | [-0.55, 2.90] | Not in primary family |
| Spectral Entropy (+PR) | MRR | +6.08 | [-0.05, 12.60] | Not in primary family |
| Spectral Entropy (+PR) | Top_1 | +7.50 | [-1.25, 17.50] | [-7.50, 22.50] |
| All Six (w/o PR) | AUROC | +3.50 | [0.65, 6.39] | [-1.10, 8.23] |
| All Six (w/o PR) | AP | +3.45 | [1.20, 5.76] | [-0.24, 7.24] |
| All Six (w/o PR) | Macro_F1 | +14.40 | [9.97, 18.82] | [7.51, 21.53] |
| All Six (w/o PR) | EZ_F1 | +7.31 | [1.34, 13.12] | Not in primary family |
| All Six (w/o PR) | NEZ_F1 | +21.49 | [14.78, 28.65] | Not in primary family |
| All Six (w/o PR) | Accuracy | +12.92 | [7.88, 18.35] | Not in primary family |
| All Six (w/o PR) | MRR | +3.15 | [-2.69, 8.98] | Not in primary family |
| All Six (w/o PR) | Top_1 | +5.00 | [-3.75, 13.75] | [-10.00, 18.75] |

## PR improvements for all seven feature sets

Values are PR minus NoPR, percentage points [nominal paired 95% CI]. Only the Full comparison on the four primary endpoints additionally belongs to the 28-comparison family above. Single-feature PR differences are descriptive. PR/NoPR use the same C grid and budget, with separately selected legal C values.

| Features | AUROC | AP | Macro_F1 | Top_1 |
|---|---:|---:|---:|---:|
| Beta | +0.90 [-0.51, 2.37] | +0.15 [-0.89, 1.25] | +10.02 [6.42, 13.79] | +0.00 [-3.75, 3.75] |
| Low Gamma | +1.41 [-0.13, 3.03] | -0.85 [-2.11, 0.34] | +13.55 [9.63, 17.58] | +0.00 [-6.25, 6.25] |
| Partial High Gamma | +10.43 [5.96, 15.35] | +5.74 [2.08, 9.49] | +11.00 [7.43, 14.56] | +1.25 [-8.75, 11.25] |
| RMS | +0.36 [-0.95, 1.64] | +0.53 [-0.35, 1.45] | +11.99 [8.60, 15.57] | +2.50 [-2.50, 7.50] |
| Line Length | -1.16 [-2.79, 0.18] | -0.13 [-1.52, 1.12] | +16.89 [12.64, 21.39] | +3.75 [-1.25, 10.00] |
| Spectral Entropy | +0.21 [-1.31, 1.82] | +0.34 [-0.97, 1.69] | +13.98 [9.94, 18.09] | +0.00 [-7.50, 7.50] |
| All Six | +3.50 [0.65, 6.39] | +3.45 [1.20, 5.76] | +14.40 [9.97, 18.82] | +5.00 [-3.75, 13.75] |

## Fold-level performance

Each cell is AUROC / Macro-F1, patient means within that Outer fold. Fold sizes are 16,16,17,15,16; the overall estimate weights patients, not the five fold means equally. All eight endpoints for all 14 groups are in results/fold_results.csv.

| Input | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| Beta (+PR) | 0.655 / 0.560 | 0.721 / 0.560 | 0.780 / 0.612 | 0.660 / 0.533 | 0.793 / 0.653 |
| Low Gamma (+PR) | 0.712 / 0.577 | 0.676 / 0.537 | 0.718 / 0.613 | 0.640 / 0.530 | 0.812 / 0.644 |
| Partial High Gamma (+PR) | 0.712 / 0.505 | 0.716 / 0.519 | 0.699 / 0.559 | 0.680 / 0.491 | 0.788 / 0.594 |
| RMS (+PR) | 0.639 / 0.569 | 0.696 / 0.585 | 0.714 / 0.623 | 0.511 / 0.492 | 0.743 / 0.629 |
| Line Length (+PR) | 0.717 / 0.614 | 0.803 / 0.626 | 0.766 / 0.650 | 0.635 / 0.558 | 0.810 / 0.703 |
| Spectral Entropy (+PR) | 0.712 / 0.621 | 0.719 / 0.594 | 0.743 / 0.637 | 0.698 / 0.588 | 0.724 / 0.612 |
| All Six (w/o PR) | 0.689 / 0.442 | 0.748 / 0.429 | 0.753 / 0.511 | 0.638 / 0.510 | 0.774 / 0.517 |
| All Six (+PR) | 0.740 / 0.621 | 0.802 / 0.622 | 0.762 / 0.639 | 0.680 / 0.548 | 0.795 / 0.692 |

## Four-source performance

Each cell is AUROC / Macro-F1. These are descriptive source subsets of the original patient-held-out evaluation, not a new LOCO study. Sources retain HUP36,LZU21,multicenter15,pediatric8. Source-level 95% intervals and all eight endpoints are provided for every group in results/source_results.csv.

| Input | HUP (36) | LZU (21) | Multicenter (15) | Pediatric (8) |
|---|---:|---:|---:|---:|
| Beta (+PR) | 0.747 / 0.624 | 0.637 / 0.502 | 0.786 / 0.608 | 0.730 / 0.580 |
| Low Gamma (+PR) | 0.749 / 0.617 | 0.660 / 0.492 | 0.730 / 0.632 | 0.657 / 0.555 |
| Partial High Gamma (+PR) | 0.760 / 0.578 | 0.622 / 0.442 | 0.745 / 0.582 | 0.742 / 0.492 |
| RMS (+PR) | 0.762 / 0.647 | 0.575 / 0.513 | 0.665 / 0.596 | 0.449 / 0.436 |
| Line Length (+PR) | 0.806 / 0.673 | 0.628 / 0.545 | 0.819 / 0.672 | 0.672 / 0.596 |
| Spectral Entropy (+PR) | 0.765 / 0.647 | 0.618 / 0.529 | 0.764 / 0.646 | 0.701 / 0.598 |
| All Six (w/o PR) | 0.774 / 0.512 | 0.619 / 0.439 | 0.823 / 0.487 | 0.571 / 0.446 |
| All Six (+PR) | 0.811 / 0.665 | 0.627 / 0.544 | 0.828 / 0.660 | 0.721 / 0.602 |
| Beta (w/o PR) | 0.742 / 0.532 | 0.619 / 0.453 | 0.786 / 0.441 | 0.707 / 0.436 |
| Low Gamma (w/o PR) | 0.752 / 0.453 | 0.624 / 0.422 | 0.707 / 0.407 | 0.638 / 0.541 |
| Partial High Gamma (w/o PR) | 0.659 / 0.428 | 0.465 / 0.370 | 0.702 / 0.482 | 0.644 / 0.444 |
| RMS (w/o PR) | 0.748 / 0.513 | 0.571 / 0.419 | 0.677 / 0.434 | 0.461 / 0.394 |
| Line Length (w/o PR) | 0.803 / 0.450 | 0.655 / 0.440 | 0.821 / 0.482 | 0.724 / 0.540 |
| Spectral Entropy (w/o PR) | 0.749 / 0.502 | 0.626 / 0.405 | 0.781 / 0.488 | 0.698 / 0.477 |

## Training-label permutation negative control

All 100 fixed full12D+PR permutations were retained. Across repetitions, patient-average AUROC was 0.487 (sample SD 0.040; central 95% reference range 0.425–0.564), compared with 0.757 for true-label training. The chance AUROC reference is 0.5.

Null AP averaged 0.339; patient-equal EZ prevalence was 0.238, not 0.5. Null Top-1 averaged 0.404. This conditional control produces learned random linear projections of the fixed physiological features, not independent uniform channel rankings. Feature geometry and finite patient channel counts can therefore yield AP/Top-1 distributions different from a uniform-ranking prevalence reference. Do not interpret their null values as 0.5 or equate them exactly with prevalence. The full 100-repetition aggregate distribution is in results/negative_control.csv, with all eight endpoint summaries in negative_control_summary.csv.

Only FIT labels were shuffled within each patient; EZ counts were preserved and sample weights were recomputed. Each fold used the legally selected true-FIT full+PR C and fixed label-free FIT preprocessing, with its own true-CAL threshold. Outer labels remained real. This conditional control distribution is not presented as an exact permutation-test p-value or a patient confidence interval.

## Research questions

### Individual-feature prediction

- Beta (+PR): AUROC 0.723 [0.683, 0.761]; above 0.5 throughout its nominal interval. These are descriptive absolute intervals, not six additional corrected hypothesis tests.
- Low Gamma (+PR): AUROC 0.713 [0.668, 0.754]; above 0.5 throughout its nominal interval. These are descriptive absolute intervals, not six additional corrected hypothesis tests.
- Partial High Gamma (+PR): AUROC 0.719 [0.681, 0.758]; above 0.5 throughout its nominal interval. These are descriptive absolute intervals, not six additional corrected hypothesis tests.
- RMS (+PR): AUROC 0.663 [0.617, 0.709]; above 0.5 throughout its nominal interval. These are descriptive absolute intervals, not six additional corrected hypothesis tests.
- Line Length (+PR): AUROC 0.748 [0.707, 0.787]; above 0.5 throughout its nominal interval. These are descriptive absolute intervals, not six additional corrected hypothesis tests.
- Spectral Entropy (+PR): AUROC 0.720 [0.678, 0.761]; above 0.5 throughout its nominal interval. These are descriptive absolute intervals, not six additional corrected hypothesis tests.

### Combining features

- AUROC: Full+PR has a positive 28-family interval versus 1/6 single-feature controls, a negative interval versus 0/6, and an interval crossing zero versus 5/6. Positive: RMS (+PR). Negative: none.
- AP: Full+PR has a positive 28-family interval versus 3/6 single-feature controls, a negative interval versus 0/6, and an interval crossing zero versus 3/6. Positive: Low Gamma (+PR), Partial High Gamma (+PR), RMS (+PR). Negative: none.
- Macro_F1: Full+PR has a positive 28-family interval versus 3/6 single-feature controls, a negative interval versus 0/6, and an interval crossing zero versus 3/6. Positive: Low Gamma (+PR), Partial High Gamma (+PR), RMS (+PR). Negative: none.
- Top_1: Full+PR has a positive 28-family interval versus 1/6 single-feature controls, a negative interval versus 0/6, and an interval crossing zero versus 5/6. Positive: Partial High Gamma (+PR). Negative: none.

### Patient-relative normalization

PR increases Macro-F1 point estimates for all seven feature sets, but is not a uniform improvement in ranking metrics. Line Length AUROC decreases from 0.760 to 0.748, and Low Gamma AP decreases from 0.507 to 0.499. These supplementary PR differences remain descriptive; the full primary comparison is reported below.

- Full+PR versus FullNoPR AUROC: +3.50 pp; nominal CI [0.65, 6.39], 28-family CI [-1.10, 8.23].
- Full+PR versus FullNoPR AP: +3.45 pp; nominal CI [1.20, 5.76], 28-family CI [-0.24, 7.24].
- Full+PR versus FullNoPR Macro_F1: +14.40 pp; nominal CI [9.97, 18.82], 28-family CI [7.51, 21.53].
- Full+PR versus FullNoPR Top_1: +5.00 pp; nominal CI [-3.75, 13.75], 28-family CI [-10.00, 18.75].

### Linear localization without raw neural representations

The all-six+PR Logistic Regression achieved AUROC 0.757 [0.718, 0.793], AP 0.554 [0.501, 0.607], Macro-F1 0.626 [0.596, 0.654] and Top-1 0.662 [0.562, 0.762] without raw iEEG neural embeddings. The supported claim is predictive information in held-out patients of this historical cohort, not external clinical validation.

## Connection to Experiment A

Experiment A measured signed within-patient association; B fits a shared cross-patient decision function. Below, A effect sizes retain their original direction, while B uses only legal FIT labels to learn coefficients. Association strength does not by itself fix cross-patient scaling, redundancy or a linear boundary.

| Feature | A mean-coordinate D | A SD-coordinate D | B +PR AUROC | B NoPR AUROC |
|---|---:|---:|---:|---:|
| Beta | 0.451 | 0.358 | 0.723 | 0.714 |
| Low Gamma | 0.422 | 0.325 | 0.713 | 0.699 |
| Partial High Gamma | 0.434 | 0.223 | 0.719 | 0.615 |
| RMS | 0.327 | 0.275 | 0.663 | 0.660 |
| Line Length | 0.504 | 0.331 | 0.748 | 0.760 |
| Spectral Entropy | 0.420 | 0.307 | 0.720 | 0.718 |

All six +PR single-feature AUROC intervals are above 0.5, consistent with A's positive patient-average associations. Nevertheless, RMS has weaker cross-patient discrimination (AUROC 0.663) than Line Length (0.748), and combining all features does not establish an improvement over Line Length. Patient heterogeneity, correlated coordinates and the shared linear boundary are possible explanations, not mechanisms isolated by this analysis. Association in A does not demonstrate independent conditional contribution in B.

## Coordinate variation and PR numerical support

All six Mean coordinates had nonzero within-patient variation in 80/80 patients (100%). Each SD coordinate had nonzero variation and numerical PR support in 73/80 patients (91.25%). The remaining seven patients had one seizure and constant SD coordinates; all stayed in the evaluation. Support counts were identical under all five FIT-specific unit scales. Full 60-row audit: results/coordinate_variation.csv.

## Convergence, coverage and interpretation

All 1,410 unique fits are recorded in fit_audit.csv. Nonconverged fits: 0; maximum observed iterations: 94. The iteration budget was not increased. Every main/supplementary group has 80 patients and 7,635 channels, with no undefined endpoint exclusions. Seven single-seizure patients were retained; their SD-coordinate variation/support is explicitly audited in coordinate_variation.csv.

Original DELTA definitions and numerical floors were preserved. PR matches current DREChannNet feature-wise patient normalization; NoPR also has FIT-only global scaling. Historical LR results are excluded because their weighting, selection and legacy PR policies differ.

The six features and this cohort were used in historical model development. This is exploratory repeated-cohort cross-patient prediction, not independent external validation. Bootstrap intervals condition on fitted models and fixed splits, and do not capture retraining uncertainty. Predictive information does not establish causal mechanisms or irreplaceable conditional contribution of every feature.

Independent verification passed 1120 main/supplementary patient rows plus 8000 null patient rows, 72960 endpoint values, all 570 score cells, 560 absolute/source intervals, 112 paired intervals and 28 additional-family intervals. Selection replay covered all 840 candidate models and 570 CAL thresholds. See INDEPENDENT_VERIFICATION.json and INTERNAL_VERIFICATION.json for numerical details.

## Reproduction and artifacts

Source/input hashes, versions and original role counts: results/INPUT_AUDIT.json. Actual dates, source commits, the metadata-only engineering repair and private log locations: RUN_AUDIT.md. Complete selected C/threshold records: results/model_selection.csv; all candidate scores: results/inner_cv.csv.

Server execution uses run_repaired.ps1 with the original source commit, separately verified engineering commit and a new run tag. It never overwrites completed logs/models. Public archive validation must run before explicit allowlist extraction. Local regeneration:

```powershell
python experiments/drechan_sixfeature_ez_prediction_v1/release_check.py --package <anonymous-archive.zip>
python experiments/drechan_sixfeature_ez_prediction_v1/render_figures.py
python experiments/drechan_sixfeature_ez_prediction_v1/publish_report.py
```

Figure B: [PDF](figures/feature_prediction.pdf), [SVG](figures/feature_prediction.svg), [600 dpi PNG](figures/feature_prediction.png).
Single-column table: [LaTeX](tables/feature_prediction_single_column.tex).
All eight endpoints/intervals: [all_results.csv](results/all_results.csv). Individual predictions, models/scalers, identities and complete execution logs remain on the trusted server.
