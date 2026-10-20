# Fixed no-Late-Gate main: what the seed42 ablation establishes

## Scope and estimand

All results below were actually run or strictly reused from the identical frozen scientific cell. There are18 unique groups, each with the original five patient-held-out folds and80 patients/7635channels/256seizures. The60 fresh fits used742 epochs;30 cells passed strict reuse gates. The main has no Variance, no Head Interaction and no Late Gate. Its initializer, original training/validation roles and selection rule were not altered after viewing outcomes.

These are equal-patient averages, not pooled-channel metrics or the best fold. The only training seed is42. Patient bootstrap uncertainty does not measure training-seed robustness. Main selection followed historical Outer viewing, making every comparison exploratory repeated-test rather than independent confirmation.

The complete18-group/nine-metric table is in [REPORT.md](public/REPORT.md) and [MAIN_RESULTS.csv](public/MAIN_RESULTS.csv). No additional experiment or threshold was selected from this table.

## Main and core controls

| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | MRR | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Full: no Late Gate | 0.653465 | 0.765286 | 0.593488 | 0.464629 | 0.846368 | 0.800000 |
| Historical previous Full with Late | 0.646339 | 0.765888 | 0.589527 | 0.447095 | 0.836034 | 0.775000 |
| Historical original B0 | 0.653828 | 0.752255 | 0.585240 | 0.467424 | 0.845674 | 0.800000 |
| w/o Feature PR | 0.640931 | 0.746600 | 0.577782 | 0.451313 | 0.825916 | 0.762500 |
| w/o Raw PR | 0.628524 | 0.761538 | 0.563517 | 0.439429 | 0.768532 | 0.675000 |
| w/o BOTH PR | 0.510425 | 0.755076 | 0.565205 | 0.365161 | 0.800646 | 0.712500 |
| w/o Window PR only | 0.648286 | 0.762667 | 0.590549 | 0.466199 | 0.839761 | 0.787500 |
| w/o Early FiLM | 0.644071 | 0.766014 | 0.568578 | 0.448547 | 0.779291 | 0.675000 |
| w/o own pre-ictal reference | 0.620060 | 0.739474 | 0.558005 | 0.415426 | 0.802719 | 0.725000 |
| Early FiLM mismatch | 0.643104 | 0.768153 | 0.566495 | 0.450395 | 0.784725 | 0.700000 |
| Raw-only | 0.621288 | 0.727387 | 0.543166 | 0.403061 | 0.774164 | 0.700000 |
| Physiology-only | 0.627572 | 0.754357 | 0.547682 | 0.429154 | 0.743619 | 0.637500 |

Full also has NEZ-F1 **0.842301**, ACC **0.772424**, and EZ-fraction bias **-0.034239**. Bias is predicted minus true EZ fraction, so proximity to zero is desirable; a larger signed bias is not automatically better. `WO_LATE_GATE=FULL` and `WO_BOTH_GUIDANCE=WO_EARLY_FILM` are aliases, not extra fits or hypotheses. All16 current-architecture groups lack Late Gate physically. Only the two labelled historical references retain it.

## PR: classification evidence is stronger than AUROC evidence

Differences below are Full minus control, in absolute metric units, with10000 seed42 paired patient-bootstrap nominal95% intervals.

| Removed operation | Delta Macro-F1 [CI] | Delta AUROC [CI] |
| --- | --- | --- |
| Feature PR, including physiological window PR | +0.012534 [-0.003239, 0.027477] | +0.018686 [0.004264, 0.034320] |
| Final Raw96D patient Z | +0.024941 [0.003263, 0.047737] | +0.003748 [-0.024511, 0.031415] |
| BOTH PR paths | +0.143040 [0.104697, 0.183579] | +0.010210 [-0.014841, 0.035335] |
| Window PR only, keeping summary PR | +0.005179 [-0.002791, 0.013610] | +0.002620 [-0.005898, 0.011517] |

All four PR removals lower both Macro-F1 and AUROC point estimates, but this does **not** establish a stable simultaneous benefit. No Full-minus-control contrast has both Macro-F1 and AUROC additional-family lower bounds above zero. In particular, the AUROC interval for removing both PR paths crosses zero even before multiplicity adjustment. The large Macro-F1 drop must not be described as an equally large ranking improvement.

Removing BOTH PR changes EZ-fraction bias from-0.034239 to+0.135118 and NEZ-F1 from0.842301 to0.655689. This pattern is consistent with a substantial decision/operating-point effect in addition to any learned-ranking changes, not a proof of a causal mechanism. Thresholds were selected only by the unchanged legal validation rule, never revised using Outer.

The PR factorial contrast `Full - NoFeaturePR - NoRawPR + NoBothPR` is-0.105566 for Macro-F1 (CI[-0.142091,-0.070785]) and+0.012224 for AUROC (CI[-0.011169,0.034607]). The negative Macro-F1 performance interaction indicates non-additive effects of the two normalizations; it is not evidence of physiological antagonism or positive synergistic biological mechanisms.

## Guidance, self-reference and branch complementarity

- Removing Late Gate relative to the inherited previous Full changes Macro-F1 by+0.007126 (CI[-0.002269,0.017415]) and AUROC by-0.000602 (CI[-0.009258,0.007626]). A point-estimate improvement is not a statistically established superiority claim. B0 changes additional preprocessing/guidance operations and is not a pure Late control.
- Removing Early FiLM or mismatching its conditioner lowers Macro-F1/AP/MRR/Top-1, but slightly raises AUROC. The results do not support claiming Early FiLM improves every metric or that the correct physiological trajectory is necessary for AUROC. Mismatch respects patient/seizure/exact-mask groups; all1471965 evaluated valid windows were eligible and changed, with zero singleton windows. No other input path was mismatched.
- The own pre-ictal reference has the clearest simultaneous nominal evidence: Full-minus-no-reference Macro-F1+0.033405 [0.015084,0.052516], AUROC+0.025813 [0.001946,0.051788]. Only the Macro-F1 additional-family interval remains wholly positive; the AUROC family interval still crosses zero.
- Full exceeds Raw-only by Macro-F1+0.032177 [0.015437,0.049684], AUROC+0.037899 [0.010345,0.067151], AP+0.050321 and EZ-F1+0.061568. Both Macro-F1/AUROC fold differences are positive in5/5folds. Full also exceeds Physiology-only in Macro-F1 by+0.025893 and AP by+0.045805, while its AUROC difference from Physiology-only is uncertain. These are empirical complementary-branch results, not matched-capacity proofs: deleting a branch changes parameter count and information access.

## Conditional feature contributions, including negative results

Each deletion was retrained, masking both summary coordinates and the corresponding window input before relative statistics. Widths, shared initialization and remaining paths stayed matched.

| Deleted feature | Macro-F1 | AUROC | AP | EZ-F1 | Full-minus Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beta | 0.642504 | 0.768603 | 0.590108 | 0.449140 | +0.010961 |
| Low gamma | 0.647285 | 0.765184 | 0.589383 | 0.465280 | +0.006180 |
| Partial high gamma,80-125Hz | 0.649565 | 0.767920 | 0.587071 | 0.454239 | +0.003900 |
| RMS | 0.646097 | 0.762832 | 0.588190 | 0.451423 | +0.007369 |
| Line Length | 0.637995 | 0.761196 | 0.590936 | 0.430656 | +0.015470 |
| Spectral Entropy,10-125Hz | 0.654035 | 0.766753 | 0.596977 | 0.467052 | -0.000570 |

Line Length has the largest Macro-F1 single-deletion loss; its nominal paired interval is[0.004904,0.026423], but the additional-family lower bound is-0.001280. None of the six feature removals establishes a positive Macro-F1 difference under this family adjustment. Removing Beta or partial high gamma raises AUROC slightly. Removing Spectral Entropy raises all four displayed point estimates, but every corresponding nominal interval spans zero. This is a retained negative finding, not a reason to silently change the fixed main.

One-at-a-time deletions test conditional contributions in this model. They cannot prove all six are essential, show that a smaller joint subset is optimal, or distinguish explicit-feature redundancy from Raw compensation. No new subset search was conducted.

## Stability and reporting boundaries

Full fold sample SD is0.033102 in Macro-F1 and0.046812 in AUROC. Main folds are reported separately in [FOLD_RESULTS.csv](public/FOLD_RESULTS.csv); there is no seed SD. The best Macro-F1 fold is fold5 (0.698278), not the all-patient result. The best AUROC fold is fold2 (0.819921), and cannot replace the five-fold average.

Full Macro-F1/AUROC by source are HUP36patients:0.686730/0.813248; LZU21:0.585990/0.665404; multicenter15:0.664055/0.774939; pediatric8:0.661041/0.793550. Removing BOTH PR lowers Macro-F1 in all four sources but does not lower AUROC in every source. Full also does not outperform Raw-only in Macro-F1 at every source (multicenter is a counterexample). Source averages and five overlapping-training-fold directions are descriptive, not independent replication or proof of uniform transportability.

The pooled-channel Full Macro-F1/AUROC are0.664438/0.710978, deliberately separate from the primary patient averages0.653465/0.765286. All nine metric coverages are80/80 for every group. Signals, labels/identities, individual predictions, weights, preprocessors, caches and runtime logs remain private.

## Conclusion

The fixed no-Late main retains Macro-F1 near0.65 and clear classification benefits over jointly removing PR. Raw and explicit physiology together outperform either branch alone on several measures. However, this one-seed, repeatedly viewed cohort does not establish a reliable PR-driven improvement in both AUROC and Macro-F1, the necessity of every guidance component, or an irreducible optimal six-feature set. Negative and uncertain directions are reported without redesigning the model.
