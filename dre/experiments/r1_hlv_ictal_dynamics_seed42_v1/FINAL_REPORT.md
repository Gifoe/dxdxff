# Formal current-80 HLV retest — validation gate failed

**Terminal: `R1_VALIDATION_GATE_FAILED`.** The predeclared gate did not authorize outer-test evaluation. This is a completed seed-42 validation experiment, not a formal held-out success or a fresh sealed confirmation.

| Fold | R0 val Macro-F1 | R1 val Macro-F1 | R1 − R0 | R0 epoch | R1 epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.678740 | 0.683260 | +0.004519 | 20 | 19 |
| 2 | 0.679668 | 0.672703 | −0.006965 | 18 | 7 |
| 3 | 0.623081 | 0.623698 | +0.000617 | 6 | 7 |
| 4 | 0.671568 | 0.676040 | +0.004473 | 15 | 23 |
| 5 | 0.600073 | 0.602354 | +0.002281 | 27 | 27 |
| Mean | **0.650626** | **0.651611** | **+0.000985** | | |

The matched R0 is noncollapsed and around 0.65 on validation, a sensible window-level reference, but it is not the historical 88-D patient-relative MLP and cannot be substituted for that baseline. R1's average gain is only **0.099 percentage points**, far below the locked **1.0-point** minimum despite 4/5 positive fold signs. Mean EZ-F1 changes by `+0.001070`; EZ-AUPRC changes by `−0.000196` and EZ-MRR by `−0.019863`. Thus the HLV mechanism does not show the required ranking benefit.

The branch did activate: mean gate `0.02136` (initial sigmoid(-4) ≈ `0.01799`), with median and q10/q90 values in `validation/GATE_DIAGNOSTICS.csv`. Its mean gated-residual/base embedding norm ratio was `0.02591`, so the residual remained conservative and did not saturate. Activation alone is not predictive improvement.

The six locked checks were: R1 mean Macro-F1 ≥0.645 **pass**; mean delta ≥+0.010 **fail**; ≥3/5 positive folds **pass**; mean EZ-F1 nondecreasing **pass**; EZ-AUPRC or MRR improves **fail**; no implementation pathology **pass**. Hence `pass=false` in `validation/VALIDATION_GATE.json`. Under the original protocol, no outer test was authorized; the formal R1 outer question and conditional R2 trigger remain **not evaluable**. R2 is not justified under this protocol.

The historical `m1_feat_HLV≈0.657` appears only in a TOOLING_CHECK table and lacks compatible fixed-80 provenance. A separate inspectable historical HLV run reports `0.654462` under an older cache, random fold/validation generation and LZU label-fraction filtering. Neither number is a current-protocol confirmation. See `HISTORICAL_HLV_AUDIT.md`.

After this validation report was completed, the user explicitly requested a **separate R0-only outer test**. The post-gate amendment `R0_ONLY_OUTER_PROTOCOL_AMENDMENT.json` was committed and pushed before that test; it froze all five R0 checkpoint hashes, selected epochs, and validation thresholds. The R0 outer test was evaluated once on all 80 patients (fold sizes 16/16/17/15/16): patient Macro-F1 `0.622504`, balanced accuracy `0.677300`, EZ-F1 `0.428478`, and EZ-AUPRC `0.550745`. See `outer/R0_ONLY_REPORT.md` and `outer/R0_OUTER_RESULTS.csv`. R1 outer test was **not** evaluated. No feature, threshold grid, gate initialization, training rule, or selected R0 checkpoint was adjusted based on the test results.

`R0_OUTER_TEST_ACCESSED = YES`; `R1_OUTER_TEST_ACCESSED = NO`; `ORIGINAL_R1_VALIDATION_GATE_PASS = NO`. This R0-only result is post-hoc exploratory, not an independent sealed confirmation; historical outer outcomes for this cohort had already been viewed.
