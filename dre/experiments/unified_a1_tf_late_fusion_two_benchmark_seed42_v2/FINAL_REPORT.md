# A1-TF independent late fusion — final report

| Model | Benchmark | AP | AUROC | Macro-F1 | MRR | Top1 |
|---|---|---:|---:|---:|---:|---:|
| Original A1 | Ictal | 0.5767 | 0.7464 | 0.6208 | 0.7400 | 0.6548 |
| A1 + TF Late | Ictal | 0.5814 | 0.7470 | 0.6205 | 0.7629 | 0.6897 |
| TF-only | Ictal | 0.5255 | 0.7148 | 0.5920 | 0.7290 | 0.6261 |
| A1-v2 | Omni | 0.5505 | 0.6887 | 0.5937 | 0.6302 | 0.5682 |
| Clean A1-only | Omni | 0.5505 | 0.6887 | 0.5937 | 0.6302 | 0.5682 |
| TF-only | Omni | 0.5964 | 0.7237 | 0.5678 | 0.6650 | 0.6136 |
| A1 + TF Late | Omni | 0.5943 | 0.6653 | 0.5731 | 0.6643 | 0.6136 |

Ictal AP is 65 matched fixed-query cells × 20 repetitions; Omni AP is the mean across estimable patients. The two AP columns are not directly comparable.

**Terminal: `STOP_TF_ROUTE`.** This is exploratory: ictal development and prior Omni official-test outcomes had already been seen before this run.

## Frozen selection and identity

- Ictal β=0 identity max error: 0; historical A1 AP replay: 0.576743.
- Omni validation selected β=0.35, threshold=0.579811 before official test; test was not used to retune either.
- Clean A1-only Nbase independently trained at seed 42, selected epoch 9; its test results reproduce N0 to numerical precision, not the early-fusion α-zero model.

## Ictal

- I2−I0 fixed-query AP: +0.0046; 47-ID paired bootstrap 95% CI [-0.0103, +0.0213]. The CI crosses zero: preservation is supported, a genuine ictal gain is not established.
- Ictal AP preservation floor 0.574743: PASS. TF-only AP=0.5255.
- Target-excluded β=0 cells: 21/65; β>0 cells: 44/65.

## Omni official test

- N2−Nbase: Macro-F1 -0.0206 (95% CI [-0.0564, +0.0257]); AUROC -0.0234 (CI [-0.0521, +0.0094]); patient AP +0.0438 (CI [+0.0164, +0.0750]).
- Sensitivity: Nbase 0.2838 → N2 0.3110 (+0.0273).
- Threshold/ranking criteria: F1≥.64/.67/.70 all False/False/False; AUROC≥.77/.80 False/False.
- TF-only has AUROC 0.7237 and patient AP 0.5964: signal exists, but the frozen late-fusion coefficient did not transfer its validation gain to official test.

## Complementarity and centers

- Omni A1 Top1 errors rescued by TF-only: 9; A1 correct cases TF-only misses: 5. Actual fused Top1 rescues/destroys: 7/3. Mean within-patient score correlation 0.318; pairwise disagreement 0.333.
- Center pattern: Open-iEEG (72 patients) improves versus Nbase in F1 and AUROC (0.5654→0.6082; 0.6513→0.6681); HUP (6) and SourceSink (14) deteriorate in both. Zurich (4) has no pathological positives, so positive-class/ranking estimates are not estimable. Full center comparisons are in `outputs/INTERICTAL_DATASET_STRATIFIED.csv`.

## Interpretation

TF-only has discrimination signal in Omni, but independent late fusion worsened pooled AUROC and Macro-F1 under the frozen validation choice. This fails the prespecified route gate. Do not tune beta, threshold or architecture on these official-test outcomes. The current evidence supports stopping this TF fusion route, not a claim of no TF information in the data.
