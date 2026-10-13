# Frozen B1 (A0 without LayerNorm) outer test — seed42

User-authorized extension after the completed PMA development study. This
evaluates **only the already trained B1**, not a redesigned or retrained model.
The original PMA gate failed and remains failed; B2/B3 receive no outer access.

Complete coverage: 80 unique held-out patients, 7,635 canonical channels,
1,743 EZ and 5,892 NEZ. Each patient appears once in the original five outer
folds. Compare against exact frozen source-free B0 outer predictions.

The user selected B1 for this extension after seeing development results.
The cohort has been viewed historically. These results are exploratory,
post-development-selected, repeated-outer evidence, **not blind confirmation**.

## Result

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|
| B0: frozen source-free A0 | 0.739768 | 0.537887 | 0.624919 | 0.419405 | 0.728620 | 0.637500 |
| B1: A0 w/o input LayerNorm | 0.740841 | 0.549391 | 0.619532 | 0.427230 | 0.759681 | 0.675000 |

No supported AUROC/AP/F1 improvement over B0 under paired patient bootstrap.
See `FINAL_REPORT.md` for all metrics, intervals, fold/center results and
the sensitivity–specificity tradeoff.

## Reproduction

Requires the original private server artifacts from the PMA experiment;
all identities, checkpoints, banks, code and thresholds are hash-bound.
Do not change files inside the original PMA code/tests/protocol directories.

```text
python tests/contract.py
python code/run.py prepare
python code/run.py evaluate
```

Use the original scientific environment above. Then run `code/run.py verify`
in the separately validated sklearn analysis environment. Verification
checks already sealed scores, without re-inference or threshold selection.
The prepare phase checks all five legal validation replays and freezes
dependencies before new outer feature access. Evaluation seals every B1
prediction before materializing complete clinical labels. Completed evaluation
refuses rerun; an interrupted run reuses only atomic hash-verified fold files.

`results/OUTER_STATUS.json` is the completed evaluation-stage snapshot with
verification pending at that time. `INDEPENDENT_VERIFICATION.json` and
`VERIFIED_RUN_STATUS.json` record the subsequent verified final state.
Only source and anonymous compact aggregates are published. Raw EEG,
features, checkpoints, patient/channel predictions and logs remain private.
