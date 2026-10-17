# Seven-feature single-head follow-up,2026-10-17

## Current answer

Three prospective bounded studies actually ran. They did not establish a
replacement for C2 under their fixed internal gates. The actual best
preselected full80single-head patient-Macro result remains
**0.6396579900303192**,not strictly above0.64. No newOuter was run in these
three studies. HistoricalOuter is exploratory repeated-test,not independent
confirmation,and was not used to tune any new candidate.

## All new configurations:legal Development only

65patient-fold appearances/47unique people,selection-conditioned estimates.
These are not newOuter or unbiased nestedCV values.

| Configuration | New training | PatientMacro-F1 | AUROC | AP | Gate |
|---|---|---:|---:|---:|---|
| C2 retained reference | Five actual old fits reused | .673076 | .762169 | .561626 | Reference |
| Mean-only extra Raw capacity |5from-scratch end-to-end fits | .673594 | .764880 | .563017 | Failed |
| Learned within-window Raw dispersion |5from-scratch end-to-end fits | .672001 | .763928 | .558274 | Failed |
| Same7D RawFiLM afterGELU |5from-scratch end-to-end fits | .668477 | .775724 | .578870 | Failed |
| Same7D RawFiLM beforeGELU |5from-scratch end-to-end fits | .670502 | .777180 | .569715 | Failed |
| Frozen C2/global hidden affine |5additional head fits | .673541 | .762474 | .562000 | Diagnostic control |
| Frozen C2/patient-context hidden affine |5additional head fits | .674710 | .762468 | .562437 | Failed |

Raw studies requiredMacro≥C2+.003 andAUROC/AP≥C2−.01. Context required
G1Macro≥max(C2,G0)+.003 andAUROC/AP≥theirmax−.01. No rule changed after
outcomes. All9metrics,allpaired intervals,fold/source/coverage and exact
epochs are in each study's RUN_REPORT.md and results/internal_v1.

Total actual new training:20Raw end-to-end fits/233epochs plus10frozen-body
head fits/275epochs. The same5existing C2 controls recur across studies;
they are not15independent new baselines or extra evidence replications.
Every study independently replayed15logical checkpoint/control cells,
195patient metric rows and54intervals. Across-study row counts do not turn
the reused47Development identities into independent samples.

## Interpretation

Adding learned response dispersion was not enough. Moving original7D
conditioning earlier increased AUROC point estimates but did not increase
Macro. Context adaptation added only an uncertain small F1 point increment;
G1−C2=.001634,nominal95%CI[-.000815,+.004211].
None supports replacing C2 as a reliably better F1 architecture.

These findings reject the tested finite additions under the registered
protocol;they do not establish a universal Raw performance ceiling or
prove original physiological variance/other7Dmeasurements useless. No
feature was deleted or fabricated,and no raw low-frequency recovery or
full80–150Hz high-gamma claim is made from250Hz caches.

The evidence does not justify another larger unconstrained architecture
sweep or repeatedOuter fishing merely to cross a numerical cutoff. Better
ranking and better thresholded patient-F1 are distinct requirements.
Any next study needs its own prospective mechanism,legal internal selection
and immutable final freeze. Even a future point estimate>.64 would not by
itself prove a significant or independently confirmed improvement.

## Reproduction and protected artifacts

All sources were pushed before formal fits. Trusted-server inputs,original
fivefold identities,7D/14coordinates,Raw masks,checkpoints,scalers and logs
are preserved. Only code,hashes and anonymous actual aggregates are public.
The old unexecuted displacement-only context source and paused Raw-only
study were not run or counted as these experiments.

This is a negative research milestone,not completion of the>0.64objective.
