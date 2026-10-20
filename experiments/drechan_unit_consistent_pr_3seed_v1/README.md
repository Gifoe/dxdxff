# Unit-consistent FeaturePR on the unchanged Raw dual-view model

**Completed and independently verified:90fresh fits/1052epochs plus45exact references. All135selections froze andall135score cells sealed before metrics. Final qualification FAIL;no newOuter ormain-model replacement.** Actual internal FULL Macro-F1=0.6083336001755916,AUROC=0.7534740689518008,versus matched B0=0.6199553281293815/0.7523440597012679. See[findings](FINDINGS.md),[nine endpoints](results/MAIN_RESULTS.csv),[paired intervals](results/INTERVALS.csv)and[execution audit](RUN_AUDIT.md). The preceding outcome-free TRAIN/CALaudit exactly reproduced mixed-unit FeaturePR inputs andbillion-scale transformed CALvalues. Only that numerical definition changes,not neural architecture,RawPR,features,loss orthreshold selection. Numerical correctness did not produce the required all-metric gain.

See the[locked protocol](PROTOCOL_LOCK.json),[preprocessing implementation](unit_pr.py)and[four unique changed-path tests](test_preprocessing.py). TRAIN-only raw global scales define coordinate units. Standard-deviation comparisons use dimensionless relativeepsilon1e-8;nondegenerate patient contrasts retain the usual formula,degenerate observed entries become0,andmissing entries remainNaN. The original second TRAIN-only imputer/scaler is retained. No feature,patient,channel orwindow is excluded orclipped.

The NoFeaturePR input is the original unchanged global TRAIN-only preprocessing,not a weakened comparator.90fresh fits cover FULL and5module controls across3seeds/original5within-FIT paths.45exact references cover B0,PRUNED_REF andLEGACY_DUALVIEW,not post-hoc winning substitutes. The30fresh noFeaturePR/bothPR cells are also checked against their prior mathematically identical procedure. All135selections freeze and135scores seal before metrics. Require independent1080patient-fold-seed rows/9720values/162nominal/14additional intervals,plus5preprocessing/30rawanchor replays and30null-control identities.

Same80patients/7635channels/256seizures/1471965valid raw250windows,all6original physiology features andVariance-only oldmask. All6new models have15473parameters,one64-hidden head andone scalar output. Exact same initialization,optimization,FP32/TF32off andwhole-patient updates. InitialRawanchors remain fixed;no dynamic normalization,additional PR,scorecalibration ornew loss is added.

Code is pushed before actual fitting. Everything runs in a new Ddrive namespace under one sustained foreground SSH owner,using existing caches in place. Genuine failure stops for diagnosis with complete atomic model/Adam/RNG checkpoints andpreserved logs. No unrelated process change,raw copying ornew extraction.

Actual admission:4unique preprocessing tests,45exact reference CAL/initialization bridges,30reused initialRawanchor bindings,andfinite complete-patient backward checks for6fresh paths. A genuine completeTRAIN NoFeaturePR update matched parent logits,loss,allgradients andAdam state exactly,max gradient error0. All5derived banks retain oldroles/inputs andNoPRmatrices bit unchanged. Fold3/5CAL active-input maxima changed from4.048493e9/1.555400e10to10.422848/11.898587without clipping orpatient/feature exclusion. These are numerical-admission findings,not prediction-performance evidence. The admissionJSON's`new_formal_fits=0`refers to its pre-training snapshot.

All findings are exploratory repeated-test. This internal study cannot replace the historical mainOuter0.653828orprove final80-patient Macro-F1>0.64/improvedAUROC/stable retained-module contributions. A point gate permits only planning final original-protocol work,not automaticOuter oractive-goal completion. Allactual negative results remain public;no better-looking control is promoted after the fact. Only source/hashes/anonymous aggregates are exported;weights,preprocessors,raw/cache,individual predictions,identities andlogs stay private.

## Do not mix the evaluation scopes

| Actual result | Assessment population and protocol | Patient-average Macro-F1 |
| --- | --- | ---: |
| Historical six-feature main model | Original80-patient fivefold Outer,seed42 | 0.6538278218104221 |
| B0,same architecture in the internal comparison | 33unique within-FIT assessment patients,three seeds | 0.6199553281293815 |
| Completed parent Raw dual-view candidate | Same33-patient internal comparison,three seeds | 0.6003079335325800 |
| Current unit-consistent candidate | Same33-patient internal comparison,three seeds;verified actual result | 0.6083336001755916 |

The0.653828and0.619955figures do not estimate the same evaluation population or training scope. Their difference is not evidence that the main Outer result fell. The parent's0.619955to0.600308internal decrease andthe current0.619955to0.608334internal decrease are real within matched comparisons;neither candidate is promoted. A numerical-admission PASS must never be presented as a predictive gain.

## Public-only release verification

These helpers are outside the12frozen scientific/execution dependencies and never import the training runtime or private inputs:

```powershell
python experiments/drechan_unit_consistent_pr_3seed_v1/verify_public.py --admission-only
python -m unittest discover -s experiments/drechan_unit_consistent_pr_3seed_v1 -p test_verify_public.py -v
# Only after the real final package exists:
python experiments/drechan_unit_consistent_pr_3seed_v1/verify_public.py
```

Admission-only mode andfull public-package verification passed against actual artifacts. Six public-helper tests passed,including source-digest tamper rejection,duplicate-key rejection,andrefusal to verify a final release without its actual manifest. These are not additional preprocessing tests,neural fits or predictive outcomes. Full mode checks33hashed anonymous payloads plusmanifest,135modelcells,1080patient-fold-seed metric rows,162nominal/14additional intervals,andthe server's independent private-verification attestation. Each arm has120patient-fold-seed rows,not135;modelcell andpatient-row counts must remain distinct. Public tables alone cannot independently reconstruct patient rankings orbootstrap draws.
