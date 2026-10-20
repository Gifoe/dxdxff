# No-Late-Gate DREChannNet: seed42 complete ablation

**Complete and independently verified on 2026-10-21 (Asia/Shanghai).** All90 fold cells completed (30 strict reuses,60 fresh fits,742 fresh training epochs), all selections froze before unfinished Outer, and all scores sealed before reporting. The independent audit passed1440 patient rows,12960 metric values,324 nominal intervals and34 additional-family intervals. The15 scientific source files are unchanged after admission. Evidence: [execution audit](RUN_AUDIT.md), [independent verification](public/INDEPENDENT_OUTER_VERIFICATION.json), [final status](public/OUTER_STATUS.json).

The fixed main remains the exact completed parent `WO_LATE_GATE`: equal-patient Macro-F1 **0.653465**, AUROC **0.765286**, AP **0.593488**, EZ-F1 **0.464629**, NEZ-F1 **0.842301**, ACC **0.772424**, MRR **0.846368**, Top-1 **0.800000**, EZ-fraction bias **-0.034239**. Variance and head interaction remain absent. Historical Outer informed this choice, so all results remain exploratory repeated-test.

[Full18-group/nine-metric report](public/REPORT.md) · [interpretation and negative findings](FINDINGS.md) · [main table](public/MAIN_RESULTS.csv) · [paired intervals](public/INTERVALS.csv) · [multiplicity-adjusted intervals](public/PRIMARY_FAMILY_INTERVALS.csv)

Removing both PR paths lowers Macro-F1 to0.510425 and AUROC to0.755076. The paired Full-minus-control differences are+0.143040 in Macro-F1 (nominal95% CI[0.104697,0.183579]) and+0.010210 in AUROC (CI[-0.014841,0.035335]). Thus PR's classification benefit is much clearer than its AUROC benefit; this study does not establish a stable simultaneous improvement in both. Removing Spectral Entropy slightly improves the point estimates (Macro-F1 0.654035, AUROC0.766753), with intervals spanning zero. No main-model reselection follows from these repeatedly viewed Outer outcomes.

16 unique current-architecture groups plus2 clearly labelled old references give18groups/90fold cells.30 strict reuses and60 fresh fits are required. All new groups remove Late Gate physically before training. `WO_BOTH_GUIDANCE` is identical to `WO_EARLY_FILM`; `WO_LATE_GATE` is identical to the main. These are explicit aliases, not extra fits/hypotheses. A guide four-cell interaction cannot be estimated when Late is fixed absent; only PR interaction is reported.

Same80patients/7635channels/256seizures/1471965validRaw250windows, original5folds,seed42, matched initializer,FP32/TF32off,4096windows,complete-patient updates,original optimizer/loss/budget/validation selection. No new extraction/features/hyperparameter search, Development gate or other seeds. Freeze all90selections before unfinishedOuter; seal all90scores before reporting.

Delivered all9patient metrics,pooled/folds/sampleSD/sources/coverage,10000patient-bootstrap324nominal/34additional-family intervals,alias/reuse records and independent1440rows/12960metric verification. Admission passed10tests,90 matched initializations,30 strict reuse gates, exact main forward/loss/all-gradient/Adam/RNG parity and18 discarded complete-patient paths. [Admission proof](public_admission/ADMISSION.json). Source and only anonymous aggregates are public; data/identities/weights/individual predictions/caches/logs remain on the trusted server.

Release validation (from the repository root, with NumPy/PyTorch available):

```powershell
python experiments/drechan_f0_no_lategate_ablation_seed42_v1/release_check.py --package <verified-anonymous-package.zip>
```

The manifest in [public](public/PACKAGE_MANIFEST.json) binds all32 anonymous payloads. The transfer ZIP SHA256 is `1f407fffafb85b0bbd1ecbf44defe485b2fc480e9bf2563a63133ad5b0ee4dc2`; ZIPs, private model artifacts and runtime logs are not committed.
