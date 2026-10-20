# Fixed no-Late DREChannNet: temporal pooling and capacity controls

**Complete and independently verified on2026-10-21 (Asia/Shanghai).** All25cells completed:10strict reuses and15fresh fits over182epochs. All selections froze before newOuter; all scores sealed before reporting. Independent verification passed400patient rows/3600metric values/90nominal/10additional-family intervals. The13 scientific files remain unchanged; the separately hash-locked masked-time execution repair is disclosed below. [Final status](public/OUTER_STATUS.json), [independent verification](public/INDEPENDENT_OUTER_VERIFICATION.json), [execution audit](RUN_AUDIT.md).

| Configuration | Macro-F1 | AUROC | AP | EZ-F1 |
| --- | ---: | ---: | ---: | ---: |
| Fixed Full | 0.653465 | 0.765286 | 0.593488 | 0.464629 |
| Original noEarly | 0.644071 | 0.766014 | 0.568578 | 0.448547 |
| P1 temporal statistics | 0.650744 | 0.773571 | 0.583444 | 0.472280 |
| P2 time-aware attention | 0.653678 | 0.768380 | 0.595115 | 0.464480 |
| A2 capacity-matched noEarly | 0.650836 | 0.774698 | 0.577704 | 0.459176 |

P2's tiny point gains do not establish superiority; paired Macro-F1/AUROC intervals cross zero. P1 is a competitive alternative but does not prove equivalence or isolate segmentation alone. Capacity-matched noEarly narrows the F1 gap and raises AUROC, weakening a claim that EarlyFiLM is indispensable on those metrics; Full retains better AP/MRR/Top-1 point estimates. No model is reselected from these repeatedly viewed Outer results. [Interpretation](FINDINGS.md), [all9metrics](public/REPORT.md), [allpaired intervals](public/INTERVALS.csv).

Admission passed6 executable tests,25 matched initializers,10 strict reuse gates, exact main forward/loss/all-gradient/clippedAdam/RNG parity and5 discarded whole-patient path checks. [Admission proof](public_admission/ADMISSION.json). Three alternatives were fixed before fitting: P1 unsegmented temporal Mean/SD then cross-seizure Mean/SD96D; P2 masked time-aware stage attention then original cross-seizure Mean/SD96D; A2 noEarlyFiLM with a73-wide ordinary head. Full and original noEarly are strict existing references. Five groups ×original5folds ×seed42 =25cells,10reused/15fresh.

Parameters: Full10377; P1 10377; P2 10403(+26); original noEarly9329; A2 10337(-40 versusFull,-0.39%). No inactive padding parameters. Main still excludes Variance, Head Interaction and Late Gate. The same six physiological measurements/directPR, original raw data/roles/normalization/pre-reference/loss/budget remain unless explicitly changed. [Frozen operation definitions](PROTOCOL_LOCK.json).

All25selections freeze before newOuter; all25scores seal before metrics. Each group must finish Outer without a performance gate. Report all9patient metrics,pooled/fold/source/coverage separately,400rows/3600metric values,10000patient draws,90nominal/10additional-family intervals. No outcome-driven architecture/threshold or main change; historical Outer viewing makes this exploratory repeated-test, not independent confirmation.

An outcome-free execution overlay fixes a masked-NaN timestamp weight-gradient bug before any P2epoch completed. It sanitizes only unsupported scorer inputs before matmul, preserving valid-support forward/derivative math, the original13scientific files and the already completed P1fold1 checkpoint. Four arithmetic cases and320label-free legal FIT/validation support checks passed, with no optimizer updates or newOuter access. [Execution audit](RUN_AUDIT.md) and [repair proof](public_admission/ENGINEERING_REPAIR_ADMISSION.json). Production continuation uses the hash-locked external `run_repaired.ps1`, not the unpatched original launcher.

P1 jointly changes segmentation and temporal statistics; P2 adds26learned coefficients; A2 matches parameter count approximately rather than exact functional capacity. Interpret those boundaries explicitly. Source is pushed before formal fitting. Only source and anonymous aggregates are uploaded; no data/identities/private predictions/weights/preprocessors/caches/logs/secrets.

The public package contains29explicit anonymous payloads plus manifest; ZIP SHA256 `f2d4bb48d8e6b21673bfa76da5702f45a808380ea422a25b1dd560f0de88631c`. ZIPs are not committed. Release checks validate all original13source hashes,5execution hashes,6tests,repair proof,preserved P1fold1 checkpoint/validation hashes and every payload/count:

```powershell
python experiments/drechan_pooling_capacity_ablation_seed42_v1/release_check.py --package <verified-public-package.zip>
```
