# Six-feature EZ/NEZ association (Task A)

Status: **COMPLETE_INDEPENDENTLY_VERIFIED_SIXFEATURE_EZ_ASSOCIATION**.

Real CPU-only analysis of the exact current-model 12D physiological summaries
before patient-relative standardization, independent of fitted classifiers.
All 80 patients / 7,635 channels / 256 seizures / 1,471,965 valid channel-windows
were retained. No classifier, feature extraction or frozen-model change.

All 12 coordinates showed positive overall patient-average effects and passed
the fixed 12-test BH family. Line Length mean had the largest observed effect,
D = 0.504 [95% CI 0.424, 0.583]. This is not universal across sources:
pediatric RMS mean and partial high-gamma SD had reversed mean directions and
intervals spanning zero. The historical feature/cohort selection makes these
exploratory associations, not independent confirmation or causality.

## Outputs

- [Full results and all 12 tests](RESULTS.md), [frozen protocol](PROTOCOL.md),
  [source-level definitions](PROTOCOL.json).
- Figure A: [PDF](figures/feature_association.pdf),
  [editable SVG](figures/feature_association.svg),
  [600 dpi PNG](figures/feature_association.png).
- [All-patient distribution PDF](figures/patient_effect_distributions.pdf),
  [four-source PDF](figures/source_association.pdf), [captions](FIGURE_CAPTIONS.md).
- [Primary CSV](results/feature_association.csv),
  [source CSV](results/source_association.csv),
  [tests CSV](results/statistical_tests.csv),
  [distribution summary](results/patient_effect_summary.csv),
  [LaTeX table](results/feature_association_table.tex),
  [permutation control](results/permutation_control.csv).
- [Independent numerical proof](results/INDEPENDENT_VERIFICATION.json),
  [input audit](results/INPUT_AUDIT.json), [prior-work audit](PRIOR_WORK_AUDIT.csv),
  [execution and privacy audit](RUN_AUDIT.md), [figure QA](FIGURE_QA.json).

Independent checks covered 960 patient effects, 60 bootstrap intervals,
12 Wilcoxon/BH tests, 48 source rows and 9.6 million permuted effects.
Maximum effect discrepancy: 1.11e-16.

## Reproduction

Server-only input paths and SHA checks are fixed in `run_analysis.py`. With the
original trusted caches mounted, from this directory in PowerShell:

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES=''
$py='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$runtime='D:\drechan_sixfeature_ez_association_v1_runtime'
& $py -m unittest discover -s . -p test_statistics.py -v
& $py run_analysis.py analyze --runtime $runtime
& $py verify_analysis.py --runtime $runtime
& $py run_analysis.py package --runtime $runtime
```

Use a fresh runtime for an intentional reproducibility run; do not overwrite
the completed production runtime. `analyze` rejects existing result files.
The actual executed wrapper used hidden subprocesses and separate stdout/stderr
files to avoid PowerShell native-stderr misclassification; see RUN_AUDIT.md.

After authenticated archive transfer, from the repository root:

```powershell
python experiments/drechan_sixfeature_ez_association_v1/release_check.py --package <anonymous-archive.zip>
python experiments/drechan_sixfeature_ez_association_v1/render_figures.py
python experiments/drechan_sixfeature_ez_association_v1/build_report.py
python experiments/drechan_sixfeature_ez_association_v1/check_public_artifacts.py
```

The public aggregate tables are sufficient to regenerate all figures and the
report. The authoritative 960 patient-coordinate rows stay server-only.
Public `patient_effect_summary.csv` deliberately contains only 12 coordinate-level
distribution summaries. Native physical units are not harmonization-certified;
no absolute amplitude units or raw measurements are inferred from these effects.
