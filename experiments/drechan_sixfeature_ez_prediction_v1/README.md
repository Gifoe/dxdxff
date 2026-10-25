# Experiment B: physiological feature predictive informativeness

Status: **COMPLETE_VERIFIED_SIXFEATURE_EZ_PREDICTION**. Real calculation and
independent reconstruction from saved model coefficients/predictions have passed.

An independent CPU-only Logistic Regression experiment:14 fixed configurations
(eight main plus six NoPR supplements), original five patient-held-out folds,
FIT-only nested C selection, CAL-only thresholds, and100fixed within-FIT-patient
label permutations for full12D+PR. All80patients/7635channels retained. No neural
model, new extraction or modification of Experiment A/other running experiments.

See PROTOCOL.md and PROTOCOL.json for the frozen design, weight normalization,
current unit-consistent PR, exact fit budget, 28-comparison primary family and
historical-cohort interpretation. HISTORICAL_PROTOCOL_AUDIT.csv explains why
old LR models/results are references only, not reused experimental cells.

## Results

All Six +PR achieved patient-average AUROC **0.756966** (nominal 95% CI
0.718381–0.793443), AP0.554238, Macro-F1 0.625590 and Top-1 0.662500.
It has the highest AUROC/AP point estimates among the eight main configurations,
but Line Length +PR has higher Macro-F1 (0.631379) and Top-1 (0.687500).
All four paired intervals against Line Length include zero: a joint-feature
advantage over the strongest single-feature control is not established.
Supplementary Line Length NoPR has AUROC0.759713, also retained in the full report.

PR increases the joint model's Macro-F1 by **14.40 percentage points**
(28-comparison-adjusted interval **7.51–21.53**). Adjusted AUROC/AP intervals
include zero. PR is not uniformly beneficial to every feature or ranking metric.
All six individual +PR AUROC nominal intervals exceed0.5. The100fixed training-label
permutations produce mean AUROC0.486998 (sample SD0.039800), versus0.756966 under
true-label training. All negative findings and all14configurations are retained.

This is exploratory prediction in unseen patients of a historically used cohort,
not independent external validation or proof that each feature is irreplaceable.

## Deliverables

- [Full results and all intervals](RESULTS.md)
- [Paper-ready English results paragraph](PAPER_PARAGRAPH.md)
- [Figure B: vector PDF](figures/feature_prediction.pdf), [editable SVG](figures/feature_prediction.svg), [600dpi PNG](figures/feature_prediction.png)
- [Single-column booktabs LaTeX table](tables/feature_prediction_single_column.tex), [standalone source](tables/feature_prediction_preview.tex), [compiled PDF](tables/feature_prediction_preview.pdf)
- [Primary CSV](results/main_results.csv), [all14groups](results/all_results.csv), [PR differences](results/pr_improvement.csv), [all seven paired comparisons](results/paired_comparisons.csv)
- [Independent numerical verification](results/INDEPENDENT_VERIFICATION.json), [selection/preprocessing verification](results/INTERNAL_VERIFICATION.json)
- [Protocol](PROTOCOL.md), [execution and provenance audit](RUN_AUDIT.md)

## Verification and reproduction

There were1,410unique formal LR fits (840inner CV,70selected final,500negative-control).
All570final models/preprocessors/thresholds froze before new Outer scores, and
all570score cells sealed before aggregate metrics. Independent checks passed
1,120main/supplementary patient rows plus8,000null rows and72,960endpoint values;
maximum probability reconstruction error was0.0. All fits converged, with at most94iterations.

Scientific source: `caceb44597097f0f4baea49d10a5c543e2bb706f`.
Metadata-only engineering adapter: `9ba8a29c7d0b6d33da48b4c293dcd233ab8aa9ef`.
Both were pushed and remote-verified before their corresponding production runs.
The initial serialization failure and unchanged saved candidate are documented,
not hidden or counted as a discarded statistical attempt.

Trusted-server reproduction requires the existing private caches and a **new**
runtime directory; do not rerun the completed canonical runtime. With the pinned
environment and source identities in RUN_AUDIT.md:

```powershell
$env:PREDICTION_SOURCE_COMMIT='caceb44597097f0f4baea49d10a5c543e2bb706f'
$env:PREDICTION_ENGINEERING_COMMIT='9ba8a29c7d0b6d33da48b4c293dcd233ab8aa9ef'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES=''
python run.py admission --runtime <new-private-runtime>
python audit_adapter.py train --runtime <new-private-runtime>
python run.py freeze --runtime <new-private-runtime>
python run.py score --runtime <new-private-runtime>
python run.py report --runtime <new-private-runtime>
python run.py verify --runtime <new-private-runtime>
python run.py package --runtime <new-private-runtime>
```

The actual audited production command and required environment are in RUN_AUDIT.md.
From the repository root, public checks and presentation regeneration use:

```powershell
python -m unittest discover -s experiments/drechan_sixfeature_ez_prediction_v1 -p test_math.py -v
python -m unittest discover -s experiments/drechan_sixfeature_ez_prediction_v1 -p test_audit_adapter.py -v
python experiments/drechan_sixfeature_ez_prediction_v1/release_check.py --package <anonymous-archive.zip>
python experiments/drechan_sixfeature_ez_prediction_v1/publish_report.py
python experiments/drechan_sixfeature_ez_prediction_v1/render_figures.py
python experiments/drechan_sixfeature_ez_prediction_v1/check_public_artifacts.py
python experiments/drechan_sixfeature_ez_prediction_v1/test_git_release.py --ref HEAD --package <anonymous-archive.zip> --output <public-check-proof.json>
```

Figure regeneration deliberately resets visual-review status until the new images
are inspected. No private patient/channel records, predictions, models, scalers,
raw EEG or execution logs are part of the public release. Experiment A, frozen
DREChannNet sources and the separate running LOCO experiment were not modified.
