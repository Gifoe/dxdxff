# Repository duplication admission

No exact A0 input-side, FIT-patient-quantile, bounded-slope, duplicate-shared
experiment was found in the inspected available source/history. Proceed with
the registered study; this is not a claim of universal mathematical novelty.

Inspected71 available local/remote-tracking Git references, relevant source
and protocols, the listed archived files, and related available checkouts
(main/upload/PCCNN/unified-A1-TF). The exact-mechanism pattern search's hits
were the posterior audit's FIT-quantile floor, not this model. Server source
search found MonotoneBasisFeature but no exact PR-Spline/KAN input module.
The requested archived patient-controls/calibrator files were read completely;
their named live server paths are not both present, so archive copies are
the evidence rather than assumed live source. No historical checkpoint or
outer outcome was executed during this search.

| Existing approach | Actual mathematical/training mechanism | Measured historical result / limitation | Exact duplication? |
|---|---|---|---|
| MonotoneBasisFeature/OutcomeHead | Fixed[-1,0,1] knots; softplus positive ReLU-basis weights; Task2 surgical-outcome head, not A0 input; no identity/zero-anchor/bounded interval slopes | No matched source-free A0 development result located for this head | No |
| Patient Z/rank controls | Unlabeled per-patient feature z; optional final NEZ percentile score; original control MLP/classical models | Establishes normalization/control implementation, not a fitted56-curve model | No |
| P2 monotonic shift calibrator | FIT inner-OOF patient-constant tanh-bounded final-logit shift; BCE/soft-F1/L2 and inner lambda selection | Ranking within patient unchanged by construction; no matched A0 input-side result located | No |
| Robust posterior R1 | FIT quantile IQR floor plus[-8,8] clipping; frozen density/MAP/decoder, no scorer training | Registered utility gates failed; clipping introduces ties | No |
| Exact current A0 | Original Patient Z/FIT preprocessing/LN88/MLP96; original patient updates and selection | Historical matched development AUROC0.710667, AP0.518235, MF10.638080 | Baseline only |
| PMA | Magnitude-statistic-gated hidden residual versus matched hidden capacity | AUROC0.711474 vs A00.710667; registered increment failed | No; relevant capacity caution |
| TabM/width | Four-member numerical MLP ensemble versus wider control, not a spline | AUROC0.720047 vs width0.719558; primary gate failed; no isolated ensemble-specific gain | No |
| Available upstream TabM numerical embeddings | `paper/lib/deep.py` wraps `rtdl_num_embeddings.PiecewiseLinearEmbeddings`; `paper/bin/model.py` supports train-derived numerical bins, including optional supervised tree bins | Generic upstream capability, not a measured A0 PR-Spline experiment; the inspected historical A0 TabM implementation uses LayerNorm/LinearBatchEnsemble, not this embedding | No exact duplication identified; piecewise-linear numerical representations are prior art |
| B1 without LayerNorm | Removes input LayerNorm, otherwise original backbone | Historical development AUROC0.736420; not the admissible A0 or this intervention | No |
| Feature-geometry/A1 adapters | Frozen representation/density probes or patient-conditioned low-rank corrections; different interface/protocol | Registered mechanisms not established; not comparable new A0 result rows | No |

Monotonic piecewise-linear feature maps already exist. The narrower untested
question is whether this specific small input reparameterization improves
the fixed current A0 beyond positive scaling and near-equal generic input
capacity, under exactly the same legal development protocol. A nonidentity
curve or gain over B0 alone cannot establish that attribution.

The upstream embedding support was additionally checked during final report
review. It qualifies the novelty statement but supplies no new model arm,
clinical training result or outcome-driven change to the frozen experiment.
Historical numbers in this table are from existing reports, not re-executed
benchmarks. This run's measurements are isolated in `results/`.
