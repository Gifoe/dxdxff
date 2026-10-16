# Physiological Discovery and Selection v2 — prospective execution design

This follow-up does not defend a fixed7D set. It tests a bounded selection
algorithm, not the retrospective highest Outer subset. The five questions are
answered in order: data eligibility, descriptive within-patient physiology,
nested low-cost cardinality/stability, equal-cardinality replacements, then
matched D1/Concat/M8 training and sealed exploratory Outer evaluation.

The previous study already ran222 neural fits and found conditional/ranking
signals without a stable general feature replacement. Joint deletions showed
nonadditive responses and no legally selected4D/5D replacement. The single-head
study completed25 fits and failed the eight-metric M8 target, despite improved
first-hit ranking. Those outcomes motivate controls, not a license to select
features from their Outer scores. No new architecture rescue is proposed here.

## Evidence plan

| Question | Executed data required | Primary evidence | Interpretation boundary |
|---|---|---|---|
| Physiological evidence | legal FIT labels; matched complete windows | patient rank-biserial effects, pre/early/late contrasts, source/seizure stability | annotation association, not causal EZ or clinical onset proof |
| Eligibility | legacy28 and32 prospective descriptors | named formulas, whole-file hashes, actual masks, finite/constant/duplicate statistics | unreliable source is not ineffective physiology |
| Cardinality | nested originalFIT patient partitions | assessment curves1–9D, selection records/frequencies/Jaccard | cheap fixed LR selector, not every possible nonlinear subset |
| Replacement | chosen equal-k sets, best single swap, two fixed randoms | same-width fully trained feature and Raw models | no universal irreplacability claim; alternatives bounded |
| Raw increment | D1-only, Raw-only, Concat and M8 | paired fixed-score contrasts and source/fold tables | physiological descriptors conditionally useful to this finite Raw representation |

The pool includes seven previously rebuilt graph-node descriptors only after
strict source/time/channel/mask admission. The legacy cached adjacency mismatch
is not hidden, nor is it a reason to call graph physiology useless. The old
nonlocal onset/burst descriptors are separately audited; their definitions are
not silently renamed into matched2s-window DELTA inputs. Low-frequency data
fidelity and125Hz Nyquist limitations remain binding.

All supervised screening stays within each originalFIT. Each inner assessment
patient is excluded from feature-choice, calibration, scaling and fitting.
Original Development is used only for final neural epoch/threshold selection.
No cross-fold global winning list is retroactively applied to the five Outer
folds. Frequency/consensus is descriptive. Outer labels are opened only after
all fixed score routes are sealed; prior historical reuse still makes the
result exploratory.

## Planned figure contract

Python follows the existing project workflow. Quantitative-grid figures,
approximately180mm double-column width, editableSVG/PDF plus600dpi PNG.

1. Physiology: signed patient effect forest/heatmap and pre-to-post contrasts;
   patient-cluster intervals, source sample counts, no independent-window error
   bars. Directional conclusions follow actual results, not a preset success.
2. Cardinality: nested assessment utility and AUROC/AP/Macro-F1/EZ-F1 versus
  1–9D, with patient/fold definitions and honest selection-conditioned scope.
3. Stability: feature frequency by k, finalfold identities of features (not
   patients), Jaccard/chance-adjusted overlap, no invented bootstrap refits.
4. Replacement/Raw increment: paired deltas with zero reference and intervals;
   same-width controls distinct from historical architecture references.

Only anonymous aggregates leave the server. Figure source data contain no
patient/channel rows. Visual QA and numerical verification are required before
the completed JBHI-oriented report or GitHub completion claim.
