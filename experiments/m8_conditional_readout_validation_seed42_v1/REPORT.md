# Conditional physiological selection did not establish nonlinear Raw increment

Supplementary internal experiment,16 October2026. This study tests transfer
from conditional ridge selection to a matched nonlinear readout of frozen Raw
representations. It does not report another Outer test or replace the full
M8 comparisons in the [main study](../m8_physiology_discovery_selection_seed42_v2/REPORT.md).

## Design and actual execution

Each of15 original own-FIT assessment paths selects its own k from the
previously computed canonical ridge feature-choice path. The smallest k
within0.005 of maximum choice utility is used. Unlike the preceding
selection-conditioned summary, this decision does not use the corresponding
assessment outcomes. Training plus feature-choice patients then fit a
256→64→1 GELU readout. Calibration patients select its epoch and threshold;
assessment patients remain outside those choices. Both192 frozen Raw
coordinates and64 fixed physiological slots are present in every arm, with
inactive physiology masked after train-only preprocessing.

Raw-only, selected physiology and original7 are compared with matched
initialization, loss, channel coverage and budget. Two fixed head seeds42/43
are evaluated, not selected. Raw encoder weights remain their original
single-seed, strictly patient-isolated states. AdamW, the original weighted
NEZ BCE/patient-soft-Macro-F1 objective and full-patient updates are used.
All90 readouts completed928 epochs. All selected k were5–9 in this procedure;
none was replaced by a favorable lower-dimensional or Outer-derived set.
Here k counts numeric measurement groups. Two representations of the same
retained band can occupy different slots, so k is not the number of distinct
physiological mechanisms. Attribute counts are supplied for every path.
All weights, thresholds and assessment logits were sealed before these
assessment metrics were computed.

This is an exploratory reused-internal evaluation: the project has previously
examined these patients and earlier linear assessment outcomes. Removing
assessment labels from the new k decision does not make the cohort new.
There are255 assessment appearances of79 identities, determined by the
original legal FIT roles, not a new patient exclusion. The primary estimand
first averages each patient's two head-seed metrics, then averages the255
appearances; unique-identity-equal results are separately supplied. Intervals
use10000 patient-ID cluster resamples and retain repeated appearances
together. They do not include full encoder or selection re-training.

## Patient-level outcomes

| Frozen-Raw readout | AUROC | AP | Macro-F1 | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Raw-only|0.723091|0.528799|0.601583|0.383966|0.819200|0.737366|0.763296|0.670588|
|Raw + selected physiology|0.724060|0.536558|0.602533|0.385705|0.819361|0.738538|0.776638|0.696078|
|Raw + original7|0.720030|0.529858|0.595371|0.381988|0.808753|0.728643|0.766601|0.674510|

| Selected minus Raw-only | Mean change | Unadjusted95% paired interval |
|---|---:|---|
|AUROC|+0.000969|[−0.007386,0.007255]|
|AP|+0.007759|[−0.000091,0.014679]|
|Macro-F1|+0.000951|[−0.008011,0.007821]|
|EZ-F1|+0.001740|[−0.008220,0.010909]|
|NEZ-F1|+0.000161|[−0.009974,0.006677]|
|ACC|+0.001172|[−0.007258,0.007321]|
|MRR|+0.013341|[−0.008001,0.032422]|
|Top-1|+0.025490|[0.000000,0.050781]|

The primary four-endpoint utility change is+0.002855
[−0.005310,0.009238]. Signed EZ-fraction bias changes from−0.045937 to
−0.049687, not an improvement in underprediction. Head seed42 reduces
Macro-F1 and EZ-F1 relative to Raw-only by0.002614 and0.006007; seed43
increases them by0.004515 and0.009486. Ranking mean changes alone therefore
do not establish reliable thresholded localization benefit.

Compared with the same readout plus original7, selected physiology improves
AP and Macro-F1 in the unadjusted paired analysis. That does not answer the
Raw increment question: Raw-only itself exceeds original7 on average AUROC,
Macro-F1, EZ-F1 and ACC. All controls are reported rather than choosing the
weaker reference to make an increment look larger.

Known-source AUROC changes versus Raw-only are+0.000538 in HUP,+0.003626
in LZU,−0.001182 in multicenter and−0.000559 in pediatric data. LZU EZ-F1
decreases by0.013528 while other source classification responses differ.
The full source/fold/seed and unique-ID-balanced tables are supplied. These
are patient-held-out results within known strata, not external-center tests.

## Definition sensitivity and selection stability

A public-only post hoc audit maps all32 definitions to measurement attributes
and broader physiological families. Different log/floor implementations may
describe the same band or attribute, but are not declared numerically or
predictively equivalent. Model inputs and selections are unchanged.

| Existing selection procedure | Exact-feature mean Jaccard | Attribute mean Jaccard | Coarse-family mean Jaccard |
|---|---:|---:|---:|
|Physiology-only final fold sets|0.250000|0.250000|0.333333|
|Raw-conditional final fold sets|0.000000|0.010000|0.050000|

The Raw-conditional comparison includes its k0 fold. Among the four nonempty
folds, attribute and coarse-family overlaps remain0.016667 and0.083333.
Thus numeric aliases alone do not explain the observed instability. Coarse
family overlap cannot show that individual measurements are interchangeable,
and averaging overlapping folds is not independent stability replication.
All exact pairs and the complete name-definition map are published.

## Allocation decision and implication

The prospective continuation rule required a positive lower utility interval,
positive mean changes in AUROC/AP/Macro-F1/EZ-F1 for both head seeds, and no
known-source AUROC change below−0.02. The first two conditions failed; the
source condition passed. **No further full Raw or M8 training is launched
from this screen.** No head, seed, threshold, subset or continuation rule was
changed after these results.

The earlier linear conditional signal did not establish transfer to this
nonlinear readout. This does not prove that physiology has zero information
or that every possible nonlinear fusion fails. It prevents treating the
earlier ridge increment as evidence for a stable minimum or final M8 benefit.
The broader research goal remains unproven. The result supports measurement
and source-admissibility work before another adaptive model search, rather
than maintaining original7 or adopting the current unstable replacements.

## Verification and reproducibility

All90 sealed logits were replayed exactly, calibration thresholds and selected
epochs matched, and1530 patient metric rows were independently recomputed.
The first replay stopped because a legacy import reset CPU threads to2,
whereas training restored1 after that import. Restoring the already specified
training setting after imports gave zero disagreement in every cell. No
training, model, input, threshold or saved prediction changed. The failure
diagnostic and versioned execution repair remain in the record.

Protocol and executable sources were pushed atc92ad7b before new head fits;
the execution-setting amendment was pushed at9e80425 before replay. No original
Development or Outer result was read by this experiment. All prior source,
prediction seals and released results remain unchanged. Private models,
identities, individual features/logits, scalers and logs are not published.
