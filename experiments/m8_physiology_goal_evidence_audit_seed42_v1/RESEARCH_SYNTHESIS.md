# DREChannNet physiological feature discovery: what the experiments establish

IEEE JBHI-oriented research synthesis,16 October2026. The complete
[main report](../m8_physiology_discovery_selection_seed42_v2/REPORT.md),
[all neural tables](../m8_physiology_discovery_selection_seed42_v2/RESULT_TABLES.md),
[feature catalogue](../m8_physiology_discovery_selection_seed42_v2/FEATURE_CATALOGUE.md)
and [Raw-conditional supplement](../m8_rawconditional_feature_selection_seed42_v1/REPORT.md)
contain executable methods, actual outcomes and figure source data.

## Main answer

The experiments identify reproducible annotation-associated physiological
differences, but not a universally stable minimum that improves the full M8.
Original7D is a reference, not an experimentally proved optimum. Sparse
physiology-only selection did not transfer into a better M8. Conditioning
selection on Raw produced a different, unstable family of subsets and a
positive internal signal, not independently validated neural localization
benefit. These outcomes do not support adopting a new fixed feature list.

## Why these attributes were considered

The32 admitted measurements describe retained-band magnitude, signal
roughness, distribution/ordinal complexity and relative network organization.
The families are physiologically motivated descriptions, not direct assays
of epileptogenic tissue. Band power and amplitude test recruitment magnitude;
line length adds temporal variation; spectral entropy and Hjorth/ordinal
statistics test organization beyond magnitude; graph-node measurements test
channel position in a simultaneous network. Their value must be distinguished
from the quality and bandwidth of the available measurement.

Patient-unit late-minus-pre rank-biserial differences were+0.513447 for line
length,+0.449710 for beta power,+0.391645 for partial high gamma and+0.406751
for spectral entropy. Their respective95% patient intervals were
[0.419240,0.608589], [0.340299,0.552808], [0.273363,0.508028] and
[0.288153,0.518694]. They met the prospectively fixed160-test BH/sign/source/
repeated-seizure rule in the primary51-patient FIT sample. Stable Hjorth
complexity and permutation entropy instead had negative contrasts. The full
160 effects and all source/seizure summaries are reported, including failures
of the reproducibility rule. These are repeated-seizure and known-source
observations within this filtered annotated cohort, not independent clinical
or causal confirmation.

## Which features were excluded, and why

| Measurement | Eligibility decision | Permitted interpretation |
|---|---|---|
|Historical delta/theta|All1471965 valid windows were zero; inherited low-frequency fidelity gate failed|Unavailable/unreliable measurement, not ineffective low-frequency physiology|
|Historical alpha|Retained only as10–13 Hz partial alpha|Full8–13 Hz was not evaluated|
|Historical high gamma|Retained only as80–125 Hz partial high gamma|Not full80–150 Hz or an HFO/PLHG measure|
|Legacy graph cache|Excluded for adjacency/source mismatch|Not evidence against network physiology|
|Source-checked rebuilt graph7|Admitted with exact signal/channel/time/mask matching|Operationally comparable node measurements within this cache|
|Eight onset/burst descriptors|No temporal-DELTA eligibility: broadcast constants within24995 record/channel units and unproven event/mask replay|Not a negative test of onset/burst information|
|Power/amplitude correlated groups|Retained in the32-group candidate pool and tested conditionally|Correlation alone did not justify declaring redundancy or removing patients|

All28 historical definitions were inspected by name. The historical88D
export's16 zero coordinates and16 exact duplicates are representation
properties, not32 independent biological discoveries. New candidates did not
change the A0 inputs, raw250 cache, invalid-boundary masks, original fivefold
identities or80-patient/7635-channel/256-seizure contract. All640 own-subFIT
admissibility checks passed for the32 prospective groups. Availability is not
the same as predictive necessity.

## Does a small subset suffice, and is it stable?

The physiology-only screen executed5388 low-cost fits before85 matched
neural fits. It evaluated1–9 groups, without requiring7D or6D. Own-FIT
selection chose1,1,1,2,2 groups: line length in three folds, stable-log beta
in two, partial alpha in one and stable Hjorth complexity in one. Mean
pairwise final-fold Jaccard was0.25. One group means two DELTA mean/SD
coordinates. No one fixed subset was inferred from the pooled Outer outcomes.

The separate Raw-conditional experiment trained20 genuinely inner-cross-fitted
encoders and executed9344 accepted ridge-head fits plus25 distinct registered
choice controls. Both0–20 s and0–30 s Raw summaries were always retained.
Own-FIT choices were1,0,9,2,1 groups; final-fold pairwise Jaccard was0.0.
Fixed-encoder/preprocessing weight re-selection overlaps ranged0.044–0.244
in nonempty folds. Even before allowing encoder seed variation, the feature
identities were unstable. The evidence therefore does not establish a
universal1–3D,4–5D,6D or7D minimum.

The two studies also show why counting the most often selected feature is
not a substitute for nested selection. A feature can explain marginal EZ–NEZ
differences yet be replaceable conditional on Raw. Different numeric
representations of the same retained band may behave differently after the
historical normalization floor. That is not a new physiological mechanism.

## Are the selected features difficult to replace?

Every main neural family used matched-width original7D, selected sets,
same-cardinality best single swaps and two fixed random combinations. Patient
folds, initialization, preprocessing, loss, budget and validation selection
were held fixed. Selected D1 exceeded randomA/B in AUROC by0.102066/0.121464,
with paired intervals[0.060929,0.145533]/[0.077764,0.166809]. Thus these
particular selected sets were more informative than the tested randoms.
Against the best-swap D1, the AUROC change was0.022262 with interval
[−0.005311,0.050994]. Irreplaceability beyond that neighborhood was not
established. Neither two random controls nor a greedy path exhausts the
feature-subset space.

Raw attenuated the disadvantage of weak physiological combinations. The
selected-minus-randomA AUROC advantage fell from0.102066 in D1 to0.020852
in M8. This fixed-score diagnostic is consistent with compensation, but is
not a causal decomposition: M8 adds a Raw branch, conditioning gate and a
different trainable objective. A deletion that M8 tolerates is not proof
that the removed measurement has no physiological or standalone information.

## Does the selected physiology improve the final Raw model?

The main table below uses one sealed exploratory Outer pass per frozen
procedure,80 patients each. All metrics are patient-equal. Original7 controls
use the same model width as their selected counterparts. Historical original
M8 has a different architecture and is not a capacity-matched feature control.

| Model | AUROC | AP | Macro-F1 | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|D1 selected|0.723879|0.489084|0.595765|0.409030|0.782499|0.700573|0.651022|0.512500|
|D1 matched old7|0.738135|0.524326|0.619432|0.425477|0.813387|0.738784|0.713672|0.612500|
|M8 selected|0.746607|0.537095|0.613517|0.418792|0.808243|0.728988|0.736549|0.625000|
|M8 matched old7|0.760509|0.549236|0.635720|0.431321|0.840120|0.767867|0.764700|0.662500|
|Concat selected|0.747303|0.557994|0.631122|0.427810|0.834433|0.758067|0.810521|0.737500|
|Concat matched old7|0.756411|0.574993|0.636987|0.436108|0.837865|0.762763|0.813762|0.725000|
|Matched Raw-only0–20|0.741296|0.550114|0.630493|0.426475|0.834511|0.757014|0.784454|0.700000|
|Matched Raw-only0–30|0.717742|0.527038|0.608706|0.378923|0.838488|0.758909|0.758779|0.675000|
|Historical original M8|0.762397|0.554067|0.634097|0.434087|0.834107|0.760166|0.749310|0.637500|

Selected M8 minus matched-old7 changed AUROC by−0.013901
[−0.037671,0.008473] and Macro-F1 by−0.022203
[−0.044935,0.000307]. Neither benefit nor noninferiority was established.
Selected Concat minus Raw-only0–20 showed a small unadjusted AUROC increase
of0.006007[0.000192,0.011899] and AP increase of0.007880
[0.001450,0.014513]. Macro-F1 changed by0.000628
[−0.006834,0.008456]; EZ-F1 by0.001335[−0.011014,0.014324].
In the later fixed-score six-primary-endpoint sensitivity, AUROC and AP
intervals widened to[−0.001902,0.013947] and[−0.000983,0.016762]. All six
intervals included zero. That sensitivity was post hoc, not a new independent
test or a change to the original estimates.

The Raw-conditional internal heads showed AUROC0.662999 versus0.654907,
Macro-F1 0.571542 versus0.562096 and EZ-F1 0.346508 versus0.324091.
Those assessments also selected cardinality. Their positive paired internal
contrasts cannot be substituted for independently held-out end-to-end neural
performance. The corresponding feature sets were not stable, pediatric
AUROC decreased, and no new candidate was sent to Outer.

### Nonlinear transfer of the conditional internal signal

The separately registered
[frozen-Raw readout experiment](../m8_conditional_readout_validation_seed42_v1/REPORT.md)
trained90 matched nonlinear heads on15 strictly internal assessment paths,
with two head seeds and no new encoder or Outer inference. Each path selected
its own k using feature-choice patients, not its assessment labels. All90
models completed928 epochs and their sealed logits replayed exactly.

Selected versus Raw-only AUROC was0.724060 versus0.723091, Macro-F1
0.602533 versus0.601583 and EZ-F1 0.385705 versus0.383966. The paired
four-endpoint utility change was0.002855[−0.005310,0.009238]. Classification
changes were not positive in both head seeds. The predeclared continuation
rule failed, so no additional full M8 training was launched. These fixed-CNN
readouts test one transfer question; they are not renamed full M8 models.

Definition grouping also did not rescue the feature identity stability.
Raw-conditional final sets had exact/attribute/coarse-family mean Jaccard
0.000/0.010/0.050; among nonempty folds, attribute/coarse-family overlap
remained0.017/0.083. This post hoc sensitivity changes no model inputs and
does not declare broad-family members interchangeable. The conditional ridge
signal cannot currently support a stable minimal set or reliable nonlinear
Raw increment.

## Decision for the model and the manuscript

The subsequent
[nonlocal-descriptor eligibility audit](../m8_nonlocal8_eligibility_audit_seed42_v1/AUDIT_REPORT.md)
resolved part of the excluded historical8 measurement gap without using
labels. All eight cached record descriptors replay exactly in256 matched
records. However, reversing channel order changes23332/23352 of24995
cached high-gamma/line-length rank values because equal latencies receive
arbitrary different ranks. Those two operators are excluded unchanged, not
declared biologically uninformative. Six valid-mask Raw250 scalar operators
are computationally reconstructable as distinct retained-band candidates;
they are not window-DELTA features or proven predictive improvements.

This audit also clarifies mask scope. The original Raw-bank mask remains
unchanged, but a native-cache metadata-based complete-window definition
flags27 different boundary entries. That changes some cached crossing and
upper-quintile descriptors. The two definitions must not be silently mixed,
and this does not independently prove invalid Raw waveform samples. Exact
creation CLI, acquisition filtering and clinical onset provenance remain
unverified. No prior features, models or Outer scores were changed.

The subsequent [six-group measurement qualification](../m8_nonlocal6_measurement_qualification_seed42_v1/REPORT.md)
constructed12 canonical-channel coordinates without labels. Slopes and post
top20% descriptors use equal-seizure mean/SD; crossing groups use observed
position plus crossing proportion, separating unobserved events and invalid
baselines from observed timing. All six groups met the fixed coverage and
own-TRAIN variation criteria over20 inner partitions. There are157/172
channels without an eligible observed high-gamma/line-length crossing and one
high-gamma channel without an eligible baseline; all channels were retained.
Independent reconstruction checked91,620 values and450 inner numeric entries.
An exact-constant float-reduction failure was disclosed and corrected before
acceptance, without changing the1e-9 gate or any previous feature/model. These
results qualify new measurements for a distinct matched comparison, not for
predictive adoption. They do not change the failed nonlinear transfer result.

The [separate six-group conditional screen](../m8_nonlocal6_conditional_selection_seed42_v1/REPORT.md)
then exhaustively tested64 subsets in four fixed contexts across20 original
inner paths. Both old and new physiology received matched unit-equivariant
preprocessing. All5,120 choice fits and574 assessment refits completed and
replayed exactly; HELD labels were omitted from the selector inputs, and all
assessment scores were sealed before evaluation. This remains a ridge screen,
not a new full M8 or Outer experiment.

Selected temporal descriptors alone reached AUROC0.707716 compared with
0.743206 for matched original7. Their Raw-only incremental AUROC was0.004779
[−0.002351,0.012482], without a clear clinical/ranking increment. Once Raw and
original7 were present, selected-new four-endpoint utility decreased by
0.005209[−0.010099,−0.000054]; final-set Jaccard was0.242. The registered
continuation rule failed. No additional neural/Outer run was launched and no
new set was adopted. Thus resolving the source gap did not itself produce a
stable predictive improvement. This excludes the selected procedures in this
matched setting, not the underlying biological attributes in every model.

No new universal feature list is adopted. The study does not substantiate
"7D is best," "these features are irreplaceable," or "the discovered minimum
improves clinical SOZ localization." It substantiates a narrower separation:
cache-supported physiological differences can be repeatable while a sparse
linear optimum fails to transfer to a nonlinear fusion model; Raw-conditional
selection can change feature identity without establishing stable neural
benefit. This is the supported manuscript result, rather than a positive
feature-selection method claim.

The original seven remain a reproducible comparator, not a newly selected
winner. Physiological discovery should next target measurement validity and
model-family-matched internal confirmation, with an explicit performance/cost
tradeoff and independent evaluation data before claiming clinical gain.
The current outputs do not authorize another search for a favorable Outer
subset. Clinical SOZ adjudication and acquisition-filter ground truth were
not reconstructed from these annotations/cache, so those conclusions remain
outside the evidence.

## Delivered evidence and verification

The project map links earlier feature engineering, pruning, normalization,
joint deletion, failed replacements and fusion experiments. The main study
provides six figures covering physiology, trajectories, cardinality, selection
stability, equal-k/Raw contrasts and redundancy. The conditional supplement
adds two figures with all32 frequencies, k0–9 curves and all four source
strata. Editable SVG/PDF,600 dpi PNG/TIFF and anonymous figure source tables
are supplied with code and Markdown reports.

All original105 sealed model cells and main patient metrics/intervals were
verified. Conditional verification replayed9179 search evaluations and165
assessment fits, independently checked2805 patient metric rows and retained
the failed numerical shortcut as failed. The135-file original release remains
byte-identical. Only source and anonymous aggregates are published. Raw EEG,
identities, individual predictions, embeddings, scalers, weights and logs
remain on the trusted server. Experimental execution and delivery are
complete for these bounded studies; a stable global minimum and a broad
Raw/M8 localization increment remain scientifically unestablished.
