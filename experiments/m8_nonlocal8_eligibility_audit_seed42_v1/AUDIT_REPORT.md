# Eligibility of the eight historical nonlocal descriptors

Mode: source/claim/numeric audit. Executed16 October2026; no classifier,
clinical channel labels, threshold fitting or new Outer outcome was used.
The source protocol was pushed at901b882 before the audit. All prior models,
features, masks, thresholds and prediction seals are unchanged.

## Main finding

The historical eight are per-record, per-channel descriptors broadcast over
time. They are not window-local trajectories. All eight replayed exactly
from the saved base features in all256 matched records. This resolves the
earlier **formula-replay** uncertainty, but not the original cache-creation
CLI, acquisition filtering or independent clinical onset provenance.

Two rank operators fail a physiological eligibility requirement: reordering
channels changes values within tied latency groups. The other six operators
can be computed on the already admitted, valid-mask Raw250 measurements.
Those are distinct retained-band descriptors, not an unqualified reuse of
native-cache columns. They are candidates for later admission, not evidence
of stable selection or predictive gain.

## Definitions that matter for interpretation

| Historical field | Actual source operation | Interpretation boundary |
|---|---|---|
|early_high_gamma_slope|Least-squares slope of saved logged high-gamma power at centers0–10 s|Feature-units per second, not an EEG voltage slope or clinical onset detector|
|early_line_length_slope|Same slope of line length|Depends on sampling/filtering and the valid temporal support|
|onset_latency_high_gamma|First post-center whose feature is at least pre-mean+2 pre-SD, divided by maximum absolute center|Dimensionless baseline-crossing fraction, not seconds or adjudicated clinical onset|
|onset_latency_line_length|Same crossing rule for line length|No-crossing cases receive a source-defined fallback beyond the last post center|
|onset_rank_high_gamma|Stable-sort order of those latencies, scaled1 to0|Equal latencies receive different ranks by input channel order|
|onset_rank_line_length|Same rank rule|Same tied-order artifact|
|high_gamma_top20pct_mean|Mean of the highest ceil(20%×post-window-count) saved logged power values|An upper-tail window statistic, not a validated burst detector or linear-power mean|
|line_length_top20pct_mean|Same upper-tail mean for line length|A static record descriptor, not an event count|

The original constructors accept a feature series and centers, but not a
valid-window mask. Their record-level outputs are broadcast back to every
window. Mathematically, subtracting the pre-mean of such a constant yields
zero. The observed nonzero numerical DELTA residuals are floating-point
roundoff, not recovered temporal physiological dynamics. Therefore a
window-DELTA experiment is not an adequate negative test of these attributes.

## Cohort and reproducibility evidence

All80 patients,7635 canonical channels,256 records and24995 observed
record-channel units were accounted for. The original Raw contract contains
1471965 channel-windows and15074 record-windows. No patient, record or channel
was selected for favorable descriptor quality. Original patient-role arrays
and source cache hashes were checked before and after the audit.

Every one of the eight saved columns exactly matched the pinned pure
constructor, in256/256 records per feature, with maximum discrepancy0.
The isolated high-gamma/line-length helper calculation also exactly matched
the constructor. The builder CLI, its label-validation functions and its
split-building functions were not executed. The trusted pickle may
materialize extra fields, but clinical label members and patient outcomes
were never indexed or used.

Independent reaggregation checked2048 private feature-record rows,
4096 permutation-audit rows and192 distribution-summary cells. These are
software/measurement checks, not independent clinical replication.

## Channel-order failures

The audit reverses the input channel order, recomputes the descriptor and
inverts the permutation. A channel-defined physiological measurement should
return the same value. The source uses stable sorting for latency ties and
assigns different ranks to tied channels. It therefore violates this property.

| Rank feature | Cached units changed /24995 | Raw250 units changed /24995 | Maximum rank change |
|---|---:|---:|---:|
|High-gamma onset rank|23332|23260|1.0|
|Line-length onset rank|23352|23266|1.0|

All six non-rank operators passed the same permutation test, on both the
saved-cache and Raw250 inputs. The rank failure is a measurement artifact,
not evidence that recruitment order is biologically uninformative. Sorting
channels into a canonical order would make the arbitrary assignment
repeatable; it would not turn tied latencies into a physiological distinction.
No rank function was repaired or replaced in this audit.

## Native-cache and Raw masks are not interchangeable

The feature-cache temporal metadata reports1000 Hz for all256 records;
the established Raw bank is250 Hz. Complete-window support derived from the
native-cache metadata flags27 boundary entries that the inherited Raw mask
does not flag:17 in multicenter and10 in pediatric records. Both mask
definitions are preserved and compared. This is a metadata/support
disagreement, not independent proof that27 Raw windows contain invalid EEG.

Removing those entries under the native definition changes the following
cached record-channel descriptors beyond the predeclared tolerance:

| Field | Changed record-channel units |
|---|---:|
|Early high-gamma slope|0|
|Early line-length slope|0|
|High-gamma crossing fraction|1218|
|Line-length crossing fraction|1283|
|High-gamma rank|26|
|Line-length rank|46|
|High-gamma top20 mean|738|
|Line-length top20 mean|939|

The two early slopes are invariant to this particular boundary change because
their early fitting interval is unaffected. That does not establish original
creation or clinical provenance. The existing CSV field `cache_safe_unchanged`
means only **formula replay plus invariance to this native-mask comparison**;
it is not a complete eligibility or clinical-safety judgment.

The earlier main-study mask statement should be read as preservation of its
original **Raw-bank** validity mask. It must not be expanded into a claim
that the Raw mask and native-cache complete-window mask are identical. No
retrospective data deletion, model refit or Outer reevaluation is made here.

## What is available for a subsequent feature study?

Six distinct valid-mask Raw250 scalar operators are computationally available
for all24995 observed record-channel units and are channel-order equivariant:
two early slopes, two baseline-crossing fractions and two upper-quintile
means. All have complete finite coverage. The two original rank operators
are not admitted unchanged. None is added to the predictive feature pool by
this audit.

Raw250 high gamma has only80–125 Hz support. The native-cache name and its
1000 Hz temporal metadata do not prove identical spectral support or cache
creation settings. High cross-channel correlations between cached and
Raw-derived descriptors cannot establish source equivalence; the rank
correlations are especially misleading because both versions share an
arbitrary tie-breaking rule. No full80–150 Hz or low-frequency recovery is
claimed.

Crossing fractions require further explicit representation policy before
prediction. In the Raw250 replay, high-gamma pre-SD is exactly zero in four
record-channel cases. No-crossing fallbacks and many tied crossing times are
also present. These are reported in `LATENCY_TIE_DIAGNOSTICS.csv`, not hidden
by excluding patients. A fallback value must not be interpreted as an observed
clinical latency. Record scalars would need an appropriate across-seizure
summary and own-FIT numerical/missingness admission, not window DELTA.

| Evidence question | Current status |
|---|---|
|Pinned arithmetic explains cached8|VERIFIED operationally, all256 records|
|Source/cache/channel/time/mask identity|VERIFIED within the original fixed contract|
|All8 are valid window-DELTA inputs|FAILED by definition; they are static record descriptors|
|Two original ranks define order-independent channel attributes|FAILED in actual data|
|Six matched-support Raw250 scalars can be computed|VERIFIED operationally, not yet predictive admission|
|Exact creation CLI/source-at-creation/acquisition filtering|UNVERIFIED; not embedded in inspected metadata|
|Independent clinical onset or SOZ truth|UNVERIFIED|
|Stable minimum or Raw/M8 performance gain|NOT TESTED by this source-only audit; remains unproven|

The next justified evidence is a separate, prospectively specified candidate
qualification for these six source-supported record measurements, including
censoring/missingness and numeric geometry. It is not a relaxed continuation
gate or another head variant after the failed readout study. No supervised
search, feature threshold or larger model is authorized by this audit alone.

## Reproduction and artifacts

`audit.py`, `nonlocal_helpers.py` and the frozen protocol implement the actual
source-only replay; `verify_audit.py` independently reaggregates private audit
arrays. Public CSVs contain only feature/source/overall summaries. Per-record
values, patient/channel identities, cache arrays and logs stay on the trusted
server. This audit changes feature eligibility knowledge, not prior results.
