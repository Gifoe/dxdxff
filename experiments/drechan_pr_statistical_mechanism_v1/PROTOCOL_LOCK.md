# Patient-relative normalization: statistical mechanism probe

## Scope fixed before new outcomes

This is a threshold-free statistical probe of the six existing physiological
DELTA summaries, not another classifier benchmark or neural training study.
All 80 patients, 7,635 channels, 1,743 EZ/5,892 NEZ channels, 256 seizures and
the original cached windows/labels are retained. Variance is excluded; the 12
active coordinates are exactly R14 indices0–7,10–13. The audited baseline
subtraction, all-valid-window and across-seizure mean/population-SD definitions
remain unchanged. No new physiological extraction or unit reinterpretation.

## All80 genuinely assessed, not merely present in TRAIN

The originally requested historical internal split has33unique assessed patients
across40appearances. Its lineage and frozen banks are audited, but those counts
cannot be represented as80patient evaluation. Before any new ASSESS endpoint was
computed, the user explicitly required all80patients to be used. The primary
protocol therefore assigns each patient exactly one NEW ASSESS path. No original global FIT/Development/
Outer roles or historical results are overwritten.

Admission01 stopped on metadata feasibility, before constructing ASSESS endpoints:
one pediatric patient NEVER belongs to original FIT in any path. Thus 80 genuinely
assessed patients cannot also be 80 strictly original-FIT-internal patients. The
minimal pre-outcome correction keeps79 assessments within original FIT and admits
the sole never-FIT patient ONLY into a path where originally validation, never
original test. This explicit exception remains held out from TRAIN/CAL. It is not
silently called internal FIT or promoted to fresh confirmation. Failed source,
inputs and logs are preserved. Zero ASSESS endpoints existed at the correction.

A metadata-only minimum-cost bipartite assignment uses five groups of16slots,
with source quotas HUP8/7/7/7/7,LZU4/5/4/4/4,multicenter3each,pediatric1/1/2/2/2.
Patient/source slots outside original FIT are forbidden except the one declared
never-FIT/original-validation exception. Legal costs
are the first eight SHA256 bytes of `42|80ASSESS|patient|fold`, divided by2^64.
Sorted patients,folds and source slots fix tie ordering. CAL8 is selected from
remaining FIT by the audited source-stratified SHA allocation, salt
`42|fold|CAL80V1`;TRAIN has base27/27/26/28/27patients, plus one in the exception
path (because it removes one fewer original FIT patient). CAL is unused.
ASSESS has80unique patients and80appearances,one per person. Patients may be in
TRAIN of another path, as in ordinary cross-fitting, but never in TRAIN/CAL of
their own evaluation path. All role assignment is label-free and frozen.

The cohort and features were used historically. This new internal assignment is
not independent external validation, a new hospital cohort or a claim of blinding.
There is no outcome-dependent promotion to original Outer.

## Clean factorial versus operational PR

Fit one common mean imputer/global StandardScaler on legal TRAIN channels only;
all-missing columns use0,constant columns have scale1. Float64 is used for
primary geometry. For each complete patient, calculate its unlabeled mean and
population SD in this common dimensionless system. Supported SD is greater than
1e-8;unsupported safeSD=1. This prevents division explosion, preserves all
coordinates and guarantees constant centered coordinates0. Near-constant
unsupported residuals are retained rather than divided by a small number.

G0=X;G1=X−mu;G2=X/safeSD;G3=(X−mu)/safeSD. No independent arm-specific downstream
scaler is allowed. The frozen matrices may use the assessed patient's complete
unlabeled channel population but never its EZ prevalence or labels.

Separately, OP_PR runs the exact actual14D feature_pr/unit_pr implementation:
TRAIN-derived units, raw-space relative support1e-8*unit,unsupported observed0,
NaNs preserved until TRAIN-only post-PR imputer/scaler,then FP32 and active12D.
OP_GLOBAL matches its no-PR TRAIN-only global FP32 representation. Both exact
implementations must agree; old-TRAIN output is also replayed against immutable
historical banks. New-TRAIN outputs are not claimed identical to old-TRAIN banks.
Operational results are diagnostic and cannot replace the factorial primary.

## TRAIN-derived directions and endpoints

EZ is positive (archive old NEZ1/EZ0 is inverted). Each TRAIN patient has equal
weight. Primary direction is the normalized mean of patient EZ-minus-NEZ class
differences. Deterministic norm floor1e-12 returns a counted zero direction or
zero alignment for degenerate cases; no silent exclusion. Patient-centered
residual arithmetic avoids cancellation of large class-independent offsets.

The TRAIN within-class covariance is the equal-patient mean of half the two
population class covariances. For direction v, TRAIN noise is v' Sigma v.
Primary epsilon=1e-8*max(noise,1e-12). Signed transferable T uses the held-out
class mean difference divided by sqrt(half the two held class population
variances+epsilon). Alignment A is cosine(held class difference,v).
Fixed epsilon sensitivities are1e-6 and1e-10;neither is selected by performance.

Secondary Fisher uses `(Sigma+lambda I)^−1 mean_delta`, normalized, with
lambda=max(0.1*trace(Sigma)/12,1e-12),TRAIN only. It remains sensitivity evidence,
not a replacement for the direct estimator if the primary finding is negative.

The class midpoint b is diagnostic,not an optimized clinical threshold. TRAIN
reference is the equal-patient mean class midpoint. Normalize displacement by
sqrt(TRAIN within-class projected noise+epsilon). Per-patient MAE=abs(z);
dispersion is sample SD of the unique-patient z values. All choices freeze before
ASSESS labels enter endpoint calculation. No AUROC/AP/F1-based selection is used.

## Identities and controls

Positive affine coordinate transforms preserve within-patient ranks/AUROC and
signed rank-biserial effects. G0/G1 share class differences,directions,T/A;
G2/G3 likewise. These are identities,not physiological discoveries. Admission
checks real TRAIN and synthetic identities,unlabeled complete-PR positive-affine
invariance,operational parity and TRAIN-only fitting before ASSESS inference.
Any support crossings are counted. Labels from ASSESS are not used at admission
for rank or other outcome checks.

Donors are selected from TRAIN only,self excluded. Prefer same source,then same
floor(log2(channel-count))bin,then nearest log2count,then deterministic SHA tie.
Donor statistics replace recipient mu/safeSD. Learn its own TRAIN direction by
the same rule. Report support,channel-count/source matches and input-variance/
effective-scale changes. A donor decline alone is not physiological specificity.

One hundred fixed within-TRAIN-patient label permutations preserve class counts
and use RNG `[42,20261026,fold,b,0]`. Recompute directions/TRAIN covariance,
references and scales; evaluate against TRUE ASSESS labels. A separate matched
H3 diagnostic uses `[42,20261026,fold,b,1]` to shuffle ASSESS labels only for its
explicitly labeled diagnostic midpoint control. It cannot be interpreted as
predictive performance. All100are retained. Null reference ranges describe
training-label randomization,not patient-bootstrap CIs or exact population tests.

## Inference fixed in advance

Primary sole comparison is direct-estimator G3−G0 in T,epsilon1e-8. Report mean,
median,positive fraction and10,000seed42paired patient-bootstrap percentile95%CI.
Always aggregate repeated appearances by patient before resampling;here each
person is assessed exactly once. Interval uncertainty conditions on directions
and overlapping TRAIN paths;it is not a re-training bootstrap.

Secondary family: six contrasts (G3−G0,G1−G0,G2−G0,G3−G1,G3−G2,G3−Donor),each
T,A,MAE reduction and SD reduction=24slots. Remove the sole primary T and four
known algebraic T/A centering identities,leaving19. Add six drift factorial
effects (centering,scaling,interaction for MAE/SD)=25declared family slots.
Keep nominal95% and additional Bonferroni percentile1−0.05/25 intervals. Some
scaling simple effects are algebraic aliases;including their declared slots is
conservative,not a claim of independent hypotheses. No p-value is calculated
for an identity. T/A factorial centering/interaction identities are labeled;
their scaling main effects alias existing comparisons and add no hypothesis.
Fisher,epsilon,operational and four-source results are descriptive sensitivities.
An additional prespecified strict79 sensitivity aggregates the same frozen direct
directions/primary epsilon over only original-FIT assessments. It reports all7
arms' four endpoints and Full-minus-Global's four endpoints,32nominal intervals.
It neither drops the exception from the primary80 nor refits or selects anything.
Extreme adjusted tails use about10bootstrap order statistics and have limited
Monte Carlo precision. No source/feature/endpoint is selected after evaluation.

## Execution, verification and publication

Freeze all7arms×2estimators×5paths×101label states=7,070directions,preprocessors,
donor assignments and numerical rules before any new ASSESS endpoint. Evaluate
179,360patient-direction rows including epsilon and matched-H3 diagnostics;
seal all private outputs before aggregate reporting. A second implementation
reconstructs directions,projected scalar moments and all717,440core endpoint
values;bootstrap intervals are independently checked using resample counts.
Verification tolerances are atol1e-7/rtol1e-8 for final endpoints,declared before
the run. Numerical identity gates are tighter and are not interpreted as effects.

CPU2,one BLAS thread,no Torch/GPU,raw encoding,feature extraction,classifier or
neural fits. Existing experiments remain immutable. Private locations are supplied
only by environment,not stored in public artifacts. No patient/channel identities,
individual statistics,predictions,directions,models,scalers or runtime logs are
exported. Public distribution figures use anonymous fixed aggregate bins.

Report outcomes A/B/C/D as specified in the user's task. Distinguish mathematical
identities,empirical evidence and the narrower tested statistical mechanism;
never infer causality,novelty,clinical threshold optimality or indispensability.
