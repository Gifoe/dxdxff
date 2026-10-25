# Results: patient-relative statistical mechanism

## Main result

The primary endpoint does not support improved transferable standardized separation.
Full PR minus Global T is **−0.06909**, 95% CI **[−0.13636, 0.00063]**;
median paired change is −0.06911 and 29/80 patients (36.25%) improve. This interval
includes zero. There is no supported improvement, or statistically resolved
decrease, in the 80-person primary analysis.

Direct mean-difference direction alignment increases by 0.07922 (nominal 95% CI
[0.04169, 0.11778]; additional 25-slot family interval [0.02230, 0.14198]).
Centering reduces midpoint displacement without changing T or alignment, as required
by the algebra. Fisher sensitivity reverses the alignment effect. Overall this is
**Outcome D: mixed, estimator-dependent effects**, not a uniformly beneficial PR mechanism.

## Cohort, partition and uncertainty

All 80 patients receive one held-out assessment: HUP36, LZU21, multicenter15,
pediatric8; 7,635 channels, 1,743 EZ and 5,892 NEZ, 256 seizures. The 12 active
original R14 DELTA coordinates are unchanged. No classifier or neural fits were run.

The old internal split assessed 33 unique patients across 40 appearances. It could
not be presented as80 evaluations. A label-free assignment was frozen before any
new endpoints. Metadata admission found one pediatric patient never in original
FIT. Consequently 79 assessments lie within original FIT and one explicit exception
uses its original-validation path, never its original-test path. All 80 are absent
from their corresponding TRAIN/CAL; TRAIN uses only original FIT and CAL is unused.
This is not an 80-person pure original-FIT-internal evaluation. A predeclared strict 79
sensitivity is reported below without replacing the 80-person primary.

Intervals use 10,000 seed42 paired unique-patient bootstrap draws, conditional on
the fixed cohort, overlapping TRAIN paths and frozen shared directions. They do not
include direction-estimation/retraining uncertainty. Absolute/source intervals
are nominal and descriptive. The secondary family contains 25 declared slots,
including conservative algebraic aliases; adjusted percentile intervals use
coverage1−0.05/25. Extreme tails contain about 10 resamples and have limited
Monte Carlo precision. No p-values are assigned to mathematical identities.

## Primary representation arms

T is signed standardized EZ-minus-NEZ separation; A is cosine alignment. MAE is
mean absolute class-midpoint displacement in the prespecified TRAIN-noise units;
SD is the sample SD of signed midpoint displacements. Smaller MAE/SD is better.
All means weight patients equally, not channels. Intervals below are nominal 95%.

| Representation | T | A | Midpoint MAE | Midpoint SD |
| --- | --- | --- | --- | --- |
| Global | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 1.32975 [1.13833, 1.54055] | 1.61096 [1.29363, 1.90590] |
| Center only | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 0.29801 [0.22961, 0.38323] | 0.46127 [0.28894, 0.60704] |
| Scale only | 0.87515 [0.70232, 1.06105] | 0.59712 [0.49278, 0.69243] | 19.20898 [12.75534, 27.27877] | 39.24242 [17.47275, 54.40656] |
| Full PR | 0.87515 [0.70232, 1.06105] | 0.59712 [0.49278, 0.69243] | 0.26445 [0.21422, 0.32218] | 0.36481 [0.26526, 0.45895] |
| Matched donor | 0.43002 [0.28455, 0.57618] | 0.16849 [0.09502, 0.23958] | 0.61530 [0.41687, 0.86169] | 1.10736 [0.66658, 1.58446] |
| Operational global | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 1.32975 [1.13833, 1.54055] | 1.61096 [1.29363, 1.90590] |
| Operational PR | 0.87492 [0.70085, 1.06155] | 0.59456 [0.49032, 0.68914] | 0.26416 [0.21389, 0.32174] | 0.36445 [0.26502, 0.45839] |


Operational PR reproduces the actual 14D Feature PR pipeline, including its own
legal TRAIN-only final imputer/scaler and FP32 rounding, then selects the 12 active
coordinates. It is separate from the common-coordinate 2×2 factorial. Its T change
is −0.06932[−0.13608,0.00038], yielding the same conclusion for T.

## All primary-estimator paired comparisons

For T/A, effect=target−control. For MAE/SD, effect=control−target, so positive
means less drift. Identity rows are floating-point zero, not hypothesis tests.
SD contrasts do not have individual paired medians or positive-patient fractions.

| Target / control | Endpoint | Effect | 95% CI | 25-slot interval | Median paired | Positive patients |
| --- | --- | --- | --- | --- | --- | --- |
| Full PR / Global | T | -0.06909 | [-0.13636, 0.00063] | — | -0.06911 | 36.25% |
| Full PR / Global | A | 0.07922 | [0.04169, 0.11778] | [0.02230, 0.14198] | 0.05047 | 76.25% |
| Full PR / Global | MAE | 1.06530 | [0.86360, 1.28561] | [0.76592, 1.41961] | 0.92814 | 93.75% |
| Full PR / Global | SD | 1.24615 | [0.91327, 1.56655] | [0.78094, 1.74819] | — | — |
| Center only / Global | T | 0 (numerical) | [0 (numerical), 0 (numerical)] | — | 0 (numerical) | 0.00% |
| Center only / Global | A | 0 (numerical) | [0 (numerical), 0 (numerical)] | — | 0 (numerical) | 0.00% |
| Center only / Global | MAE | 1.03174 | [0.83911, 1.24684] | [0.73515, 1.38145] | 0.86350 | 91.25% |
| Center only / Global | SD | 1.14969 | [0.83316, 1.49041] | [0.68691, 1.69729] | — | — |
| Scale only / Global | T | -0.06909 | [-0.13636, 0.00063] | [-0.17017, 0.04617] | -0.06911 | 36.25% |
| Scale only / Global | A | 0.07922 | [0.04169, 0.11778] | [0.02230, 0.14198] | 0.05047 | 76.25% |
| Scale only / Global | MAE | -17.87923 | [-25.94910, -11.42858] | [-32.15442, -9.17109] | -10.79793 | 1.25% |
| Scale only / Global | SD | -37.63145 | [-52.88302, -15.91934] | [-61.23328, -7.92140] | — | — |
| Full PR / Center only | T | -0.06909 | [-0.13636, 0.00063] | [-0.17017, 0.04617] | -0.06911 | 36.25% |
| Full PR / Center only | A | 0.07922 | [0.04169, 0.11778] | [0.02230, 0.14198] | 0.05047 | 76.25% |
| Full PR / Center only | MAE | 0.03357 | [-0.02564, 0.09866] | [-0.05975, 0.13668] | 0.01480 | 60.00% |
| Full PR / Center only | SD | 0.09646 | [-0.05909, 0.22987] | [-0.12976, 0.28360] | — | — |
| Full PR / Scale only | T | 0 (numerical) | [0 (numerical), 0 (numerical)] | — | 0 (numerical) | 0.00% |
| Full PR / Scale only | A | 0 (numerical) | [0 (numerical), 0 (numerical)] | — | 0 (numerical) | 0.00% |
| Full PR / Scale only | MAE | 18.94453 | [12.48969, 27.03034] | [10.17692, 33.19640] | 12.11142 | 100.00% |
| Full PR / Scale only | SD | 38.87760 | [17.09818, 54.05113] | [9.18946, 62.41858] | — | — |
| Full PR / Matched donor | T | 0.44513 | [0.26231, 0.63859] | [0.16912, 0.75817] | 0.22249 | 72.50% |
| Full PR / Matched donor | A | 0.42862 | [0.30601, 0.54393] | [0.24338, 0.61390] | 0.55852 | 85.00% |
| Full PR / Matched donor | MAE | 0.35085 | [0.14550, 0.60624] | [0.05443, 0.78614] | 0.01743 | 55.00% |
| Full PR / Matched donor | SD | 0.74255 | [0.28556, 1.26738] | [0.18102, 1.62465] | — | — |
| Operational PR / Operational global | T | -0.06932 | [-0.13608, 0.00038] | — | -0.06613 | 35.00% |
| Operational PR / Operational global | A | 0.07666 | [0.03960, 0.11487] | — | 0.04928 | 75.00% |
| Operational PR / Operational global | MAE | 1.06559 | [0.86406, 1.28615] | — | 0.92839 | 93.75% |
| Operational PR / Operational global | SD | 1.24651 | [0.91378, 1.56673] | — | — | — |


## Complete 2×2 factorial decomposition

Main effects average the two corresponding simple effects; interaction is
G3−G2−G1+G0, with drift signs reversed so positive means reduction. T/A scaling
effects alias already reported pairs and are not additional independent hypotheses.
The large drift interaction is dominated by the failure of scaling without
centering: Scale-only MAE=19.20898 and SD=39.24242 versus Global 1.32975/1.61097.
It is not evidence of a new physiological interaction. Full PR adds only0.03357
MAE reduction over center-only; its adjusted interval[−0.05975,0.13668] includes zero.

| Component | Endpoint | Effect | 95% CI | 25-slot interval | Identity |
| --- | --- | --- | --- | --- | --- |
| CENTER | T | 0 (numerical) | [0 (numerical), 0 (numerical)] | — | True |
| CENTER | A | 0 (numerical) | [0 (numerical), 0 (numerical)] | — | True |
| CENTER | MAE | 9.98813 | [6.74888, 14.01293] | [5.58830, 17.10245] | False |
| CENTER | SD | 20.01365 | [9.08260, 27.58790] | [5.13586, 31.85047] | False |
| SCALE | T | -0.06909 | [-0.13636, 0.00063] | — | False |
| SCALE | A | 0.07922 | [0.04169, 0.11778] | — | False |
| SCALE | MAE | -8.92283 | [-12.96207, -5.69482] | [-16.06049, -4.57526] | False |
| SCALE | SD | -18.76750 | [-26.39150, -7.89797] | [-30.57628, -3.92273] | False |
| INTERACTION | T | 0 (numerical) | [0 (numerical), 0 (numerical)] | — | True |
| INTERACTION | A | 0 (numerical) | [0 (numerical), 0 (numerical)] | — | True |
| INTERACTION | MAE | 17.91280 | [11.46405, 25.97662] | [9.22351, 32.18786] | False |
| INTERACTION | SD | 37.72791 | [15.97888, 52.97732] | [8.04785, 61.31401] | False |


## Four-source descriptive results

Each source retains its actual patient count. These are not four independent
hospital-validation studies. Full-minus-Global T is negative in every source;
direct A increases in every source. Source differences are descriptive, not
multiplicity-adjusted subgroup discoveries. Intervals apply to each arm, not to
the difference of two arm means.

| Source | n | Global T | Full T | ΔT | Global A | Full A | ΔA | Full MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hup | 36 | 1.09858 | 0.98461 | -0.11398 | 0.61805 | 0.72906 | 0.11101 | 0.25657 |
| lzu | 21 | 0.39489 | 0.38345 | -0.01144 | 0.36601 | 0.42884 | 0.06283 | 0.24580 |
| multicenter | 15 | 1.32276 | 1.27408 | -0.04868 | 0.61865 | 0.67389 | 0.05523 | 0.28785 |
| pediatric | 8 | 0.98201 | 0.92530 | -0.05671 | 0.27700 | 0.30117 | 0.02417 | 0.30498 |


All seven arms, including center-only, scale-only, donor and operational probes:

| Source | n | Arm | T [95% CI] | A [95% CI] | MAE [95% CI] | SD [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| hup | 36 | Global | 1.09858 [0.86783, 1.34044] | 0.61805 [0.50228, 0.71435] | 1.53637 [1.22293, 1.91031] | 1.89656 [1.30078, 2.36513] |
| lzu | 21 | Global | 0.39489 [0.15786, 0.61787] | 0.36601 [0.15499, 0.55395] | 1.06839 [0.78544, 1.37400] | 1.29884 [0.93970, 1.55713] |
| multicenter | 15 | Global | 1.32276 [0.81547, 1.87464] | 0.61865 [0.45337, 0.74709] | 1.17241 [0.85128, 1.53291] | 1.40014 [0.87612, 1.72148] |
| pediatric | 8 | Global | 0.98201 [0.11876, 2.11599] | 0.27700 [-0.07553, 0.58048] | 1.38105 [0.92042, 1.72728] | 1.25921 [0.26152, 1.72503] |
| hup | 36 | Center only | 1.09858 [0.86783, 1.34044] | 0.61805 [0.50228, 0.71435] | 0.32712 [0.19706, 0.48613] | 0.55462 [0.25908, 0.75636] |
| lzu | 21 | Center only | 0.39489 [0.15786, 0.61787] | 0.36601 [0.15499, 0.55395] | 0.24593 [0.20383, 0.28693] | 0.17879 [0.07776, 0.24885] |
| multicenter | 15 | Center only | 1.32276 [0.81547, 1.87464] | 0.61865 [0.45337, 0.74709] | 0.33322 [0.17063, 0.53560] | 0.50101 [0.19586, 0.67639] |
| pediatric | 8 | Center only | 0.98201 [0.11876, 2.11599] | 0.27700 [-0.07553, 0.58048] | 0.23775 [0.18945, 0.28883] | 0.16010 [0.04330, 0.22708] |
| hup | 36 | Scale only | 0.98461 [0.79342, 1.17722] | 0.72906 [0.60752, 0.83034] | 20.54056 [11.56954, 33.09030] | 39.53308 [14.12226, 59.45131] |
| lzu | 21 | Scale only | 0.38345 [0.20715, 0.56333] | 0.42884 [0.18442, 0.64267] | 11.37825 [9.32364, 13.36178] | 5.56722 [3.54255, 7.11906] |
| multicenter | 15 | Scale only | 1.27408 [0.80095, 1.76309] | 0.67389 [0.49135, 0.82042] | 10.95617 [8.44037, 13.27528] | 8.08828 [4.71917, 9.37671] |
| pediatric | 8 | Scale only | 0.92530 [0.12804, 2.01127] | 0.30117 [-0.10916, 0.67129] | 49.24657 [6.62920, 103.94001] | 81.43388 [8.23806, 99.68161] |
| hup | 36 | Full PR | 0.98461 [0.79342, 1.17722] | 0.72906 [0.60752, 0.83034] | 0.25657 [0.18311, 0.33933] | 0.33880 [0.24821, 0.40607] |
| lzu | 21 | Full PR | 0.38345 [0.20715, 0.56333] | 0.42884 [0.18442, 0.64267] | 0.24580 [0.21209, 0.27801] | 0.14612 [0.05570, 0.22179] |
| multicenter | 15 | Full PR | 1.27408 [0.80095, 1.76309] | 0.67389 [0.49135, 0.82042] | 0.28785 [0.13065, 0.51563] | 0.49191 [0.16729, 0.73636] |
| pediatric | 8 | Full PR | 0.92530 [0.12804, 2.01127] | 0.30117 [-0.10916, 0.67129] | 0.30498 [0.17395, 0.48645] | 0.39732 [0.06598, 0.57916] |
| hup | 36 | Matched donor | 0.55414 [0.33922, 0.78372] | 0.20113 [0.10876, 0.29392] | 0.80866 [0.43421, 1.30230] | 1.45986 [0.70108, 2.18115] |
| lzu | 21 | Matched donor | 0.26277 [0.04378, 0.48562] | 0.13872 [-0.04348, 0.31011] | 0.47950 [0.22903, 0.77421] | 0.72348 [0.42647, 0.88026] |
| multicenter | 15 | Matched donor | 0.48200 [0.17266, 0.83123] | 0.11850 [-0.03206, 0.26650] | 0.49937 [0.21845, 0.84184] | 0.70716 [0.26781, 0.86852] |
| pediatric | 8 | Matched donor | 0.21299 [-0.40120, 0.68012] | 0.19352 [0.01965, 0.41440] | 0.31900 [0.06109, 0.73566] | 0.62316 [0.06516, 0.90247] |
| hup | 36 | Operational global | 1.09858 [0.86783, 1.34044] | 0.61805 [0.50228, 0.71435] | 1.53637 [1.22293, 1.91031] | 1.89656 [1.30078, 2.36513] |
| lzu | 21 | Operational global | 0.39489 [0.15786, 0.61787] | 0.36601 [0.15499, 0.55395] | 1.06839 [0.78544, 1.37400] | 1.29884 [0.93970, 1.55713] |
| multicenter | 15 | Operational global | 1.32276 [0.81547, 1.87464] | 0.61865 [0.45337, 0.74709] | 1.17241 [0.85128, 1.53291] | 1.40014 [0.87612, 1.72148] |
| pediatric | 8 | Operational global | 0.98201 [0.11876, 2.11599] | 0.27700 [-0.07553, 0.58048] | 1.38105 [0.92042, 1.72728] | 1.25921 [0.26152, 1.72503] |
| hup | 36 | Operational PR | 0.98210 [0.79003, 1.17513] | 0.72506 [0.60309, 0.82674] | 0.25710 [0.18328, 0.33999] | 0.33978 [0.24864, 0.40760] |
| lzu | 21 | Operational PR | 0.38883 [0.21474, 0.56739] | 0.42978 [0.18653, 0.64134] | 0.24473 [0.21064, 0.27721] | 0.14618 [0.05579, 0.22177] |
| multicenter | 15 | Operational PR | 1.27229 [0.79662, 1.76068] | 0.67064 [0.48557, 0.81899] | 0.28643 [0.13012, 0.51280] | 0.48984 [0.16684, 0.73389] |
| pediatric | 8 | Operational PR | 0.92355 [0.10906, 2.03152] | 0.29717 [-0.11419, 0.66892] | 0.30516 [0.17491, 0.48491] | 0.39601 [0.06423, 0.57731] |


## Negative controls

All 100 prespecified within-patient TRAIN-label permutations were retained.
Representations remain fixed; each repeat reconstructs TRAIN directions,
covariances, references and noise scales. True ASSESS labels are used for C2.
The intervals below are central95% reference ranges across100 repeats, NOT
patient-bootstrap CIs or exact population-level permutation tests.

Observed Global T=0.94424 exceeds its shuffled-TRAIN reference upper endpoint0.61807;
Full T=0.87515 likewise exceeds0.63883. Both observed A values exceed their null
reference ranges. This supports a label-dependent transferable direction within
this historical probe, but does not show a benefit of PR over Global.

| Arm | Endpoint | Observed | Null mean | Null sample SD | Null95% reference |
| --- | --- | --- | --- | --- | --- |
| Matched donor | T | 0.43002 | -0.12636 | 0.19867 | [-0.49694, 0.22113] |
| Matched donor | A | 0.16849 | -0.01936 | 0.06002 | [-0.13327, 0.10663] |
| Matched donor | MAE | 0.61530 | 0.44697 | 0.12081 | [0.21207, 0.63779] |
| Matched donor | SD | 1.10736 | 0.82425 | 0.19414 | [0.43210, 1.15242] |
| Global | T | 0.94424 | 0.01527 | 0.32100 | [-0.57436, 0.61807] |
| Global | A | 0.51790 | 0.00176 | 0.11689 | [-0.22831, 0.20481] |
| Global | MAE | 1.32975 | 0.57525 | 0.13429 | [0.34878, 0.83694] |
| Global | SD | 1.61096 | 0.94683 | 0.23496 | [0.53563, 1.39053] |
| Center only | T | 0.94424 | 0.01527 | 0.32100 | [-0.57436, 0.61807] |
| Center only | A | 0.51790 | 0.00176 | 0.11689 | [-0.22831, 0.20481] |
| Center only | MAE | 0.29801 | 0.12796 | 0.02991 | [0.07221, 0.18193] |
| Center only | SD | 0.46127 | 0.22300 | 0.05925 | [0.11968, 0.34310] |
| Scale only | T | 0.87515 | 0.00957 | 0.33233 | [-0.73455, 0.63883] |
| Scale only | A | 0.59712 | 0.00608 | 0.17507 | [-0.38898, 0.34060] |
| Scale only | MAE | 19.20898 | 38.37045 | 12.52011 | [20.32617, 65.83344] |
| Scale only | SD | 39.24242 | 89.11698 | 44.66917 | [34.70686, 213.78780] |
| Full PR | T | 0.87515 | 0.00957 | 0.33233 | [-0.73455, 0.63883] |
| Full PR | A | 0.59712 | 0.00608 | 0.17507 | [-0.38898, 0.34060] |
| Full PR | MAE | 0.26445 | 0.26691 | 0.03495 | [0.19933, 0.34094] |
| Full PR | SD | 0.36481 | 0.39728 | 0.04797 | [0.29951, 0.49550] |
| Operational global | T | 0.94424 | 0.01527 | 0.32100 | [-0.57436, 0.61807] |
| Operational global | A | 0.51790 | 0.00176 | 0.11689 | [-0.22831, 0.20481] |
| Operational global | MAE | 1.32975 | 0.57525 | 0.13429 | [0.34878, 0.83694] |
| Operational global | SD | 1.61096 | 0.94683 | 0.23496 | [0.53563, 1.39053] |
| Operational PR | T | 0.87492 | 0.00931 | 0.33039 | [-0.73422, 0.63506] |
| Operational PR | A | 0.59456 | 0.00557 | 0.17420 | [-0.38472, 0.33799] |
| Operational PR | MAE | 0.26416 | 0.26561 | 0.03511 | [0.19884, 0.33968] |
| Operational PR | SD | 0.36445 | 0.39565 | 0.04814 | [0.29801, 0.49495] |


The separate MATCHED_H3_LABEL_NULL also shuffles ASSESS labels within patients,
preserving prevalence, solely for its diagnostic class midpoint. It is not a
predictive evaluation. Generic centering still reduces MAE under shuffled labels.
For center-only, observed MAE reduction 1.03174 exceeds the matched-null central
range[0.28400,0.69397]; Full 1.06530 exceeds[0.23237,0.67875]. Center-only SD
reduction 1.14969 remains inside[0.43415,1.24332], whereas Full 1.24615 lies only
slightly above[0.38996,1.22552]. These reference comparisons are not adjusted tests
of an observed-minus-null population effect. They do not establish a clinical threshold.

For a patient with EZ fraction π and projected all-channel mean m, the class
midpoint satisfies b=m+(0.5−π)(mu_EZ−mu_NEZ). Removing m therefore changes b even
without creating EZ-specific ranking information. This identity explains why
the generic label-null centering effect matters.

| Target / control | Endpoint | Observed | Null mean | Null95% reference |
| --- | --- | --- | --- | --- |
| Full PR / Global | T | -0.06909 | 0.00472 | [-0.06494, 0.07640] |
| Full PR / Global | A | 0.07922 | 0.00380 | [-0.08873, 0.08789] |
| Full PR / Global | MAE | 1.06530 | 0.45674 | [0.23237, 0.67875] |
| Full PR / Global | SD | 1.24615 | 0.78860 | [0.38996, 1.22552] |
| Center only / Global | T | 0 (numerical) | 0 (numerical) | [-0.00000, 0.00000] |
| Center only / Global | A | 0 (numerical) | 0 (numerical) | [-0.00000, 0.00000] |
| Center only / Global | MAE | 1.03174 | 0.48050 | [0.28400, 0.69397] |
| Center only / Global | SD | 1.14969 | 0.80670 | [0.43415, 1.24332] |
| Scale only / Global | T | -0.06909 | 0.00472 | [-0.06494, 0.07640] |
| Scale only / Global | A | 0.07922 | 0.00380 | [-0.08873, 0.08789] |
| Scale only / Global | MAE | -17.87923 | -37.77785 | [-65.45553, -19.71689] |
| Scale only / Global | SD | -37.63145 | -88.19922 | [-213.03022, -33.81024] |
| Global / Absolute | T | 0.94424 | -0.00160 | [-0.08769, 0.06774] |
| Global / Absolute | A | 0.51790 | 0.00053 | [-0.08988, 0.09387] |
| Global / Absolute | MAE | 1.32975 | 0.54592 | [0.33711, 0.76923] |
| Global / Absolute | SD | 1.61096 | 0.90789 | [0.53296, 1.33758] |
| Center only / Absolute | T | 0.94424 | -0.00160 | [-0.08769, 0.06774] |
| Center only / Absolute | A | 0.51790 | 0.00053 | [-0.08988, 0.09387] |
| Center only / Absolute | MAE | 0.29801 | 0.06543 | [0.04660, 0.09810] |
| Center only / Absolute | SD | 0.46127 | 0.10120 | [0.05431, 0.18245] |
| Scale only / Absolute | T | 0.87515 | 0.00312 | [-0.06600, 0.06716] |
| Scale only / Absolute | A | 0.59712 | 0.00433 | [-0.09180, 0.11257] |
| Scale only / Absolute | MAE | 19.20898 | 38.32377 | [20.27583, 65.84236] |
| Scale only / Absolute | SD | 39.24242 | 89.10711 | [34.66691, 213.78818] |
| Full PR / Absolute | T | 0.87515 | 0.00312 | [-0.06600, 0.06716] |
| Full PR / Absolute | A | 0.59712 | 0.00433 | [-0.09180, 0.11257] |
| Full PR / Absolute | MAE | 0.26445 | 0.08918 | [0.06848, 0.11064] |
| Full PR / Absolute | SD | 0.36481 | 0.11929 | [0.08570, 0.18834] |


## Correct versus matched-donor reference

All 80 donor pairs match source and74/80 match the floor-log2 channel-count bin.
Correct-reference Full PR exceeds donor T by 0.44513 (adjusted interval
[0.16912,0.75817]) and A by 0.42862[0.24338,0.61390]. Drift also improves.
However, mismatching causes severe input-variance distortion: fold-average donor
variance ranges from 149.91 to618,239.51 versus correct-reference 0.90625–1.0.
This control supports operational reference matching under this construction;
it cannot distinguish physiological specificity from artificial distribution shift.
The donors are source/channel-count matched, not variance-distribution matched.

| fold | assessment_patients | supported_coordinates | donor_supported_coordinates | same_source_pairs | same_channel_bin_pairs | median_log2_channel_distance | median_effective_scale_ratio | mean_input_variance_correct | mean_input_variance_donor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 186 | 192 | 16 | 16 | 0.05878 | 0.75676 | 0.96875 | 690.82727 |
| 2 | 16 | 174 | 192 | 16 | 13 | 0.07149 | 0.82913 | 0.90625 | 618239.51402 |
| 3 | 16 | 186 | 186 | 16 | 14 | 0.06842 | 1.18831 | 0.96875 | 149.91219 |
| 4 | 16 | 180 | 180 | 16 | 15 | 0.07185 | 0.63261 | 0.93750 | 916.44072 |
| 5 | 16 | 192 | 186 | 16 | 16 | 0.06426 | 1.60876 | 1.00000 | 1415.77199 |


## Direction-estimator and epsilon sensitivities

The Fisher estimator does not support improved T either: Full−Global 0.03595,
nominal 95% CI[−0.04665,0.12386]. Its alignment changes by−0.11456
[−0.16547,−0.06170], opposite to the primary direct estimator. Thus angular
alignment improvement is probe-dependent, not a universal geometric property.
Changing the prespecified epsilon factor to1e−6 or1e−10 leaves the conclusions
unchanged. Fisher results do not replace the primary direct result.

| Estimator | Epsilon factor | Endpoint | Full−Global effect (drift reduction) | 95% CI |
| --- | --- | --- | --- | --- |
| DIRECT | 1e-10 | T | -0.06909 | [-0.13636, 0.00063] |
| DIRECT | 1e-10 | A | 0.07922 | [0.04169, 0.11778] |
| DIRECT | 1e-10 | MAE | 1.06530 | [0.86360, 1.28561] |
| DIRECT | 1e-10 | SD | 1.24615 | [0.91327, 1.56655] |
| DIRECT | 1e-08 | T | -0.06909 | [-0.13636, 0.00063] |
| DIRECT | 1e-08 | A | 0.07922 | [0.04169, 0.11778] |
| DIRECT | 1e-08 | MAE | 1.06530 | [0.86360, 1.28561] |
| DIRECT | 1e-08 | SD | 1.24615 | [0.91327, 1.56655] |
| DIRECT | 1e-06 | T | -0.06909 | [-0.13636, 0.00062] |
| DIRECT | 1e-06 | A | 0.07922 | [0.04169, 0.11778] |
| DIRECT | 1e-06 | MAE | 1.06530 | [0.86360, 1.28561] |
| DIRECT | 1e-06 | SD | 1.24615 | [0.91326, 1.56655] |
| FISHER | 1e-10 | T | 0.03595 | [-0.04665, 0.12386] |
| FISHER | 1e-10 | A | -0.11456 | [-0.16547, -0.06170] |
| FISHER | 1e-10 | MAE | 1.52741 | [1.20818, 1.87560] |
| FISHER | 1e-10 | SD | 1.92069 | [1.36493, 2.45731] |
| FISHER | 1e-08 | T | 0.03595 | [-0.04665, 0.12386] |
| FISHER | 1e-08 | A | -0.11456 | [-0.16547, -0.06170] |
| FISHER | 1e-08 | MAE | 1.52741 | [1.20818, 1.87560] |
| FISHER | 1e-08 | SD | 1.92069 | [1.36493, 2.45731] |
| FISHER | 1e-06 | T | 0.03595 | [-0.04665, 0.12386] |
| FISHER | 1e-06 | A | -0.11456 | [-0.16547, -0.06170] |
| FISHER | 1e-06 | MAE | 1.52741 | [1.20818, 1.87560] |
| FISHER | 1e-06 | SD | 1.92069 | [1.36493, 2.45731] |


All absolute arms for both estimators and all three numerical factors:

| Estimator | Factor | Arm | T [95% CI] | A [95% CI] | MAE [95% CI] | SD [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| DIRECT | 1e-10 | Global | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 1.32975 [1.13833, 1.54055] | 1.61097 [1.29363, 1.90590] |
| DIRECT | 1e-10 | Center only | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 0.29801 [0.22961, 0.38323] | 0.46127 [0.28894, 0.60704] |
| DIRECT | 1e-10 | Scale only | 0.87515 [0.70232, 1.06105] | 0.59712 [0.49278, 0.69243] | 19.20898 [12.75534, 27.27877] | 39.24242 [17.47275, 54.40656] |
| DIRECT | 1e-10 | Full PR | 0.87515 [0.70232, 1.06105] | 0.59712 [0.49278, 0.69243] | 0.26445 [0.21422, 0.32218] | 0.36481 [0.26526, 0.45895] |
| DIRECT | 1e-10 | Matched donor | 0.43936 [0.29245, 0.58619] | 0.16849 [0.09502, 0.23958] | 0.61530 [0.41687, 0.86169] | 1.10736 [0.66658, 1.58446] |
| DIRECT | 1e-10 | Operational global | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 1.32975 [1.13833, 1.54055] | 1.61097 [1.29363, 1.90590] |
| DIRECT | 1e-10 | Operational PR | 0.87492 [0.70085, 1.06155] | 0.59456 [0.49032, 0.68914] | 0.26416 [0.21389, 0.32174] | 0.36445 [0.26502, 0.45839] |
| DIRECT | 1e-08 | Global | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 1.32975 [1.13833, 1.54055] | 1.61096 [1.29363, 1.90590] |
| DIRECT | 1e-08 | Center only | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 0.29801 [0.22961, 0.38323] | 0.46127 [0.28894, 0.60704] |
| DIRECT | 1e-08 | Scale only | 0.87515 [0.70232, 1.06105] | 0.59712 [0.49278, 0.69243] | 19.20898 [12.75534, 27.27877] | 39.24242 [17.47275, 54.40656] |
| DIRECT | 1e-08 | Full PR | 0.87515 [0.70232, 1.06105] | 0.59712 [0.49278, 0.69243] | 0.26445 [0.21422, 0.32218] | 0.36481 [0.26526, 0.45895] |
| DIRECT | 1e-08 | Matched donor | 0.43002 [0.28455, 0.57618] | 0.16849 [0.09502, 0.23958] | 0.61530 [0.41687, 0.86169] | 1.10736 [0.66658, 1.58446] |
| DIRECT | 1e-08 | Operational global | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 1.32975 [1.13833, 1.54055] | 1.61096 [1.29363, 1.90590] |
| DIRECT | 1e-08 | Operational PR | 0.87492 [0.70085, 1.06155] | 0.59456 [0.49032, 0.68914] | 0.26416 [0.21389, 0.32174] | 0.36445 [0.26502, 0.45839] |
| DIRECT | 1e-06 | Global | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 1.32975 [1.13833, 1.54055] | 1.61096 [1.29363, 1.90590] |
| DIRECT | 1e-06 | Center only | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 0.29801 [0.22961, 0.38323] | 0.46127 [0.28894, 0.60704] |
| DIRECT | 1e-06 | Scale only | 0.87515 [0.70232, 1.06105] | 0.59712 [0.49278, 0.69243] | 19.20897 [12.75533, 27.27876] | 39.24240 [17.47274, 54.40653] |
| DIRECT | 1e-06 | Full PR | 0.87515 [0.70232, 1.06105] | 0.59712 [0.49278, 0.69243] | 0.26445 [0.21422, 0.32218] | 0.36481 [0.26526, 0.45895] |
| DIRECT | 1e-06 | Matched donor | 0.40085 [0.26354, 0.54106] | 0.16849 [0.09502, 0.23958] | 0.61530 [0.41687, 0.86169] | 1.10736 [0.66658, 1.58446] |
| DIRECT | 1e-06 | Operational global | 0.94424 [0.74718, 1.15027] | 0.51790 [0.42600, 0.60417] | 1.32975 [1.13833, 1.54055] | 1.61096 [1.29363, 1.90590] |
| DIRECT | 1e-06 | Operational PR | 0.87492 [0.70085, 1.06155] | 0.59456 [0.49032, 0.68914] | 0.26416 [0.21389, 0.32174] | 0.36445 [0.26502, 0.45839] |
| FISHER | 1e-10 | Global | 1.06763 [0.83956, 1.31305] | 0.43682 [0.35254, 0.51630] | 1.85857 [1.55434, 2.19804] | 2.37420 [1.83553, 2.88242] |
| FISHER | 1e-10 | Center only | 1.06763 [0.83956, 1.31305] | 0.43682 [0.35254, 0.51630] | 0.37903 [0.29490, 0.47933] | 0.57207 [0.37861, 0.73725] |
| FISHER | 1e-10 | Scale only | 1.10358 [0.86967, 1.35633] | 0.32226 [0.25938, 0.37964] | 17.72054 [10.96127, 26.51521] | 40.40315 [14.37939, 60.81633] |
| FISHER | 1e-10 | Full PR | 1.10358 [0.86967, 1.35633] | 0.32226 [0.25938, 0.37964] | 0.33116 [0.26907, 0.40142] | 0.45351 [0.32636, 0.56609] |
| FISHER | 1e-10 | Matched donor | 0.76461 [0.58516, 0.94695] | 0.30427 [0.23338, 0.37444] | 1.32539 [0.93862, 1.75169] | 2.20885 [1.48919, 2.97189] |
| FISHER | 1e-10 | Operational global | 1.06763 [0.83956, 1.31305] | 0.43682 [0.35254, 0.51630] | 1.85857 [1.55434, 2.19804] | 2.37420 [1.83553, 2.88242] |
| FISHER | 1e-10 | Operational PR | 1.10259 [0.86898, 1.35501] | 0.32142 [0.25883, 0.37872] | 0.33121 [0.26920, 0.40139] | 0.45325 [0.32651, 0.56574] |
| FISHER | 1e-08 | Global | 1.06763 [0.83956, 1.31305] | 0.43682 [0.35254, 0.51630] | 1.85857 [1.55434, 2.19804] | 2.37420 [1.83553, 2.88242] |
| FISHER | 1e-08 | Center only | 1.06763 [0.83956, 1.31305] | 0.43682 [0.35254, 0.51630] | 0.37903 [0.29490, 0.47933] | 0.57207 [0.37861, 0.73725] |
| FISHER | 1e-08 | Scale only | 1.10358 [0.86967, 1.35633] | 0.32226 [0.25938, 0.37964] | 17.72054 [10.96127, 26.51521] | 40.40315 [14.37939, 60.81633] |
| FISHER | 1e-08 | Full PR | 1.10358 [0.86967, 1.35633] | 0.32226 [0.25938, 0.37964] | 0.33116 [0.26907, 0.40142] | 0.45351 [0.32636, 0.56609] |
| FISHER | 1e-08 | Matched donor | 0.76446 [0.58513, 0.94684] | 0.30427 [0.23338, 0.37444] | 1.32539 [0.93862, 1.75169] | 2.20885 [1.48919, 2.97189] |
| FISHER | 1e-08 | Operational global | 1.06763 [0.83956, 1.31305] | 0.43682 [0.35254, 0.51630] | 1.85857 [1.55434, 2.19804] | 2.37420 [1.83553, 2.88242] |
| FISHER | 1e-08 | Operational PR | 1.10259 [0.86898, 1.35501] | 0.32142 [0.25883, 0.37872] | 0.33121 [0.26920, 0.40139] | 0.45325 [0.32651, 0.56574] |
| FISHER | 1e-06 | Global | 1.06763 [0.83956, 1.31305] | 0.43682 [0.35254, 0.51630] | 1.85857 [1.55434, 2.19804] | 2.37420 [1.83552, 2.88242] |
| FISHER | 1e-06 | Center only | 1.06763 [0.83956, 1.31305] | 0.43682 [0.35254, 0.51630] | 0.37903 [0.29490, 0.47933] | 0.57207 [0.37861, 0.73725] |
| FISHER | 1e-06 | Scale only | 1.10358 [0.86967, 1.35633] | 0.32226 [0.25938, 0.37964] | 17.72053 [10.96127, 26.51520] | 40.40313 [14.37939, 60.81630] |
| FISHER | 1e-06 | Full PR | 1.10358 [0.86967, 1.35633] | 0.32226 [0.25938, 0.37964] | 0.33116 [0.26907, 0.40142] | 0.45351 [0.32636, 0.56609] |
| FISHER | 1e-06 | Matched donor | 0.75405 [0.57620, 0.93683] | 0.30427 [0.23338, 0.37444] | 1.32539 [0.93862, 1.75169] | 2.20885 [1.48919, 2.97189] |
| FISHER | 1e-06 | Operational global | 1.06763 [0.83956, 1.31305] | 0.43682 [0.35254, 0.51630] | 1.85857 [1.55434, 2.19804] | 2.37420 [1.83552, 2.88242] |
| FISHER | 1e-06 | Operational PR | 1.10259 [0.86898, 1.35501] | 0.32142 [0.25883, 0.37872] | 0.33121 [0.26920, 0.40139] | 0.45325 [0.32651, 0.56574] |


## Prespecified strict 79 sensitivity

The never-original-FIT validation-only exception remains in the 80-person primary analysis.
The prespecified79-person sensitivity was fully computed and independently checked
without refitting or changing directions. It does not support improved transferable
T and does not replace the 80-person primary inference. Its32 exact statistics/intervals
remain trusted-side: combining exact80 and79 means reveals the omitted patient's
endpoint by subtraction. Publishing those complementary aggregates would violate
the individual-statistics privacy boundary. The public privacy review commits to
the verified private table and retains this qualitative negative conclusion.


## Identities, numerical verification and complete files

G0/G1 and G2/G3 have equal class differences, directions, T and A within floating
point error. This is an algebraic consequence, not a zero-effect discovery.
Single-coordinate rank-biserial effects remain unchanged because their AUC is
unchanged under positive affine transformations. Supported Full PR is likewise
affine-invariant; admission maximum error3.53×10⁻11, no support crossings.
All 80 patients and all 12 coordinates remain;42/960 patient-coordinate supports
are absent (seven single-seizure patients, six SD coordinates each), with the
prespecified safe-scale fallback. These patients are not dropped.

Independent arithmetic reconstructs 7,070 TRAIN directions and 179,360 patient-direction
rows, covering717,440 core endpoint values. All 1,112 nominal interval checks and
25 additional-family interval checks pass; maximum endpoint discrepancy
1.14284×10⁻10. Nine synthetic/isolation tests also pass under both local and
trusted-server runtimes. No new classifier metrics were calculated.

The 18 privacy-filtered aggregate/verification payloads plus manifest are in `results/`.
`paired_intervals.csv` contains all 168 paired rows; `factorial_contrasts.csv`
contains all 72 factorial rows, including sensitivities and identities.
`negative_controls.csv` contains every one of the 2,200 null-aggregate rows;
`negative_control_summary.csv` contains all 136 summaries, including Fisher.
`source_results.csv` contains all 168 source/arm/estimator/factor rows.
`patient_distribution_bins.csv` publishes only fixed anonymous bins, not individual
effects. Exact individual results, the complementary79-person table and runtime
logs remain on the trusted server. The publication-only privacy filter changes no
scientific result, endpoint, direction, source hash or 80-person primary analysis.

The cohort and feature set were used in historical model development. Findings
are exploratory associations conditional on this protocol, not external
validation, clinical causality, a proof of method novelty or PR indispensability.
