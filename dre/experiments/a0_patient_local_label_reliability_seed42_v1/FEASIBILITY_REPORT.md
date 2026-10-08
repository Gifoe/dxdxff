# FIT-only PLLR feasibility: PASS

All five folds passed the pre-outcome operational feasibility gate. No validation
or outer-test outcomes were used for PCA, tau, prototypes, weights or the gate.
See [PLLR_FEASIBILITY_AUDIT.json](audit/PLLR_FEASIBILITY_AUDIT.json) and
[PROTOTYPE_STABILITY.csv](results/PROTOTYPE_STABILITY.csv).

| Fold | FIT patients | Eligible | Eligible fraction | Eligible channels | Median patient mean rank rho |
|---|---:|---:|---:|---:|---:|
| 1 | 51 | 45 | .882353 | 4,244 | .729446 |
| 2 | 51 | 49 | .960784 | 4,748 | .764763 |
| 3 | 50 | 46 | .920000 | 4,473 | .752262 |
| 4 | 52 | 45 | .865385 | 4,444 | .782697 |
| 5 | 51 | 47 | .921569 | 4,716 | .755929 |

These are FIT patient/channel **appearances across folds**, not unique cohort
counts. Original cohort membership remains 80 patients / 7,635 unique channels.

| Center | FIT patient appearances | Eligible appearances | Eligible channel appearances |
|---|---:|---:|---:|
| HUP | 116 | 109 | 11,124 |
| LZU | 70 | 70 | 7,224 |
| multicenter | 49 | 36 | 2,364 |
| pediatric | 20 | 17 | 1,913 |
| Total | 255 | 232 | 22,625 |

The 23 insufficient patient-fold groups retain all-one raw/final weights; no
missing prototypes are imputed. Class support must be >=5 in both classes and
the query is excluded, leaving >=4 same-class channels. PCA8 uses the original
FIT-preprocessed88D vectors, label-blind full SVD and FIT-only projected
population mean/std. No PCA feature or prototype enters the model forward pass.
Fold tau values are .123906/.113324/.151133/.163570/.168324, from strictly positive
FIT conflicts. None of the five folds is degenerate with no positive conflict.

## Stability and interpretation

Each eligible patient has100 prototype resamples: bootstrap other same/opposite
class channels, always exclude the query, keep original FIT tau, recompute and
class-normalize weights. Compare each resampled weight ranking with the original
within-patient ranking, then summarize patients equally. Identical constant
vectors are assigned rho1, a single constant vector rho0.

The pre-outcome conservative gate requires every fold's median patient mean rho
>=.5 and eligible fraction >=30%. All pass. Nevertheless 8.2%–15.6% of eligible
patient-fold groups have mean rho<.5; fold q10 rho ranges .455–.550. Mean absolute
weight changes are .0324–.0382 and conflict-sign changes .0892–.0964. This is
moderate resampling stability, not proof of physiological separation or clinical
label validity. Some small center groups remain particularly uncertain.

## Weight mass, minority behavior and leverage

Final weights range **.514117–1.362857**. Float64 patient/class weight-sum error
is <=5.69e-14; each observed class retains mean1, with no post-normalization
clipping. Sum of within-patient/class effective sample sizes is 23,951.22 over
24,267 FIT channel appearances (about98.7%). Full per-center/class ranges,
within-patient standard deviations and downweighted fractions are in
[RELIABILITY_WEIGHT_SUMMARY.csv](results/RELIABILITY_WEIGHT_SUMMARY.csv).

EZ and NEZ use the same support, distance and weighting rule, with no rarity
term, and class normalization preserves both class coefficient sums. However,
observed EZ channels are often downweighted more frequently than NEZ channels
(for example HUP fold1 raw fractions .391 vs .152). Symmetric construction does
not prove the geometry is unbiased against difficult minority-class channels.
That concern must be judged by the matched validation sensitivity/EZ-F1 results,
not dismissed as impossible because class mass is normalized.

One channel-mean optimizer update per patient protects against large channel
counts dominating an epoch. Final multipliers are bounded by the raw .5–1
construction and class normalization, not extra clipping. Center/class mass
shares in the CSV identify small-subgroup concentration; they are not an
assertion that per-channel loss/gradient leverage is eliminated.

## B1 control

Every patient/class B1 weight multiset exactly equals B2. Stable IDs/seeds and
canonical channel order ensure reproducibility and channel-order equivariance.
Every nonconstant group changes at least some correspondence; degenerate
all-one groups remain unchanged. Across patient groups, mean changed-channel
fraction is .333151. Many channels share equal weights, so permutation does not
change every channel; this limitation is reported, not repaired after outcomes.
See [B1_PERMUTATION_AUDIT.json](audit/B1_PERMUTATION_AUDIT.json).

Feasibility therefore admits model training. It is **not** evidence that lower
weights identify wrong labels or that PLLR will improve localization.
