# Actual joint-moment Raw internal experiment

## Scope and current conclusion

All **39** registered fresh fits / **1,061** epochs and independent internal
verification completed. This is **not** the complete80-patient Outer result
or an achievement of Macro-F1>0.65. Original Development/Outer were not used
for these internal fits or their procedure selection. The separately frozen
final stage subsequently completed; its actual negative results are reported
separately in [RUN_REPORT.md](RUN_REPORT.md), not substituted for internal metrics.

The complete first-path assessment has **86 appearances / 59 unique patients
per arm**, not86independent patients. Repeated appearances remain together in
10,000patient-ID-cluster bootstrap draws. These reused/overlapping internal
data and nominal intervals do not establish independent confirmation.

## Complete first-path matrix

| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Native | 0.601974 | 0.718595 | 0.517026 | 0.370762 | 0.833186 | 0.751410 | 0.738937 | 0.604651 | -0.042319 |
| Fixed precompand | 0.604982 | 0.723093 | 0.535747 | 0.382907 | 0.827057 | 0.745545 | 0.746988 | 0.627907 | -0.027114 |
| PRE_MEAN_DUP | 0.615241 | 0.756458 | 0.564861 | 0.410108 | 0.820373 | 0.739705 | 0.770248 | 0.662791 | +0.003345 |
| PRE_ENERGY_DUP | 0.609969 | 0.733490 | 0.526831 | 0.398902 | 0.821037 | 0.741584 | 0.740174 | 0.627907 | -0.013031 |
| PRE_JOINT | 0.624301 | 0.758347 | 0.571843 | 0.419517 | 0.829084 | 0.753567 | 0.773648 | 0.674419 | -0.018165 |

Allthree fresh models have14,577parameter positions and exactly matched
initial CNN/head tensors per fold. They retain the same pre-only waveform
normalization, pre-latent reference, two phases, available-seizure mean/SD and
full-patientZ. One learned CNN supplies (mean,mean), (energy,energy), or
(mean,energy) responses to one192to64to1head. Energy means
`log1p(mean_t(learned_response^2))`, not a manually measured physiological
band. Duplicating a statistic adds no information and matches tensor/head
shape; it does not make effective information capacity identical. Fixed
historical references have different training histories and budgets.

## Mechanism contrasts, not a forced success claim

| Patient Macro-F1 difference | Difference | Nominal paired95% CI |
|---|---:|---|
| JOINT - Native | +0.022327 | [-0.005250, 0.051821] |
| JOINT - frozen precompand | +0.019319 | [-0.004665, 0.045624] |
| JOINT - MEAN_DUP | +0.009060 | [-0.006059, 0.026495] |
| JOINT - ENERGY_DUP | +0.014331 | [-0.002168, 0.031349] |
| ENERGY_DUP - MEAN_DUP | -0.005271 | [-0.020838, 0.009145] |

The joint point is highest within this matched first-path matrix, but the
paired intervals do **not** establish a Macro-F1 advantage over either
duplicated control or either fixed reference. Its fivefold Macro sampleSD is
0.037955, versus0.020125 forMEAN_DUP and0.016653 forENERGY_DUP. A higher mean
point is not proof of better fold stability. Full nine-metric fold/source,
coverage and interval tables are in [results/internal_v1](results/internal_v1).

### Strong prior compact controls are not hidden

The preceding fresh PRE_MEAN and PRE_LOGENERGY first-path Macro-F1 points were
0.625949 and0.624857 on these same86appearances/59people. They are not the
weaker frozenNative references above. After both internal programs completed,
an additional descriptive comparison matched the hash-verified retained
first-path records from allthree new arms with both prior compact PRE arms.

| Post-hoc internal context | Difference | Nominal paired95% CI |
|---|---:|---|
| New JOINT - prior compact PRE_MEAN | -0.001649 | [-0.013528, 0.011539] |
| New JOINT - prior compact PRE_LOGENERGY | -0.000556 | [-0.015523, 0.014184] |

Thus this experiment does not establish an improvement over the stronger
prior compact configurations either. These comparisons differ in head
shape/reparameterization and are not equal-capacity attribution controls.
All54metric rows for all3x2comparisons are provided, not only the joint rows.
This is explicitly **post-hoc internal context**, never an input to selection.
The already frozen fivefold choice hash was identical before and after the
comparison. It loaded no new Development/Outer samples, trained no model,
made no new prediction and changed no threshold.

## Prespecified race and retained choices

| OwnFIT fold | First candidate | Advanced? | Final choice after legal assessments |
|---|---|---|---|
| 1 | PRE_MEAN_DUP | Yes | FROZEN_PRE |
| 2 | PRE_JOINT | No | NATIVE |
| 3 | PRE_JOINT | Yes | PRE_JOINT |
| 4 | PRE_ENERGY_DUP | Yes | PRE_ENERGY_DUP |
| 5 | PRE_JOINT | Yes | PRE_JOINT |

The first15fits cover allthree fresh arms in allfive ownFITs. Onlyfolds1,3,4,5
passed the fixed gate and ran allthree arms on two additional paths:
15+4x6=39fits. The secondfold's negative result is retained and its extra fits
were not run. Selected replication tables are separate from the complete
first-path matrix; unrun cells are not imputed. No global winner from
overlapping folds is imposed on everyOuter fold. The positive firstfold
MEAN_DUP candidate did not survive replication; its final choice is the
unchanged fixed precompand reference.

These choices were frozen before new original Development access. Final
primary is the selected procedure, not a retrospectively chosen best fixed
Outer configuration. Allthree fixed fresh arms and both unchanged references
remain predeclared secondary complete fivefold comparisons.

## Actual verification and compute

- 39 fresh fits,65 logical cells including references,1,105 private patient metric
  rows; every fresh retained checkpoint is an actually trained epoch>=1.
- Allretained CAL/HELD states were independently replayed with maximum logit
  error**0**; decisions on all201CAL threshold candidates were identical.
- Allnine patient metrics, races, fold-specific choices and all126absolute/
  paired CI metric rows were independently checked. Maximum bootstrap
  difference:**9.80118763926896e-17**.
- Five unique changed-path tests passed. The generalized aggregation matched
  the original24D output **and gradients** exactly. Allthree models passed
  actual largest complete training-patient backpropagation:164channels /
  48,380validwindows, finite full gradients, no admission optimizer updates.
- The1,061formal epochs totaled2,162.357103seconds, mean2.038037s/epoch. This
  excludes admission, inference, replay, orchestration and reporting; it is not
  a whole-pipeline speed benchmark. No completed internal epoch was discarded.
- Local empty-stderr SSH pump timeouts did not stop the same live server
  trainer/coordinator. No native or Python internal-training restart occurred.

## Final admission: original roles, not an invented80-person FIT union

After the internal PASS, the first final admission stopped at an incorrect
assertion that the union of original FIT patients must be80. Role diagnostics
confirmed unchanged FIT counts51/51/50/52/51,13Development patients per fold,
and Outer16/16/17/15/16. The original FIT union is**79**, while Outer union is
**80**; one patient is never in FIT. There was no final optimizer update or
new test evaluation before that stop. No patient was added or reassigned.

The diagnosed handler `admit_exact_roles_v2.py` checks the exact original
cohort/FIT identity hashes and performs largest legal-FIT backward admission
for allthree byte-unchanged models. It passed with164channels/48,380windows,
finite gradients and zero updates. The original final input freeze and all
scientific sources remain unchanged. Original failedadmit01 logs/code are
preserved; `run_final_exact_roles_v2.ps1 -RunTag 02` uses new logs and replaces
only this admission entry. The old one-shot handoff correctly stopped on the
failure, rather than retrying it. Final fitting and subsequent30cell score
seal,480patientmetricrow verification and Raw census use the original frozen
implementation. At this report's original creation, final FIT/Development
fitting was active. The completed final performance and the later native
post-verification exit are now documented in RUN_REPORT.md.

Key SHA256 locks:

- Internal result seal: `99800c1d64c9ca68e7bd7a25ef153543f77e56e61f519208480199bf10154e6e`
- OwnFIT choices: `51963534e666e5bae8a657a3674cdb9895ce56d683507da0925b5c5c268faad3`
- Original final input: `cd08eeec8272affa4ac7dac448b47b206a725f4ec0c74abe00faf27c25b0d6f2`
- Exact original FIT identities: `574217d8957f84f472b270c325e8092f231eaf2b9804bbdb4f05c8240d29c0bc`
- 16-file compact internal export: `d17583735e5483129e0e51d3b754195107e61215307c371da161bd794cea316e`

Only code, hashes and actual anonymous aggregates are public. Raw, labels,
identities, weights, per-patient/channel predictions, private caches and
runtime logs stay on the trusted server. Historical Outer has been viewed
repeatedly; any forthcoming final test is exploratory repeated-test, not
independent confirmation. Full80patient-equal Macro-F1>0.65 remains unachieved.
