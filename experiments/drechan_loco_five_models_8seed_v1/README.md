# Four-source LOCO: five models, fixed seeds42 through49

Prospective fixed additional batch45-49 AFTER viewing historical42-44 results.
The user requested seeds that make Full win; this protocol does not treat test
wins as a selection or stopping rule. All8 seeds remain in the primary analysis.
No guaranteed result. No threshold, model, source or seed subset promotion.

160 logical cells:60 strict historical42/43/44 reuses and100 fresh main fits,
plus60 fresh FIT-only OOF SVM calibration auxiliaries. All5 models and4 sources
retained. ORIGINAL seed42 FIT/validation/test roles stay fixed across all8 seeds.
All160 choices freeze before ANY NEW test; all160 score cells seal before metrics.
Patient endpoints are averaged over8 seeds BEFORE source/patient averaging.
All9 endpoints, perseed, per-source, source-equal, pooled separately, sampleSD,
coverage,405 nominal and8 adjusted main-family intervals are reported. Independent
verification must check3200 patient-seed rows /28800 endpoint values.

No new data/extraction/features/architecture/search/sampling/budget changes.
Same 80patients/7635channels/256seizures/1471965validRaw250windows.
Original FP32 DREChannNet and BF16train/FP32eval original TimeConv grad-cache.
FIT-only preprocessing and SVM OOF calibration; validation-only epoch/threshold.
Same noVariance/noEarly/noLate/noHeadInteraction six-feature single-head main.

Server code/runtime/weights on D, cached inputs read in place. Source, real
admission proof and GitHub remote commit verified before formal fitting.
Only anonymous explicit-allowlist aggregate exports; identities, waves, patient
scores, checkpoints, scalers, caches, logs and credentials never leave server.
Old three-seed sources/results remain unchanged. No installed research/extraction
skills or subagents. Experiment remains exploratory repeated-cohort leave-one-
SOURCE-out; source labels are not verified individual hospitals.

Status: COMPLETE_VERIFIED_LOCO_5_MODELS_4_SOURCES_8SEED on2026-10-26.
Independent PASS:160 cells (60 strict reuses /100 fresh),3200 patient-seed-model
rows,28800 endpoint values,405 nominal and8 main-family intervals; maximum
aggregate numeric error1.1102230246251565e-16. These are80 independent patients,
not3200 patients. Production02 exactly resumed one host-memory failure; see
[RUN_AUDIT](RUN_AUDIT.md). All old42-44 artifacts and frozen science remain unchanged.

## Actual eight-seed results

The full model has the highest80-patient mean Macro-F1 (0.623160), but not the
highest AUROC (Feature-only0.748471 versus Full0.746108). It does not lead every
source. Each cell below is Macro-F1 / EZ-F1, averaging patient endpoints over
ALL seeds42-49 before patient means; no probabilities are ensembled.

| Held-out source | n | Raw + Feature | Raw-only | Feature-only | Linear SVM | TimeConvCNN |
|---|---:|---:|---:|---:|---:|---:|
| hup | 36 | 0.648870 / 0.459778 | 0.609630 / 0.415091 | 0.657995 / 0.450493 | 0.662056 / 0.465651 | 0.422361 / 0.124739 |
| lzu | 21 | 0.558922 / 0.375998 | 0.539120 / 0.343898 | 0.563502 / 0.410307 | 0.522811 / 0.319013 | 0.454456 / 0.316290 |
| multicenter | 15 | 0.655373 / 0.447934 | 0.627618 / 0.397295 | 0.597527 / 0.419897 | 0.615796 / 0.433833 | 0.476975 / 0.234798 |
| pediatric | 8 | 0.615692 / 0.376332 | 0.617398 / 0.379968 | 0.589796 / 0.322467 | 0.594418 / 0.346032 | 0.342109 / 0.155341 |

Both source-wise F1 rankings favor SVM on HUP, Feature-only on LZU, Full on
multicenter, and Raw-only on pediatric. Rankings are descriptive point estimates,
not proof of source-level superiority. Pediatric has only8 patients.

For Full minus its own Raw-only, Macro-F1 +2.79pp (8-comparison-family-adjusted
CI[+1.20,+4.55]) and AUROC +3.77pp (CI[+1.28,+6.67]) remain positive. Both primary
intervals cross zero against Feature-only and linear SVM. This is a complete-input-
pipeline comparison, not isolated causal attribution to PR or any one feature.

See [FINDINGS](FINDINGS.md) for all9 metrics, all8 individual seeds, mean/sampleSD,
paired intervals, source-equal results and limitations. Exact anonymous source×seed,
pooled-per-seed, coverage and validation replay tables are under [public](public/REPORT.md).
[public_admission](public_admission/PACKAGE_MANIFEST.json) retains21 pre-fit proof
payloads; the final [manifest](public/PACKAGE_MANIFEST.json) authenticates42
explicit anonymous payloads. Final archive SHA256:
`c879e83e667f4e1f02c6f45218667587000bc004229b00cb1f605ef1696e7aa1`
(112318bytes). It was authenticated and release-checked BEFORE allowlist extraction.

## Reproducible release checks

Run against the actual Git-index/committed-tree archive, not only loose files:

```powershell
python experiments/drechan_loco_five_models_8seed_v1/release_check.py --package <authenticated-public-archive.zip>
python tools/drechan_loco_3seed_release_tests_v1/verify_parent_math.py
python tools/drechan_loco_8seed_release_tests_v1/verify_public.py
```

The six CPU tests use the prior3seed common/pipeline/svm/timeconv_engine, original
seed42 parent and original m8_raw_backbones_pr_seed42_v1/evaluate.py from that
same archive. The release-only verifier checks payload/source bytes, all60 old
selection bindings,75 old seed/source aggregate rows and complete80-patient
coverage. It imports no private server data. Never publish individual scores,
signals, IDs, weights, scalers, caches, embeddings or runtime logs.
