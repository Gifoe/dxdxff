# Additional AUROC-positive TP/FP case

Requested after CASE-F01: keep TP gain and low FP increment, and require positive
DRE three-seed mean AUROC change. Original thresholds, all three seeds and the
complete original Outer cohort stay fixed. Exclude all four earlier cases.
The requested retrospective extension is not independent confirmation.

Protocol/source commit: `493a7d35ba70d419c788a5e990a7460b0e112f38`.
Whole source archive SHA256:
`825160ee9334d1a9fd9f3c310b37062852964a86c1096963cbdc0861ffd5586c`.
Seven explicit files were verified and deployed under Codex_Research. A single
foreground process used the already verified Python endpoint implementation
from the beginning. Its unchanged source SHA256 was checked:
`f53ca3f7814ac209e3e2723471e4f10fda75bfdaac23dd44924b3a31e93c8cee`.
The original NumPy-based admission reader still verifies frozen inputs. There
was no new model fit, calibration, inference, threshold selection or ensemble.

All 640 patient-condition rows and 400 patient-change rows replayed within
1e-12 against the completed frozen-score tables; all protected parent hashes
remained unchanged. Eleven base candidates met TP and mean AUROC direction
criteria; FP tier counts were [0,3]. CASE-A01 maximizes mean TP gain in the
three-person eligible tier, using the locked secondary criteria for ties.

Selected case: HUP, Outer fold4, 49 channels, 10 EZ / 39 NEZ. Mean TP change
+1.6666667, FP change +0.3333333, AUROC change +0.0170940; maximal per-seed FP
change +1. Two seeds increase TP and two increase AUROC. Seed44 AUROC declines;
mean precision declines and fixed-budget mean Recall10/20 is unchanged.
These outcomes are retained in the report and all per-seed data.

Successful trusted runtime:
`D:\Codex_Research\experiments\drechan_pr_outer_tp_fp_auroc_case_v1_runtime`.
Only the exact anonymous public package was fetched, whole SHA256:
`5030436c5008cfa5cb8158f1f7c6040ad071a832e4d6741127e50da63bc9b0fc`.
Thirteen payloads plus manifest passed release_check.py BEFORE extraction.
Local independent verification reconstructed 25 metrics, 10 confusion rows,
392 channel-score rows/ranks and all 400 Recall@K rows. Wide tables have 49
channels and 10 EZ-only rows. Identity and original channel names stay private.
No old case result was modified. The user subsequently requested seed42 on
GitHub. Publish only the DREChanNet seed42 derivative under results/seed42,
with five metrics, two confusion rows, 98 score rows, 100 Recall@K rows and
49-channel / 10-EZ wide tables. The subset records the whole parent archive
SHA and original payload manifest, along with the case and seed selection
disclosure. Complete other-seed outputs stay local and on the trusted server.
No identity, original contact name, private profile or transport ZIP is included.
