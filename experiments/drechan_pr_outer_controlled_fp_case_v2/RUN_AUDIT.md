# TP/FP case selection and verified engineering recovery

The user requested a patient with increased EZ detection and only a small FP
increase, subsequently asking for the best TP/FP tradeoff. Original frozen
validation thresholds were retained throughout, with no new fits or inference.

V1 source commit: `9894c7b02544bca6b2f093258e5e4d0a656af793`.
Its extra nondecreasing precision and stable ranking requirements left three
base candidates, none satisfying either all-seeds FP<=0 or FP<=2. No case was
selected or exported. V1's directory and source remain preserved.

A separate, private aggregate diagnostic recomputed all 640 frozen endpoints:
of 77 remaining patients after excluding the three earlier cases, 26 had mean
TP gain and at least two TP-positive seeds. Six had <=2 additional FP in each
seed; zero had all three seeds FP nonincreasing. The profile table containing
identities stayed server-private; only aggregate counts were viewed here.

V2 removes the extra ranking and precision eligibility restrictions explicitly,
after those aggregate diagnostics and before identifying/exporting this case.
Frozen rule commit: `49dba1624e887ced72358a26621cd706cfe9c938`.
Source archive SHA256:
`79d643c5c55ac8734676328f3007a2113a8c2c62a77597151124fdd37206f837`.
Choose the first nonempty FP tier, maximize mean TP gain, then minimize FP change,
then use ranking gain and a fixed hash for ties. All outcomes remain exploratory.

The first V2 run terminated with native access violation -1073741819. The
faulthandler stack ended in parent endpoint line46, constructing the independent
channel ordering. Only PREANALYSIS_LOCK.json existed in the failed runtime;
there was no selected case or export. Underlying cause is not established.
The failed runtime was preserved. No OS, environment or threshold changes made.

Engineering recovery source: `engineering_pure_python.py`, commit `8d8042c`.
Its distinct deployment archive SHA256:
`98fe389779b4ae13eaebc1fe3bb4f571ce1d9c0d7766669fb1d1286872d038bd`.
The identical frozen V2 main/selector ran under a distinct output directory with
independent Python endpoint, rank, confusion and percentile formulas. The
unchanged original admission reader still uses NumPy for input loading/checks.
All 640 endpoints and 400 patient-change rows replayed within 1e-12 against the
previously completed frozen-score tables; the change-table hash matched its
original selection lock. Old failure lock rehashed unchanged after recovery.
An engineering binding with code hashes and runtime override stays private.

Successful runtime:
`D:\Codex_Research\experiments\drechan_pr_outer_controlled_fp_case_v2_runtime_pure01`.
Selected CASE-F01: HUP, fold2, 84 channels / 16 EZ. Tier counts [0,6]. Mean TP
change +4.6666667, mean FP change +0.6666667, maximum per-seed FP change +1;
all three seeds increase TP. Selection maximum independently agreed.

Only the exact public archive was downloaded, whole SHA256:
`dbd107fe9781f8d06607368ef9588f6223037af80d3661653e4b1d189c76a9d8`.
The 13 anonymous payloads plus manifest passed release_check.py BEFORE extraction.
Local independent verification reconstructed all 25 original metrics, 10 confusion
rows, all 672 score rows/ranks, and all 680 Recall@K rows. Wide files contain
84 channels and 16 EZ-only channels. No original identity or contact name was
downloaded; old cases remain unchanged. Following the user's explicit upload
request, the GitHub release contains anonymous case outputs, protocols, scripts,
the report and the V1 negative-screening record. Original identities, channel
names, private cohort profiles, runtime logs and transport archives are excluded.
