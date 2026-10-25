# Task A execution, provenance and privacy audit

## Isolation and pre-computation freeze

Work was isolated in the independent branch
`codex/drechan-sixfeature-ez-association-v1` and worktree
`D:\chenyu-iclr\dxdxff_sixfeature_ez_association_v1`.
The ongoing LOCO worktree, production coordinator, scientific sources,
training/checkpoints and heartbeat were not changed or relaunched.
No old experiment report, data or frozen model was edited.

The initial protocol/statistics commit was
`0f4a225d970ae45f5a1d1d6bcb3c271573de45c9` (2026-10-25 17:54:02 +08:00).
An encoding-only input-reader correction and figure renderer were pushed in
`f753d677e00bbde4ff335ca45decdedf5fe776cc` (18:00:48 +08:00), before any real
association outcome was calculated. Both pushes were verified by git ls-remote.
The fixed 12 coordinates, effect definition, direction, seed, test family,
bootstrap, permutation procedure and inference thresholds did not change.
The five authoritative analysis source hashes are recorded in
results/INPUT_AUDIT.json and results/PACKAGE_MANIFEST.json.

Corrected source archive SHA256:
`d03b8936b05cdf7bcc4fac72d238970fb4e51c04933a6987a2bdc0f73c89fb62`.
Only this task's new server code/runtime paths were moved to preservation
locations before fresh deployment; original feature/model/LOCO directories
were not moved or overwritten.

## Engineering incidents before association calculation

1. The first PowerShell wrapper classified unittest's normal stderr progress
   output as NativeCommandError under ErrorActionPreference=Stop. It aborted
   before running association statistics. No worker was blindly duplicated.
2. The second wrapper used hidden Start-Process workers with independent stdout
   and stderr files. Tests passed, but reading UTF-8 PROTOCOL.json through
   Windows' default GBK encoding raised UnicodeDecodeError before INPUT_FREEZE
   or any effect calculation. Explicit UTF-8 readers corrected the transport/
   serialization boundary, not the statistics or feature definitions.

Before restarting, the exact owned paths were resolved and checked, no active
Task A Python worker existed, and no feature_association.csv or INPUT_FREEZE.json
existed. Failed source/runtime/logs remain on the trusted server as:

```text
D:\drechan_sixfeature_ez_association_v1_encoding01_preserved
D:\drechan_sixfeature_ez_association_v1_runtime_encoding01_preserved
```

No outcome was used to choose a repair, direction, threshold or rerun seed.
No formal association calculation was discarded.

## Actual successful execution

Server code: `D:\drechan_sixfeature_ez_association_v1`.
Runtime: `D:\drechan_sixfeature_ez_association_v1_runtime`.
Python: `D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe`.
CPU analysis/verifier affinity: CPU0; OMP/MKL/OpenBLAS threads: 1 each;
CUDA_VISIBLE_DEVICES empty; no torch import, neural fit or raw extraction.
Existing native OpenSSH/SCP pinned Ed25519 verification was retained without
relaxing authentication/host checks. No credentials entered code or reports.

Actual stage argument vectors (with -u and separate log/err redirection):

```text
python -u -m unittest discover -s D:\drechan_sixfeature_ez_association_v1 -p test_statistics.py -v
python -u D:\drechan_sixfeature_ez_association_v1\run_analysis.py analyze --runtime D:\drechan_sixfeature_ez_association_v1_runtime
python -u D:\drechan_sixfeature_ez_association_v1\verify_analysis.py --runtime D:\drechan_sixfeature_ez_association_v1_runtime
python -u D:\drechan_sixfeature_ez_association_v1\run_analysis.py package --runtime D:\drechan_sixfeature_ez_association_v1_runtime
```

The successful inline foreground wrapper used hidden Start-Process workers,
retained the process handle, waited for actual exit codes and stopped on any
failed stage. The equivalent reusable run_cpu.ps1 is supplied for a deliberate
reproduction **in a new runtime**, not as a reason to rerun completed statistics.

Observed server timestamps (+08:00):

| Stage/log prefix | Started | Last write | Outcome |
|---|---|---|---|
| tests03 | 18:04:37 | 18:04:38 | Six tests passed; normal unittest progress in stderr |
| analysis03 | 18:04:42 | 18:04:48 | 960 effects; stderr 0 bytes |
| verification03 | 18:04:52 | 18:04:58 | Independent PASS; stderr 0 bytes |
| package03 | 18:05:02 | 18:05:03 | Anonymous archive complete; stderr 0 bytes |

Date: 2026-10-25. Full .log/.err files remain in the runtime on the trusted
server and are not published. A subsequent read-only process check found zero
remaining Task A workers. Source/read-only cache hashes were checked again
after computation.

Environment: Python3.13.9, NumPy2.3.5, SciPy1.16.3, pandas2.3.3. Local rendering
and release checks used Python3.10, NumPy2.2.6, SciPy1.15.3, pandas2.3.3,
matplotlib3.10.7. Wilcoxon approximation/tie/zero behavior was fixed explicitly,
not delegated to version-dependent automatic method selection.

## Independent numerical evidence

PASS covers all 960 patient-coordinate effects using direct pairwise signs and
sklearn ROC AUC versus average-rank U; maximum effect difference 1.11e-16.
All 60 patient-bootstrap intervals were independently recomputed with resample
count matrices, all 12 Wilcoxon tests with tie-corrected rank variance and
normal-tail probabilities, and all 12 BH q-values with SciPy's separate FDR
implementation. All 48 source rows and 9.6 million permuted effects were also
checked; pairwise dominance scores verified the rank-based permutation engine.
The 10,000 joint label permutations preserved each patient's EZ count exactly.
All inputs, 80 summary caches, labels and mappings remained unchanged.

## Authenticated aggregate release

Only the explicit anonymous aggregate archive was downloaded:
`D:\drechan_sixfeature_ez_association_v1_runtime\public_aggregate_export_v1.zip`.

Archive: 13,613 bytes; SHA256
`1f51f82d385a71683b4e10d729b364a05a6877093a53a884934d18a637352537`.

The authenticated server package hash, transfer hash and local hash matched.
release_check.py verified the schema, exact 10-payload-plus-manifest allowlist,
all payload hashes, five immutable analysis source hashes, independent PASS,
coverage and aggregate-only CSV headers **before extraction** into results/.
The locally generated LaTeX table is not misrepresented as an extra server
payload; its numbers are generated from the verified CSV.

Public patient_effect_summary.csv has 12 coordinate-level distribution summaries.
The actual 960 individual rows, exact patient effects/matrices, labels,
identities, raw data, channel rows and logs remain server-only. Fixed-bin
histograms contain no patient identifier or exact individual measurements.
No weights, preprocessors, scores, waveforms or caches are published.

## Figure and document QA

The nature-figure workflow fixed 178 mm double-column sizing, editable SVG/PDF
text, 7–10 pt fonts and 600 dpi PNG exports. All three rendered PNGs were
visually inspected for labels, signs, zero lines, intervals, q-values and
source reversals. SVG text and embedded PDF TrueType fonts were checked.
Actual raster and vector dimensions agree; all text stays on the canvas.

Static figure preflight: 12 PASS, 2 WARN, 0 FAIL. The PNG-without-TIFF warning
is expected because PDF/SVG/PNG were requested. The width warning is a static
parser false positive: it reads 178*MM as 178 inches; actual checked width is
178 mm in all three formats. Warnings do not alter data or hide a rendering
failure. Detailed checks are in FIGURE_QA.json and ARTIFACT_CHECK.json.

The report and LaTeX table are generated from verified aggregates, retaining all
12 results and all source reversals. Local QA checks report numbers against
CSVs, source-weighted means against the overall patient mean, direction-count
percentages, public schemas, figure dimensions and every server-payload hash.
Final actual Git-index/committed-tree release checks are recorded separately
in RELEASE_AUDIT.json.
