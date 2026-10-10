# Implementation contract

Scientific protocol written before new analysis. The only two representations
are R0 identity and R1 FIT quantile(.10) IQR floor >=.001 plus [-8,8] clipping.
All other density/MAP/decoder functions are copied from the frozen predecessor;
test-only historical transform substitution proves exact moment, hierarchy,
prior and inference parity. No scorer or teacher training entrypoint exists.

The source manifest binds 101 prior private files. Original teacher checkpoints,
OOF files, identity digests, coverage and TRAIN preprocessor hashes are checked
again. Frozen control predictions exactly reproduce every historical metric.
Previous independent model/teacher forward replays are retained and hash-gated;
this run does not regenerate scoring outputs or replace cached predictions.

Full FIT densities and patient-held-out LOO checks precede the admission seal.
Each query is absent from fitting labels, source priors and the R1 floor.
Other teachers' upstream TRAIN/selection exposure is counted and disclosed.
No fully nested independence claim is made. Class/source counts are operational
metadata, not clinical outcome predictors. Unknown source uses global fallback.

C0–C9 controls cover constant/tiny-IQR/reversed/correct mixture/unknown source,
clipping and saturation ranking ties, target-label isolation, membership guards,
all-k parity and deterministic execution. Repeated appearances are retained by
10,000 unique-ID paired cluster draws. Missing metrics are never filled.

The severe-collapse term has a pre-outcome operational definition: any invalid
LOO/numerical/MAP result, or >20% near-boundary/all-or-none decoded counts in any
outer FIT group blocks admission. This is screening, not an optimized cutoff.
No outcome-driven lower bound, alternative Gaussian or fallback is permitted.

Only admitted combinations have a VAL inference path. Unlabeled stored VAL
score transport is audited after admission and cannot feed back into fitting.
Public artifacts contain only hashes, counts, aggregate metrics and diagnostics.
Patient/channel records, proportions, clinical labels and logs stay private.

## Completed verification

Twelve pre-analysis tests passed: C0–C9, exact predecessor-family parity for
both transforms, and real FIT-only floor/bound smoke. The protocol SHA-256 is
199dde0bd3df875b0f57c3204ddde882ef3aae396a0927bf0fbe8ee96f7a6ef9.
Both immutable controls reproduce all eleven historical development metrics.
Twenty full-FIT numerical checks and 1,020 LOO patient appearances completed;
none of the four utility gates passed. No posterior VAL inference ran.

The exact second execution has identical aggregate bytes and private density,
floor, LOO and validation objects. All predecessor private files remain intact.
LOO clipping exceedances are diagnostic; the specified clipping-quality gate
is applied to each full outer-FIT fold, without adding a new LOO cutoff.
Transport code discards labels before normalization summaries and performs
no VAL class-conditional fitting. No code or scientific rule changed after
outcomes were read. Reporting-only README/ledger additions summarize results.
