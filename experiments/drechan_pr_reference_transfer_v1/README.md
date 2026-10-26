# Patient reference transfer across disjoint channel sets

Completed on all **80 patients and 7,635 channels**, with 100 fixed bidirectional
unlabeled reference/query partitions per patient. No classifier, raw encoder or
new extraction was used. Independent numerical verification passed.

| Background probe | CRPS (lower is better) |
| --- | ---: |
| Global | 0.450934 |
| Same source | 0.452609 |
| Center only | 0.313221 |
| Scale only | 0.435714 |
| Full PR | 0.173431 |

The primary Global-minus-Full gain is **0.277503**, 95% CI **[0.221522, 0.351705]**.
The ratio of pooled mean errors decreases by **61.54%**, and **79/80** patients
improve. Full also improves on same-source references and center-only estimates
under the prespecified six-comparison adjustment. Scale-only versus Global is
inconclusive.

The EZ-minus-NEZ relative-deviation gap is 0.958464, but its adjusted interval
**[-0.250800, 2.561005] includes zero**. The complete mechanism chain is therefore
not established. Results support transferable patient-specific background
estimation; they do not establish conserved EZ information or biological causality.

See [RESULTS.md](RESULTS.md), [FINDINGS.md](FINDINGS.md),
[中文结论](CHINESE_SUMMARY.md), and the [paper figure](figures/pr_reference_transfer.pdf).

Reuse the exact parent80-person assessment mapping and legal TRAIN masks.79 people
are originally FIT in their assessment path;one never-FIT person follows the
previously declared original-validation exception. All80 are absent from their own
TRAIN/CAL;CAL is unused. This historical cohort is not independent confirmation.

Read `PROTOCOL_LOCK.md` and `PROTOCOL_LOCK.json` for complete definitions. Frozen
scientific files include an independent replay and mathematical/isolation tests.

Trusted inputs use existing environment aliases `PRM_RECONSTRUCTION`, `PRM_LABELS`,
`PRM_FEATURE_PR`, `PRM_UNIT_PR`, `PRM_INTERNAL_BANKS`. Set `PRREF_SOURCE_COMMIT` to
the pushed source revision. Exact private locations and identifiers remain on the
server. Run with a new,previously nonexistent private runtime:

```powershell
python -m unittest discover -s experiments/drechan_pr_reference_transfer_v1/tests -v
powershell -NoProfile -File experiments/drechan_pr_reference_transfer_v1/run_all.ps1 -Runtime <private-runtime> -Python <trusted-python>
```

Stages:admit,freeze,evaluate,verify,report,package. Every stage binds input and
source hashes. References are frozen before endpoints;all endpoints sealed before
verification/report. Only an explicit anonymous aggregate allowlist is exported.
Existing experiments and the earlier negative transferable-SNR finding remain
unchanged. The source commit frozen before inference is
`30549f7cbc4391f6574d661bb514e91273ab9f18`.

Verification reconstructs 180,260 endpoint values, 1,040 aggregate interval rows,
six adjusted intervals and 5,000 aggregate permutation values. Maximum endpoint
discrepancy is 1.82e-12. The authenticated archive contains 12 anonymous payloads
plus manifest, SHA256 `85f52aa7a4b9f9811844b9f825671bfb1881350134c9a0a917531b6a9719456e`.
Eight scientific files and two parent dependency files remain byte-identical.
