# Independent-channel patient reference transfer

Implementation and protocol are locked before real outcomes. This isolated CPU
experiment evaluates all80 patients/7,635 channels with100 fixed bidirectional
unlabeled reference/query partitions. No classifier,raw encoder or new extraction.
The primary endpoint is held-out marginal Gaussian CRPS reduction (GLOBAL-FULL),
with same-source and centering/scaling controls. A separately controlled EZ/NEZ
relative-atypicality gap tests residual association, not information conservation.

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
unchanged. Completion and scientific interpretation require actual verified data.
