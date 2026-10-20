# Execution audit

- Scientific source pushed and remote-verified before any new fit: `86039bce7ad5e8f0990d39061667b5cae12b7f1f`.
- Real-server admission proof pushed and remote-verified before any new fit: `02bfef51cf2ae630898509e43f9a613aa7a3c6b3`.
- Admission:11 executable tests passed;285 matched initialization cells checked;45 current seed42 cells and5 historical Raw-only cells passed all strict reuse gates.235 fresh fits,285 logical cells.
- All3 seeds matched the original unchanged no-PR forward/loss/all gradients/clipped Adam/RNG update.19 complete-patient paths passed two discarded admission updates each. These discarded checks are not formal fits or performance estimates.
- Foreground-owned sequential production started at2026-10-20T22:25:27+08:00, RunTag `production01`. No Raw copying, subsampling, reduced patient count, reduced budget, simultaneous model training or changes to frozen scientific sources.
- All285 selection cells completed:235 fresh fits,50 strict reuses and2493 fresh training epochs. The model-selection archive hash is `43e7d8778b65c6ecd959b4a44a2159c6d0ed3a79145458900f70e684ab44aeed`.
- All285 validation replays, selected epochs/thresholds and independent FIT-only preprocessing checks passed before unfinished Outer. Pre-Outer freeze SHA256: `c0bfea48a8ce3c4d6ffaa5783afdc20f3b621bbe00606086451c29149ca6834c`.
- The local foreground monitor socket reset once; the original server coordinator and training workers continued and advanced with stderr0. A new monitor attached to the existing coordinator. No formal trainer was restarted, no completed epoch was discarded, and no scientific file, optimizer, seed, loss, budget or threshold rule changed. A later new-connection SSH banner error resolved on the next read-only connection; it did not affect the running job.
- All285 Outer score cells were sealed before reporting. The score-seal SHA256 is `27ea77daec81231170c9686a54496d3d61b63383e19d1441c6ca06ef9c3ad011`.
- Completion criteria:19groups ×3seeds ×80patients,4560 patient/seed rows,41040 metric values,351 nominal and36 additional-family paired interval checks. Final anonymous aggregate archive is accepted only after independent verification and release hash/schema checks.
- All conclusions remain exploratory repeated-test. No ablation is required to deteriorate; adverse results are retained and do not trigger changes to the fixed Full or its thresholds.
- Release-only `release_check.py` does not participate in training, preprocessing, scoring or source bindings; it checks the17 immutable scientific-file hashes, executable tests and the final explicit anonymous-export schema.

## Final verification and anonymous release

- Final status: `COMPLETE_VERIFIED_CFA3_19_GROUP_3SEED_FIVE_FOLD_OUTER`.
- Independent verification PASS:4560 patient/seed rows,41040 metric values,351 nominal intervals and36 additional-family intervals. Maximum independent interval discrepancy: `5.551115123125783e-16`.
- All17 frozen scientific files remained byte-unchanged from admission through completion. All6 production-stage stderr files have length0; the transport-monitor errors were separate from these stage logs.
- The completed public archive has287630bytes and35 members (34 explicit anonymous payloads plus manifest). SHA256: `7ace5189f176054b33c1e6da7ba86231d944fea59ec457b5c42f2694cb2ed16b`.
- Only this archive was fetched after independent PASS. Local archive SHA, exact member allowlist, payload hashes, privacy headers and expected aggregate row counts all passed; the11 executable tests passed again. No raw signals, identities, individual scores, weights, preprocessing objects, caches or runtime logs were fetched or committed.
- An actual staged Git-index archive containing this study plus the4 immutable parent `model.py` dependencies passed the11 executable tests and all17 scientific-source hash checks. Its34 staged public payloads matched the server-export manifest byte-for-byte; the public directory contained exactly35 manifest-bound members.
- Full three-seed equal-patient averages:Macro-F1 `0.6376412950985306`,AUROC `0.764881739568939`. Thus experiment execution/verification is complete, but the mean-F1≥0.64 performance objective is not met. No Outer-selected alternative replaces the fixed Full.
