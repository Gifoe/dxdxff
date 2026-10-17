# Verification-only native failure,2026-10-17

The first verifier completed actual Raw replay of all15retained model/control
cells and their per-patient metrics,then died in the pandas group/cluster
bootstrap stage. Python faulthandler reported Windows access violation.
ApplicationError1000 identified pandas internals.cp313-win_amd64.pyd,
exception0xc0000005 in that process. This localizes the observed native
fault;it does not establish the underlying hardware/library root cause.

All10fits/108epochs and all checkpoints were already complete. Model-freeze
SHA9aeb0770c177993048dff7585ca23cb22acd06147766b4797f61eb607108d361
and candidateSHA24eb24e9dcbdb38665a69b122b979cf7fc2e2e601d0b56c35b29ad3c40810658
matched their pre-failure seals. The worker was confirmed gone.

RunTag02 invoked the exact same unchanged run_internal.ps1. Admission and
training terminal gates skipped those stages;only verify_internal.py ran
in a new process with fresh02stdout/stderr. No retraining,checkpoint/seed/
threshold/objective/source/environment change or discarded epoch occurred.
All original01logs and sources are preserved. Second verification exited0
with all1755metric/54interval checks passing and empty stderr.

No native fault was silently called a successful first run. No invalid-data
or scientific gate was retried away. There was no Outer evaluation in either
attempt,and no automatic blind retry loop was installed.
