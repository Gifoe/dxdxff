# Historical nonlocal8 eligibility audit

This is a strictly unlabeled source/mask/operator audit, not another
classifier experiment. It closes the formula-replay part of the original
historical28 eligibility gap. See [AUDIT_REPORT.md](AUDIT_REPORT.md) for actual
counts, interpretation and the remaining provenance limits.

Actual outcome: all8 saved descriptors replay exactly across256 records;
six valid-mask Raw250 scalar operators are computable with complete coverage;
the two tied-latency rank operators fail channel-order equivariance. They
are not passed to a model. No Raw waveform/EDF is extracted, no label member
is indexed, no model or threshold is fitted and no Outer outcome is read.

The trusted-server environment is Python3.10.20, NumPy2.2.6, pandas2.3.3 and
SciPy1.15.3, one BLAS thread. Existing licensed caches and their original
reconstruction/Raw/pool registries must exist at the pinned audit locations.
The three measurement-source SHA values are fixed in `PROTOCOL_LOCK.json`.
Source functions are AST-selected so historical builder label-validation or
split code is never executed. Exact cache/source identity gates run before
any descriptor calculation.

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& 'E:\Anaconda\envs\persist_stable_251\python.exe' -u `
  'D:\m8_nonlocal8_eligibility_audit_seed42_v1\audit.py'
& 'E:\Anaconda\envs\persist_stable_251\python.exe' -u `
  'D:\m8_nonlocal8_eligibility_audit_seed42_v1\verify_audit.py'
& 'E:\Anaconda\envs\persist_stable_251\python.exe' -u `
  'D:\m8_nonlocal8_eligibility_audit_seed42_v1\export_public.py'
```

Inspect live workers before invoking a stage. Hold the foreground SSH owner
and use unique logs; do not restart an active audit. The audit recognizes
an already completed identical input freeze and does not recompute it.
No previous model, cache, feature mask or result is overwritten.

Public-only release verification:

```powershell
python experiments/m8_nonlocal8_eligibility_audit_seed42_v1/verify_release.py
```

All source/overall summaries are in `results/`. The field
`cache_safe_unchanged` is a narrow formula/native-mask invariance check, not
proof of acquisition provenance, biological validity or predictive utility.
The six Raw250 measurements are distinct candidates requiring their own
prospective numerical/missingness qualification and selection protocol.
