# Execution-only engineering record

## Before integrated production training: wrong-channel axis indexing

The first integrated program stopped during its prospective whole-FIT admission, before any formal D1 or M8 training epoch and before any new Outer inference. NumPy separated advanced indices in `v[:, permutation, :, feature]` moved the channel axis ahead of seizure, yielding C×S×W rather than intended S×C×W for the wrong-pair negative control. This was a deterministic engineering error, not a failed efficacy result.

Original code, `integrated_M8` input freeze, all existing smoke artifacts, and `integrated_01.log/.err` remain preserved. The sustained wrapper verified that no formal `LAST_PRIVATE.pt` exists in that initial runtime. No completed training epoch was discarded.

`integrated_data_v2.py` uses `v[..., feature][:, permutation, :]`, explicitly selecting the scalar feature before applying the channel permutation. The independent3×64×59×21 nested-loop regression proves exact intended pairing and unchanged old7 coordinates. `train_integrated_v2.py` additionally binds both original and repaired sources and the regression, with a separate `integrated_M8_v2` runtime. The original scientific feature map, permutation, candidate lists, model, loss, budgets, optimizer, patient masks and outcome gates do not change. All whole-FIT admissions are rerun before production fits; previous artifacts are not overwritten.

The second sustained foreground command keeps an observable live process, emitting a heartbeat every20seconds. It performs no automatic failure retries. Scientific production uses the original fold trainers and exact complete-epoch resume rules; no unrelated processes are stopped.

## Completed training and process-observation repairs

All80 original-format model pairs and45 logical integrated cells completed. Original code,all completed epochs,checkpoints and logs remain unchanged. Windows PowerShell returned an empty `Start-Process` exit-code property in some observers even after a successful terminal artifact. Later observers materialized the process handle while live and validated the actual exit code and atomic terminal audit. The600second silent-stderr Paramiko observation timeout was not taken as evidence of a dead training worker. No model was retrained because an observer failed.

## After all130 Outer score files were sealed: native statistical-runtime failures

The frozen evaluator completed all130 armfold score files and the score seal. It then exited0xC0000005 during aggregation; Windows Application Error and subsequent faulthandler diagnostics localized failures to native NumPy Unicode lexsort and Pandas grouping/sorting paths. Reporting-only recoveries preserved all scores and never created a model or called a predictor. A NumPy-independent ranking implementation passed33 synthetic equivalence cases but did not fix subsequent Pandas failures; it was **not adopted** in the final reports.

A CPU-only reporter executes the exact `report` and `classification` AST extracted from the hash-frozen evaluator and imports the original unmodified metric/ranking source. The existing Python3.10/NumPy2.2.6/Pandas2.3.3/sklearn1.7.2 runtime was used read-only; no package/environment used by another job was changed.35 synthetic classification/ranking cases plus10000 bootstrap draws matched the original reporting runtime at maximum metric difference0. The final report process alone was pinned to logical processor16 (mask65536), with no system/BIOS/driver or unrelated-job affinity change. This produced a successful report. The lower-level native cause is **not proven**; processor affinity is an execution workaround, not evidence that a specific hardware component is faulty.

The9 partial anonymous aggregate tables produced before an earlier reporting crash were preserved and compared with the final tables: maximum numeric difference0, no changed cell above1e-12. Independent final verification reconstructs all actual feature-transformed cache inputs in the original model runtime, then independently recalculates all2080 patient metrics,pooled/fold/source summaries,rankings and33 bootstrap comparisons in the parity-gated CPU process. No statistical definition,threshold,feature list,data cohort or prediction was changed during recovery. All failed intermediate source/log versions remain available for provenance,not as recommended production entrypoints.
