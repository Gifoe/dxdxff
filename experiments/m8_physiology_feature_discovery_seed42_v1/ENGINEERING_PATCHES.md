# Execution-only engineering record

## Before integrated production training: wrong-channel axis indexing

The first integrated program stopped during its prospective whole-FIT admission, before any formal D1 or M8 training epoch and before any new Outer inference. NumPy separated advanced indices in `v[:, permutation, :, feature]` moved the channel axis ahead of seizure, yielding C×S×W rather than intended S×C×W for the wrong-pair negative control. This was a deterministic engineering error, not a failed efficacy result.

Original code, `integrated_M8` input freeze, all existing smoke artifacts, and `integrated_01.log/.err` remain preserved. The sustained wrapper verified that no formal `LAST_PRIVATE.pt` exists in that initial runtime. No completed training epoch was discarded.

`integrated_data_v2.py` uses `v[..., feature][:, permutation, :]`, explicitly selecting the scalar feature before applying the channel permutation. The independent3×64×59×21 nested-loop regression proves exact intended pairing and unchanged old7 coordinates. `train_integrated_v2.py` additionally binds both original and repaired sources and the regression, with a separate `integrated_M8_v2` runtime. The original scientific feature map, permutation, candidate lists, model, loss, budgets, optimizer, patient masks and outcome gates do not change. All whole-FIT admissions are rerun before production fits; previous artifacts are not overwritten.

The second sustained foreground command keeps an observable live process, emitting a heartbeat every20seconds. It performs no automatic failure retries. Scientific production uses the original fold trainers and exact complete-epoch resume rules; no unrelated processes are stopped.
