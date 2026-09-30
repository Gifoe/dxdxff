# Engineering repair ledger

## Aggregation contradiction

The frozen official evaluation first applies sigmoid to each segment and then
averages probabilities within `(EDF, channel)`. Its frozen AUROC is
0.7987673466. The experiment prompt also labels `sigmoid(mean(logit))` as the
official aggregation, but that transformation gives 0.7901418977 on the same
frozen predictions and therefore fails the prompt's own exact-replay gate.
The primary implementation retains the actual frozen evaluator: one
channel-level residual is added to every segment logit, followed by sigmoid
and the original mean. The mean-logit calculation is diagnostic only. No test
outcome was used to choose this resolution; it is forced by the predeclared
baseline identity requirement.

## Native extraction exits

The stable PyTorch 2.11.0+cu128 runtime on the Windows RTX 5090 host can exit
natively between EDF files without Python stderr. Each EDF cache is written
atomically and bound to the protocol, source, and checkpoint hashes. A bounded
supervisor repeats the exact frozen extraction command only when stderr is
empty and completed-file count increased. It stops on Python error, three
consecutive no-progress exits, or its fixed attempt cap. Batch size was reduced
from 32 to 8 to reduce peak CUDA memory; this changes neither preprocessing,
weights, outputs, nor aggregation.

## TEST extraction transport and sharding

The first TEST extraction connection was closed by the local SSH wrapper after
60 seconds without returned stdout. Windows OpenSSH then cleaned up its child
process at 67/237 EDFs; there was no Python or CUDA error. A foreground watcher
with a 20-second remote heartbeat was used thereafter. The extractor's existing
hash-bound atomic resume retained completed EDFs. To overlap CPU preprocessing
with intermittent GPU kernels, the remaining fixed file list was partitioned by
the pre-existing `files[shard_index::3]` rule into three disjoint batch-8
workers. The workers shared no EDF and did not alter inputs, weights,
preprocessing, aggregation, or selection. Peak observed allocation was about
24.2/32.6 GiB. No metric or prediction outcome had been read before this
engineering-only restart.

## Aggregate evaluator native exits and user stop

The legacy Python 3.11 environment exited with Windows `0xC0000005` during
module import and produced no output. The stable Python 3.10 environment then
exited in NumPy `_multiarray_umath` during cache loading. A third compatible
environment reconstructed all 237 caches and atomically wrote `TEST_METRICS`
and `PATIENT_METRICS`, then exited after the 10,000-draw loop but before the
bootstrap table was written. Each failed process had empty Python error output;
Windows Application Error 1000 identified the native NumPy module. No model,
checkpoint, threshold, cohort, or prediction was changed, and no test result
was used for selection. A NumPy-only export of both frozen heads was checked
against PyTorch at max absolute error `2.98e-8` as a recovery path.

After the primary metric table showed PR-Residual AUROC below both FrozenCNN
and ABS-only, the user explicitly directed the run to stop and upload. The
unfinished bootstrap, center, relative-zero, and shuffle diagnostics were not
resumed; their absence is recorded rather than filled with partial or
fabricated values.
