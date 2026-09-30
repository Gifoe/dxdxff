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
