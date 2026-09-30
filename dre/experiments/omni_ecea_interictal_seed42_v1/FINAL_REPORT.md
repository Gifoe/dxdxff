# ECEA final report

| Model | New params | AUROC | AP | Macro-F1 | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen TimeConv-CNN | 0 | 0.798767 | not re-estimated | 0.659754 at 0.5 | not re-estimated | not re-estimated |
| Tail only | 1 | not run | not run | not run | not run | not run |
| Event pool only | 1 | not run | not run | not run | not run | not run |
| Event rate only | 1 | not run | not run | not run | not run | not run |
| Full ECEA | 3 | not run | not run | not run | not run | not run |

## Terminal

`EVENT_EVIDENCE_UNAVAILABLE`

This is not a negative ECEA result. It is a protocol-mandated availability stop. The official code requires an external event-model checkpoint, but the official repository does not bundle it. The server has 1,014 HFO candidate CSVs and 54 expert-annotation parquet files; neither is a frozen pathological-event inference product covering official Task 2. No server result file contains `3label_pred` or `pathological_probability`, and no official event checkpoint was found.

Treating every HFO candidate as pathological would materially change the scientific question and violate the explicit fallback rule. Training a new event classifier would also violate the locked protocol. Both shortcuts were rejected.

## Required questions

1. **Baseline exact replay?** Yes. Frozen predictions reproduce AUROC `0.7987673466324111`, absolute error `0`, on 8,104 labeled EDF-channel pairs.
2. **TRAIN OOF delta AUROC?** Not estimable; OOF fitting was prohibited after the event-source gate failed.
3. **Event pool / event rate / CNN tail contributions?** Not estimable. Running tail-only after the event-source terminal would answer a different experiment.
4. **Final lambda, beta, gamma?** None were fitted.
5. **Is lambda positive?** Not applicable.
6. **TEST ECEA AUROC?** Not run.
7. **Does ECEA exceed 0.798767?** Unknown.
8. **Does ECEA exceed published 0.8061?** Unknown.
9. **Does paired bootstrap support a gain?** Not run because no ECEA TEST predictions exist.
10. **Does event shuffle reduce AUROC?** Not run.
11. **Are event burden and CNN segment scores more coupled in pathological channels?** Not testable without pathological-event predictions.
12. **Which component supplies the gain?** No gain was estimated.
13. **Does the evidence support “uniform temporal averaging discards sparse pathological evidence”?** No. The hypothesis remains untested; the required event evidence is absent.

## Data-access and tuning statement

- Frozen historical CNN predictions were reused only for the locked identity replay.
- No new ECEA official TEST evaluation was performed.
- No Task 2 label was used to train or modify an event detector.
- No test-driven tuning or v2 iteration occurred.
