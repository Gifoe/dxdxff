# Score semantics

- Positive channel class: official Omni Task 2 pathological channel.
- Frozen CNN segment score: `sigmoid(cnn_logit)` after the already-audited official label and score orientation.
- Baseline channel score: arithmetic mean of segment probabilities within `(edf_name, channel)`.
- Prohibited alternative: `sigmoid(mean(cnn_logit))`; it does not reproduce the frozen AUROC.
- Official event loader mapping at revision `57c22a75...` assigns class 2 to `artifact == 1 and spike == 1`; ECEA's predeclared pathological event score would therefore use `softmax(logits)[:, 2]`. The source comments and dataset-card wording for classes 0/1 are not fully consistent, so this audit does not relabel them.
- Raw HFO candidate presence is not a pathological-event prediction and is not assigned score 1.

No ECEA score was constructed because no frozen class-2 event checkpoint or output was available.
