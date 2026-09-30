# ECEA seed42

This directory implements the frozen-source audit and the soft-probability-only serialization change for ECEA.

The run terminated at the predeclared event-source gate with `EVENT_EVIDENCE_UNAVAILABLE`. The server contains raw HFO candidate timestamps and a limited expert annotation corpus, but neither an official trained three-class event checkpoint nor existing `3label_pred`/`pathological_probability` inference outputs covering Task 2.

No CNN was retrained, no event detector was trained, no adapter was fitted, and no new ECEA official TEST result was read.

`code/inference_3label_soft.py` is intentionally dormant: it exposes class-2 softmax probability from an existing checkpoint without modifying the model, but the required checkpoint is absent.
