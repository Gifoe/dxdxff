# Implementation audit

- Source A1 and its five FIT/validation folds were replayed exactly; all 150 validation grids agreed bitwise at reported metric precision.
- S-F1 and S-RANK used other 12 validation patients for all epoch/threshold choices; excluded labels never entered selection.
- Exact source A1 reused as CTX0/OBJ0/CTR0/VIEW0. Seven independent new variants each trained 5 folds × 30 fixed epochs without early stopping or mechanism combinations.
- Common model parameters loaded from the same fold initialization. VIEW1 and VIEW2 used the precommitted label-free partition/semantic partition and identical architecture/parameter counts.
- CTR1 weights came only from FIT center counts and were not normalized within each minibatch. Center ID was absent from model.forward and inference.
- Frozen T0/T0.5/T1 interventions used no labels or retraining. The T1 replay matched A1. Channel masks were label-free, seed-derived, and identical across structural variants.
- Layerwise linear scorers saw FIT embeddings/labels only, trained for 20 fixed epochs, and were never selected on validation.
- Patient/channel-level records, labels, predictions, checkpoints, validation grids and runtime logs remain private on the server. Public files contain aggregate rows only.
- No outer-test loader was constructed or used. The historical source constructor indexes cohort metadata but no outer label/prediction/performance was read for this audit.
