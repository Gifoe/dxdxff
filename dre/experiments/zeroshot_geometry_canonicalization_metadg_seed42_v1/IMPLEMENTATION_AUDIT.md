# Implementation audit

- Protocol lock was pushed before new target outcomes. A1 150-checkpoint/R4 replay and matched B0 AP were rechecked.
- Identical source A1 architecture, per-fold initial state, patient-equal BCE, optimizer, 30 epochs, and 19-threshold VLOO. Geometry heads/adversary are training-only.
- FIT-only patient-ID-disjoint 80/20 hyperparameter selection; full-FIT retraining after selection. Z4 geometry and meta parameters derive solely from FIT selections.
- All 900 new-model epoch/fold/variant score and R4 snapshots, 30 FIT-selection files, and five exact A1 epoch-30 R4 snapshots were hash-frozen before the new target-label evaluation pass. Candidate pools were ignored for B=0 fixed-query prediction.
- A pre-outcome diagnostic amendment fixes epoch 30 as the common checkpoint within each fold for every variant. Cross-patient R4 cosine uses only within-fold pairs from that same model state; VLOO-selected states are never mixed for geometry diagnostics. Primary VLOO performance is unchanged.
- Literal strict target-label sequencing is false: exact A1 VLOO uses a patient's label to select another patient's checkpoint. The patient's own label never enters its own selection; all candidate score grids were frozen first. The legacy loader also materializes all 80 labels, making strict outer label non-materialization false. No outer predictions, metrics or selection were computed.
- Geometry diagnostics, target-specific OOF headroom and FIT-direction transfer are post-score exploratory at common epoch 30; patient-specific heads are nondeployable. Winner/gates are retrospective across multiple variants without family-wise correction.
