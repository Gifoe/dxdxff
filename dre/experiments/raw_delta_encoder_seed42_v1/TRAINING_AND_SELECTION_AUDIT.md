# Training and selection

Original five folds and seed 42+1009*fold are preserved. Own-fold FIT and validation are disjoint from own TEST; bank provenance and identities are checked. No global feature-archive y member or Outer result is read. Raw alignment may include all 80 patients label-blind; no model is evaluated on new Outer data.

E1/D1 invoke the pinned original trainer with the compact source-free factory and zero source penalty: AdamW lr0.001/wd0.0001, NEZ-positive FIT class-weighted BCE, one channel-mean update per FIT patient, clip1, maximum30 epochs/minimum stop6/patience6/dropout0.15. Both arms have identical corresponding scratch states. Training reseeds after construction, preserving patient-order RNG semantics.

D2/D3 use AdamW lr0.0003/wd0.001, the same one-patient update/class weighting/clip/stopping rules, and loss BCE(final)+0.25BCE(raw). D1 parameters and buffers are frozen. Completed epoch model/optimizer/RNG states are atomically saved with hash-bound private selections for exact recovery after engineering failures.

Checkpoint priority is original validation patient Macro-F1, EZ-F1, earlier epoch. Threshold selection uses the original NEZ probability grid 0,0.005,...,1 with patient Macro-F1, EZ-F1, pooled balanced accuracy, nearest0.5, then smaller threshold tie order. Every reported model retains its own selected numeric threshold; no threshold is optimized inside bootstrap or on Outer.

The primary estimand is 65 original Development patient-fold appearances representing 47 unique IDs. The unique-47 summary first averages repeated appearances within ID. Paired 10,000-draw seed42 bootstrap resamples unique IDs and retains repeated appearances together. Intervals are unadjusted exploratory intervals on extensively reused Development, not independent confirmation or clinical-validation evidence.

No unfavorable intermediate fold was dropped. The same model/bands/window support/loss/hyperparameters were kept for all five folds. Only the documented arithmetic and native-execution repairs preceded training. `MODEL_SELECTION.csv`, both training-history CSVs, input/code hashes and private freezes preserve actual execution lineage.
