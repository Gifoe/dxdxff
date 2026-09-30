# Engineering repair ledger

All repairs were made before accepting the final TRAIN-OOF result. None accessed official TEST data or changed the frozen feature definitions, model, optimizer, loss weights, split seed, gate, or evaluation endpoints.

1. The initial center×prevalence stratification contained a center stratum with fewer than five patients. The deterministic fallback merges all prevalence strata within only the affected center. This preserves center stratification where feasible and yields five patient-disjoint folds. The failed preflight produced no model result.
2. The original patient-equal loss loop repeatedly scanned all rows for every patient and epoch. It was replaced by exactly equivalent `numpy.bincount` aggregation. A finite-difference gradient test verifies the vectorized gradient.
3. Exhaustive threshold evaluation recomputed sklearn metrics once per unique score, giving quadratic work. It was replaced by an exact cumulative-confusion implementation. Random and tied-score tests compare it against the original exhaustive definition, including the lower-threshold tie break.
4. The server's legacy `MNElab` Python crashed natively in `python310.dll` after two folds. No complete result was written. The final run used the existing project Python 3.11 environment after the full unit suite passed there.
5. Before accepting results, an audit found that weighted patient BCE was divided by the sum of class weights. The prompt requires applying the fixed 2:1 class weight and then taking the arithmetic channel mean per patient. The denominator was corrected to channel count, an explicit objective-value test was added, and the full TRAIN-only OOF run was repeated. Only this corrected run is retained in the published artifacts.

Final validation: `7/7` implementation tests passed in both the local environment and the server execution environment.
