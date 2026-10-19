# F0 single-head complete ablation, seed42

User-approved eighteen-arm experiment: exact frozen F0 and C2 reuse, sixteen newly trained configurations across all five original patient-held-out folds (80 fresh fits). Every arm completes full80 Outer, irrespective of Development performance. Internal validation remains exclusively responsible for the unchanged epoch/threshold rule.

The source/protocol pins the original80patients/7635channels/256seizures and1471965valid Raw250 windows, original seven measurements, one scalar readout and matched shared initializer. No feature extraction, hyperparameter search, trained-full-model initialization of ablations or window subsampling. Seven leave-one-out masks affect direct14 plus BOTH early FiLM and late Gate inputs during training and inference. The early mismatch control preserves whole-trajectory support within each patient/seizure; Raw-only has no explicit physiology path and physiology-only consumes no Raw.

Run the immutable server source with `run_all.ps1 -StopAfter Admission -RunTag 01`, push the admitted source to GitHub, then `run_all.ps1 -StopAfter All -RunTag 02 -SourceCommit <verified-40-character-commit>` in a sustained foreground server session. The coordinator locks its private runtime, rejects duplicate workers, preserves logs and never blindly retries. All90 selected cells freeze before new Outer prediction, and all90 score cells seal before metric reporting.

Output includes nine patient-average metrics, separate pooled/fold/source/coverage, 324 nominal patient-bootstrap intervals, eighteen additional Macro-F1 family intervals, independent endpoint/interval verification, feature contribution and paired uncertainty figures. No results are asserted until actual execution and verification.

This is **exploratory repeated-test evidence**: F0 itself was chosen after earlier Outer viewing. The fixed matrix is not an opportunity to choose a new winner or retune on this cohort. Feature deletion measures conditional explicit-input utility, not absence of correlated physiological information in Raw or proof of global7D optimality. No private identities, signals, weights, predictions or logs are published.
