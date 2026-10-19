# F0 single-head complete ablation, seed42

**Completed and independently verified on 2026-10-19.** All 18 configurations completed the original five-fold Outer protocol: 80 actual fresh fits plus ten exact frozen F0/C2 reuses, 90 model/fold cells, 997 completed fresh-training epochs, 1,440 patient-result rows and 12,960 independently checked metric values. All 324 nominal bootstrap intervals and 18 additional Macro-F1 family intervals passed independent verification.

Full F0 patient-average Macro-F1 is **0.647977**, AUROC **0.754932**, AP **0.587757**, EZ-F1 **0.453985**, NEZ-F1 **0.841968**, ACC **0.770914**, MRR **0.831379**, Top-1 **0.775000**, and EZ-fraction bias **−0.041987**. These are equal-patient averages, not pooled-channel results.

## Results and interpretation

- [Complete English report: all 18 configurations, nine metrics and paired intervals](results/REPORT.md)
- [中文结果解读：模块、条件特征贡献、容量与分支互补](FINDINGS.md)
- [Exact patient-average results](results/MAIN_RESULTS.csv), [pooled results](results/POOLED_RESULTS.csv), [five folds](results/FOLD_RESULTS.csv), [fold SD](results/FOLD_SD.csv), [sources](results/SOURCE_RESULTS.csv)
- [All 324 nominal intervals](results/INTERVALS.csv), [18 Macro-F1 family intervals](results/MACRO_FAMILY_INTERVALS.csv), [coverage](results/RAW_COVERAGE.csv)
- [Actual execution and native-resume audit](RUN_AUDIT.md), [visible figure QA](FIGURE_QA.md)
- [Feature contribution figure](results/figures/FEATURE_CONTRIBUTION.svg), [paired core intervals](results/figures/PAIRED_CORE_INTERVALS.svg); vector PDF and PNG previews are alongside them.

The largest Macro-F1 point drops are removal of Raw PR (−0.035736) and Raw pre-reference (−0.028594). Only the pre-reference comparison has an entirely positive Full-minus-ablation interval after the predefined 18-comparison approximate adjustment. Removing Variance gives 0.650225, but its paired interval crosses zero: this is **not evidence that a reduced feature set is superior**. Early FiLM, head interaction and most individual feature contributions remain uncertain.

## Fixed protocol and execution

The [protocol](PROTOCOL_LOCK.json) pins the original 80 patients, 7,635 channels, 256 seizures and 1,471,965 valid Raw250 windows, seven measurements, one scalar head and matched shared initializer. No new features, data extraction, hyperparameter search, trained-F0 initialization of new fits or window subsampling. Seven leave-one-out masks affect direct14 and both early FiLM/late Gate inputs during training and inference. Raw-only removes every explicit physiology route; physiology-only consumes no Raw. No Development eligibility screen was applied; legal internal validation still selected epochs and thresholds by the unchanged rule.

Scientific sources were pushed before formal fits at `c8cd777d5b84d6cd7f17ebad1649aafc7dcfd537` and remained byte unchanged. The existing immutable server driver was run with `run_all.ps1 -StopAfter Admission -RunTag 01`, then `-StopAfter All -RunTag 02 -SourceCommit <verified-source-commit>`. After a diagnosed native serialization failure, the exact command resumed with fresh log tag `03`; completed cells and complete-epoch optimizer/RNG checkpoints were preserved. See the execution audit, not a claim of failure-free execution.

All 90 selected checkpoints, preprocessors, epochs and thresholds froze before new Outer inference; all 90 logical score cells sealed before metric reporting. On a checkout containing these public outputs, run `python experiments/m8_7d_f0_complete_ablation_seed42_v1/verify_public.py` to independently check the aggregate/report/manifest contracts. Private-data execution requires the original trusted server caches and hash-bound parent experiments; this public release does not distribute those data.

After pushing this branch, `python experiments/m8_7d_f0_complete_ablation_seed42_v1/verify_git_release.py` also checks remote-tip equality and the exact committed Git blob hashes. Immutable server exports retain their original bytes, including CRLF and generated SVG path spacing, rather than being reformatted after hashing.

This is **exploratory repeated-test evidence**: F0 itself was chosen after historical Outer viewing. No new model or feature subset was selected from these results. Leave-one-feature-out effects do not prove global 7D optimality, absence of correlated physiology in Raw, or training-seed robustness. Only code, hashes and anonymous aggregates/figures/reports are public; identities, signals, weights, predictions, mappings and logs remain on the server.
