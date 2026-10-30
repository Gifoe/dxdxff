# Deliberately high-gain case from frozen Ictal Outer predictions

User requested another patient prioritizing PR improvement, with no minimum EZ
channel count. The protocol maximizes the equal-weight mean of DREChanNet's
three-seed mean AUROC and Recall@10% changes, requiring both to improve and at
least two seeds to improve both jointly. Previously selected cases are excluded.
The original >=20 total-channel criterion is retained. SVM results are reported
without influencing case choice. This is a retrospective high-gain illustration,
not a median or representative case. No new training or inference is performed.

Only anonymous CASE-H01 and CH aliases may leave the trusted server. Original
contact names and the identity mapping remain private. Parent data and results
are read-only; the prior median-improvement and declining cases stay unchanged.

Completed and independently verified on 2026-10-30. See the
[Chinese report](results/REPORT_ZH.md),
[all-channel table](results/readable/CASE-H01_CHANNELS_WIDE.csv), and
[EZ-only table](results/readable/CASE-H01_EZ_CHANNELS_ONLY.csv).

CASE-H01: HUP, Outer fold 2, 176 channels / 9 EZ. All three DRE seeds improve
AUROC and Recall@10%. SVM Recall@10% and Macro-F1 decline, and DRE seed43
Macro-F1 declines; all are retained. The GitHub release contains this anonymous
case, the fixed selection protocol, verification scripts and Chinese report.
