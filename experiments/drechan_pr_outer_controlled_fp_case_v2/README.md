# TP gain with limited FP increase: exploratory case

V1 added precision and ranking eligibility rules and produced no qualified case.
Aggregate diagnostics showed six candidates when using the requested TP/FP
criteria only. V2 freezes this simpler descriptive rule before selecting an
identity: mean TP increases, >=2 seeds increase TP, and each seed FP increase
<=2. Prefer all seeds FP <=0 if possible. Rank by mean TP gain, then FP change.

Retain original frozen thresholds and all seeds/endpoints. Only anonymous
CASE-F01/CH data may leave the trusted server. V1 remains preserved.

Completed and independently verified on 2026-10-30. See
[Chinese report](results/REPORT_ZH.md) and
[actual TP/FP/FN/TN table](results/public/CASE_CONFUSION.csv).

CASE-F01: HUP, fold 2, 84 channels / 16 EZ. DRE seed42/43/44 TP changes
1->7 / 2->7 / 5->8; FP changes 0->0 / 1->2 / 1->2. The three-seed mean
AUROC is unchanged. No claim of consistent ranking improvement is made.
The original native endpoint path failed; preserved same-rule engineering
recovery with Python formulas replayed all 640 endpoints and 400 changes.
See RUN_AUDIT.md. The GitHub release contains anonymous results, fixed protocols,
verification/recovery scripts and the report. Identities and original contact
names remain on the trusted server.
