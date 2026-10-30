# TP gain with limited FP increase: exploratory case

V1 added precision and ranking eligibility rules and produced no qualified case.
Aggregate diagnostics showed six candidates when using the requested TP/FP
criteria only. V2 freezes this simpler descriptive rule before selecting an
identity: mean TP increases, >=2 seeds increase TP, and each seed FP increase
<=2. Prefer all seeds FP <=0 if possible. Rank by mean TP gain, then FP change.

Retain original frozen thresholds and all seeds/endpoints. Only anonymous
CASE-F01/CH data may leave the trusted server. V1 remains preserved.
