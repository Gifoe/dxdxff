# Frozen Outer case with TP gain and controlled FP

Retrospective selection under unchanged original validation thresholds. Prefer
all three seeds with no FP increase, otherwise allow at most two additional FP
per seed. Require mean TP gain, at least two seeds with TP gain, nondecreasing
mean precision, and stable AUROC/Recall10 gains. Maximize mean TP increase within
the first nonempty tier; retain every seed and SVM comparison. Exclude all three
previous cases. No new training, inference or threshold adjustment.

Only CASE-F01 and CH aliases leave the trusted server. Original identities and
contact names stay private. Full details in PROTOCOL_LOCK.json.
