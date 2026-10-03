# Label semantics audit

The official Task2 raw construction is `-1` for excluded channels, `1` for a normal channel (`patient_outcome == 1 and channel_resection == 0`), and `0` for an SOZ/pathological channel (`channel_soz == 1`). The frozen representation extractor applied `pathological_labels = where(official_label >= 0, 1 - official_label, -1)` before this evaluation.

This run read only the resulting frozen `pathological_labels` field. Its values were restricted to `-1, 0, 1`; the `-1` rows were excluded. The retained test cohort contains `7297` normal (`y_pathological=0`) and `807` pathological/SOZ (`y_pathological=1`) EDF-channel units. No labels were recomputed, repaired, or reinterpreted.

The source population/filtering was already frozen in the cache: official Task2 `test`, frequency >900, interictal, length >=62, dataset != Multicenter, and official good channels. This script did not reread raw recordings or test metadata to alter membership.
