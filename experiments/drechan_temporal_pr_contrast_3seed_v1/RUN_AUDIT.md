# Execution ledger

## 2026-10-20, before formal fits

- Branch based on075305f1aa372a9ac34faa19e689d2cca96fe7bd; original experiment bytes unchanged.
- User tolerance interpreted as at most0.01absolute loss simultaneously in patient AUROC/AP/Macro-F1/EZ-F1 versus matched B0. Added fixed global-reference comparison gate prevents capacity-only gains being relabeled as PR evidence. No success is assumed.
- Historical correction: PatientZ+0.028655 and[0.010792,0.049047] are factorial mean/min/max, not a patient-paired95% CI. The original file explicitly describes8backgrounds of frozen-CNN logistic screening.
- Historical relation/affine/ranking negative studies read, not overwritten or omitted. Stop-gradient backgrounds, original fully differentiable final PR, exact current-CNN TRAIN reference per update, fixed wrong-time support and3seeds are explicit first-round decisions.
- Local5reference tests PASS; server5reference+3model tests PASS. Source bundle SHA2563440cdd42e927a7cb8d64788d1f5e2cf4d2ab3ada4d7804f69d13f3a653923a7.
- Real admission PASS at approximately02:35Asia/Shanghai. All5shared initializers match the reference exactly; all5old legal-validation selected-model logits replay bit exactly. No newOuter inference was run. Every new arm initially reproduces complete B0 logits exactly on a genuine TRAIN patient, with125channels and36,580valid windows.
- Each new arm completed3discarded admission-only optimizer updates, with finite/live CNN, head and fusion gradients; first fusion layer becomes active after the zero-initial output layer begins updating. No admission weight is saved/reused as a formal model initializer.
- B2 admission used all35genuine TRAIN patients/566,008valid windows, current CNN and exact per-patient4096chunk boundaries. One initial global-reference refresh measured0.571seconds;3updates including per-step current references measured1.052seconds. These are short admission measurements, not full-experiment timing guarantees. Peak process allocation reached2.15GB in this admission; no Raw disk copy created.
- Internal roles, independent of training seed: folds1/2/5have35TRAIN+8CAL+8ASSESS, fold3has34+8+8, fold4has36+8+8. Role identity hashes/private preprocessing remain on trusted server; anonymous counts/hashes exported.
- No formal fits or new assessment/Outer outcomes existed at this source/admission checkpoint. Source must be pushed and remote-tip-verified before formal launch. All subsequent checkpoints/logs and patient-level predictions stay private.
