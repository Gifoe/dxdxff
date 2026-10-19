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

## 2026-10-20, first native interruption

- Original13scientific files were pushed/remote-tip-verified at4c3eb8468e9857ac91a5dc87827f8993797c75ac before production. Every Git source blob matches the real admission hashes. Production started approximately02:36Asia/Shanghai.
-30internal cells/307complete epochs finished before a native access violation during fold3/B2/seed43, next epoch7. Windows Application Error1000 identifies torch_cuda.dll/0xc0000005; faulthandler locates the current global-reference quantile path. No Python-level exception, assessment metrics or Outer outcomes were read.
- Current cell epoch6/LAST remains loadable with full model, optimizer, RNG and6-entry history; SHA25680a21e7169e3384c1cb51f016db2c7f25dfae5a577d5360c1df7adde470f3cfe. Selected best epoch3/reference preserved. Completed cells are retained and hash-checked, not retrained.
- Read-only `diagnose_native.py` is outside the original13file science binding. Five exact current-model global-reference replays passed, and selected-best reference replay is bit exact. All34TRAIN patients/625,864valid windows remain present. No formal optimizer update, source/configuration change, CAL/ASSESS/Outer scoring or checkpoint overwrite occurred in diagnosis. See [native diagnosis](results/NATIVE_DIAGNOSIS_01.json).
- GPU had ample free memory; no matching recent Display/nvlddmkm/WHEA System event was returned. The underlying native runtime cause is not proven; this is not described as a resolved library bug.
- One bounded same-command recovery is permitted after the above diagnosis, with a whole-file-hash-verified private checkpoint backup and fresh production02 logs. It resumes epoch7 from exact epoch6 model/optimizer/RNG, preserving all completed epochs and old logs. No automatic retry loop, backend change, relaxed equivalence tolerance, model/reference/threshold/gate change or window sampling is introduced. A recurrence requires further diagnosis, not blind restarts.
