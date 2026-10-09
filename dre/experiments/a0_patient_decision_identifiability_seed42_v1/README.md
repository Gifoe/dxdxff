# A0 Patient Decision Identifiability — seed42

Complete development-only audit, **not a new improved model**. Terminal: `PATIENT_ORACLE_HEADROOM_NOT_RECOVERABLE` **under the tested fixed 4D probes and budgets**, not an impossibility theorem.

Frozen A0 Macro-F1 0.638080; exhaustive patient threshold oracle 0.706285 (+0.068205). Eight uncertainty-selected labels did not produce reliable bias or direction gains. Half-channel supervision improved ranking AP but not decision Macro-F1. Different operational clinical target definitions are confirmed; their causal role is not established.

Read [FINAL_REPORT.md](FINAL_REPORT.md), [SOURCE_AUDIT.md](SOURCE_AUDIT.md), [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md), [clinical provenance](CLINICAL_LABEL_PROVENANCE.md), and [limitations](LABEL_AUDIT_LIMITATIONS.md). Machine-readable gates are in `audit/FINAL_DECISION_MATRIX.json`; all aggregate metric/contrast tables are in `results/`.

No A0 retraining, label changes, new EEG extraction, or outer TEST evaluation. All intervals are exploratory on a repeatedly used development cohort. No clinical deployment claim.

## Reproduction on the original trusted server

The private assets are intentionally not distributed: sealed TabM FIT/VAL banks, hash-locked A0 checkpoints, PR-UAS FIT-patient-OOF artifacts, and original clinical cache. `PROTOCOL_LOCK.json` hard-gates them. Run from this experiment's directory after placing the code in the matching server experiment folder:

```powershell
$py='C:\pr_uncertainty_aware_supervision_seed42_runtime\runtime312\Scripts\python.exe'
$code='E:\DRE-nips\new-pipeline\7-11\a0_patient_decision_identifiability_seed42_v1\code'
$rt='C:\a0_patient_decision_identifiability_seed42_runtime'
& $py -m pytest "$code\..\tests\test_core.py" -q
& $py "$code\run.py" --source 'E:\DRE-nips\new-pipeline\7-11' --runtime $rt --protocol "$code\..\PROTOCOL_LOCK.json" --bank-runtime 'C:\a0_tabm_ensemble_seed42_runtime' --oof-runtime 'C:\pr_uncertainty_aware_supervision_seed42_runtime'
& $py "$code\supplement.py" --runtime $rt --bank-runtime 'C:\a0_tabm_ensemble_seed42_runtime' --oof-runtime 'C:\pr_uncertainty_aware_supervision_seed42_runtime'
foreach ($auditFold in 1..5) {
  & $py "$code\verify_completed.py" --runtime $rt --bank-runtime 'C:\a0_tabm_ensemble_seed42_runtime' --fold $auditFold
}
```

Run into an empty private audit runtime for a new provenance binding. Never substitute training-in-sample scores for OOF scores. Reporting supplements must run only after COMPLETE. `clinical_inventory_v1.py` and `core_before_vectorized_metric.py` are inactive archived engineering source versions retained to resolve hashes in the original execution binding; they are not entry points. Subsequent reporting utilities have separate source hashes.

Runtime was Python 3.12 on CPU with two math threads. Native long-process failures were isolated with per-fold verification without changing predictions. See implementation audit. Only code, compact aggregate results and non-identifying metadata summaries are public.
