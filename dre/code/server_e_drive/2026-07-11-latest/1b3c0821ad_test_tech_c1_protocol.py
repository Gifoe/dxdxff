from pathlib import Path
from neuroez_c.task2.tech_outcome_c1.audit import protocol_audit
def test_protocol_and_third_party_notice():
 a=protocol_audit();assert a['model_version']=='TECH_OUTCOME_C1' and not a['uses_task1'] and not a['uses_ez_nez_labels'] and a['checkpoint_monitor']=='validation_bce';assert (Path(__file__).parents[2]/'THIRD_PARTY_NOTICES.md').exists()
def test_no_forbidden_imports():
 root=Path(__file__).parents[2]/'neuroez_c/task2/tech_outcome_c1';text='\n'.join(p.read_text() for p in root.glob('*.py'));assert 'neuroez_c.task1' not in text and 'trace_rawmil' not in text and 'true_dualset' not in text
def test_runner_has_overfit_gate_and_no_epoch_test_inference():
 runner=(Path(__file__).parents[2]/'scripts/task2/run_tech_outcome_c1.py').read_text();assert '--overfit-audit' in runner and "cp/'best_val_bce.pt'" in runner and 'len(fold_rows)!=5' in runner
