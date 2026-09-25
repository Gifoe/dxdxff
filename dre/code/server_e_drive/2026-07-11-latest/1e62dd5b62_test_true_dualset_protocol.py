from pathlib import Path
def test_no_task1_or_trace_checkpoint_dependency_and_direct_protocol():
 root=Path(__file__).parents[1];text='\n'.join(p.read_text(encoding='utf8') for p in (root/'neuroez_c/task2/true_dualset').glob('*.py'))
 assert 'trace_dre_lite' not in text and 'p2_q10' not in text and 'localization loss' not in text
 runner=(root/'scripts/task2/run_true_ez_nez_dualset.py').read_text(encoding='utf8')
 assert 'StratifiedShuffleSplit' not in runner and 'early_stopping' in runner and "selected_model_type':'ema_final" in runner and 'infer(final,cohort,test' in runner
