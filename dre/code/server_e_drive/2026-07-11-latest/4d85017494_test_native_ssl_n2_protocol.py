from pathlib import Path
def test_no_forbidden_dependencies():
 t='\n'.join(x.read_text() for x in (Path(__file__).parents[1]/'neuroez_c/task2/native_ssl_n2').glob('*.py'));assert 'from neuroez_c.task1' not in t and 'P2_TEMPORAL_Q10' not in t and 'true_dualset' not in t
