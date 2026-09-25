from pathlib import Path
def test_protocol_has_no_forbidden_models():
 t=(Path(__file__).parents[1]/'scripts/task2/run_trace_aux_stable.py').read_text();assert 'task1' not in t.lower() and 'vdr' not in t.lower() and 'top_q10' not in t.lower()
