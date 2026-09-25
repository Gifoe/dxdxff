def test_overfit_gate_is_present():
 from pathlib import Path
 assert '--overfit-audit' in (Path(__file__).parents[1]/'scripts/task2/run_trace_aux_stable.py').read_text()
