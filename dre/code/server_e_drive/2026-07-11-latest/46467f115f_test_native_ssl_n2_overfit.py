from pathlib import Path
def test_overfit_cli_is_exposed():
 script=(Path(__file__).parents[1]/'scripts/task2/run_task2_native_ssl_n2.py').read_text(); assert "--overfit-audit" in script
