from pathlib import Path
def test_no_task1_dependency():
 text='\n'.join(x.read_text() for x in (Path(__file__).parents[1]/'neuroez_c/task2/native_ssl_n2').glob('*.py')); assert 'task1' not in text.lower().replace('uses_task1', '')
