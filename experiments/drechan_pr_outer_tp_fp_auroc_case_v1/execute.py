"""Use the already verified independent endpoint formulas, unchanged."""
import hashlib,os,runpy,sys,types
from pathlib import Path
BASE=Path(r'D:\Codex_Research\experiments')
SOURCE=BASE/'drechan_pr_outer_tp_fp_auroc_case_v1'
RUNTIME=BASE/'drechan_pr_outer_tp_fp_auroc_case_v1_runtime'
sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE/'drechan_pr_representation_vs_score_v1'))
from path_adapter import install
install(r'D:\Codex_Research\_maintenance\cleanup_20261028\LEGACY_PATHS.json',[RUNTIME])
engine=BASE/'drechan_pr_outer_controlled_fp_case_v2_engineering_pure01/engineering_pure_python.py'
assert hashlib.sha256(engine.read_bytes()).hexdigest()=='f53ca3f7814ac209e3e2723471e4f10fda75bfdaac23dd44924b3a31e93c8cee'
run=runpy.run_path
pure=run(str(engine),run_name='verified_pure_formulas')
m=run(str(SOURCE/'run.py'),run_name='new_auroc_case');ns=m['main'].__globals__
def bridge(path,*args,**kwargs):
    assert Path(path)==BASE/'drechan_pr_outer_representative_cases_v1/run.py'
    g=run(path,*args,**kwargs);g['compute']=pure['compute'];g['pct']=pure['pct'];return g
ns.update(np=pure['MathShim'],runpy=types.SimpleNamespace(run_path=bridge))
ns['main']()
