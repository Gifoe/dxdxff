"""Read-only original dependencies; predeclared source/bank/code bindings."""
import json
from pathlib import Path
from baseline_admission import args,dependencies,checkpoint,patient_rows,load


def setup(a):
 ref,c,core,original,trainer,helper,banks,original_binding=dependencies(a)
 baseline=json.loads((a.runtime/'public/BASELINE_REPRODUCTION.json').read_text())
 assert baseline['status']=='PASS' and baseline['binding']==original_binding
 lock=json.loads(a.protocol.read_text());assert lock['arms']==['B0','B1','B2','B3'] and lock['no_new_outer']
 here=Path(__file__).parent;paths={'protocol':a.protocol,'baseline_reproduction':a.runtime/'public/BASELINE_REPRODUCTION.json'}
 for path in sorted(here.glob('*.py')):paths['code_'+path.name]=path
 for path in sorted(here.parent.joinpath('tests').glob('*.py')):paths['test_'+path.name]=path
 manifest=a.previous/'public/FEATURE_GROUP_MANIFEST.json'
 assert c.sha(manifest)==lock['manifest_sha256'];paths['feature_manifest']=manifest
 stats=load('spline_knot_statistics',here/'knot_statistics.py');models=load('spline_model',here/'model.py')
 binding={'original':original_binding,'new':{k:c.sha(v) for k,v in paths.items()}}
 return lock,ref,c,core,original,trainer,models,stats,banks,binding
