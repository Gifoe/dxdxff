"""Reporting-only fix for the parent's early-bound means(keys=old_metrics).

Frozen training source, model artifacts, thresholds and sealed predictions stay
byte-identical. This script refuses to run without all completed private scores.
"""
import sys
from pathlib import Path
TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK))
import run as R

a=R.argparse.Namespace(runtime=Path(r'D:\m8_14d_pr_linear_svm_adaboost_seed42_v1_runtime'),
    baseline=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'),
    features=Path(r'D:\d1_m8_progressive_ablation_seed42_v1_runtime'),
    references=Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime'))
protocol,oldlock,core,rec,banks,x,binding=R.initialize(a)
fr=R.jread(a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json')
assert fr['binding_sha256']==R.digest(binding) and len(fr['selections'])==20
fh=R.sha(a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json')
seal=R.jread(a.runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
assert seal['freeze_sha256']==fh and len(seal['files'])==20
protected=[a.runtime/p for p in ['INPUT_FREEZE_PRIVATE.json','MODEL_SELECTION_PRIVATE.pkl',
    'FROZEN_BEFORE_OUTER_PRIVATE.json','OUTER_PREDICTIONS_SEAL_PRIVATE.json','PREDICTIONS_PRIVATE.pkl']]
for s in fr['selections']:
    assert R.sha(s['path'])==s['model_sha256'];protected.append(Path(s['path']))
    key=f"{s['model']}_{s['version']}_{s['fold']}"
    p=a.runtime/f"outer_scores_private/{s['model']}_{s['version']}_fold{s['fold']}.pkl"
    assert R.sha(p)==seal['files'][key];protected.append(p)
before={str(p):R.sha(p) for p in protected}
frame=R.read(a.runtime/'PREDICTIONS_PRIVATE.pkl')
for s in fr['selections']:
    p=a.runtime/f"outer_scores_private/{s['model']}_{s['version']}_fold{s['fold']}.pkl"
    q=R.read(p)
    g=frame[(frame.model==s['model'])&(frame.version==s['version'])&(frame.fold==s['fold'])]
    assert R.np.array_equal(q['indices'],g.global_index)
    assert R.np.array_equal(q['probability'],g.probability) and R.np.array_equal(q['prediction'],g.prediction)
# Parent aggregate was imported before metric expansion, so its function default
# retained the old list even after module-level METRICS was replaced. Repair the
# reporting function's default only; metric arithmetic itself is unchanged.
R.REPORT.means=lambda g,keys=R.KEYS:{m:R.MET.nanmean(g[m]) for m in keys}
R.write(a.runtime/'public/REPORTING_REPAIR_AUDIT.json',dict(status='PASS',
    cause='Parent means default argument retained old classification list without NEZ-F1 and EZ-fraction bias',
    repair='Replace reporting means default with expanded metric list; recompute aggregate tables from sealed scores only',
    original_training_source_sha256=R.sha(TASK/'run.py'),reporting_helper_sha256=R.sha(__file__),
    no_training_or_inference_called=True,twenty_score_hashes_verified=True))
R.verify_and_report(a,frame,banks,fr['selections'],binding,fh)
for p,h in before.items():assert R.sha(p)==h
print('REPORT_ONLY_RECOVERY_ALL_FROZEN_ARTIFACTS_UNCHANGED',flush=True)
