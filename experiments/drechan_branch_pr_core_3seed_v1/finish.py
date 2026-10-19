"""After both studies finish training, join exact parentFULL references and score."""
from common import *

def main():
    binding,core,fac,records,rec,banks=setup()
    assert jr(PUB/'NEW_TRAIN_STATUS.json')['status']=='ALL75_NEW_COMPLETE'
    assert jr(parent.PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    pf=parent.frozen();ps=jr(parent.ROOT/'SCORE_SEAL_PRIVATE.json');assert len(ps['files'])==90 and ps['all90_before_metrics']
    q=jr(ROOT/'NEW_MODEL_FREEZE_PRIVATE.json');assert q['input']==binding and len(q['cells'])==75
    references=[]
    for r in pf['cells']:
        if r['arm']=='FULL':references.append(dict(r,reused=True,parent_arm='FULL'))
    assert len(references)==15
    cells=q['cells']+references;assert {(r['arm'],r['seed'],r['fold']) for r in cells}==cells_expected()
    csv_out('TRAIN_CELLS',[{k:r[k] for k in ['arm','seed','fold','epochs','epoch','threshold','parameters','checkpoint_sha256','calibration_sha256','reused']} for r in cells])
    with (PUB/'NEW_TRAIN_HISTORY.csv').open(newline='',encoding='utf-8') as f:history=list(csv.DictReader(f))
    with (parent.PUB/'TRAIN_HISTORY.csv').open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['arm']=='FULL':history.append(dict(r,reused='True'))
    csv_out('TRAIN_HISTORY',history)
    freeze(ROOT/'MODEL_FREEZE_PRIVATE.json',dict(input=binding,cells=cells,parent_freeze=sha(parent.ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json'),parent_seal=sha(parent.ROOT/'SCORE_SEAL_PRIVATE.json')))
    write(PUB/'TRAIN_STATUS.json',dict(status='ALL90_COMPLETE',fresh_fits=75,reused=15,new_epochs=jr(PUB/'NEW_TRAIN_STATUS.json')['new_epochs'],reused_epochs=sum(r['arm']=='FULL' for r in history)))
    verify=module('core_selection_and_metric_verifier',HERE/'verify.py');verify.selection()
    assess=module('core_all_score_and_report',HERE/'assess.py');assess.main()
    verify.metrics()
    package=module('core_public_package',HERE/'package_public.py');package.main()
    print('CORE_ALL90_VERIFIED_NO_NEW_OUTER_GOAL_NOT_ACHIEVED',flush=True)

if __name__=='__main__':main()
