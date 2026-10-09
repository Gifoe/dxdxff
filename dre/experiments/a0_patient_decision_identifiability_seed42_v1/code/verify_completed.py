"""Independent deterministic replay of completed private probes, no parameter selection."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core import sha,write_json,episode,seed,fit_pca,METRICS

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--bank-runtime',type=Path,required=True)
    p.add_argument('--fold',type=int,choices=range(1,6));a=p.parse_args();pub=a.runtime/'public'; status=json.loads((pub/'RUN_STATUS.json').read_text());assert status['status']=='COMPLETE'
    lockpath=Path(r'E:\DRE-nips\new-pipeline\7-11\a0_patient_decision_identifiability_seed42_v1\PROTOCOL_LOCK.json')
    assert sha(lockpath)==status['binding']['protocol']; lock=json.loads(lockpath.read_text())
    freeze=json.loads((pub/'PRE_VAL_FREEZE.json').read_text());assert freeze['binding']==status['binding']
    reg=json.loads((pub/'REGULARIZATION_SELECTION_AUDIT.json').read_text());assert sha(pub/'REGULARIZATION_SELECTION_AUDIT.json')==freeze['regularization_sha256']
    private=torch.load(a.runtime/'AUDIT_PRIVATE.pt',map_location='cpu',weights_only=False);counts=Counter();maxdrift=0.
    folds=[a.fold] if a.fold else range(1,6)
    for fold in folds:
        bp=a.bank_runtime/f'fold{fold}/DEVELOPMENT_PRIVATE.pt';assert sha(bp)==lock['bank_sha256'][fold-1]
        bank=torch.load(bp,map_location='cpu',weights_only=False);assert sha(bank['A0_checkpoint'])==lock['checkpoint_sha256'][fold-1]
        _,z=fit_pca(bank['x'],bank['train']);ref=pd.read_csv(a.bank_runtime/f'fold{fold}/A0_CHANNEL_PRIVATE.csv')
        lam=reg['choices'][fold-1]; mapping={(str(p),str(c)):i for i,(p,c) in enumerate(zip(bank['patient'],bank['channel']))}
        for patient,g in ref.groupby('patient',sort=True):
            g=g.sort_values('channel',kind='stable'); ix=np.asarray([mapping[(str(patient),str(c))] for c in g.channel]); score=g.score_nez.to_numpy()
            for policy,reps,budget,kind in [('B8_uncertainty',1,8,'uncertainty'),('B8_random',20,8,'random'),('half_random',10,len(ix)//2,'random')]:
                for rep in range(reps):
                    key=hashlib.sha256(json.dumps([status['binding'],freeze['regularization_sha256'],fold,str(patient),policy,rep],sort_keys=True).encode()).hexdigest()
                    cell=json.loads((a.runtime/'episodes_private'/(key+'.json')).read_text());assert cell['key']==key
                    out,check=episode(bank['y'][ix],score,z[ix],bank['channel'][ix],bank['A0_threshold'],budget,kind,seed(fold,str(patient),policy,rep),(lam['P1'],lam['P2']))
                    assert np.array_equal(check['support'],cell['check']['support']) and np.array_equal(check['query'],cell['check']['query'])
                    for m in out:
                        for metric,value in out[m].items():
                            old=cell['metrics'][m][metric]
                            if np.isnan(value):assert old is None
                            else:
                                delta=abs(value-old);maxdrift=max(maxdrift,delta);assert delta<1e-12
                    sy=bank['y'][ix][check['support']]
                    permutation=np.random.default_rng(seed(seed(fold,str(patient),policy,rep),'permute')).permutation(sy)
                    counts[policy+'_episodes']+=1
                    counts[policy+'_one_class']+=int(len(set(sy))==1)
                    counts[policy+'_mixed_class_identical_permutation']+=int(len(set(sy))>1 and np.array_equal(sy,permutation))
    expected=403 if a.fold else 2015
    assert sum(v for k,v in counts.items() if k.endswith('_episodes'))==expected
    clinical=json.loads((pub/'LABEL_ALIGNMENT_AUDIT.json').read_text())
    assert sum(r['patients'] for r in clinical['label_source_counts'])==80 and sum(r['channels'] for r in clinical['label_source_counts'])==7635
    assert sum(r['run_union_disagree'] for r in clinical['target_comparison'])==0
    assert not clinical['alternative_task2_function_used']
    output='FINAL_INTEGRITY_AUDIT.json' if a.fold is None else f'FINAL_INTEGRITY_FOLD{a.fold}.json'
    write_json(pub/output,{'status':'PASS','fold':a.fold,'episodes_replayed_exactly':expected,'max_metric_drift':maxdrift,
        'all_original_A0_checkpoint_and_bank_hashes_unchanged':True,'clinical_export_index_run_union_complete':True,
        'counts':dict(counts),'permutation_identity_allowed_under_randomization':'mixed-class chance identity is reported, not misrepresented as a changed assignment; one-class explicitly degenerate',
        'protocol_sha256':sha(lockpath),'regularization_freeze_unchanged':True,'source_sha256':sha(__file__),
        'no_retraining_no_query_tuning_no_outer_test':True})
    print('FINAL_INTEGRITY_PASS',maxdrift,json.dumps(dict(counts)),flush=True)

if __name__=='__main__':main()
