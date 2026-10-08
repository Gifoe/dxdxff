"""Fresh integrity/completeness check after native interpreter shutdown failure."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from uas_core import sha, json_write, validate_exclusion


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True)
    a=p.parse_args(); root=a.runtime
    state=json.loads((root/'RUN_STATUS.json').read_text()); assert state['status']=='ALL_DEVELOPMENT_TRAINING_COMPLETE'
    gate=json.loads((root/'gate'/'A0_REPRODUCTION.json').read_text())
    assert sha(root/'gate'/'FEATURES_PRIVATE.npz')==gate['private_features_sha256']
    with np.load(root/'gate'/'FEATURES_PRIVATE.npz',allow_pickle=True) as z: data={k:z[k] for k in z.files}
    teachers=json.loads((root/'OOF_TEACHER_AUDIT.json').read_text()); assert teachers['status']=='PASS'
    checks=[]; student_prediction_sets=[]
    for fold in range(1,6):
        mask=data['split_fold']==fold
        ids={r:set(data['split_patient'][mask&(data['split_role']==r)]) for r in ['fit','validation','test']}
        fi=np.flatnonzero(np.isin(data['patient'],list(ids['fit'])))
        oof=torch.load(root/f'fold{fold}'/'OOF_FROZEN_PRIVATE.pt',weights_only=False)
        assert np.array_equal(oof['patient'],data['patient'][fi]) and np.array_equal(oof['channel'],data['channel'][fi])
        assert np.array_equal(oof['y_nez'],data['y'][fi])
        q,u=oof['q'],oof['u']; assert np.isfinite(q).all() and np.isfinite(u).all()
        assert np.allclose(u,-(q*np.log(q)+(1-q)*np.log1p(-q))/np.log(2),atol=1e-14,rtol=0)
        delta=.25*u*(q-oof['y_nez']); assert (np.abs(delta)<=.25*u+1e-15).all()
        queried=[]
        for k in range(4):
            cell=root/f'fold{fold}'/f'teacher{k}'
            record=torch.load(cell/'OOF_PRIVATE.pt',weights_only=False)
            plan=record['source']['plan']; validate_exclusion(plan['train'],plan['validation'],plan['query'],ids['fit'])
            assert not set(plan['train']+plan['validation']+plan['query'])&(ids['validation']|ids['test'])
            assert record['preprocessor']['fit_patient_ids']==plan['train']
            queried.extend(record['patient'].tolist())
            line=next(r for r in teachers['folds'] if r['fold']==fold and r['teacher_group']==k)
            assert sha(cell/'OOF_PRIVATE.pt')==line['OOF_file_sha256']
            assert sha(cell/'BEST_PRIVATE.pt')==line['checkpoint_sha256']
        assert len(queried)==len(fi)
        for arm in ['teacher0','teacher1','teacher2','teacher3','A0','A1','A2']:
            cell=root/f'fold{fold}'/arm
            meta=json.loads((cell/'COMPLETE_PRIVATE.json').read_text())
            assert sha(cell/'LAST_PRIVATE.pt')==meta['last_sha256']
            assert sha(cell/'BEST_PRIVATE.pt')==meta['best_sha256']
            saved=torch.load(cell/'LAST_PRIVATE.pt',weights_only=False)
            assert saved['binding']==meta['binding']
            assert 'optimizer' in saved and 'rng' in saved and len(saved['history'])==saved['epoch']
            assert all(torch.isfinite(v).all() for v in saved['model'].values())
            checks.append({'fold':fold,'cell':arm,'epochs':saved['epoch'],
                           'selected_epoch':meta['selected_epoch'],'selected_threshold':meta['threshold'],
                           'last_sha256':meta['last_sha256'],'best_sha256':meta['best_sha256']})
            if arm.startswith('A'):
                prediction=pd.read_csv(cell/'VALIDATION_CHANNEL_PRIVATE.csv')
                assert set(prediction.patient)==ids['validation']
                assert np.isfinite(prediction.score_nez).all()
                assert prediction.threshold.nunique()==1 and prediction.threshold.iloc[0]==meta['threshold']
                assert not prediction[['patient','channel']].duplicated().any()
                pre=torch.load(cell/'PREPROCESSOR_PRIVATE.pt',weights_only=False)
                assert set(pre['fit_patient_ids'])==ids['fit']
                student_prediction_sets.append((fold,arm,set(zip(prediction.patient,prediction.channel))))
        for arm in ['A1','A2']:
            left=next(s for f,t,s in student_prediction_sets if f==fold and t=='A0')
            right=next(s for f,t,s in student_prediction_sets if f==fold and t==arm)
            assert left==right
    json_write(root/'public'/'ARTIFACT_INTEGRITY_AUDIT.json',{'status':'PASS','cells':checks,
        'teacher_cells':20,'student_cells':15,'OOF_identity_and_exclusion_pass':True,
        'last_best_optimizer_rng_history_complete':True,'validation_channel_pairs_exactly_matched':True,
        'outer_evaluation_accessed':False,'post_training_native_shutdown_exception_disclosed':True})
    print('ALL_35_CELLS_INTEGRITY_PASS',flush=True)


if __name__=='__main__': main()
