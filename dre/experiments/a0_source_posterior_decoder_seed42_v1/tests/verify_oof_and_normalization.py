"""Read-only post-completion replay and aggregate FIT normalization-tail diagnosis.

No normalization, model, density, prior, decision or label is changed. Contribution
arithmetic is diagnostic, not a refitted density excluding difficult patients.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True); a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
    sys.path.insert(0,str(a.source/'pr_uncertainty_aware_supervision_seed42_v1/code'))
    from uas_core import prepare,PRMLP,oof_plan,validate_exclusion
    from common import sha,digest,write_json
    from model import SourceA0
    from posterior import normalize
    torch.set_num_threads(2); prior=Path(r'C:\pr_uncertainty_aware_supervision_seed42_runtime'); tails=[]; replay=[]
    for fold in range(1,6):
        root=a.runtime/f'fold{fold}'; d=torch.load(root/'BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        of=torch.load(root/'D2_OOF_PRIVATE.pt',map_location='cpu',weights_only=False)
        tr=d['train']; fp=d['patient'][tr]; fit=set(fp); fx=d['raw_x'][tr]; fg=d['g'][tr]
        assert d['plans']==oof_plan(fit,fold)
        for plan in d['plans']:
            validate_exclusion(plan['train'],plan['validation'],plan['query'],fit)
            ti=np.flatnonzero(np.isin(fp,plan['train'])); qi=np.flatnonzero(np.isin(fp,plan['query']))
            tx,pre=prepare(fx,fp,ti); assert pre['fit_patient_ids']==plan['train']
            ph=digest({q:pre[q].tolist() for q in ['mean','scale','var','imputer_statistics']})
            rec=of['audits'][plan['group']]; assert rec['preprocessor_hash']==ph
            tfile=root/f'D2_teacher{plan["group"]}/BEST_PRIVATE.pt'; assert sha(tfile)==rec['checkpoint_sha256']
            ck=torch.load(tfile,map_location='cpu',weights_only=False); assert ck['binding']['source'].endswith(digest(plan)+ph)
            model=SourceA0(plan['seed']).cuda().eval(); model.load_state_dict(ck['model'])
            with torch.no_grad():logits=model(torch.as_tensor(tx[qi],device='cuda'),torch.as_tensor(fg[qi],device='cuda')).cpu().numpy()
            drift=float(np.abs(logits-of['logits'][qi]).max()); assert drift<=1e-6
            oldfile=prior/f'fold{fold}/teacher{plan["group"]}/BEST_PRIVATE.pt'
            oldck=torch.load(oldfile,map_location='cpu',weights_only=False); oldmodel=PRMLP().eval(); oldmodel.load_state_dict(oldck['model'])
            with torch.no_grad():oldlogit=oldmodel(torch.from_numpy(tx[qi])).numpy()
            olddrift=float(np.abs(oldlogit-d['a0_oof'][qi]).max()); assert olddrift<=1e-6
            replay.append({'fold':fold,'group':plan['group'],'D2_logit_max_drift':drift,'A0_logit_max_drift':olddrift,
                'D2_teacher_sha256':sha(tfile),'A0_teacher_sha256':sha(oldfile),'TRAIN_preprocessor_exact':True,
                'query_train_selection_outer_overlap':0,'query_channels':len(qi)})
        for method,logits in [('D1',d['a0_oof']),('D3',of['logits'])]:
            all_rows=[]
            for pat in sorted(fit):
                ix=np.flatnonzero(fp==pat); z,r=normalize(logits[ix]); y=d['y'][tr][ix]
                for cls in [0,1]:
                    v=z[y==cls]
                    if len(v):all_rows.append({'class':cls,'mean':float(v.mean()),'second':float(np.mean(v*v)),
                        'low_iqr':r<=1e-5,'iqr':r,'max_abs_z':float(np.abs(z).max())})
            for cls in [0,1]:
                rows=[r for r in all_rows if r['class']==cls]; n=len(rows)
                tails.append({'fold':fold,'method':method,'class':['EZ','NEZ'][cls],'class_bearing_patients':n,
                    'low_iqr_patients':sum(r['low_iqr'] for r in rows),
                    'global_mean':float(np.mean([r['mean'] for r in rows])),
                    'low_iqr_contribution_to_global_mean':sum(r['mean'] for r in rows if r['low_iqr'])/n,
                    'other_patients_contribution_to_global_mean':sum(r['mean'] for r in rows if not r['low_iqr'])/n,
                    'low_iqr_contribution_to_global_second_moment':sum(r['second'] for r in rows if r['low_iqr'])/n,
                    'patients_max_abs_z_above_100':sum(r['max_abs_z']>100 for r in rows),
                    'patient_max_abs_z_median':float(np.median([r['max_abs_z'] for r in rows])),
                    'patient_max_abs_z_max':max(r['max_abs_z'] for r in rows),
                    'minimum_IQR':min(r['iqr'] for r in rows),'normalization_or_density_altered':False})
        print('INDEPENDENT_OOF_AND_TAIL_AUDIT_PASS',fold,flush=True)
    pub=a.runtime/'public'
    write_json(pub/'INDEPENDENT_OOF_REPLAY.json',{'status':'PASS','teachers':replay,'all_A0_D2_query_predictions_replayed':40,
        'stored_OOF_not_modified':True,'no_invalid_density_inference':True,'reporting_source_sha256':sha(__file__)})
    pd.DataFrame(tails).to_csv(pub/'NORMALIZATION_TAIL_AUDIT.csv',index=False)
    status=json.loads((pub/'RUN_STATUS.json').read_text()); status['independent_OOF_replay_passed']=True
    status['aggregate_files']={q.name:sha(q) for q in pub.glob('*') if q.name!='RUN_STATUS.json'}
    write_json(pub/'RUN_STATUS.json',status)

if __name__=='__main__':main()
