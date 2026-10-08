"""15 matched students; new A0 must reproduce PR-UAS development before B1/B2."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from baseline import PRMLP, sha, json_write, torch_write, seed_all, state_hash, patient_metrics
from train import train_weighted
from prepare import binding


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--prior-runtime',type=Path,required=True)
    parser.add_argument('--device',default='cuda'); a=parser.parse_args(); root=a.runtime
    torch.set_num_threads(2)
    feasible=json.loads((root/'public/PLLR_FEASIBILITY_AUDIT.json').read_text())
    smoke=json.loads((root/'public/TEST_AND_SMOKE_AUDIT.json').read_text())
    bind=binding(a.protocol)
    assert feasible['pass'] and smoke['status']=='PASS'
    assert feasible['binding']==smoke['binding']==bind
    assert smoke['runtime']['torch']==torch.__version__ and smoke['runtime']['numpy']==np.__version__
    a0checks=[]; selections=[]
    # Complete and independently compare all fresh A0 runs before admitting B1/B2.
    for arm in ['A0','B1','B2']:
        for fold in range(1,6):
            foldroot=root/f'fold{fold}'
            sealed=foldroot/'DEVELOPMENT_PRIVATE.pt'; reliability=foldroot/'PLLR_FROZEN_PRIVATE.pt'
            fr=next(r for r in feasible['folds'] if r['fold']==fold)
            assert sha(reliability)==fr['PLLR_sha256']
            d=torch.load(sealed,weights_only=False); w=torch.load(reliability,weights_only=False)
            assert w['binding']=={**bind,'sealed':sha(sealed)}
            tr,va=d['train'],d['val']
            assert np.array_equal(w['patient'],d['patient'][tr])
            assert np.array_equal(w['channel'],d['channel'][tr]) and np.array_equal(w['y'],d['y'][tr])
            assert set(d['patient'][tr]).isdisjoint(d['patient'][va])
            weights=np.ones(len(d['x']))
            if arm!='A0':weights[tr]=w['artifact'][arm]
            seed=42+1009*fold; seed_all(seed); initial=copy.deepcopy(PRMLP().state_dict())
            assert sum(v.numel() for v in initial.values())==8817
            cb={**bind,'sealed':sha(sealed),'PLLR':sha(reliability),
                'smoke':sha(root/'public/TEST_AND_SMOKE_AUDIT.json')}
            json_write(root/'RUN_STATUS.json',{'status':'TRAINING_DEVELOPMENT','fold':fold,'arm':arm,
                'outer_evaluation_run':False,'binding':bind})
            cell=foldroot/arm
            model,best,history=train_weighted(d['x'],d['y'],d['patient'],tr,va,cell,seed,initial,
                json.dumps(cb,sort_keys=True),a.device,arm,weights)
            with torch.no_grad():score=torch.sigmoid(model(torch.as_tensor(d['x'][va],device=a.device))).cpu().numpy()
            patients=patient_metrics(d['y'][va],score,d['patient'][va],d['center'][va],best['threshold'])
            patients['arm']=arm; patients['fold']=fold
            patients.to_csv(cell/'VALIDATION_PATIENT_PRIVATE.csv',index=False)
            pred=pd.DataFrame({'patient':d['patient'][va],'channel':d['channel'][va],'center':d['center'][va],
                'y_nez':d['y'][va],'score_nez':score,'threshold':best['threshold'],'arm':arm,'fold':fold})
            pred.to_csv(cell/'VALIDATION_CHANNEL_PRIVATE.csv',index=False)
            selection={'arm':arm,'fold':fold,'epoch':best['epoch'],'threshold':best['threshold'],
                'initial_hash':state_hash(initial),'parameters':8817,'checkpoint_sha256':sha(cell/'BEST_PRIVATE.pt'),
                'completed_epochs':len(history),'PLLR_sha256':sha(reliability)}
            json_write(cell/'SELECTION.json',selection); selections.append(selection)
            if arm=='A0':
                reference=pd.read_csv(a.prior_runtime/f'fold{fold}/A0/VALIDATION_CHANNEL_PRIVATE.csv')
                joined=pred.merge(reference,on=['patient','channel','fold','center','y_nez'],validate='one_to_one',suffixes=('_new','_old'))
                assert len(joined)==len(pred)==len(reference)
                drift=float(np.abs(joined.score_nez_new-joined.score_nez_old).max()); assert drift<=2e-6
                old=json.loads((a.prior_runtime/f'fold{fold}/A0/SELECTION.json').read_text())
                assert best['epoch']==old['epoch'] and best['threshold']==old['threshold']
                assert selection['initial_hash']==old['initial_hash']
                a0checks.append({'fold':fold,'max_probability_drift':drift,'selected_epoch':best['epoch'],
                    'selected_threshold':best['threshold'],'matched_IDs_labels_and_initial_state':True,
                    'reference_prediction_sha256':sha(a.prior_runtime/f'fold{fold}/A0/VALIDATION_CHANNEL_PRIVATE.csv'),
                    'macro_f1':float(patients.macro_f1.mean())})
            print('STUDENT_COMPLETE',fold,arm,flush=True)
        if arm=='A0':
            mean=float(np.mean([r['macro_f1'] for r in a0checks]))
            assert abs(mean-.6380797828499001)<1e-10
            report=json.loads((root/'public/A0_REPRODUCTION.json').read_text())
            report.update({'status':'PASS','matched_development_training_PASS':True,
                'matched_development_macro_f1':mean,'fresh_A0_folds':a0checks})
            json_write(root/'public/A0_REPRODUCTION.json',report)
            print('MATCHED_A0_DEVELOPMENT_REPRODUCTION_PASS',flush=True)
    json_write(root/'public/SELECTION_AUDIT.json',{'selections':selections,
        'same_initial_state_within_fold':True,'no_outer_test_evaluation':True})
    json_write(root/'RUN_STATUS.json',{'status':'ALL_15_STUDENTS_COMPLETE','student_cells':15,
        'outer_evaluation_run':False,'binding':bind})
    print('ALL_15_STUDENTS_COMPLETE',flush=True)


if __name__=='__main__':main()
