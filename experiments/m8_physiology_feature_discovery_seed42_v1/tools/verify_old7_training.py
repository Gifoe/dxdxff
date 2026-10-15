"""Replay all80 legal validation cells; no fits and no Outer labels/predictions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_old7_m8 import *
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score


def main():
    lock,c,core,fac,trainer,records,rec,banks,binding=setup_new()
    assert jr(RUN/'public/STATUS.json')['status']=='ALL80_MATCHED_OLD7_D1_AND_M8_DEVELOPMENT_CELLS_COMPLETE'
    selection=read(RUN/'MODEL_SELECTION_PRIVATE.pkl');assert selection['binding']==binding
    rows=selection['selections'];assert len(rows)==80
    store=P3.Inputs(BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),RUN);store.arm='M1_SCALED_DGM'
    maximum=0.;count=0
    for r in rows:
        fold=r['fold'];d=torch.load(r['bank'],map_location='cpu',weights_only=False);original=banks[fold-1]
        assert d['binding']==binding and sha(r['bank'])==r['bank_sha256']
        legacy=torch.load(BASELINE/f'D1/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        expected=legacy['x']*np.repeat(np.asarray(r['keep'],np.float32),2)[None]
        assert np.array_equal(d['x'],expected)
        assert set(d['patient'][d['train']])==original['roles']['fit'] and set(d['patient'][d['val']])==original['roles']['validation']
        assert set(d['patient']).isdisjoint(original['roles']['test'])
        ck={}
        for head in ['D1','M8']:
            cp=Path(r[head+'_checkpoint']);assert sha(cp)==r[head+'_checkpoint_sha256']
            ck[head]=torch.load(cp,map_location='cpu',weights_only=False)
            assert ck[head]['epoch']==r[head+'_epoch'] and ck[head]['threshold']==r[head+'_threshold']
            done=jr(cp.parent/'COMPLETE_PRIVATE.json');assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt')
            last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            assert 6<=len(last['history'])<=30 and [h['epoch'] for h in last['history']]==list(range(1,len(last['history'])+1))
            winner=max(last['history'],key=lambda h:(h['macro_f1'],h['ez_f1'],-h['epoch']))
            assert winner['epoch']==ck[head]['epoch'] and winner['threshold']==ck[head]['threshold']
        model,original_model=build(core,fac,records,Path(r['D1_checkpoint']),fold,r['keep']);del original_model
        model.load_state_dict(ck['M8']['model']);model.eval()
        assert core.state_hash(model.base.state_dict())==core.state_hash(ck['D1']['model'])
        scale=torch.tensor(read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda')
        pred=P3.EV.predict(model,d,d['val'],store,scale);q=read(r['validation_path'])
        assert q['D1_checkpoint_sha256']==r['D1_checkpoint_sha256'] and q['M8_checkpoint_sha256']==r['M8_checkpoint_sha256']
        assert np.array_equal(q['validation_archive_indices'],d['archive_indices'][d['val']])
        with torch.inference_mode():base=model.base(torch.as_tensor(d['x'][d['val']],device='cuda')).cpu().numpy()
        for head,logit in [('D1',base),('M8',pred['logit'])]:
            error=float(np.max(np.abs(q[head+'_logit']-logit)));maximum=max(maximum,error);assert error<=1e-6
            pn=torch.sigmoid(torch.from_numpy(q[head+'_logit'])).numpy();sel=core.select_threshold(d['y'][d['val']],pn,d['patient'][d['val']])
            assert sel['threshold']==r[head+'_threshold']
            frame=pd.DataFrame(dict(model=r['variant'],version=head,fold=fold,patient=d['patient'][d['val']],channel=d['channel'][d['val']],
                center=d['center'][d['val']],y=(d['y'][d['val']]==0).astype(int),probability=1-pn.astype(float),prediction=pn<sel['threshold']))
            patient=EV.patient_rows(frame)
            for row in patient:
                h=frame[frame.patient==row['patient']];y=h.y.to_numpy();p=h.probability.to_numpy();v=h.prediction.to_numpy()
                checks=dict(macro_f1=f1_score(y,v,labels=[0,1],average='macro',zero_division=0),f1=f1_score(y,v,zero_division=0),
                    nez_f1=f1_score(1-y,1-v.astype(int),zero_division=0),accuracy=accuracy_score(y,v))
                if len(set(y))==2:checks.update(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p))
                for k,value in checks.items():assert abs(row[k]-value)<1e-12,(r['variant'],head,k)
                count+=1
        print('OLD7_INDEPENDENT_MODEL_REPLAY_PASS',r['variant'],fold,flush=True)
        del model,d,ck;gc.collect();torch.cuda.empty_cache()
    assert count==16*2*65
    for p,h in binding['protected'].items():assert sha(p)==h
    write(RUN/'public/INDEPENDENT_RETRAIN_VERIFICATION.json',dict(status='PASS',pairs=80,actual_models=160,
        all80_M8_validation_inference_replayed=True,all80_D1_validation_inference_replayed=True,
        maximum_logit_replay_error=maximum,registered_absolute_tolerance=1e-6,independent_patient_metric_rows=count,
        all_best_epochs_and_thresholds_reproduced=True,all_original_FIT_validation_test_roles_verified=True,
        no_training_or_Outer_read=True,protected_artifacts_unchanged=True,verifier_sha256=sha(__file__),full_goal_complete=False))
    print('ALL160_OLD7_D1_M8_MODELS_INDEPENDENT_VALIDATION_REPLAY_PASS',flush=True)


if __name__=='__main__':main()
