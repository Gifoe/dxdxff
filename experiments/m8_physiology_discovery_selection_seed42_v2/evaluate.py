"""Verify all validation states, then seal one frozen Outer without reading labels."""
import gc
import time
from common import *
from model_io import Inputs,bank,masked,make,transform
from model import FeatureHead
predict=load('physiology_v2_pinned_frozen_trainer',HERE/'train.py').predict
from support_control import promote

HISTORY=Path(r'D:\m8_single_head_concat_seed42_v1_runtime')


def main():
    _,core,fac,trainer,records,rec,banks,deps=legal()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    status=jr(PUB/'NEURAL_STATUS.json');assert status['status']=='ALL80_LOGICAL_MODELS_FROZEN'
    assert jr(PUB/'INDEPENDENT_SELECTION_VERIFICATION.json')['status']=='PASS'
    assert sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl')==status['model_freeze_sha256']
    fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');binding=fr['binding'];selection=jr(ROOT/'SELECTED_FEATURES_PRIVATE.json')
    support=read(ROOT/'SUPPORT_MODEL_FREEZE_PRIVATE.pkl');assert jr(PUB/'SUPPORT30_STATUS.json')['status']=='ALL5_SUPPORT_MATCHED_RAW_MODELS_FROZEN'
    for name,h in support['binding']['source'].items():assert sha(HERE/name)==h
    allselections=fr['selections']+support['selections']
    assert sha(ROOT/'SELECTED_FEATURES_PRIVATE.json')==binding['selected_features_sha256']
    for name,h in binding['source'].items():assert sha(HERE/name)==h
    assert sha(Path(r'D:\d1_small_module_exploration_seed42_v1\phase3\code\run.py'))==binding['original_raw_trainer_sha256']
    assert sha(Path(trainer.__file__))==binding['D1_trainer_sha256']
    reg=read(ROOT/'POOL_REGISTRY_PRIVATE.pkl');pool=read(ROOT/'POOL_PRIVATE.pkl');inputs=Inputs(reg)
    # Replay only original validation, before any newOuter scores.
    maximum=0.;checks=[]
    for row in allselections:
        f=row['fold'];d=torch.load(ROOT/f'fold{f}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert d['binding']==binding and set(d['pre']['fit_patient_ids'])==d['roles']['fit']
        assert set(d['patient']).isdisjoint(d['roles']['test'])
        assert row['groups']==([] if row['recipe'] in ['RAW_ONLY','RAW_SUPPORT30'] else selection['folds'][str(f)][row['recipe']])
        assert sha(row['checkpoint'])==row['checkpoint_sha256'] and sha(row['validation'])==row['validation_sha256']
        cp=torch.load(row['checkpoint'],map_location='cuda',weights_only=False)
        assert cp['epoch']==row['epoch'] and cp['threshold']==row['threshold']
        md=masked(d,row['groups']);va=d['val'];ss=torch.tensor(d['scale'],device='cuda')
        if row['family']=='D1':
            m=FeatureHead(42+1009*f).cuda();m.load_state_dict(cp['model']);m.eval()
            with torch.inference_mode():logit=m(torch.as_tensor(md['x'][va],device='cuda')).cpu().numpy()
        else:
            m=make(core,fac,records,f,row['family'],row['groups'])
            if row['recipe']=='RAW_SUPPORT30':m=promote(m)
            m.load_state_dict(cp['model']);logit=predict(m,md,va,inputs,ss)
        saved=read(row['validation']);assert saved['checkpoint_sha256']==row['checkpoint_sha256'] and np.array_equal(saved['archive_indices'],d['archive_indices'][va])
        error=float(np.max(np.abs(saved['logit']-logit)));assert error<=1e-6;maximum=max(maximum,error)
        assert core.select_threshold(d['y'][va],torch.sigmoid(torch.from_numpy(logit)).numpy(),d['patient'][va])['threshold']==row['threshold']
        checks.append(dict(model=row['family']+'__'+row['recipe'],fold=f,maximum_logit_error=error))
        del m,cp;gc.collect();torch.cuda.empty_cache()
    write(PUB/'NEURAL_VALIDATION_VERIFICATION.json',dict(status='PASS',logical_models=85,maximum_logit_error=maximum,ownFIT_VAL_roles_and_features_checked=True,
        all_original_sources_unchanged=True,checks=checks,model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl')))
    prior=jr(HISTORY/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');assert prior['freeze_sha256']==sha(HISTORY/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert jr(HISTORY/'public/INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    historical=[]
    for name,old in jr(HERE/'EVALUATION_LOCK.json')['historical_references'].items():
        for f in range(1,6):
            path=HISTORY/'outer_scores_private'/f'{old}_{f}.pkl';assert sha(path)==prior['files'][f'{old}_{f}']
            historical.append(dict(model=name,fold=f,path=str(path),sha256=sha(path)))
    freeze_object=dict(model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl'),support_freeze_sha256=sha(ROOT/'SUPPORT_MODEL_FREEZE_PRIVATE.pkl'),input_binding=binding,feature_freeze_sha256=sha(ROOT/'SELECTED_FEATURES_PRIVATE.json'),
        evaluation_code={n:sha(HERE/n) for n in ['EVALUATION_LOCK.json','evaluate.py','common.py','model.py','model_io.py','train.py','support_control.py','TIME_SUPPORT_AMENDMENT.json']},historical=historical,
        preprocessing={str(f):sha(ROOT/f'fold{f}/BANK_PRIVATE.pt') for f in range(1,6)},source_metric_sha256=sha(Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py')))
    freeze(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json',freeze_object);fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    done={};cost=[];meta=rec['meta']
    for row in allselections:
        f=row['fold'];model_name=row['family']+'__'+row['recipe'];key=model_name+f'_{f}';target=ROOT/'outer_scores_private'/(key+'.pkl')
        d=torch.load(ROOT/f'fold{f}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False);ix=np.flatnonzero(np.isin(meta['patient'],sorted(d['roles']['test'])))
        assert set(meta['patient'][ix]).isdisjoint(d['roles']['fit']|d['roles']['validation'])
        if target.exists():
            q=read(target);assert q['freeze_sha256']==fh and q['checkpoint_sha256']==row['checkpoint_sha256'];done[key]=sha(target);continue
        raw=pool['summary'][ix];x=transform(raw,meta['patient'][ix],d['pre'],core)
        keep=np.zeros(32,np.float32);keep[row['groups']]=1;x*=np.repeat(keep,2)
        cp=torch.load(row['checkpoint'],map_location='cuda',weights_only=False)
        if row['family']=='D1':m=FeatureHead(42+1009*f).cuda()
        else:
            m=make(core,fac,records,f,row['family'],row['groups'])
            if row['recipe']=='RAW_SUPPORT30':m=promote(m)
        m.load_state_dict(cp['model']);m.eval();ss=torch.tensor(d['scale'],device='cuda');scores=np.empty(len(ix),np.float32);tick=time.perf_counter();windows=0;reused=0
        with torch.inference_mode():
            for pid in sorted(set(meta['patient'][ix])):
                jj=np.flatnonzero(meta['patient'][ix]==pid);ai=ix[jj];identity=digest(dict(freeze=fh,model=model_name,fold=f,patient=str(pid)))
                cache=ROOT/'patient_predictions_private'/(identity+'.pkl')
                pb=dict(freeze_sha256=fh,checkpoint_sha256=row['checkpoint_sha256'],indices_sha256=array_sha(ai),input_sha256=array_sha(x[jj]),source_pool_sha256=reg['files'][pid]['sha256'])
                if cache.exists():q=read(cache);assert q['binding']==pb;pn=q['score_nez'];reused+=1
                else:
                    xx=torch.as_tensor(x[jj],device='cuda')
                    if row['family']=='D1':logit=m(xx)
                    else:
                        rv,mask,times,delta,perms=inputs.get(str(pid),meta['channel'][ai].astype(str).tolist(),'cuda');windows+=int(mask.sum())
                        logit=m(xx,rv,mask,times,delta,ss)[0]
                        del rv,mask,times,delta,perms
                    pn=torch.sigmoid(logit).cpu().numpy();assert np.isfinite(pn).all();save(cache,dict(binding=pb,score_nez=pn))
                scores[jj]=pn
        save(target,dict(freeze_sha256=fh,checkpoint_sha256=row['checkpoint_sha256'],indices=ix,score_nez=scores,threshold=row['threshold']))
        done[key]=sha(target);cost.append(dict(model=model_name,fold=f,patients=len(set(meta['patient'][ix])),channels=len(ix),new_valid_CNN_windows=windows,seconds=time.perf_counter()-tick,reused_patient_predictions=reused))
        write(PUB/'OUTER_STATUS.json',dict(status='SEALING_SCORES_LABELS_NOT_READ',completed_cells=len(done),total_cells=105,freeze_sha256=fh))
        print('FROZEN_DISCOVERY_OUTER_CELL',model_name,f,len(done),'of105',flush=True)
        del m,cp;gc.collect();torch.cuda.empty_cache()
    for row in historical:
        key=row['model']+f"_{row['fold']}";path=ROOT/'outer_scores_private'/(key+'.pkl');q=read(row['path'])
        assert sha(row['path'])==row['sha256'] and q['freeze_sha256']==prior['freeze_sha256']
        save(path,dict(freeze_sha256=fh,historical_sha256=row['sha256'],indices=q['indices'],score_nez=q['score_nez'],threshold=q['threshold']))
        done[key]=sha(path)
    assert len(done)==105
    freeze(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json',dict(freeze_sha256=fh,files=done,all105_score_cells_sealed_before_full_labels=True))
    pd.DataFrame(cost).to_csv(PUB/'OUTER_FORWARD_COST.csv',index=False)
    write(PUB/'OUTER_STATUS.json',dict(status='ALL105_FROZEN_SCORE_CELLS_SEALED',freeze_sha256=fh,seal_sha256=sha(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),new_models=85,historical_reused_cells=20,scientific_status='EXPLORATORY_REPEATED_TEST'))
    print('ALL105_DISCOVERY_SCORE_CELLS_SEALED_NO_FULL_LABEL_READ',flush=True)


if __name__=='__main__':main()
