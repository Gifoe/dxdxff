"""Label-free Outer prediction after the complete corresponding stage freeze."""
import sys
import numpy as np
import torch
from scipy.special import logit
from runtime import *
from svm_adapter import logits,probabilities
from train_svm_plugin import make as make_a,output_z
from train_drechan_joint import make as make_b
from training import predict

def simple(score,probability=None):
    z=np.zeros_like(score,dtype=float)
    out=dict(anchor=score.copy(),delta_rank=z.copy(),delta_cal=z.copy(),rank=score.copy(),final=score.copy(),rank_score=score.copy())
    if probability is not None:out['probability_exact']=probability.copy();out['rank_score']=probability.copy()
    return out

def historical_svm(cell,data):
    arm='Linear_SVM_'+('PR' if cell['arm']=='S1_LEGACY_PR' else 'NO_PR');f=cell['fold']
    path=SVMROOT/'scores_private'/f'{arm}_{f}.pkl';seal=read_json(SVMROOT/'TABULAR_SCORE_SEAL_PRIVATE.json')
    assert sha(path)==seal['files'][f'{arm}_{f}'];q=load(path)
    assert q['checkpoint']==cell['checkpoint_sha256'] and np.array_equal(q['indices'],data['archive_indices'])
    x=data['raw12'].astype(float)
    if cell['arm']=='S1_LEGACY_PR':x=context()['core'].patient_z(x,data['patient'])
    replay=probabilities(load(cell['checkpoint'])['bundle'],x)
    assert np.max(abs(replay-q['probability_ez']))<=1e-12
    for p in sorted(set(data['patient'])):
        ix=data['patient']==p;assert np.array_equal(replay[ix]>=cell['threshold'],q['prediction_ez'][ix])
        assert np.array_equal(np.lexsort((data['channel'][ix],-replay[ix])),np.lexsort((data['channel'][ix],-q['probability_ez'][ix])))
    score=logit(np.clip(q['probability_ez'],1e-15,1-1e-15))
    return simple(score,q['probability_ez'])

def intervention(out,model,data,variant):
    q={k:v.copy() for k,v in out.items() if isinstance(v,np.ndarray)}
    if variant=='NO_CAL':q['delta_cal'][:]=0
    else:
        q['delta_rank'][:]=0
        if variant=='NO_BOTH':q['delta_cal'][:]=0
        else:
            for patient in sorted(set(data['patient'])):
                ix=data['patient']==patient
                t=torch.as_tensor(q['anchor'][ix],device=next(model.parameters()).device,dtype=torch.float32)
                s=torch.stack([t.mean(),t.std(unbiased=False),t.new_tensor(len(t)).log1p()])
                with torch.no_grad():q['delta_cal'][ix]=float(model.plugin.calibration(s))
    q['rank']=q['anchor']+q['delta_rank'];q['final']=q['rank']+q['delta_cal'];q['rank_score']=q['rank'].copy()
    return q

def main(stage):
    check_admission();fr=read_json(ROOT/(stage+'_FREEZE.json'));assert fr['source']==source_hashes()
    c=context();seal=[]
    for cell in fr['cells']:
        fold=cell['fold'];arm=cell['arm'];data=bank_data(fold,outer=True,labels=False)
        assert 'y' not in data
        target=ROOT/'scores_private'/stage/f'{arm}_fold{fold}.pkl'
        binding=dict(freeze=sha(ROOT/(stage+'_FREEZE.json')),checkpoint=cell['checkpoint_sha256'],indices=digest(data['archive_indices'].tolist()))
        model=None
        if target.exists():
            q=load(target);assert q['binding']==binding;out=q['components']
        else:
            if stage=='A':
                if arm in ['S0_NO_PR','S1_LEGACY_PR']:out=historical_svm(cell,data)
                else:
                    base=load(SVMROOT/'tabular_models_private/Linear_SVM_NO_PR'/f'fold{fold}.pkl')
                    data['base']=logits(base['bundle'],data['raw12'].astype(float))
                    if arm=='S2_OUTPUT_Z':out=simple(output_z(data['base'],data['patient']))
                    else:
                        assert sha(cell['checkpoint'])==cell['checkpoint_sha256'];model=make_a(arm,fold)
                        model.load_state_dict(torch.load(cell['checkpoint'],weights_only=False,map_location='cpu')['model'])
                        out=predict(model,data,np.arange(len(data['patient'])),fold,'A')
            elif arm=='D_HISTORICAL_FULL':
                old,oldsha,_=c['parent'].reference_score(cell['historical_cell'])
                assert np.array_equal(old['indices'],data['archive_indices'])
                p=1-old['score_nez'].astype(float);out=simple(logit(np.clip(p,1e-15,1-1e-15)),p)
            else:
                assert sha(cell['checkpoint'])==cell['checkpoint_sha256'];model=make_b(arm,fold)
                model.load_state_dict(torch.load(cell['checkpoint'],weights_only=False,map_location='cpu')['model'])
                out=predict(model,data,np.arange(len(data['patient'])),fold,'B')
            save(target,dict(binding=binding,components=out,archive_indices=data['archive_indices']))
        seal.append(dict(arm=arm,fold=fold,path=str(target),sha256=sha(target),threshold=cell['threshold'],
                         threshold_kind=cell['threshold_kind'],inference_only=False))
        variants=(['NO_CAL'] if stage=='A' and arm=='S5_PR2_FULL' else
                  ['NO_RANK','NO_CAL','NO_BOTH'] if stage=='B' and arm=='D2_PR2_FULL' else [])
        if variants:
            if model is None:
                model=make_a(arm,fold) if stage=='A' else make_b(arm,fold)
                model.load_state_dict(torch.load(cell['checkpoint'],weights_only=False,map_location='cpu')['model'])
            for variant in variants:
                name=('S5_' if stage=='A' else 'D2_')+variant
                ipath=target.with_name(f'{name}_fold{fold}.pkl');output=intervention(out,model,data,variant)
                value=dict(binding=dict(binding,intervention=variant),components=output,archive_indices=data['archive_indices'])
                if ipath.exists():
                    prior=load(ipath);assert prior['binding']==value['binding']
                    assert all(np.array_equal(prior['components'][k],output[k]) for k in output)
                else:save(ipath,value)
                seal.append(dict(arm=name,fold=fold,path=str(ipath),sha256=sha(ipath),threshold=cell['threshold'],
                                 threshold_kind=cell['threshold_kind'],inference_only=True))
        if model is not None:del model;gc.collect();torch.cuda.empty_cache()
        print('PR2_OUTER_SEALED',stage,arm,fold,len(seal),flush=True)
    assert len(seal)==(35 if stage=='A' else len(fr['cells'])+15)
    write(ROOT/(stage+'_SCORE_SEAL.json'),dict(freeze=sha(ROOT/(stage+'_FREEZE.json')),files=seal,all_scores_sealed_before_metrics=True))
    print('PR2_ALL_OUTER_SCORES_SEALED',stage,len(seal),flush=True)

if __name__=='__main__':main(sys.argv[1])
