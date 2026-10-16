"""15 strict inner paths, three matched frozen-CNN readouts, two head seeds."""
import copy
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

HERE=Path(__file__).resolve().parent
COND=Path(r'D:\m8_rawconditional_feature_selection_seed42_v1')
PARENT=Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_dense_reference_runtime')
ROOT=Path(r'D:\m8_conditional_readout_validation_seed42_v1_runtime')
PUB=ROOT/'public'
sys.path.insert(0,str(COND))
import conditional_io as io
from readout import Readout


def prepared(x,train):
    good=np.isfinite(x[train]);count=good.sum(0)
    impute=np.divide(np.nansum(x[train],axis=0),count,out=np.zeros(x.shape[1]),where=count>0)
    filled=np.where(np.isfinite(x),x,impute);mean=filled[train].mean(0);scale=filled[train].std(0,ddof=0);scale[scale==0]=1
    z=((filled-mean)/scale).astype(np.float32)
    assert np.isfinite(z).all()
    return z,dict(imputer=impute,mean=mean,scale=scale)


def plan(core,banks,names):
    original=io.jr(PARENT/'INPUT_FREEZE_PRIVATE.json')
    assert original['original_scientific_binding']==io.bindings()
    assert io.jr(PARENT/'public/INTERNAL_REPLAY_VERIFICATION.json')['status']=='PASS'
    bound=dict(source={n:io.sha(HERE/n) for n in ['PROTOCOL_LOCK.json','readout.py','run_readouts.py','readout_report.py','attribute_audit.py']},
        parent_binding=original,parent_encoder_registry=io.sha(PARENT/'ENCODER_REGISTRY_PRIVATE.pkl'),
        parent_screens={str(f):io.sha(PARENT/f'fold{f}_SCREEN_PRIVATE.pkl') for f in range(1,6)})
    io.freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',bound)
    entries=[];public=[]
    for fold,bank in enumerate(banks,1):
        fit=bank['train'];pid=bank['patient'][fit]
        for screen in range(1,4):
            q=io.read(PARENT/f'fold{fold}_SCREEN_PRIVATE.pkl')['paths'][screen]
            assert [step['k'] for step in q['path']]==list(range(10))
            maximum=max(step['utility'] for step in q['path'])
            step=next(step for step in q['path'] if step['utility']>=maximum-.005)
            roles=q['roles'];sets=[set(pid[ix]) for ix in roles.values()]
            assert all(sets[i].isdisjoint(sets[j]) for i in range(len(sets)) for j in range(i))
            assert set.union(*sets)==bank['roles']['fit'] and set.union(*sets).isdisjoint(bank['roles']['validation']|bank['roles']['test'])
            raw=io.read(Path(q['raw_entry']['path']));assert io.sha(Path(q['raw_entry']['path']))==q['raw_entry']['sha256']
            assert np.array_equal(raw['archive_indices'],bank['archive_indices'][fit])
            assert raw['binding']['role_sha256']==original['original_scientific_binding']['original_role_files'][f'{fold}_{screen}']
            entry=dict(fold=fold,screen=screen,groups=step['groups'],k=step['k'],roles=roles,raw_entry=q['raw_entry'])
            entries.append(entry)
            public.append(dict(fold=fold,screen=screen,k=step['k'],groups=','.join(map(str,step['groups'])),
                features=';'.join(names[g] for g in step['groups']),
                choice_utility=step['utility'],assessment_used_for_this_choice=False,
                role_sha256=raw['binding']['role_sha256'],encoder_artifact_sha256=q['raw_entry']['sha256']))
    assert len(entries)==15
    if (ROOT/'PLAN_PRIVATE.pkl').exists():
        old=io.read(ROOT/'PLAN_PRIVATE.pkl');assert old['binding']==bound
        for old_entry,new_entry in zip(old['entries'],entries):
            assert old_entry['groups']==new_entry['groups']
            for role in old_entry['roles']:assert np.array_equal(old_entry['roles'][role],new_entry['roles'][role])
    else:io.save(ROOT/'PLAN_PRIVATE.pkl',dict(binding=bound,entries=entries))
    pd.DataFrame(public).to_csv(PUB/'FROZEN_READOUT_PLAN.csv',index=False)
    io.freeze(ROOT/'PLAN_FREEZE_PRIVATE.json',dict(binding=bound,plan_sha256=io.sha(ROOT/'PLAN_PRIVATE.pkl')))
    return bound,entries


def prediction(model,x):
    model.eval()
    with torch.inference_mode():logit=model(torch.from_numpy(x)).numpy()
    assert np.isfinite(logit).all()
    return logit


def fit(model,x,refit,cal,y_train,y_cal,pid_train,pid_cal,seed,cell,binding,core,p,objective):
    cell.mkdir(parents=True,exist_ok=True)
    initial=core.state_hash(model.network.state_dict());bb=dict(binding,initial=initial)
    last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json';best_path=cell/'BEST_PRIVATE.pt'
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    positive=max(int((1-y_train).sum()),1)/max(int(y_train.sum()),1)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive,dtype=torch.float32))
    train_x=torch.from_numpy(x[refit]);train_y=torch.from_numpy(y_train.astype(np.float32))
    patients=[np.flatnonzero(pid_train==who) for who in sorted(set(pid_train))]
    best=key=None;stale=0;history=[];start=1
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False);assert saved['binding']==bb
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best=saved['best'];key=saved['key'];stale=saved['stale'];history=saved['history'];start=saved['epoch']+1
        if done.exists():
            assert io.jr(done)==dict(binding=bb,best_sha256=io.sha(best_path),last_sha256=io.sha(last))
            model.load_state_dict(best['model']);return best,history,initial
        if saved['epoch']>=6 and stale>=6:start=31
    # Actual complete-patient input admission, not a discarded optimization run.
    model.eval();sample=train_x[patients[0]].clone()
    with torch.inference_mode():
        expected=model(sample);sample[:,model.mask==0]+=123
        assert torch.equal(expected,model(sample))
    for epoch in range(start,31):
        tick=time.perf_counter();model.train();losses=[]
        for step,j in enumerate(np.random.default_rng(seed+epoch).permutation(len(patients))):
            ii=patients[int(j)];optimizer.zero_grad(set_to_none=True)
            logit=model(train_x[ii]);loss=objective(logit,train_y[ii],loss_fn,'M8_MACRO_AWARE')
            assert torch.isfinite(loss);loss.backward()
            assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
            if epoch==1 and step==0:
                assert model.network[0].weight.grad.norm()>0 and model.network[2].weight.grad.norm()>0
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        probability=expit(-prediction(model,x[cal]).astype(float))
        threshold=p.threshold(1-y_cal,probability,pid_cal)
        rows=p.metric_rows(1-y_cal,probability,pid_cal,np.arange(len(cal)).astype(str),np.full(len(cal),'calibration'),threshold)
        macro=float(np.mean([r['macro_f1'] for r in rows]));ez=float(np.mean([r['f1'] for r in rows]))
        candidate=(macro,ez,-epoch)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-tick,
                            macro_f1=macro,ez_f1=ez,threshold=threshold))
        if key is None or candidate>key:
            key=candidate;stale=0;best=dict(model=copy.deepcopy(model.state_dict()),epoch=epoch,threshold=threshold,macro_f1=macro,ez_f1=ez)
        else:stale+=1
        core.torch_write(last,dict(binding=bb,model=model.state_dict(),optimizer=optimizer.state_dict(),
            best=best,key=key,stale=stale,history=history,epoch=epoch))
        if epoch>=6 and stale>=6:break
    core.torch_write(best_path,dict(binding=bb,**best))
    io.write(done,dict(binding=bb,best_sha256=io.sha(best_path),last_sha256=io.sha(last)))
    model.load_state_dict(best['model']);return best,history,initial


def main():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    common,data,_,_,p=io.old_modules();_,core,_,_,_,_,banks,_=common.legal()
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    bound,entries=plan(core,banks,common.NAMES)
    objective=data.P3.objective
    dependency=dict(objective_sha256=io.sha(Path(inspect.getfile(objective))),torch=torch.__version__,numpy=np.__version__)
    io.freeze(ROOT/'READOUT_ENVIRONMENT_PRIVATE.json',dependency)
    if (ROOT/'READOUT_SCORE_SEAL_PRIVATE.json').exists():
        seal=io.jr(ROOT/'READOUT_SCORE_SEAL_PRIVATE.json');assert seal['binding']==bound
        for name,h in seal['files'].items():assert io.sha(ROOT/name)==h
        print('READOUTS_ALREADY_SEALED_NO_REFIT',flush=True);return
    pool=io.read(io.OLDROOT/'POOL_PRIVATE.pkl');saved=[];public=[];epochs=actual=0
    for entry in entries:
        fold=entry['fold'];screen=entry['screen'];bank=banks[fold-1];fit_indices=bank['train']
        pid=bank['patient'][fit_indices];raw=io.read(Path(entry['raw_entry']['path']))
        original=np.c_[raw['raw'],core.patient_z(pool['summary'][bank['archive_indices'][fit_indices]],pid)]
        roles=entry['roles'];refit=np.r_[roles['train'],roles['choice']];cal=roles['cal'];held=roles['held']
        x,pre=prepared(original,refit)
        preprocessing=ROOT/f'fold{fold}_screen{screen}_PREPROCESS_PRIVATE.pkl'
        if preprocessing.exists():
            prior=io.read(preprocessing);assert prior['binding']==bound and np.array_equal(prior['refit_indices'],refit)
            for key in pre:assert np.array_equal(prior['pre'][key],pre[key])
        else:io.save(preprocessing,dict(binding=bound,pre=pre,refit_indices=refit))
        # The trainer receives TRAIN/CAL label vectors only, not held labels.
        y_train=bank['y'][fit_indices[refit]];y_cal=bank['y'][fit_indices[cal]]
        for head_seed in [42,43]:
            seed=head_seed+1009*fold+50021*screen;initializers=[];seen={}
            for arm,groups in [('RAW_ONLY',[]),('SELECTED',entry['groups']),('OLD7',list(range(7)))]:
                mask_signature=common.digest(groups);cell=ROOT/'models_private'/f'fold{fold}_screen{screen}_seed{head_seed}_{mask_signature[:12]}'
                model=Readout(groups,seed);initializers.append(core.state_hash(model.network.state_dict()))
                binding=dict(input=bound,environment=dependency,fold=fold,screen=screen,seed=seed,groups=groups,
                    preprocessing_sha256=io.sha(ROOT/f'fold{fold}_screen{screen}_PREPROCESS_PRIVATE.pkl'),raw_sha256=entry['raw_entry']['sha256'])
                was_alias=mask_signature in seen
                if not was_alias:
                    best,history,initial=fit(model,x,refit,cal,y_train,y_cal,pid[refit],pid[cal],seed,cell,binding,core,p,objective)
                    actual+=1;epochs+=len(history)
                    output=cell/'ASSESSMENT_LOGITS_PRIVATE.pkl'
                    logit=prediction(model,x[held])
                    payload=dict(binding=binding,checkpoint_sha256=io.sha(cell/'BEST_PRIVATE.pt'),
                        logit_nez=logit,archive_indices=bank['archive_indices'][fit_indices[held]],threshold=best['threshold'])
                    if output.exists():
                        prior=io.read(output);assert prior['binding']==binding and prior['checkpoint_sha256']==payload['checkpoint_sha256']
                        assert prior['threshold']==payload['threshold'] and np.array_equal(prior['logit_nez'],logit)
                    else:io.save(output,payload)
                    seen[mask_signature]=(best,history,initial,output)
                best,history,initial,output=seen[mask_signature]
                saved.append(dict(fold=fold,screen=screen,head_seed=head_seed,arm=arm,groups=groups,cell=str(cell),
                    score=str(output),score_sha256=io.sha(output),checkpoint_sha256=io.sha(cell/'BEST_PRIVATE.pt'),
                    threshold=best['threshold'],best_epoch=best['epoch'],epochs=len(history),initial=initial,
                    held_indices=held,alias=was_alias))
                public.append({k:v for k,v in saved[-1].items() if k not in ['cell','score','held_indices']})
                io.write(PUB/'READOUT_STATUS.json',dict(status='TRAINING_INTERNAL_READOUTS',logical_cells=len(saved),total_cells=90,
                    actual_readout_fits=actual,epochs=epochs,no_new_Raw_encoder_or_Outer=True))
                print('INTERNAL_READOUT_COMPLETE',fold,screen,head_seed,arm,len(saved),'of90',flush=True)
            assert len(set(initializers))==1
    assert len(saved)==90 and io.bindings()==bound['parent_binding']['original_scientific_binding']
    io.save(ROOT/'READOUT_REGISTRY_PRIVATE.pkl',dict(binding=bound,entries=saved))
    files={str(Path(e['score']).relative_to(ROOT)):e['score_sha256'] for e in saved}
    files['READOUT_REGISTRY_PRIVATE.pkl']=io.sha(ROOT/'READOUT_REGISTRY_PRIVATE.pkl')
    io.freeze(ROOT/'READOUT_SCORE_SEAL_PRIVATE.json',dict(binding=bound,files=files,assessment_labels_not_used_for_selection=True))
    pd.DataFrame(public).to_csv(PUB/'READOUT_SELECTION.csv',index=False)
    io.write(PUB/'READOUT_STATUS.json',dict(status='ALL90_INTERNAL_READOUTS_SEALED_BEFORE_ASSESSMENT_REPORT',
        logical_cells=90,actual_readout_fits=actual,epochs=epochs,parameters_per_readout=16513,
        matched_initializers=True,actual_input_mask_admission=True,formal_gradient_admission=True,neural_Raw_encoder_training=0,
        original_Development_or_Outer_used=False,assessment_not_used_for_new_cardinality=True,
        not_full_M8_or_end_to_end_Concat=True,score_seal_sha256=io.sha(ROOT/'READOUT_SCORE_SEAL_PRIVATE.json')))
    print('ALL90_INTERNAL_READOUTS_SEALED',actual,epochs,flush=True)


if __name__=='__main__':
    main()
