"""One posthoc combined deletion; reuse exact previous audit/training helpers."""
import argparse
import copy
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ARM='w/o Source + w/o ZDELTA'
REFS=['Full D2','w/o Source','w/o ZDELTA']


def setup(a):
    oldcode=a.source/'d2_ictal_ablation_seed42_v1/code'
    sys.path.insert(0,str(oldcode))
    import run_study as old
    from ablation_core import mask_for,masked,source_free_factory
    base=copy.copy(a);base.protocol=oldcode.parent/'PROTOCOL_LOCK.json';base.runtime=a.previous
    o=old.setup(base);oldlock,ref,c,core,model,trainer,helper,original_bind=o
    lock=json.loads(a.protocol.read_text());assert lock['arm']==ARM and lock['parameters']==8817
    assert c.sha(a.previous/'FROZEN_BEFORE_OUTER.json')==lock['old_study_freeze_sha256']
    assert c.sha(a.previous/'OUTER_PREDICTIONS_SEAL.json')==lock['old_study_prediction_seal_sha256']
    oldfreeze=json.loads((a.previous/'FROZEN_BEFORE_OUTER.json').read_text())
    assert oldfreeze['binding']==original_bind
    assert c.sha(a.previous/'public/FEATURE_GROUP_MANIFEST.json')==oldfreeze['feature_manifest_sha256']
    manifest=json.loads((a.previous/'public/FEATURE_GROUP_MANIFEST.json').read_text())
    mask=mask_for(manifest['columns'],'w/o ZDELTA');assert mask.sum()==62
    for col in manifest['columns']:
        if 'duplicate_of' in col:assert mask[col['index']]==mask[col['duplicate_of']]
    bind={'original_study_binding':original_bind,'new_protocol':c.sha(a.protocol),'runner':c.sha(__file__),
        'old_freeze':c.sha(a.previous/'FROZEN_BEFORE_OUTER.json'),'old_prediction_seal':c.sha(a.previous/'OUTER_PREDICTIONS_SEAL.json')}
    a.runtime.mkdir(parents=True,exist_ok=True);(a.runtime/'public').mkdir(exist_ok=True)
    return lock,ref,c,core,model,trainer,helper,old,base,oldfreeze,mask,masked,source_free_factory,bind


def prepare(a,o):
    lock,ref,c,core,model,trainer,helper,old,base,oldfreeze,mask,masked,factory,bind=o
    out=a.runtime/'public/PRETRAIN_AUDIT.json';assert not out.exists()
    raw=old.raw_without_y(base);rows=[]
    for fold in range(1,6):
        d=torch.load(a.original/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        ids=old.roles(raw,fold);di=np.flatnonzero(np.isin(raw['patient'],list(ids['fit']|ids['validation'])))
        assert np.array_equal(raw['x'][di],d['raw_x'],equal_nan=True)
        assert not set(d['patient'])&ids['test']
        assert set(d['patient'][d['train']])==ids['fit'] and set(d['patient'][d['val']])==ids['validation']
        x,pre=core.prepare(d['raw_x'],d['patient'],d['train']);assert np.array_equal(x,d['x'])
        for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pre[k],d['pre'][k])
        xx=masked(x,mask);seed=42+1009*fold
        core.seed_all(seed);full=model.SourceA0(seed)
        core.seed_all(seed);tm=factory(model.SourceA0)(seed)
        assert core.state_hash(full.network.state_dict())==core.state_hash(tm.network.state_dict())
        z=torch.from_numpy(xx[:12]);g=torch.from_numpy(d['g'][:12]);tm.eval()
        assert torch.equal(tm(z,g),tm(z,(g+1)%4))
        assert all(k.startswith('network.') for k in tm.state_dict()) and sum(p.numel() for p in tm.parameters())==8817
        original_ck=torch.load(a.original/f'fold{fold}/D2/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert original_ck['binding']['initial_hash']==core.state_hash(full.state_dict())
        masked_ref=torch.load(a.previous/f'no_zdelta/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert np.array_equal(xx,masked_ref['x']) and np.array_equal(mask,masked_ref['mask'])
        bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert not bp.exists()
        cell={k:d[k] for k in ['patient','channel','center','features','train','val','y','g','pre']}
        cell.update({'x':xx,'mask':mask,'binding':bind,'initial_hash':core.state_hash(tm.state_dict())})
        core.torch_write(bp,cell)
        rows.append({'fold':fold,'bank_sha256':c.sha(bp),'initial_hash':cell['initial_hash'],
            'shared_initial_parity':True,'retained_columns':62,'excluded_columns':26,'source_free_parameters':8817,
            'FIT_only_preprocessor':True,'outer_overlap':0,'same_masked_inputs_as_previous_no_ZDELTA':True})
    # Combined changed path: gradient and exact synthetic CUDA epoch resume gate.
    rng=np.random.default_rng(42);x=masked(rng.normal(size=(24,88)).astype('float32'),mask)
    y=np.tile([0,1],12);pid=np.repeat(['synthetic_FIT_a','synthetic_FIT_b','synthetic_VAL_a','synthetic_VAL_b'],6)
    g=np.arange(24)%4;tr=np.arange(12);va=np.arange(12,24);seed=1051
    core.seed_all(seed);tm=factory(model.SourceA0)(seed).cuda()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(tm(torch.tensor(x,device='cuda'),torch.tensor(g,device='cuda')),torch.tensor(y,dtype=torch.float32,device='cuda'))
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in tm.parameters())
    directory=Path(tempfile.mkdtemp(prefix='synthetic_combo_resume_',dir=a.runtime))
    original_factory=trainer.SourceA0;trainer.SourceA0=factory(model.SourceA0)
    try:
        trainer.train(x,y,pid,g,tr,va,directory/'full',seed,'synthetic',max_epochs=6,penalty=0.,device='cuda')
        trainer.train(x,y,pid,g,tr,va,directory/'resume',seed,'synthetic',max_epochs=6,stop_after=2,penalty=0.,device='cuda')
        trainer.train(x,y,pid,g,tr,va,directory/'resume',seed,'synthetic',max_epochs=6,penalty=0.,device='cuda')
    finally:trainer.SourceA0=original_factory
    p=torch.load(directory/'full/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
    q=torch.load(directory/'resume/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,q['model'][k]) for k,v in p['model'].items()) and p['history']==q['history'] and p['best_key']==q['best_key']
    assert all(torch.equal(v,q['optimizer']['state'][k][n]) for k,slot in p['optimizer']['state'].items() for n,v in slot.items())
    assert torch.equal(p['rng']['torch'],q['rng']['torch']) and all(torch.equal(v,w) for v,w in zip(p['rng']['cuda'],q['rng']['cuda']))
    c.write_json(out,{'status':'PASS','binding':bind,'folds':rows,'combined_path_CUDA_gradient_and_exact_resume':True,'new_outer_y_loaded':False})
    print('COMBINATION_PRETRAIN_AND_RESUME_PASS',flush=True)


def train_freeze(a,o):
    lock,ref,c,core,model,trainer,helper,old,base,oldfreeze,mask,masked,factory,bind=o
    audit=json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text());assert audit['binding']==bind
    rows=[];histories=[]
    for fold in range(1,6):
        bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==audit['folds'][fold-1]['bank_sha256']
        d=torch.load(bp,map_location='cpu',weights_only=False)
        original_factory=trainer.SourceA0;trainer.SourceA0=factory(model.SourceA0)
        try:
            trainer.train(d['x'],d['y'],d['patient'],d['g'],d['train'],d['val'],bp.parent/'TRAIN',42+1009*fold,
                c.digest({'bank':c.sha(bp),'binding':bind}),penalty=0.,device='cuda')
        finally:trainer.SourceA0=original_factory
        cp=bp.parent/'TRAIN/BEST_PRIVATE.pt';ck=torch.load(cp,map_location='cpu',weights_only=False)
        assert ck['binding']['initial_hash']==d['initial_hash']
        done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==c.sha(cp)
        rows.append({'arm':ARM,'fold':fold,'epoch':ck['epoch'],'nez_threshold':ck['threshold'],
            'checkpoint_sha256':c.sha(cp),'bank_sha256':c.sha(bp),'mask':mask.tolist()})
        last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
        histories.extend({'fold':fold,**h} for h in last['history'])
        print('COMBO_FOLD_SELECTED',fold,ck['epoch'],ck['threshold'],flush=True)
    reference=[r for r in oldfreeze['selections'] if r['arm'] in REFS]
    for r in reference:
        bp,cp=old.bank_ck(base,r['arm'],r['fold']);assert c.sha(bp)==r['bank_sha256'] and c.sha(cp)==r['checkpoint_sha256']
    path=a.runtime/'FROZEN_BEFORE_OUTER.json';assert not path.exists()
    c.write_json(path,{'status':'PASS','binding':bind,'selections':rows,'reference_selections':reference,
        'all_20_models_and_thresholds_locked_before_new_inference':True,'new_full_y_loaded':False})
    c.write_json(a.runtime/'public/FREEZE_AUDIT.json',{'status':'PASS','freeze_sha256':c.sha(path),'new_models':5,'reused_models':15})
    pd.DataFrame([{k:v for k,v in r.items() if k!='mask'} for r in rows]).to_csv(a.runtime/'public/MODEL_SELECTION.csv',index=False)
    pd.DataFrame(histories).to_csv(a.runtime/'public/TRAINING_HISTORY.csv',index=False)


def evaluate(a,o):
    lock,ref,c,core,model,trainer,helper,old,base,oldfreeze,mask,masked,factory,bind=o;pub=a.runtime/'public'
    assert not (pub/'RUN_STATUS.json').exists(),'Completed outer test cannot rerun'
    frozen=json.loads((a.runtime/'FROZEN_BEFORE_OUTER.json').read_text());assert frozen['binding']==bind
    fs=c.sha(a.runtime/'FROZEN_BEFORE_OUTER.json');raw=old.raw_without_y(base);cells={};files={}
    oldseal=json.loads((a.previous/'OUTER_PREDICTIONS_SEAL.json').read_text())['files']
    from ablation_core import KEYS
    for arm in [ARM]+REFS:
        visited=[]
        for fold in range(1,6):
            ix=np.flatnonzero(np.isin(raw['patient'],list(old.roles(raw,fold)['test'])))
            if arm==ARM:
                rec=frozen['selections'][fold-1];root=a.runtime/f'fold{fold}';bp=root/'BANK_PRIVATE.pt';cp=root/'TRAIN/BEST_PRIVATE.pt'
                assert c.sha(bp)==rec['bank_sha256'] and c.sha(cp)==rec['checkpoint_sha256'] and np.array_equal(mask,rec['mask'])
                path=root/'OUTER_PREDICTIONS_PRIVATE.pt'
                if path.exists():
                    assert json.loads(path.with_suffix('.sha256.json').read_text())=={'binding':fs,'sha256':c.sha(path)}
                    cell=torch.load(path,map_location='cpu',weights_only=False);assert cell['binding']==fs and np.array_equal(cell['indices'],ix)
                else:
                    d=torch.load(bp,map_location='cpu',weights_only=False);ck=torch.load(cp,map_location='cpu',weights_only=False)
                    x=masked(helper.frozen_transform(raw['x'][ix],raw['patient'][ix],d['pre'],core),mask)
                    tm=factory(model.SourceA0)().eval();tm.load_state_dict(ck['model'],strict=True)
                    with torch.inference_mode():
                        logits=tm(torch.from_numpy(x),torch.from_numpy(model.source_index(raw['center'][ix]))).numpy()
                        scores=torch.sigmoid(torch.from_numpy(logits)).numpy()
                    cell={'binding':fs,'indices':ix,'scores':scores,'logits':logits,'threshold':ck['threshold']}
                    core.torch_write(path,cell);c.write_json(path.with_suffix('.sha256.json'),{'binding':fs,'sha256':c.sha(path)})
            else:
                path=old.prediction_path(base,arm,fold);assert c.sha(path)==oldseal[KEYS[arm]+'_'+str(fold)]
                cell=torch.load(path,map_location='cpu',weights_only=False);assert np.array_equal(cell['indices'],ix)
            assert np.isfinite(cell['scores']).all();cells[(arm,fold)]=cell;files[arm+'_'+str(fold)]=c.sha(path);visited.extend(ix.tolist())
        assert sorted(visited)==list(range(7635))
    c.write_json(a.runtime/'OUTER_PREDICTIONS_SEAL.json',{'binding':fs,'files':files,'all_20_predictions_before_full_y':True})
    with np.load(a.features,allow_pickle=True) as f:y=f['y']
    assert len(y)==7635 and int((y==0).sum())==1743
    rows=[]
    for (arm,fold),cell in cells.items():
        ix=cell['indices']
        for pat in sorted(set(raw['patient'][ix])):
            k=np.flatnonzero(raw['patient'][ix]==pat);k=k[np.argsort(raw['channel'][ix][k].astype(str),kind='stable')];si=ix[k]
            rows.append({'arm':arm,'fold':fold,'patient':pat,'center':str(raw['center'][si[0]]),**c.metric(y[si],cell['scores'][k],cell['threshold'])})
    df=pd.DataFrame(rows);core.torch_write(a.runtime/'PATIENT_METRICS_PRIVATE.pt',rows);summary=[]
    previous=pd.read_csv(a.previous/'public/ABLATION_SUMMARY.csv')
    for arm in [ARM]+REFS:
        part=df[df.arm==arm];assert len(part)==part.patient.nunique()==80 and part.channels.sum()==7635
        if arm in REFS:
            for key in c.METRICS:assert abs(part[key].mean()-previous[previous.arm==arm][key].iloc[0])<1e-12
        summary.append({'arm':arm,**part[c.METRICS].mean().to_dict(),'parameters':8817 if arm in [ARM,'w/o Source'] else 9221,
            'retained_columns':62 if arm in [ARM,'w/o ZDELTA'] else 88,'patients':80,'channels':7635,
            'training_status':'new_five_fold_retraining' if arm==ARM else 'frozen_reuse'})
    pd.DataFrame(summary).to_csv(pub/'OUTER_SUMMARY.csv',index=False)
    for field,name in [('fold','OUTER_BY_FOLD.csv'),('center','OUTER_BY_CENTER.csv')]:
        grouped=[{'arm':arm,field:v,'patients':len(part),'channels':int(part.channels.sum()),**part[c.METRICS].mean().to_dict()} for (arm,v),part in df.groupby(['arm',field])]
        pd.DataFrame(grouped).to_csv(pub/name,index=False)
    new=df[df.arm==ARM].set_index('patient').sort_index();effects=[]
    for arm in REFS:
        baseline=df[df.arm==arm].set_index('patient').sort_index();assert np.array_equal(new.index,baseline.index)
        delta=new[c.METRICS].to_numpy()-baseline[c.METRICS].to_numpy();boot=c.cluster_bootstrap(new.index,delta,10000)
        for j,key in enumerate(c.METRICS):effects.append({'reference':arm,'direction':'Combination minus reference','metric':key,
            'delta':np.mean(delta[:,j]),'ci_low':np.percentile(boot[:,j],2.5),'ci_high':np.percentile(boot[:,j],97.5),
            'draws':10000,'seed':42,'threshold_reoptimized':False})
    pd.DataFrame(effects).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False)
    assert setup(a)[-1]==bind
    c.write_json(pub/'IMMUTABILITY_AUDIT.json',{'status':'PASS','old_checkpoints_banks_and_predictions_unchanged':True,
        'freeze_sha256':fs,'prediction_seal_sha256':c.sha(a.runtime/'OUTER_PREDICTIONS_SEAL.json'),
        'all_new_selections_before_inference':True,'all_predictions_before_full_y':True,'patients_per_arm':80,'channels_per_arm':7635})
    c.write_json(pub/'RUN_STATUS.json',{'status':'COMBINED_DELETION_OUTER_COMPLETE','scientific_status':lock['scientific_status'],
        'new_training_cells':5,'reused_cells':15,'no_post_result_tuning':True})
    print('COMBINED_DELETION_OUTER_COMPLETE',flush=True)


def verify(a,o):
    # Use the previously admitted analysis-only runtime, never for model inference.
    from sklearn.metrics import f1_score,recall_score,accuracy_score,roc_auc_score,average_precision_score
    lock,ref,c,core,model,trainer,helper,old,base,oldfreeze,mask,masked,factory,bind=o;pub=a.runtime/'public'
    assert json.loads((pub/'RUN_STATUS.json').read_text())['status']=='COMBINED_DELETION_OUTER_COMPLETE'
    freeze=json.loads((a.runtime/'FROZEN_BEFORE_OUTER.json').read_text());seal=json.loads((a.runtime/'OUTER_PREDICTIONS_SEAL.json').read_text())
    assert freeze['binding']==bind and seal['binding']==c.sha(a.runtime/'FROZEN_BEFORE_OUTER.json')
    with np.load(a.features,allow_pickle=True) as f:y,pid,ch=f['y'],f['patient'],f['channel']
    recorded=pd.DataFrame(torch.load(a.runtime/'PATIENT_METRICS_PRIVATE.pt',map_location='cpu',weights_only=False))
    rows=[];drift=0.;epochs=0
    for arm in [ARM]+REFS:
        for fold in range(1,6):
            if arm==ARM:
                root=a.runtime/f'fold{fold}';cp=root/'TRAIN/BEST_PRIVATE.pt';bp=root/'BANK_PRIVATE.pt';pp=root/'OUTER_PREDICTIONS_PRIVATE.pt';rec=freeze['selections'][fold-1]
                assert c.sha(cp)==rec['checkpoint_sha256'] and c.sha(bp)==rec['bank_sha256']
                d=torch.load(bp,map_location='cpu',weights_only=False);orig=torch.load(a.original/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
                assert np.all(d['x'][:,~mask]==0) and np.array_equal(d['x'][:,mask],orig['x'][:,mask])
                ck=torch.load(cp,map_location='cpu',weights_only=False);assert all(k.startswith('network.') for k in ck['model'])
                assert sum(v.numel() for v in ck['model'].values())==8817 and ck['binding']['initial_hash']==d['initial_hash']
                assert ck['binding']['seed']==42+1009*fold and ck['binding']['penalty']==0. and ck['binding']['device']=='cuda'
                assert ck['epoch']==rec['epoch'] and ck['threshold']==rec['nez_threshold']
                last=root/'TRAIN/LAST_PRIVATE.pt';done=json.loads((last.parent/'COMPLETE_PRIVATE.json').read_text())
                assert c.sha(last)==done['last_sha256'] and c.sha(cp)==done['best_sha256']
                epochs+=torch.load(last,map_location='cpu',weights_only=False)['epoch']
            else:pp=old.prediction_path(base,arm,fold)
            assert c.sha(pp)==seal['files'][arm+'_'+str(fold)]
            cell=torch.load(pp,map_location='cpu',weights_only=False);ix=cell['indices'];s=cell['scores'].astype(float)
            if arm==ARM:assert cell['threshold']==rec['nez_threshold']
            for pat in sorted(set(pid[ix])):
                k=np.flatnonzero(pid[ix]==pat);k=k[np.argsort(ch[ix][k].astype(str),kind='stable')]
                yy,ss=y[ix[k]],s[k];assert len(set(yy))==2;pred=(ss>=cell['threshold']).astype(int)
                r0=recall_score(yy,pred,pos_label=0);r1=recall_score(yy,pred,pos_label=1);rank=np.argsort(ss,kind='stable');ez=np.flatnonzero(yy[rank]==0)
                values={'macro_f1':f1_score(yy,pred,average='macro',labels=[0,1]),'ez_f1':f1_score(yy,pred,pos_label=0),
                    'nez_f1':f1_score(yy,pred,pos_label=1),'balanced_accuracy':(r0+r1)/2,'ez_auprc':average_precision_score(1-yy,1-ss),
                    'ez_auroc':roc_auc_score(1-yy,1-ss),'ez_mrr':1/(ez[0]+1),'top1_is_ez':float(yy[rank[0]]==0),
                    'sensitivity':r0,'specificity':r1,'accuracy':accuracy_score(yy,pred)}
                rr=recorded[(recorded.arm==arm)&(recorded.patient==pat)].iloc[0]
                for key,v in values.items():drift=max(drift,abs(v-rr[key]));assert abs(v-rr[key])<1e-12
                rows.append({'arm':arm,'patient':pat,**values})
    df=pd.DataFrame(rows);assert len(df)==320
    summary=pd.read_csv(pub/'OUTER_SUMMARY.csv');effects=pd.read_csv(pub/'PAIRED_BOOTSTRAP.csv');keys=c.METRICS
    new=df[df.arm==ARM].set_index('patient').sort_index()[keys].to_numpy();ix=np.random.default_rng(42).integers(0,80,size=(10000,80))
    for arm in [ARM]+REFS:
        values=df[df.arm==arm].set_index('patient').sort_index()[keys].to_numpy()
        rr=summary[summary.arm==arm].iloc[0]
        for j,key in enumerate(keys):assert abs(values[:,j].mean()-rr[key])<1e-12
        if arm==ARM:continue
        delta=new-values;draws=delta[ix].mean(1)
        for j,key in enumerate(keys):
            rr=effects[(effects.reference==arm)&(effects.metric==key)].iloc[0]
            assert abs(delta[:,j].mean()-rr.delta)<1e-12 and abs(np.percentile(draws[:,j],2.5)-rr.ci_low)<1e-12 and abs(np.percentile(draws[:,j],97.5)-rr.ci_high)<1e-12
    out=pub/'INDEPENDENT_VERIFICATION.json';assert not out.exists()
    import sklearn
    c.write_json(out,{'status':'PASS','model_patient_cells':320,'metrics_per_cell':11,'maximum_metric_drift':drift,
        'new_training_epochs':epochs,'paired_effects_checked':33,'bootstrap_draws':10000,
        'analysis_only_numpy':np.__version__,'analysis_only_sklearn':sklearn.__version__,'no_model_replay':True,'no_threshold_changes':True})
    print('COMBO_INDEPENDENT_VERIFICATION_PASS',drift,epochs,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','verify'])
    p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
    p.add_argument('--runtime',type=Path,default=Path(r'C:\d2_no_source_no_zdelta_outer_seed42_v1_runtime'))
    p.add_argument('--previous',type=Path,default=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'))
    p.add_argument('--original',type=Path,default=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'))
    p.add_argument('--full',type=Path,default=Path(r'C:\d2_ictal_outer_seed42_runtime'))
    p.add_argument('--noz',type=Path,default=Path(r'C:\d2_no_patient_z_outer_seed42_runtime'))
    p.add_argument('--features',type=Path,default=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'))
    p.add_argument('--protocol',type=Path,default=Path(__file__).parents[1]/'PROTOCOL_LOCK.json')
    a=p.parse_args();torch.set_num_threads(2);o=setup(a);start=time.monotonic()
    if a.phase=='prepare':prepare(a,o)
    elif a.phase=='run':train_freeze(a,o);evaluate(a,o)
    else:verify(a,o)
    print('ELAPSED_SECONDS',time.monotonic()-start,flush=True)


if __name__=='__main__':main()
