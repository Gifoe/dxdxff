"""B2-based single-view Development only; no main retrain and no outer access."""
import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from folding import Compact,factory,inflate
from views import configurations

METRICS=['ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr','top1_is_ez','sensitivity','specificity','nez_f1','balanced_accuracy','accuracy']
ALL=METRICS+['predicted_ez_fraction','TP','FP','TN','FN']


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def patient_rows(c,b,score,tau,arm,fold):
    va=b['val'];rows=[]
    for pid in sorted(set(b['patient'][va])):
        k=np.flatnonzero(b['patient'][va]==pid);k=k[np.argsort(b['channel'][va][k].astype(str),kind='stable')]
        rows.append(dict(arm=arm,fold=fold,patient=pid,center=str(b['center'][va[k[0]]]),**c.metric(b['y'][va[k]],score[k],tau)))
    return rows


def setup():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));p.add_argument('--prior',type=Path,default=Path(r'D:\a0_stable_feature_pruning_seed42_v1_runtime'));a=p.parse_args()
    a.root=Path(__file__).parents[1];a.original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime');a.previous=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime');a.features=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    dep=load('singleview_dependencies',a.source/'a0_cove_seed42_v1/code/dependencies.py');ref,c,core,orig,trainer,helper,banks,authoritative=dep.dependencies(a)
    lock=json.loads((a.root/'PROTOCOL_LOCK.json').read_text());assert lock['no_new_outer'] and lock['main_is_not_a_newly_proven_winner']
    for name,digest in lock['prior_private_hashes'].items():assert c.sha(a.prior/name)==digest
    prior=json.loads((a.prior/'INPUT_FREEZE_PRIVATE.json').read_text());assert prior['binding']['authoritative']==authoritative
    oldroot=a.source/'a0_stable_feature_pruning_seed42_v1'
    for name,digest in prior['binding']['new'].items():assert c.sha(oldroot/name)==digest
    assert c.sha(a.source/'a0_cove_seed42_v1/code/dependencies.py')==prior['binding']['dependency_loader']
    assert c.sha(a.root/'code/folding.py')==prior['binding']['new']['code\\folding.py']
    assert c.sha(a.root/'AUTHORITATIVE_FEATURE_MANIFEST.json')==lock['manifest_sha256']==c.sha(a.previous/'public/FEATURE_GROUP_MANIFEST.json')
    assert json.loads((a.prior/'public/INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
    z=torch.load(a.prior/'PREPARED_PRIVATE.pt',weights_only=False);assert z['binding']==prior['binding']
    configs=configurations(json.loads((a.root/'AUTHORITATIVE_FEATURE_MANIFEST.json').read_text())['columns'])
    for b,d in zip(z['banks'],banks):
        for k in ['patient','channel','center','y','g','train','val','features']:assert np.array_equal(b[k],d[k])
        for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(b['pre'][k],d['pre'][k])
        assert np.array_equal(b['x'],helper.masked(d['x'],np.array([q['view']!='zdelta' for q in json.loads((a.root/'AUTHORITATIVE_FEATURE_MANIFEST.json').read_text())['columns']])))
    bound=dict(authoritative=authoritative,prior=lock['prior_private_hashes'],prior_executed_sources=prior['binding']['new'],new={str(p.relative_to(a.root)):c.sha(p) for p in a.root.rglob('*') if p.is_file() and (p.suffix=='.py' or p.name in ['PROTOCOL_LOCK.json','AUTHORITATIVE_FEATURE_MANIFEST.json'])},auxiliary_code={n:c.sha(a.source/'a0_cove_seed42_v1/code'/n) for n in ['dependencies.py','metrics.py','bootstrap.py']})
    fr=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if fr.exists():assert json.loads(fr.read_text())['binding']==bound
    else:c.write_json(fr,dict(status='PASS',binding=bound,no_new_outer=True))
    pub=a.runtime/'public';c.write_json(pub/'DEPENDENCY_HASHES.json',dict(status='PASS',binding=bound));torch.set_num_threads(2)
    return SimpleNamespace(a=a,c=c,core=core,orig=orig,trainer=trainer,banks=z['banks'],configs=configs,lock=lock,binding=bound,pub=pub)


def admit(o):
    a,c,core=o.a,o.c,o.core;out=a.runtime/'BASELINE_PRIVATE.pt'
    if out.exists():
        q=torch.load(out,weights_only=False);assert q['binding']==o.binding and c.sha(out)==json.loads(out.with_suffix('.sha256.json').read_text())['sha256'];return q['rows']
    oldfreeze=json.loads((a.prior/'DEVELOPMENT_FREEZE_PRIVATE.json').read_text());seal=json.loads((a.prior/'PREDICTION_SEAL_PRIVATE.json').read_text());assert oldfreeze['binding']==seal.get('binding',oldfreeze['binding']);fs=c.sha(a.prior/'DEVELOPMENT_FREEZE_PRIVATE.json');assert seal['freeze']==fs
    tests=load('pruned_math_test',a.root/'tests/test_folding.py');tests.test_fold();tests=load('pruned_views_test',a.root/'tests/test_views.py');tests.test_actual_manifest_views()
    rows=[];audits=[];manifest=[];selections=[]
    for arm,rec in o.configs.items():
        for q in rec:manifest.append(dict(arm=arm,index=q['index'],name=q['name'],feature=q['physiological_feature'],view=q['view'],aggregation=q['aggregation'],multiplicity=q['multiplicity']))
    for fold,b in enumerate(o.banks,1):
        record=next(q for q in oldfreeze['selections'] if q['arm']=='B2' and q['fold']==fold);cp=Path(record['checkpoint']);assert c.sha(cp)==record['checkpoint_sha256'];vp=a.prior/f'final/fold{fold}/B2_VAL_PRIVATE.pt';assert c.sha(vp)==seal['files'][f'B2_{fold}'];cell=torch.load(vp,weights_only=False)
        assert cell['freeze']==fs and cell['checkpoint_sha256']==c.sha(cp) and np.array_equal(cell['indices'],b['val']);tau=record['threshold'];assert cell['threshold']==tau
        ck=torch.load(cp,map_location='cpu',weights_only=False);seed=42+1009*fold;core.seed_all(seed);m=factory(o.orig.SourceA0,o.configs['MAIN_B2'])(seed).eval();assert core.state_hash(m.state_dict())==ck['binding']['initial_hash'];m.load_state_dict(ck['model'],strict=True)
        va=b['val'];tx=torch.as_tensor(b['x'][va]);m=m.cuda()
        with torch.no_grad():s=torch.sigmoid(m(tx.cuda())).cpu().numpy()
        drift=float(np.max(np.abs(s-cell['scores'])));assert drift<=2e-7 and np.array_equal(s>=tau,cell['scores']>=tau)
        frozen=patient_rows(c,b,cell['scores'],tau,'MAIN_B2',fold);fresh=patient_rows(c,b,s,tau,'MAIN_B2',fold)
        for r,q in zip(frozen,fresh):
            for key in ALL:assert abs(r[key]-q[key])<1e-12
        rows.extend(frozen);selections.append(dict(arm='MAIN_B2',fold=fold,epoch=ck['epoch'],threshold=tau,coordinates=36,parameters=3649,checkpoint_sha256=c.sha(cp),prediction_sha256=c.sha(vp),historical_reuse=True))
        # Untrained subset identity on actual legal FIT/VAL; no labels used.
        for arm,rec in o.configs.items():
            core.seed_all(seed);net=o.orig.SourceA0(seed).network.eval();model=Compact(net,rec).eval();xx=torch.as_tensor(b['x'])[:,[q['index'] for q in rec]];expanded=inflate(xx,rec)
            with torch.no_grad():fd=float((model(xx)-net(expanded).squeeze(-1)).abs().max());assert fd<=3e-6;nd=copy.deepcopy(net).double();md=Compact(nd,rec).double().eval();dd=float((md(xx.double())-nd(inflate(xx.double(),rec)).squeeze(-1)).abs().max());assert dd<=1e-10
            other=torch.as_tensor(b['x']).clone();outside=[i for i in range(88) if i not in [q['index'] for q in rec]];other[:,outside]=torch.randn_like(other[:,outside])*1000
            assert torch.equal(model(torch.as_tensor(b['x'])),model(other))
            audits.append(dict(fold=fold,arm=arm,coordinates=len(rec),parameters=96*len(rec)+193,virtual_nonzero_positions=sum(q['multiplicity'] for q in rec),float64_max_logit=dd,float32_max_logit=fd,excluded_input_invariance=True,main_replay_probability_drift=drift if arm=='MAIN_B2' else None))
        del m
    expected=[.719569975390029,.5173307071740828,.6427219679229726,.43763109261853367,.6912190315700592]
    assert np.allclose(pd.DataFrame(rows)[METRICS[:5]].mean(),expected,atol=1e-12,rtol=0)
    c.write_json(o.pub/'MAIN_REUSE_AUDIT.json',dict(status='PASS',selections=selections,metrics=dict(zip(METRICS[:5],expected)),binding=o.binding,no_main_retraining=True,no_new_outer=True))
    pd.DataFrame(audits).to_csv(o.pub/'IDENTITY_AND_INPUT_ISOLATION.csv',index=False);pd.DataFrame(manifest).to_csv(o.pub/'FEATURE_RETENTION_MANIFEST.csv',index=False)
    core.torch_write(out,dict(binding=o.binding,rows=rows));c.write_json(out.with_suffix('.sha256.json'),dict(sha256=c.sha(out)))
    return rows


def train(o):
    a,c,core=o.a,o.c,o.core;fr=a.runtime/'MODEL_FREEZE_PRIVATE.json'
    if fr.exists():
        q=json.loads(fr.read_text());assert q['binding']==o.binding;return q
    records=[];history=[];resources=[]
    for fold,b in enumerate(o.banks,1):
        for arm in ['ABS_ONLY','DELTA_ONLY','RATIO_ONLY']:
            rec=o.configs[arm];cell=a.runtime/f'fold{fold}/{arm}';was_complete=(cell/'COMPLETE_PRIVATE.json').exists();seed=42+1009*fold;core.seed_all(seed);initial=core.state_hash(factory(o.orig.SourceA0,rec)(seed).state_dict())
            old=o.trainer.SourceA0;o.trainer.SourceA0=factory(o.orig.SourceA0,rec);t=time.perf_counter()
            try:model,best,h=o.trainer.train(b['x'],b['y'],b['patient'],b['g'],b['train'],b['val'],cell,seed,c.digest(dict(binding=o.binding,arm=arm,fold=fold)),device='cuda',penalty=0)
            finally:o.trainer.SourceA0=old
            seconds=time.perf_counter()-t;cp=cell/'BEST_PRIVATE.pt';ck=torch.load(cp,map_location='cpu',weights_only=False);done=json.loads((cell/'COMPLETE_PRIVATE.json').read_text());assert c.sha(cp)==done['best_sha256'] and c.sha(cell/'LAST_PRIVATE.pt')==done['last_sha256'] and ck['binding']['initial_hash']==initial
            tx=torch.as_tensor(b['x'][b['val']],device='cuda');changed=tx.clone();outside=[i for i in range(88) if i not in [q['index'] for q in rec]];changed[:,outside]=torch.randn_like(changed[:,outside])*1000
            with torch.no_grad():assert torch.equal(model(tx),model(changed))
            r=dict(arm=arm,fold=fold,coordinates=len(rec),parameters=96*len(rec)+193,checkpoint=str(cp),checkpoint_sha256=c.sha(cp),epoch=best['epoch'],threshold=best['threshold'],initial_hash=initial)
            records.append(r);history.extend(dict(arm=arm,fold=fold,**q) for q in h);resources.append(dict(arm=arm,fold=fold,epochs=len(h),seconds=seconds,completed_task_reused=was_complete))
            c.write_json(o.pub/'RUN_STATUS.json',dict(status='TRAIN_VALIDATION',cells_complete=len(records),total=15,no_new_outer=True));print('VIEW_SELECTED',fold,arm,len(rec),best['epoch'],flush=True);del model
    q=dict(status='PASS',binding=o.binding,selections=records,all15_frozen_before_comparative_summary=True,no_new_outer=True);c.write_json(fr,q)
    pd.DataFrame([{k:v for k,v in q.items() if k!='checkpoint'} for q in records]).to_csv(o.pub/'MODEL_SELECTION.csv',index=False);pd.DataFrame(history).to_csv(o.pub/'TRAINING_HISTORY.csv',index=False);pd.DataFrame(resources).to_csv(o.pub/'RESOURCE_AUDIT.csv',index=False)
    return q


def evaluate(o,baseline,frozen):
    a,c,core=o.a,o.c,o.core;fs=c.sha(a.runtime/'MODEL_FREEZE_PRIVATE.json');rows=list(baseline);files={};ind=load('pruned_independent_metrics',a.source/'a0_cove_seed42_v1/code/metrics.py');checked=0
    for fold,b in enumerate(o.banks,1):
        va=b['val']
        for arm in ['ABS_ONLY','DELTA_ONLY','RATIO_ONLY']:
            r=next(q for q in frozen['selections'] if q['fold']==fold and q['arm']==arm);cp=Path(r['checkpoint']);assert c.sha(cp)==r['checkpoint_sha256'];out=a.runtime/f'fold{fold}/{arm}/VAL_PRIVATE.pt';tau=r['threshold']
            if out.exists():
                cell=torch.load(out,weights_only=False);assert cell['freeze']==fs and c.sha(out)==json.loads(out.with_suffix('.sha256.json').read_text())['sha256'];score=cell['scores']
            else:
                m=factory(o.orig.SourceA0,o.configs[arm])().cuda().eval();ck=torch.load(cp,map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True)
                with torch.no_grad():score=torch.sigmoid(m(torch.as_tensor(b['x'][va],device='cuda'))).cpu().numpy()
                selected=core.select_threshold(b['y'][va],score,b['patient'][va]);assert selected['threshold']==tau and selected['macro_f1']==ck['macro_f1'] and selected['ez_f1']==ck['ez_f1']
                cell=dict(freeze=fs,scores=score,indices=va,threshold=tau,checkpoint_sha256=c.sha(cp));core.torch_write(out,cell);c.write_json(out.with_suffix('.sha256.json'),dict(sha256=c.sha(out)));del m
            new=patient_rows(c,b,score,tau,arm,fold);rows.extend(new);files[f'{arm}_{fold}']=c.sha(out)
            for pid in sorted(set(b['patient'][va])):
                k=np.flatnonzero(b['patient'][va]==pid);k=k[np.argsort(b['channel'][va][k].astype(str),kind='stable')];v=ind.independent(b['y'][va[k]],score[k],tau);stored=next(q for q in new if q['patient']==pid)
                for key in ALL:assert abs(v[key]-stored[key])<1e-12
                checked+=1
    df=pd.DataFrame(rows);assert len(df)==260 and df.patient.nunique()==47
    # Also independently check baseline frozen scores, without new model inference.
    for fold,b in enumerate(o.banks,1):
        cell=torch.load(a.prior/f'final/fold{fold}/B2_VAL_PRIVATE.pt',weights_only=False);score=cell['scores'];va=b['val']
        for pid in sorted(set(b['patient'][va])):
            k=np.flatnonzero(b['patient'][va]==pid);k=k[np.argsort(b['channel'][va][k].astype(str),kind='stable')];v=ind.independent(b['y'][va[k]],score[k],cell['threshold']);stored=df[(df.arm=='MAIN_B2')&(df.fold==fold)&(df.patient==pid)].iloc[0]
            for key in ALL:assert abs(v[key]-stored[key])<1e-12
            checked+=1
    summary=df.groupby('arm')[ALL].mean().reset_index();summary['coordinates']=summary.arm.map(o.lock['coordinates']);summary['parameters']=summary.arm.map(o.lock['parameters']);summary.to_csv(o.pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    df.groupby(['arm','fold'])[ALL].mean().reset_index().to_csv(o.pub/'DEVELOPMENT_BY_FOLD.csv',index=False)
    df.groupby(['arm','patient'])[ALL].mean().groupby('arm').mean().reset_index().to_csv(o.pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
    center=[dict(arm=arm,center=center,appearances=len(q),unique_patients=q.patient.nunique(),**q[ALL].mean().to_dict()) for (arm,center),q in df.groupby(['arm','center'])];pd.DataFrame(center).to_csv(o.pub/'DEVELOPMENT_BY_CENTER.csv',index=False)
    bootmod=load('pruned_bootstrap',a.source/'a0_cove_seed42_v1/code/bootstrap.py');boot=[]
    for left,right in o.lock['bootstrap']['comparisons']:
        l=df[df.arm==left];r=df[df.arm==right];one=bootmod.compare(c,l,r,METRICS);two=bootmod.independent(l,r,METRICS)
        for q in one:assert np.allclose([q['delta'],q['ci_low'],q['ci_high']],two[q['metric']],atol=1e-12,rtol=0)
        boot.extend(dict(comparison=left+' minus '+right,**q) for q in one)
    pd.DataFrame(boot).to_csv(o.pub/'PAIRED_BOOTSTRAP.csv',index=False);core.torch_write(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pt',rows);c.write_json(a.runtime/'PREDICTION_SEAL_PRIVATE.json',dict(freeze=fs,files=files))
    c.write_json(o.pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',patient_metric_cells=checked,paired_bootstrap_metrics=len(boot),binding=o.binding,no_outer_access=True,all15_isolated_inputs=True))
    c.write_json(o.pub/'RUN_STATUS.json',dict(status='COMPLETE',terminal='B2_BASED_SINGLEVIEW_DEVELOPMENT_COMPLETE_EXPLORATORY',new_cells=15,main_models_reused=5,validation_appearances=65,unique_ids=47,no_new_outer=True))
    print(summary[['arm']+METRICS[:8]].to_string(index=False),flush=True);print('B2_BASED_SINGLEVIEW_DEVELOPMENT_COMPLETE_EXPLORATORY',flush=True)


def main():
    o=setup();baseline=admit(o);frozen=train(o);evaluate(o,baseline,frozen)


if __name__=='__main__':main()
