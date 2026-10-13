"""All feasible stages on legal FIT/VAL banks; never loads outer features/scores.

The unchanged authoritative trainer is imported and constructor-swapped. Nested
screening preprocessors fit only inner training patients. All subset choices are
frozen before any comparative new Development outcome is computed.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np  # before Torch: verified server OpenMP-compatible order
import pandas as pd
import torch
from folding import mapping,factory,retained,Compact,inflate

METRICS=['ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr','top1_is_ez','sensitivity','specificity','nez_f1','balanced_accuracy','accuracy']


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def setup():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));a=p.parse_args()
    a.root=Path(__file__).parents[1];a.original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime');a.previous=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime');a.features=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    dep=load('prior_dependencies',a.source/'a0_cove_seed42_v1/code/dependencies.py')
    ref,c,core,orig,trainer,helper,banks,bind=dep.dependencies(a)
    lock=json.loads((a.root/'PROTOCOL_LOCK.json').read_text());assert lock['no_new_outer']
    paths={'manifest':a.previous/'public/FEATURE_GROUP_MANIFEST.json','F0_registry':Path(r'C:\d2_no_source_no_zdelta_outer_seed42_v1_runtime/FROZEN_BEFORE_OUTER.json'),
           'validation_registry':Path(r'D:\a0_compact_delta_ratio_only_seed42_v1_runtime/EXPERIMENT_FREEZE.json'),
           'validation_verification':Path(r'D:\a0_compact_delta_ratio_only_seed42_v1_runtime/public/INDEPENDENT_VERIFICATION.json')}
    for k,path in paths.items():assert c.sha(path)==lock[k+'_sha256']
    assert json.loads(paths['validation_verification'].read_text())['status']=='PASS'
    manifest=json.loads(paths['manifest'].read_text());mp=mapping(manifest['columns'])
    bind=dict(authoritative=bind,new={str(p.relative_to(a.root)):c.sha(p) for p in sorted(a.root.rglob('*')) if p.is_file() and (p.suffix=='.py' or p.name=='PROTOCOL_LOCK.json')},provenance={k:c.sha(v) for k,v in paths.items()},dependency_loader=c.sha(a.source/'a0_cove_seed42_v1/code/dependencies.py'))
    pub=a.runtime/'public';pub.mkdir(exist_ok=True)
    fr=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if fr.exists():assert json.loads(fr.read_text())['binding']==bind
    else:c.write_json(fr,dict(status='PASS',binding=bind,no_outer_access=True))
    torch.set_num_threads(2)
    return SimpleNamespace(a=a,c=c,core=core,orig=orig,trainer=trainer,helper=helper,banks=banks,lock=lock,mp=mp,bind=bind,pub=pub,paths=paths)


def patient_rows(o,d,ix,score,tau,arm,fold):
    rows=[]
    for pid in sorted(set(d['patient'][ix])):
        k=np.flatnonzero(d['patient'][ix]==pid);k=k[np.argsort(d['channel'][ix][k].astype(str),kind='stable')]
        rows.append(dict(arm=arm,fold=fold,patient=pid,**o.c.metric(d['y'][ix[k]],score[k],tau)))
    return rows


def original_mask(o,x):
    return o.helper.masked(x,np.array([q['view']!='zdelta' for q in json.loads(o.paths['manifest'].read_text())['columns']]))


def status(o,**q):o.c.write_json(o.pub/'RUN_STATUS.json',q)


def prepare(o):
    a,c,core=o.a,o.c,o.core;target=a.runtime/'PREPARED_PRIVATE.pt'
    if target.exists():
        z=torch.load(target,weights_only=False);assert z['binding']==o.bind;assert c.sha(target)==json.loads(target.with_suffix('.sha256.json').read_text())['sha256'];return z
    baseline=[];parity=[];bb=[];f0reg=json.loads(o.paths['F0_registry'].read_text())['selections'];vreg=json.loads(o.paths['validation_registry'].read_text())['historical_files'];diagnostics=[];corr=[]
    test=load('numeric_folding_tests',a.root/'tests/test_folding.py');test.test_fold()
    for fold,d in enumerate(o.banks,1):
        r=f0reg[fold-1];root=Path(r'C:\d2_no_source_no_zdelta_outer_seed42_v1_runtime')/f'fold{fold}'
        bp=root/'BANK_PRIVATE.pt';cp=root/'TRAIN/BEST_PRIVATE.pt';vp=Path(r'D:\a0_compact_delta_ratio_only_seed42_v1_runtime')/f'scores/F0_fold{fold}_VAL_PRIVATE.pt'
        assert c.sha(bp)==r['bank_sha256'] and c.sha(cp)==r['checkpoint_sha256'] and c.sha(vp)==vreg[f'F0_validation{fold}']
        b=torch.load(bp,weights_only=False);ck=torch.load(cp,map_location='cpu',weights_only=False);cell=torch.load(vp,weights_only=False);tau=r['nez_threshold'];va=d['val']
        assert np.array_equal(b['x'],original_mask(o,d['x']))
        for k in ['patient','channel','y','g','train','val','features']:assert np.array_equal(b[k],d[k])
        assert np.array_equal(cell['indices'],va) and tau==ck['threshold']==cell['threshold']
        for q in o.mp['unique']:
            for i in q['copies']:assert np.array_equal(b['x'][:,i],b['x'][:,q['index']])
        assert np.all(b['x'][:,o.mp['zero']+o.mp['masked']]==0)
        seed=42+1009*fold;core.seed_all(seed);orig=o.helper.source_free_factory(o.orig.SourceA0)(seed).eval();orig.load_state_dict(ck['model'],strict=True)
        compact=Compact(orig.network,o.mp['unique']).eval();tx=torch.as_tensor(b['x'])
        with torch.no_grad():
            aa=orig(tx,torch.tensor(b['g']));vv=compact(tx);drift=float((aa-vv).abs().max());assert drift<=o.lock['parity']['float32_logit_atol']
            score=torch.sigmoid(vv[va]).numpy();score_drift=float(np.max(np.abs(score-cell['scores'])));assert score_drift<=o.lock['parity']['frozen_probability_atol'];assert np.array_equal(score>=tau,cell['scores']>=tau)
            od=copy.deepcopy(orig).double();cd=Compact(od.network,o.mp['unique']).double().eval();dd=float((od(tx.double(),torch.tensor(b['g']))-cd(tx.double())).abs().max());assert dd<=o.lock['parity']['float64_logit_atol']
            cg=copy.deepcopy(compact).cuda();gpu=torch.sigmoid(cg(tx.cuda())[va]).cpu().numpy();assert np.max(np.abs(gpu-cell['scores']))<=o.lock['parity']['frozen_probability_atol'];assert np.array_equal(gpu>=tau,cell['scores']>=tau)
        before=patient_rows(o,b,va,cell['scores'],tau,'B0',fold);after=patient_rows(o,b,va,score,tau,'B0',fold)
        for u,v in zip(before,after):
            for k in METRICS:assert abs(u[k]-v[k])<1e-12
        sel=core.select_threshold(b['y'][va],score,b['patient'][va]);assert sel['threshold']==tau and sel['macro_f1']==ck['macro_f1'] and sel['ez_f1']==ck['ez_f1']
        baseline.extend(before);parity.append(dict(fold=fold,float64_max_logit=dd,float32_max_logit=drift,frozen_probability_max=score_drift,gpu_probability_max=float(np.max(np.abs(gpu-cell['scores']))),threshold_decisions_identical=True,metrics_exact=True,parameters=4225))
        tr=d['train'];raw=d['raw_x'];pp=d['patient'];fitx=b['x'][tr][:,[q['index'] for q in o.mp['unique']]].astype(float);w=np.array([1/np.sum(pp[tr]==p) for p in pp[tr]])
        mu=(fitx*w[:,None]).sum(0)/w.sum();cov=((fitx-mu)*w[:,None]).T@(fitx-mu)/w.sum();sd=np.sqrt(np.diag(cov));den=sd[:,None]*sd[None,:];cc=np.divide(cov,den,out=np.zeros_like(cov),where=den>0)
        for i in range(42):
            for j in range(i+1,42):corr.append(dict(fold=fold,left=o.mp['unique'][i]['name'],right=o.mp['unique'][j]['name'],FIT_patient_equal_correlation=float(cc[i,j]),selection_use=False))
        for q in o.mp['unique']:
            vals=[];floors=[]
            for pid in sorted(set(pp[tr])):
                xx=raw[pp==pid,q['index']];s=float(np.nanstd(xx));vals.append(s);floors.append(s<=1e-8)
            diagnostics.append(dict(fold=fold,name=q['name'],feature=q['physiological_feature'],view=q['view'],aggregation=q['aggregation'],original_patientZ_floor_fraction=float(np.mean(floors)),mean_raw_patient_sd=float(np.mean(vals)),predefined_candidate=q['view']=='delta' and q['physiological_feature'] in ['log_bp_beta','log_bp_low_gamma','log_bp_high_gamma'],selection_use=False))
        # Bind all screening assignments before held-out performance is computed.
        ids=np.array(sorted(set(pp[tr])));parts=np.array_split(np.random.default_rng(seed).permutation(ids),3);plans=[]
        for h,screen in enumerate(parts):
            remaining=np.array(sorted(set(ids)-set(screen)));order=np.random.default_rng(seed+10000+h).permutation(remaining);n=int(np.ceil(.2*len(order)));select=order[:n];train=order[n:]
            assert len(train)>0 and len(select)>0 and set(train).isdisjoint(screen) and set(select).isdisjoint(screen)
            plans.append(dict(screen=screen,select=select,train=train,seed=seed+10000+h))
        bb.append(dict(b,raw_x=d['raw_x'],plans=plans))
        print('FOLDING_PARITY_PASS',fold,flush=True)
    expected=[.7165694393593033,.5085590459690381,.6420669456105904,.43351598043908673,.6765905768221185]
    assert np.allclose(pd.DataFrame(baseline)[METRICS[:5]].mean().to_numpy(),expected,atol=1e-12,rtol=0)
    c.write_json(o.pub/'FOLDING_PARITY.json',dict(status='PASS',rows=parity,synthetic_PASS=True,binding=o.bind,no_outer_access=True))
    pd.DataFrame(diagnostics).to_csv(o.pub/'NUMERICAL_RELIABILITY_AUDIT.csv',index=False);pd.DataFrame(corr).to_csv(o.pub/'FIT_CORRELATION_AUDIT.csv',index=False)
    c.write_json(o.pub/'DEPENDENCY_HASHES.json',dict(status='PASS',binding=o.bind))
    z=dict(binding=o.bind,banks=bb,baseline=baseline,mapping=o.mp);core.torch_write(target,z);c.write_json(target.with_suffix('.sha256.json'),dict(sha256=c.sha(target)));return z


def train_cell(o,d,train,select,records,cell,seed,binding):
    old=o.trainer.SourceA0;o.trainer.SourceA0=factory(o.orig.SourceA0,records)
    try:
        t=time.perf_counter();m,best,h=o.trainer.train(d['x'],d['y'],d['patient'],d['g'],train,select,cell,seed,o.c.digest(dict(binding=o.bind,cell=binding,records=[q['index'] for q in records])),device='cuda',penalty=0)
    finally:o.trainer.SourceA0=old
    return m,best,h,time.perf_counter()-t


def screen(o,z):
    a,c,core=o.a,o.c,o.core;frozen=a.runtime/'FEATURE_SELECTION_PRIVATE.json'
    if frozen.exists():
        f=json.loads(frozen.read_text());assert f['binding']==o.bind and f['status']=='PASS';return f
    selections=[];importance=[];screenrows=[];resources=[]
    for fold,b in enumerate(z['banks'],1):
        tr=b['train'];d={k:b[k][tr] for k in ['y','patient','channel','g']};raw=b['raw_x'][tr]
        for h,plan in enumerate(b['plans']):
            ti=np.flatnonzero(np.isin(d['patient'],plan['train']));vi=np.flatnonzero(np.isin(d['patient'],plan['select']));qi=np.flatnonzero(np.isin(d['patient'],plan['screen']))
            assert set(np.r_[ti,vi,qi])==set(range(len(raw)))
            xx,pre=core.prepare(raw,d['patient'],ti);d['x']=original_mask(o,xx)
            assert set(pre['fit_patient_ids'])==set(plan['train'])
            full=None
            for group in [-1]+list(range(21)):
                cell=a.runtime/f'screen/fold{fold}/inner{h}/group{group}';rec=retained(o.mp,[] if group==-1 else [group])
                m,best,history,seconds=train_cell(o,d,ti,vi,rec,cell,plan['seed'],dict(fold=fold,inner=h,group=group,preprocessor=c.digest(pre),train_ids_hash=c.digest(plan['train'].tolist()),screen_ids_hash=c.digest(plan['screen'].tolist())))
                with torch.inference_mode():score=torch.sigmoid(m(torch.as_tensor(d['x'][qi],device='cuda'))).cpu().numpy()
                rows=patient_rows(o,d,qi,score,best['threshold'],str(group),fold);vals=pd.DataFrame(rows)[METRICS].mean().to_dict();screenrows.extend(dict(inner=h,group=group,**r) for r in rows)
                core.torch_write(cell/'SCREEN_PRIVATE.pt',dict(binding=o.bind,indices=qi,scores=score,threshold=best['threshold'],rows=rows,preprocessor=pre,checkpoint_sha256=c.sha(cell/'BEST_PRIVATE.pt')))
                resources.append(dict(phase='SCREEN',fold=fold,inner=h,group=group,epochs=len(history),seconds=seconds,parameters=96*len(rec)+193,reused_complete_cell=seconds<.05))
                if group==-1:full=vals
                else:importance.append(dict(fold=fold,inner=h,group=group,feature=o.mp['groups'][group]['feature'],view=o.mp['groups'][group]['view'],training_patients=len(plan['train']),selection_patients=len(plan['select']),screening_patients=len(plan['screen']),selected_epoch=best['epoch'],threshold=best['threshold'],**vals,**{'delta_'+k:vals[k]-full[k] for k in METRICS}))
                del m
                status(o,status='FIT_SCREENING',cells_complete=len(resources),total=330,last_fold=fold,last_inner=h,last_group=group,no_new_outer=True)
            print('SCREEN_COMPLETE',fold,h,len(resources),'of330',flush=True)
        candidates=[]
        for group in range(21):
            q=pd.DataFrame([r for r in importance if r['fold']==fold and r['group']==group]);means=q[['delta_ez_auroc','delta_ez_auprc','delta_macro_f1']].mean();positive=int((q.delta_ez_auroc>0).sum())
            eligible=means.delta_ez_auroc>=.002 and means.delta_ez_auprc>=0 and means.delta_macro_f1>=-.002 and positive>=2
            if eligible:candidates.append((group,float(means.delta_ez_auroc),float(means.delta_ez_auprc)))
        selected=[g for g,auc,ap in sorted(candidates,key=lambda q:(-q[1],-q[2],q[0]))[:3]]
        random=sorted(np.random.default_rng(42+1009*fold+20000).choice(21,size=len(selected),replace=False).tolist())
        selections.append(dict(fold=fold,selected_groups=selected,random_groups=random,eligible_groups=[q[0] for q in candidates]))
        print('FIT_SELECTION_FROZEN',fold,'selected_count',len(selected),flush=True)
    assert len(resources)==330 and len(importance)==315
    pd.DataFrame(importance).to_csv(o.pub/'FEATURE_IMPORTANCE_BY_INNER_FOLD.csv',index=False)
    pd.DataFrame(resources).to_csv(o.pub/'SCREEN_RESOURCE_AUDIT.csv',index=False)
    core.torch_write(a.runtime/'SCREEN_METRICS_PRIVATE.pt',screenrows)
    f=dict(status='PASS',binding=o.bind,selections=selections,no_Development_labels_used=True,all_selections_before_new_Development=True)
    c.write_json(frozen,f)
    public=[]
    for r in selections:
        for group in o.mp['groups']:
            q=pd.DataFrame([v for v in importance if v['fold']==r['fold'] and v['group']==group['group']])
            public.append(dict(fold=r['fold'],group=group['group'],feature=group['feature'],view=group['view'],eligible=group['group'] in r['eligible_groups'],selected=group['group'] in r['selected_groups'],random_selected=group['group'] in r['random_groups'],positive_AUROC_screens=int((q.delta_ez_auroc>0).sum()),**q[['delta_ez_auroc','delta_ez_auprc','delta_macro_f1']].mean().to_dict()))
    pd.DataFrame(public).to_csv(o.pub/'FEATURE_SELECTION_BY_OUTER_FOLD.csv',index=False)
    pd.DataFrame(public).groupby(['group','feature','view'])[['selected','eligible','random_selected']].sum().reset_index().to_csv(o.pub/'FEATURE_SELECTION_FREQUENCY.csv',index=False)
    return f


def final_train(o,z,f):
    a,c=o.a,o.c;freeze=a.runtime/'DEVELOPMENT_FREEZE_PRIVATE.json'
    if freeze.exists():
        r=json.loads(freeze.read_text());assert r['binding']==o.bind and r['selection_sha256']==c.sha(a.runtime/'FEATURE_SELECTION_PRIVATE.json');return r
    selections=[];resources=[];history=[];man=[];counts=[]
    predef=[q['group'] for q in o.mp['groups'] if q['view']=='delta' and q['feature'] in ['log_bp_beta','log_bp_low_gamma','log_bp_high_gamma']];assert len(predef)==3
    for fold,b in enumerate(z['banks'],1):
        s=next(r for r in f['selections'] if r['fold']==fold)
        configs={'B1':[],'B2':predef,'B3':s['selected_groups'],'B4':s['random_groups']}
        for arm,deleted in configs.items():
            rec=retained(o.mp,deleted);alias='B1' if arm in ['B3','B4'] and not deleted else None
            if alias:
                r=next(r for r in selections if r['fold']==fold and r['arm']==alias);selections.append(dict(r,arm=arm,alias=alias));continue
            cp=a.runtime/f'final/fold{fold}/{arm}';seed=42+1009*fold
            m,best,h,sec=train_cell(o,b,b['train'],b['val'],rec,cp,seed,dict(fold=fold,arm=arm,selection_sha256=c.sha(a.runtime/'FEATURE_SELECTION_PRIVATE.json')))
            # Prove deleted and duplicate auxiliary values cannot enter inference.
            xx=torch.as_tensor(b['x'][b['val']],device='cuda');other=xx.clone();ix=[i for i in range(88) if i not in [q['index'] for q in rec]];other[:,ix]=torch.randn_like(other[:,ix])
            with torch.inference_mode():assert torch.equal(m(xx),m(other))
            selections.append(dict(arm=arm,fold=fold,alias=None,deleted=deleted,coordinates=len(rec),parameters=96*len(rec)+193,epoch=best['epoch'],threshold=best['threshold'],checkpoint=str(cp/'BEST_PRIVATE.pt'),checkpoint_sha256=c.sha(cp/'BEST_PRIVATE.pt')))
            resources.append(dict(phase='FINAL',arm=arm,fold=fold,epochs=len(h),seconds=sec,parameters=96*len(rec)+193));history.extend(dict(arm=arm,fold=fold,**q) for q in h)
            del m;print('FINAL_CELL_SELECTED',fold,arm,'coordinates',len(rec),flush=True)
        counts.append(dict(arm='B0',fold=fold,coordinates=88,unique_nonconstant=42,parameters=8817,alias='historical frozen F0'))
        for arm in configs:
            r=next(q for q in selections if q['arm']==arm and q['fold']==fold)
            counts.append(dict(arm=arm,fold=fold,coordinates=r['coordinates'],unique_nonconstant=r['coordinates'],parameters=r['parameters'],alias=r['alias']))
            for q in o.mp['unique']:man.append(dict(fold=fold,arm=arm,original_index=q['index'],name=q['name'],multiplicity=q['multiplicity'],retained=q['index'] in [v['index'] for v in retained(o.mp,configs[arm])]))
    r=dict(status='PASS',binding=o.bind,selection_sha256=c.sha(a.runtime/'FEATURE_SELECTION_PRIVATE.json'),selections=selections,no_new_outer=True)
    c.write_json(freeze,r);pd.DataFrame(resources).to_csv(o.pub/'FINAL_RESOURCE_AUDIT.csv',index=False);pd.DataFrame(history).to_csv(o.pub/'TRAINING_HISTORY.csv',index=False);pd.DataFrame(man).to_csv(o.pub/'FEATURE_RETENTION_MANIFEST.csv',index=False);pd.DataFrame(counts).to_csv(o.pub/'PARAMETER_AND_FEATURE_COUNT.csv',index=False)
    return r


def finish(o,z,frozen):
    a,c,core=o.a,o.c,o.core;rows=list(z['baseline']);files={};fs=c.sha(a.runtime/'DEVELOPMENT_FREEZE_PRIVATE.json')
    for fold,b in enumerate(z['banks'],1):
        va=b['val'];local={}
        for arm in ['B1','B2','B3','B4']:
            r=next(q for q in frozen['selections'] if q['arm']==arm and q['fold']==fold);tau=r['threshold'];cp=Path(r['checkpoint']);assert c.sha(cp)==r['checkpoint_sha256'];out=a.runtime/f'final/fold{fold}/{arm}_VAL_PRIVATE.pt'
            if out.exists():
                cell=torch.load(out,weights_only=False);assert cell['freeze']==fs and c.sha(out)==json.loads(out.with_suffix('.sha256.json').read_text())['sha256'];score=cell['scores']
            elif r['alias']:
                score=local[r['alias']];core.torch_write(out,dict(freeze=fs,scores=score,indices=va,threshold=tau,checkpoint_sha256=r['checkpoint_sha256']));c.write_json(out.with_suffix('.sha256.json'),dict(sha256=c.sha(out)))
            else:
                model=factory(o.orig.SourceA0,retained(o.mp,r['deleted']))().cuda().eval();ck=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
                with torch.inference_mode():score=torch.sigmoid(model(torch.as_tensor(b['x'][va],device='cuda'))).cpu().numpy()
                sel=core.select_threshold(b['y'][va],score,b['patient'][va]);assert sel['threshold']==tau and sel['macro_f1']==ck['macro_f1'] and sel['ez_f1']==ck['ez_f1']
                core.torch_write(out,dict(freeze=fs,scores=score,indices=va,threshold=tau,checkpoint_sha256=r['checkpoint_sha256']));c.write_json(out.with_suffix('.sha256.json'),dict(sha256=c.sha(out)));del model
            local[arm]=score;files[f'{arm}_{fold}']=c.sha(out);rows.extend(patient_rows(o,b,va,score,tau,arm,fold))
    assert len(rows)==325
    df=pd.DataFrame(rows);assert df.patient.nunique()==47
    # Independent sklearn score-only metric reconstruction, no new inference.
    independent=load('independent_metrics',o.a.source/'a0_cove_seed42_v1/code/metrics.py');checked=0
    for fold,b in enumerate(z['banks'],1):
        va=b['val']
        for arm in ['B1','B2','B3','B4']:
            cell=torch.load(a.runtime/f'final/fold{fold}/{arm}_VAL_PRIVATE.pt',weights_only=False)
            for pid in sorted(set(b['patient'][va])):
                k=np.flatnonzero(b['patient'][va]==pid);k=k[np.argsort(b['channel'][va][k].astype(str),kind='stable')];q=independent.independent(b['y'][va[k]],cell['scores'][k],cell['threshold']);r=df[(df.arm==arm)&(df.fold==fold)&(df.patient==pid)].iloc[0]
                for m in METRICS:assert abs(q[m]-r[m])<1e-12
                checked+=1
    summary=df.groupby('arm')[METRICS].mean().reset_index();summary['appearances']=65;summary['unique_IDs']=47;summary.to_csv(o.pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    df.groupby(['arm','patient'])[METRICS].mean().groupby('arm').mean().reset_index().to_csv(o.pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
    df.groupby(['arm','fold'])[METRICS].mean().reset_index().to_csv(o.pub/'DEVELOPMENT_BY_FOLD.csv',index=False)
    bootstrap=load('paired_bootstrap',a.source/'a0_cove_seed42_v1/code/bootstrap.py');boot=[]
    for l,r in o.lock['bootstrap']['comparisons']:
        one=bootstrap.compare(c,df[df.arm==l],df[df.arm==r],METRICS);two=bootstrap.independent(df[df.arm==l],df[df.arm==r],METRICS)
        for v in one:assert np.allclose([v['delta'],v['ci_low'],v['ci_high']],two[v['metric']],atol=1e-12,rtol=0)
        boot.extend(dict(comparison=l+' minus '+r,**v) for v in one)
    pd.DataFrame(boot).to_csv(o.pub/'PAIRED_BOOTSTRAP.csv',index=False);core.torch_write(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pt',rows)
    s=summary.set_index('arm');ff=df.groupby(['arm','fold'])[METRICS].mean();g=o.lock['usefulness'];gates={}
    for arm in ['B2','B3']:
        d=s.loc[arm,METRICS]-s.loc['B1',METRICS];nonnegative=int((ff.loc[arm,'ez_auroc']>=ff.loc['B1','ez_auroc']).sum());groups=sum(len(r['selected_groups']) for r in json.loads((a.runtime/'FEATURE_SELECTION_PRIVATE.json').read_text())['selections']) if arm=='B3' else 15
        conditions=dict(AUROC_nondecrease=d.ez_auroc>=0,AP_stable=d.ez_auprc>=-g['AP_max_loss'],MacroF1_stable=d.macro_f1>=-g['MacroF1_max_loss'],EZF1_stable=d.ez_f1>=-g['EZF1_max_loss'],nonnegative_folds=nonnegative>=4,actual_pruning=groups>0,identity_leakage_verified=True)
        if arm=='B3':conditions['beats_random']=s.loc['B3','ez_auroc']>s.loc['B4','ez_auroc']
        gates[arm]=dict(useful=bool(all(conditions.values())),conditions=conditions,nonnegative_folds=nonnegative,AUROC_delta=float(d.ez_auroc),target_met=bool(d.ez_auroc>=.010))
    terminal='USEFUL_PRUNING_DEVELOPMENT_ONLY' if any(q['useful'] for q in gates.values()) else 'STRUCTURAL_COMPRESSION_VERIFIED_PREDICTIVE_PRUNING_NOT_ESTABLISHED'
    c.write_json(o.pub/'SCIENTIFIC_GATES.json',dict(terminal=terminal,arms=gates,no_outer_run=True));c.write_json(o.pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',score_only_metric_cells=checked,paired_bootstrap_metrics=len(boot),binding=o.bind,no_outer_access=True))
    c.write_json(a.runtime/'PREDICTION_SEAL_PRIVATE.json',dict(freeze=fs,files=files));status(o,status='COMPLETE',terminal=terminal,screening_cells=330,final_cells=len(pd.read_csv(o.pub/'FINAL_RESOURCE_AUDIT.csv')),aliases=20-len(pd.read_csv(o.pub/'FINAL_RESOURCE_AUDIT.csv')),validation_appearances=65,unique_ids=47,no_new_outer=True)
    print(summary[['arm','ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr']].to_string(index=False),flush=True);print(terminal,flush=True)


def main():
    o=setup();z=prepare(o);f=screen(o,z);frozen=final_train(o,z,f);finish(o,z,frozen)


if __name__=='__main__':main()
