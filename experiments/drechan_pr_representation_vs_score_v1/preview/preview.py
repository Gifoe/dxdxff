"""User-requested early CPU outer preview; no training and no parent writes."""
import os,sys,types,inspect,json,csv,hashlib,zipfile,gc,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
sys.dont_write_bytecode=True
SOURCE=Path(r'D:\Codex_Research\experiments\drechan_pr_representation_vs_score_v1')
MAIN=Path(r'D:\Codex_Research\experiments\drechan_pr_representation_vs_score_v1_runtime')
ROOT=MAIN/'partial_outer_20261028'
os.environ['PR_REPR_RUNTIME']=str(ROOT)
sys.path.insert(0,str(SOURCE))
from path_adapter import install
install(r'D:\Codex_Research\_maintenance\cleanup_20261028\LEGACY_PATHS.json',[ROOT])
import numpy as np
import torch
assert not torch.cuda.is_available()
torch.nn.Module.cuda=lambda self,*a,**k:self.cpu()
torch.Tensor.cuda=lambda self,*a,**k:self.cpu()
original_rand,original_tensor=torch.rand,torch.tensor
def cpu_rand(*a,**k):
    if k.get('device')=='cuda':k['device']='cpu'
    return original_rand(*a,**k)
def cpu_tensor(*a,**k):
    if k.get('device')=='cuda':k['device']='cpu'
    return original_tensor(*a,**k)
torch.rand=cpu_rand;torch.tensor=cpu_tensor
from task_io import *
from parents import Parents
import neural,evaluation,reporting
from verify import independent_values
from math_core import KEEP12
MODELS=['LINEAR_SVM','DRECHANNET']


def freeze(p):
    lock=jr(MAIN/'SOURCE_LOCK.json')
    assert lock['source']==sources() and lock['source_commit']=='eede217c80985bbb4f91f8df5580d4fcfbef0f6b'
    q=read(MAIN/'SELECTED_MODELS_PRIVATE.pkl');assert q['binding']==digest(lock)
    cells=[r for r in q['cells'].values() if r['model'] in MODELS]
    assert len(cells)==70 and {(r['model'],r['condition'],r['fold']) for r in cells}=={(m,a,f) for m in MODELS for a in ARMS for f in range(1,6)}
    for r in cells:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
        if r.get('preprocessor'):assert sha(r['preprocessor'])==r['preprocessor_sha256']
    obj=dict(source=sources(),source_commit=lock['source_commit'],cells=cells,
             scope='REQUESTED_EARLY_TWO_MODEL_CPU_OUTER',main_three_model_training_unchanged=True,
             all70_selected_before_preview_test=True,preview_source_sha256=sha(Path(__file__)),
             amendment_sha256=sha(Path(__file__).with_name('PROTOCOL_AMENDMENT.md')))
    fp=ROOT/'FROZEN_BEFORE_TEST.json'
    if fp.exists():assert jr(fp)==obj
    else:write(fp,obj)
    write(ROOT/'SOURCE_LOCK.json',lock)
    p.verify_protected();return obj


def frozen():
    q=jr(ROOT/'FROZEN_BEFORE_TEST.json');assert q['source']==sources() and len(q['cells'])==70
    for r in q['cells']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        if r.get('preprocessor'):assert sha(r['preprocessor'])==r['preprocessor_sha256']
    return q


def cpu_raw(self,m,d,ix,fold,bypass=False):
    raw,mask,times,delta,_=self.inputs.get(str(d['patient'][ix[0]]),d['channel'][ix].astype(str).tolist(),'cpu')
    z,_=(m._base_features if bypass else m.features)(raw,mask,times,delta,self.own.scale(fold))
    return z,int(mask.sum())


def cpu_forward(self,m,d,ix,fold):
    z,n=self.own_raw(m,d,ix,fold)
    return m.classify(z,torch.as_tensor(d['x'][ix],device='cpu')),n


def validate_cpu(p,fr):
    refs=read(MAIN/'admission_private/REFERENCE_VALIDATION.pkl');audit=[]
    for r in fr['cells']:
        if r['model']!='DRECHANNET' or r['condition'].startswith('SCORE_'):continue
        fold,arm=r['fold'],r['condition'];d=p.bank(fold)
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        if arm in ['NO_PR','FULL_PR']:
            m=p.own_model(fold,'FULL' if arm=='FULL_PR' else 'WO_BOTH_PR')
            if arm=='FULL_PR':d=p.bank(fold,True)
            target=refs[key('DRECHANNET',arm,fold)];expected_ll=-target['logit_ez'];expected_prob=target['probability_ez']
        else:
            d,_=neural.data(p,'DRECHANNET',fold,arm,read(r['preprocessor']))
            m=neural.make(p,'DRECHANNET',fold,arm);m._fit_stats=cp['statistics']
            expected_ll=cp['validation_logit'];expected_prob=cp['validation_probability']
        m.load_state_dict(cp['model'])
        ll=neural.predict(p,m,d,d['val'],'DRECHANNET',fold);prob=1-p.own.probability(ll)
        le=float(np.max(np.abs(ll-expected_ll)));pe=float(np.max(np.abs(prob-expected_prob)))
        equal=bool(np.array_equal(p.own.probability(ll)<r['threshold'],(1-expected_prob)<r['threshold']))
        assert le<=1e-5 and pe<=2e-6 and equal,(fold,arm,le,pe,equal)
        audit.append(dict(model='DRECHANNET',condition=arm,fold=fold,maximum_logit_error=le,maximum_probability_error=pe,validation_classifications_exact=equal))
        print('PREVIEW_CPU_VALIDATED',arm,fold,flush=True)
        del m;gc.collect()
    assert len(audit)==25
    write(PUB/'CPU_VALIDATION_REPLAY.json',dict(status='PASS',cells=audit))


def score_and_endpoints(p):
    evaluation.MODELS=MODELS;evaluation.assert_frozen=frozen
    # Only count constants and the declared CPU replay tolerance change. Probability,
    # threshold, index, preprocessing and scoring code stays the frozen implementation.
    refsrc=inspect.getsource(evaluation.references).replace('<=1e-7','<=2e-6')
    exec(compile(refsrc,'<preview_cpu_reference_adapter>','exec'),evaluation.__dict__)
    src=inspect.getsource(evaluation.score).replace('105','70')
    exec(compile(src,'<preview_70_cell_score_adapter>','exec'),evaluation.__dict__)
    src=inspect.getsource(evaluation.endpoints).replace('105','70').replace('1680','1120').replace('15120','10080')
    exec(compile(src,'<preview_70_cell_endpoint_adapter>','exec'),evaluation.__dict__)
    evaluation.score(p)
    return evaluation.endpoints(p)


def report_verify(p,rows):
    ids=sorted({r['patient'] for r in rows});assert len(ids)==80 and len(rows)==1120
    lookup={(r['model'],r['condition'],r['patient']):r for r in rows};src=np.array([lookup[MODELS[0],'NO_PR',pid]['source'] for pid in ids])
    mat={(m,a):np.array([[lookup[m,a,pid][metric] for metric in METRICS] for pid in ids]) for m in MODELS for a in ARMS}
    seal=jr(ROOT/'SCORE_SEAL_PRIVATE.json');assert len(seal['files'])==70
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==p.rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    for r in frozen()['cells']:
        k=key(r['model'],r['condition'],r['fold']);path=ROOT/'scores_private'/(k+'.pkl');assert sha(path)==seal['files'][k];q=read(path);ii=q['indices']
        for pid in np.unique(p.meta['patient'][ii]):
            jj=np.flatnonzero(p.meta['patient'][ii]==pid);v=independent_values(y[ii[jj]],q['ranking_anchor'][jj],q['prediction_ez'][jj],p.meta['channel'][ii[jj]])
            e=lookup[r['model'],r['condition'],str(pid)];assert all(abs(v[metric]-e[metric])<1e-12 for metric in METRICS)
            if r['condition'].startswith('SCORE_'):assert all(e[metric]==lookup[r['model'],'NO_PR',str(pid)][metric] for metric in ['auroc','auprc','mrr','hit_at_1'])
    draw=reporting.draws(ids,src);counts=np.array([np.bincount(q,minlength=80) for q in draw])
    main=[];sources=[];summary=[];ci=[];family=[]
    for m in MODELS:
        for a in ARMS:
            value=mat[m,a];assert np.isfinite(value).all() and sum(lookup[m,a,p]['channels'] for p in ids)==7635
            main.append(dict(model=m,condition=a,patients=80,channels=7635,**dict(zip(METRICS,value.mean(0).tolist()))))
            sp=[]
            for s in reporting.SOURCES:
                v=value[src==s].mean(0);sp.append(v);sources.append(dict(model=m,condition=a,source=s,patients=int(np.sum(src==s)),**dict(zip(METRICS,v.tolist()))))
            sp=np.array(sp);summary.append(dict(model=m,condition=a,**{'center_mean_'+metric:float(sp[:,j].mean()) for j,metric in enumerate(METRICS)},**{'worst_'+metric:float(sp[:,j].min()) for j,metric in enumerate(METRICS) if metric!='ez_fraction_bias'},worst_absolute_ez_fraction_bias=float(np.abs(sp[:,-1]).max())))
        definitions=[('absolute',a,mat[m,a]) for a in ARMS]+[('FULL_MINUS',a,mat[m,'FULL_PR']-mat[m,a]) for a in ARMS if a!='FULL_PR']+[('POST_MINUS_NOPR',a,mat[m,a]-mat[m,'NO_PR']) for a in ['SCORE_CENTER','SCORE_Z']]
        for kind,a,value in definitions:
            sample=counts@value/80
            independent=np.concatenate([value[draw[k:k+250]].mean(1) for k in range(0,10000,250)])
            assert np.max(np.abs(sample-independent))<1e-12
            bounds=np.quantile(sample,[.025,.975],axis=0);ibounds=np.quantile(independent,[.025,.975],axis=0);assert np.max(np.abs(bounds-ibounds))<1e-12
            ci.extend(dict(kind=kind,model=m,condition=a,metric=metric,mean=float(value[:,j].mean()),low=float(bounds[0,j]),high=float(bounds[1,j])) for j,metric in enumerate(METRICS))
            if kind=='FULL_MINUS' and a=='SCORE_Z':
                b=np.quantile(sample,[.05/12,1-.05/12],axis=0);ib=np.quantile(independent,[.05/12,1-.05/12],axis=0);assert np.max(np.abs(b-ib))<1e-12
                family.extend(dict(model=m,metric=metric,mean=float(value[:,j].mean()),low=float(b[0,j]),high=float(b[1,j]),planned_family_size=6) for j,metric in enumerate(METRICS) if metric in ['macro_f1','auroc'])
    assert len(ci)==270 and len(family)==4 and np.array_equal(mat['LINEAR_SVM','GLOBAL'],mat['LINEAR_SVM','NO_PR'])
    # Fresh independent draw reconstruction verifies the resampling, not just summaries.
    rng=np.random.default_rng(42);parts=[]
    for s in ['hup','lzu','multicenter','pediatric']:
        ii=np.array([i for i,pid in enumerate(ids) if lookup[MODELS[0],'NO_PR',pid]['source']==s]);parts.append(ii[rng.integers(0,len(ii),(10000,len(ii)))])
    assert np.array_equal(draw,np.concatenate(parts,axis=1))
    for name,rr in [('MAIN_RESULTS',main),('SOURCE_RESULTS',sources),('SOURCE_SUMMARY',summary),('INTERVALS',ci),('PRIMARY_FAMILY_INTERVALS',family)]:csv_out(name,rr)
    p.verify_protected()
    write(PUB/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',logical_cells=70,patient_condition_rows=1120,endpoint_values=10080,nominal_intervals=270,reported_primary_intervals=4,planned_primary_family_size=6,patients=80,channels_per_condition=7635,inference_device='CPU_FP32',main_gpu_training_unchanged=True))
    write(PUB/'OUTER_STATUS.json',dict(status='VERIFIED_EARLY_TWO_MODEL_OUTER_PREVIEW',all_three_model_complete=False,inference_device='CPU_FP32'))
    print('PREVIEW_ANONYMOUS_RESULTS',json.dumps(dict(main=main,summary=summary,family=family),separators=(',',':')),flush=True)
    files=['MAIN_RESULTS.csv','SOURCE_RESULTS.csv','SOURCE_SUMMARY.csv','INTERVALS.csv','PRIMARY_FAMILY_INTERVALS.csv','CPU_VALIDATION_REPLAY.json','INDEPENDENT_VERIFICATION.json','OUTER_STATUS.json']
    manifest={n:dict(sha256=sha(PUB/n),bytes=(PUB/n).stat().st_size) for n in files}
    path=ROOT/'public_preview.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for n in files:z.write(PUB/n,'public/'+n)
        z.writestr('PUBLIC_MANIFEST.json',json.dumps(dict(files=manifest,private_data_included=False),sort_keys=True))
    print('PREVIEW_PACKAGE_SHA256',sha(path),flush=True)


def main():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    p=Parents();torch.rand=original_rand;torch.set_num_threads(1)
    p.own_raw=types.MethodType(cpu_raw,p);p.own_forward=types.MethodType(cpu_forward,p)
    fr=freeze(p)
    validate_cpu(p,fr)
    rows=score_and_endpoints(p)
    report_verify(p,rows)


if __name__=='__main__':main()
