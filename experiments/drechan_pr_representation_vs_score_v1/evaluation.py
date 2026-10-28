"""All choices freeze, then score sealing, then label/endpoint computation."""
import gc
import hashlib
import numpy as np
import torch
from scipy.special import expit
from task_io import *
from math_core import KEEP12,query_feature,score_transform
from parents import NGA_ROOT,SVM_ROOT,RAW_ROOT
import neural

def assert_frozen():
    fr=jr(ROOT/'FROZEN_BEFORE_TEST.json');assert fr['source']==sources() and len(fr['cells'])==105
    assert fr['source_commit']==jr(ROOT/'SOURCE_LOCK.json')['source_commit']
    for r in fr['cells']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        if r.get('preprocessor'):assert sha(r['preprocessor'])==r['preprocessor_sha256']
    return fr

def references(p,model,condition,fold,d,indices):
    r=p.ref(model,condition,fold);anchor=None;logit=None
    if model=='DRECHANNET':
        k=f"{r['arm']}_seed42_fold{fold}";path=NGA_ROOT/'scores_private'/(k+'.pkl');seal=jr(NGA_ROOT/'OUTER_SCORE_SEAL_PRIVATE.json');assert sha(path)==seal['files'][k]
        q=read(path);assert np.array_equal(q['indices'],indices) and q['checkpoint']==r['checkpoint_sha256']
        anchor=1-q['score_nez'].astype(float);pred=q['score_nez']<r['threshold']
        if condition=='NO_PR':
            cache=ROOT/'reuse_logits_private'/f'DRECHANNET_fold{fold}.pkl'
            if cache.exists():
                cached=read(cache);assert cached['checkpoint']==r['checkpoint_sha256'] and cached['freeze']==sha(ROOT/'FROZEN_BEFORE_TEST.json');logit=cached['logit_ez']
            else:
                m=p.own_model(fold);m.load_state_dict(torch.load(r['checkpoint'],weights_only=False,map_location='cpu')['model'])
                raw=p.rec['R14'][indices];pre=p.bank(fold)['pre'];x=p.own.query_features(raw,p.meta['patient'][indices],pre,'WO_BOTH_PR',p.core)
                td=dict(x=x,patient=p.meta['patient'][indices],channel=p.meta['channel'][indices]);ll=neural.predict(p,m,td,np.arange(len(indices)),model,fold)
                assert np.max(np.abs(p.own.probability(ll)-q['score_nez']))<=1e-7
                logit=-ll.astype(float);save(cache,dict(checkpoint=r['checkpoint_sha256'],freeze=sha(ROOT/'FROZEN_BEFORE_TEST.json'),logit_ez=logit));del m;gc.collect();torch.cuda.empty_cache()
    elif model=='LINEAR_SVM':
        path=SVM_ROOT/'scores_private'/f"{r['arm']}_{fold}.pkl";seal=jr(SVM_ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
        assert sha(path)==seal['files'][f"{r['arm']}_{fold}"]
        q=read(path);assert np.array_equal(q['indices'],indices);anchor=q['probability_ez'].astype(float);pred=q['prediction_ez'].astype(bool)
        if condition=='NO_PR':
            bundle=read(r['checkpoint'])['bundle'];raw=np.asarray(p.rec['R14'][indices][:,KEEP12],np.float64);prob,margin=p.bl.predict(bundle,raw)
            assert np.max(np.abs(prob-anchor))<=1e-12
            logit=bundle['calibration'].decision_function(margin[:,None])
    else:
        version='Original' if condition=='NO_PR' else 'PR';path=RAW_ROOT/'TimeConvCNN/outer_private'/f'{version}_fold{fold}.pkl';q=read(path)
        assert np.array_equal(q['indices'],indices) and q['checkpoint_sha256']==r['checkpoint_sha256']
        anchor=q['score_ez'].astype(float);pred=anchor>=r['threshold']
        if condition=='NO_PR':
            logit=np.empty(len(indices),float);seal=jr(RAW_ROOT/'TimeConvCNN/OUTER_PREDICTIONS_SEAL_PRIVATE.json')
            for n,pid in enumerate(sorted(set(p.meta['patient'][indices])),1):
                jj=np.flatnonzero(p.meta['patient'][indices]==pid);token=hashlib.sha256(str(pid).encode()).hexdigest()[:24]
                cell=RAW_ROOT/f'TimeConvCNN/outer_private/{version}/fold{fold}/{token}.pkl';assert sha(cell)==seal['files'][f'{version}_{fold}_{n}']
                z=read(cell);assert np.array_equal(z['indices'],indices[jj])
                logit[jj]=-z['logit_nez'].astype(float)
    assert anchor.shape==pred.shape==(len(indices),) and np.isfinite(anchor).all()
    return dict(probability_ez=anchor,prediction_ez=pred,logit_ez=logit,ranking_anchor=anchor)

def score(p):
    fr=assert_frozen();fh=sha(ROOT/'FROZEN_BEFORE_TEST.json');meta=p.meta;seal={}
    for model in MODELS:
        for fold in range(1,6):
            d=p.bank(fold);indices=np.flatnonzero(np.isin(meta['patient'],list(d['roles']['test'])))
            ref={a:references(p,model,a,fold,d,indices) for a in ['NO_PR','FULL_PR']}
            for condition in ARMS:
                k=key(model,condition,fold);path=ROOT/'scores_private'/(k+'.pkl');r=next(q for q in fr['cells'] if key(q['model'],q['condition'],q['fold'])==k)
                if path.exists():
                    q=read(path);assert q['freeze']==fh and np.array_equal(q['indices'],indices)
                else:
                    if condition in ref:out=ref[condition]
                    elif model=='LINEAR_SVM' and condition=='GLOBAL':out=ref['NO_PR']
                    elif condition.startswith('SCORE_'):
                        z=score_transform(ref['NO_PR']['logit_ez'],meta['patient'][indices],condition)
                        prob=expit(z);pred=expit(-z)<r['threshold'] if r['orientation']=='NEZ_LT' else prob>=r['threshold']
                        for pid in sorted(set(meta['patient'][indices])):
                            jj=np.flatnonzero(meta['patient'][indices]==pid)
                            a=ref['NO_PR']['logit_ez'][jj];b=z[jj];order=np.argsort(a,kind='stable')
                            assert np.all(np.diff(b[order])>=0) and np.array_equal(a[order][1:]==a[order][:-1],b[order][1:]==b[order][:-1])
                        out=dict(probability_ez=prob,prediction_ez=pred,logit_ez=z,ranking_anchor=ref['NO_PR']['ranking_anchor'])
                    elif model=='LINEAR_SVM':
                        from svm import predict
                        q=read(r['checkpoint']);raw=p.rec['R14'][indices][:,KEEP12];prob,ll=predict(q['bundle'],raw,meta['patient'][indices]);out=dict(probability_ez=prob,prediction_ez=prob>=r['threshold'],logit_ez=ll,ranking_anchor=prob)
                    else:
                        cp=torch.load(r['checkpoint'],weights_only=False,map_location='cpu');m=neural.make(p,model,fold,condition);m.load_state_dict(cp['model']);m._fit_stats=cp['statistics']
                        td=dict(patient=meta['patient'][indices],channel=meta['channel'][indices])
                        if model=='DRECHANNET':td['x']=query_feature(p.rec['R14'][indices].copy()*np.array([1]*8+[0,0]+[1]*4),td['patient'],read(r['preprocessor'])).astype(np.float32)
                        ll=neural.predict(p,m,td,np.arange(len(indices)),model,fold);prob=1-p.own.probability(ll) if model=='DRECHANNET' else 1-torch.sigmoid(torch.as_tensor(ll)).numpy()
                        pred=p.own.probability(ll)<r['threshold'] if r['orientation']=='NEZ_LT' else prob>=r['threshold']
                        out=dict(probability_ez=prob,prediction_ez=pred,logit_ez=-ll.astype(float),ranking_anchor=prob)
                        del m;gc.collect();torch.cuda.empty_cache()
                    q=dict(freeze=fh,indices=indices,model=model,condition=condition,fold=fold,threshold=r['threshold'],orientation=r['orientation'],checkpoint=r['checkpoint_sha256'],**out);save(path,q)
                assert len(q['probability_ez'])==len(indices) and np.isfinite(q['ranking_anchor']).all()
                seal[k]=sha(path);status('SCORING',completed_cells=len(seal),logical_cells=105);print('PR_REPR_SCORE_SEALED',model,condition,fold,len(seal),flush=True)
    assert len(seal)==105;p.verify_protected()
    write(ROOT/'SCORE_SEAL_PRIVATE.json',dict(freeze=fh,files=seal,all105_cells_sealed_before_new_metrics=True))
    write(PUB/'SCORE_STATUS.json',dict(status='ALL105_SCORE_CELLS_SEALED',cells=105,seal_sha256=sha(ROOT/'SCORE_SEAL_PRIVATE.json')))

def endpoints(p):
    fr=assert_frozen();seal=jr(ROOT/'SCORE_SEAL_PRIVATE.json');assert len(seal['files'])==105
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==p.rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    assert len(y)==7635 and int(y.sum())==1743
    rows=[]
    source_values={str(p.meta['center'][p.meta['patient']==pid][0]) for pid in set(p.meta['patient'])}
    mapping={}
    size_to_source={36:'hup',21:'lzu',15:'multicenter',8:'pediatric'}
    for value in source_values:
        n=len(set(p.meta['patient'][p.meta['center'].astype(str)==value]));mapping[value]=size_to_source[n]
    for r in fr['cells']:
        k=key(r['model'],r['condition'],r['fold']);path=ROOT/'scores_private'/(k+'.pkl');assert sha(path)==seal['files'][k];q=read(path);ii=q['indices']
        for pid in sorted(set(p.meta['patient'][ii])):
            ix=np.flatnonzero(p.meta['patient'][ii]==pid);truth=y[ii[ix]];anchor=q['ranking_anchor'][ix];pred=q['prediction_ez'][ix]
            v=dict(**p.ev.classification(truth,anchor,pred),**p.ev.ranking(truth,anchor,p.meta['channel'][ii[ix]]))
            assert all(np.isfinite(v[m]) for m in METRICS)
            rows.append(dict(model=r['model'],condition=r['condition'],fold=r['fold'],patient=str(pid),source=mapping[str(p.meta['center'][ii[ix[0]]])],channels=len(ix),**{m:float(v[m]) for m in METRICS}))
    assert len(rows)==1680
    save(ROOT/'PATIENT_ENDPOINTS_PRIVATE.pkl',rows)
    write(PUB/'ENDPOINT_STATUS.json',dict(status='ALL_PATIENT_ENDPOINTS_SEALED',rows=1680,values=15120,sha256=sha(ROOT/'PATIENT_ENDPOINTS_PRIVATE.pkl')))
    return rows
