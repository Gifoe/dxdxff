"""Real FIT/VAL-only admission. No new outer scores or endpoint inspection."""
import gc
import shutil
import numpy as np
import torch
from task_io import *
from math_core import KEEP12,score_transform,StreamingMoments,preserved_eval
import neural

def reference_validation(p,model,condition,fold):
    r=p.ref(model,condition,fold);assert sha(r['checkpoint'])==r['checkpoint_sha256']
    d=p.bank(fold,condition=='FULL_PR');va=d['val'];ll=None
    if model=='LINEAR_SVM':
        q=read(r['checkpoint']);raw=np.asarray(p.rec['R14'][d['archive_indices']][:,KEEP12],np.float64)
        view=p.core.patient_z(raw,d['patient']) if condition=='FULL_PR' else raw
        prob,margin=p.bl.predict(q['bundle'],view[va]);ll=q['bundle']['calibration'].decision_function(margin[:,None])
        expected=q['validation_probability'];sel=p.ev.threshold(1-d['y'][va],prob,d['patient'][va]);threshold=sel['threshold'];orientation='EZ_GE'
    elif model=='DRECHANNET':
        m=p.own_model(fold,'FULL' if condition=='FULL_PR' else 'WO_BOTH_PR');initial=p.own.state_hash(m.state_dict());q=torch.load(r['checkpoint'],weights_only=False,map_location='cpu');assert initial==q['binding']['initial'];m.load_state_dict(q['model'])
        ll_nez=neural.predict(p,m,d,va,model,fold);ll=-ll_nez.astype(float);s=p.core.select_threshold(d['y'][va],p.own.probability(ll_nez),d['patient'][va]);threshold=s['threshold'];orientation='NEZ_LT';prob=1-p.own.probability(ll_nez)
        v=read(r['validation']);assert 'logit' in v;expected=1-p.own.probability(v['logit']);sel=s
        del m;gc.collect();torch.cuda.empty_cache()
    else:
        m=p.raw_model(fold,condition=='FULL_PR');assert p.own.state_hash(m.state_dict())==r['initial_sha256'];q=torch.load(r['checkpoint'],weights_only=False,map_location='cpu');m.load_state_dict(q['model'])
        ll_nez=neural.predict(p,m,d,va,model,fold);ll=-ll_nez.astype(float);prob=1-torch.sigmoid(torch.as_tensor(ll_nez)).numpy();s=p.ev.threshold(1-d['y'][va],prob,d['patient'][va]);sel=s;threshold=s['threshold'];orientation='EZ_GE'
        v=read(Path(r['checkpoint']).parent/'VALIDATION_PRIVATE.pkl');expected=v['score_ez']
        del m;gc.collect();torch.cuda.empty_cache()
    error=float(np.max(np.abs(prob-expected)));assert error<=1e-6,(model,condition,fold,error)
    assert threshold==r['threshold'],(model,condition,fold,'threshold drift')
    assert np.isfinite(ll).all()
    return dict(logit_ez=ll,probability_ez=prob,indices=d['archive_indices'][va],patient=d['patient'][va],selection=sel,threshold=threshold,orientation=orientation,maximum_replay_error=error)

def post_selection(p,model,ref,condition,fold):
    d=p.bank(fold);va=d['val'];z=score_transform(ref['logit_ez'],d['patient'][va],condition)
    from scipy.special import expit
    if model=='DRECHANNET':
        q=expit(-z);s=p.core.select_threshold(d['y'][va],q,d['patient'][va]);orientation='NEZ_LT'
    else:
        q=expit(z);s=p.ev.threshold(1-d['y'][va],q,d['patient'][va]);orientation='EZ_GE'
    return dict(model=model,condition=condition,fold=fold,threshold=s['threshold'],orientation=orientation,parent_selection=s,reused=False,postprocess=True)

def cache_parity(p):
    """Changed normalization Jacobian under the original two-pass BF16 engine."""
    from models.backbones import aggregate
    torch.manual_seed(224);torch.cuda.manual_seed_all(224)
    raw=torch.randn(1,7,20,500,device='cuda');mask=torch.ones(1,7,20,dtype=torch.bool,device='cuda');target=torch.tensor([0,1,1,0,1,1,1],device='cuda',dtype=torch.float32);pos=torch.tensor(.4,device='cuda')
    rows=[]
    for condition in ['GLOBAL','CENTER','SCALE']:
        a=neural.make(p,'TIMECONV',1,condition);b=neural.make(p,'TIMECONV',1,condition);b.load_state_dict(a.state_dict())
        stats={'mean':np.linspace(-.1,.1,a.window_dim*2),'sd':np.linspace(.7,1.3,a.window_dim*2),'n':100}
        a._fit_stats=b._fit_stats=stats;a.train();b.train();state=p.engine.rng()
        loss_a,logit_a=p.engine.forward(a,raw,mask,128,True,target,pos)
        cached_rng=p.engine.rng()
        grads={k:v.grad.detach().float().clone() for k,v in a.named_parameters() if v.grad is not None}
        p.engine.restore(state);r,m,chunks=p.engine.layout(raw,mask,False,128);parts=[]
        for ii in chunks:
            with p.engine.amp():parts.append(b.encode(r[ii]).float())
        z=aggregate(p.engine.dense(torch.cat(parts),mask,False),mask)
        with p.engine.amp():logit_b=b.classify(z)
        loss_b=torch.nn.functional.binary_cross_entropy_with_logits(logit_b.float(),target,pos_weight=pos);loss_b.backward()
        assert torch.equal(logit_a,logit_b.float()) and loss_a==float(loss_b.detach())
        errors=[]
        for name,v in b.named_parameters():
            if v.grad is None:assert name not in grads;continue
            ga=grads[name];gb=v.grad.float();err=float((ga-gb).abs().max());errors.append(err)
            assert torch.allclose(ga,gb,atol=1e-5,rtol=2e-4),(condition,name,err)
        for (ka,va),(kb,vb) in zip(a.named_buffers(),b.named_buffers()):assert ka==kb and torch.equal(va,vb)
        direct_rng=p.engine.rng();assert all(torch.equal(a,b) for a,b in zip(cached_rng,direct_rng))
        rows.append(dict(condition=condition,logits_exact=True,loss_exact=True,bn_exact=True,max_gradient_error=max(errors)))
        del a,b;gc.collect();torch.cuda.empty_cache()
    return rows

def admit(p,commit):
    from test_core import CoreTests
    import unittest,sklearn
    assert sklearn.__version__=='1.7.2' and torch.__version__=='2.11.0+cu128'
    import sys
    result=unittest.TextTestRunner(stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests));assert result.wasSuccessful()
    assert shutil.disk_usage(ROOT).free>=12*1024**3,'7 GiB task budget plus5GiB reserve unavailable'
    references={};rows=[]
    cache=ROOT/'admission_private';cache.mkdir(parents=True,exist_ok=True)
    for model in MODELS:
        for fold in range(1,6):
            for condition in ['NO_PR','FULL_PR']:
                k=key(model,condition,fold);r=reference_validation(p,model,condition,fold);references[k]=r
                rows.append(dict(model=model,condition=condition,fold=fold,maximum_replay_error=r['maximum_replay_error'],threshold_verified=True,checkpoint_sha256=p.ref(model,condition,fold)['checkpoint_sha256']))
                save(cache/'REFERENCE_VALIDATION.pkl',references)
                print('PR_REPR_REFERENCE_REPLAY',model,condition,fold,r['maximum_replay_error'],flush=True)
    feature_parity=[]
    from math_core import fit_feature
    for fold in range(1,6):
        d=p.bank(fold,True);raw=p.rec['R14'][d['archive_indices']].copy();raw[:,8:10]=0
        x,_=fit_feature(raw,d['patient'],d['train'],'FULL_PR',True)
        error=float(np.max(np.abs(x[:,KEEP12].astype(np.float32)-d['x'][:,KEEP12])))
        assert error<=1e-6
        original=p.core.patient_z(raw[:,KEEP12],d['patient']);_,pr=fit_feature(raw[:,KEEP12],d['patient'],d['train'],'FULL_PR',False)
        viewed=__import__('math_core').feature_view(raw[:,KEEP12],d['patient'],'FULL_PR',pr['mu'],pr['unit'],False)
        assert np.max(np.abs(original-viewed))<=1e-12
        feature_parity.append(dict(fold=fold,own_full_feature_max_error=error,svm_full_view_max_error=float(np.max(np.abs(original-viewed)))))
    parity=cache_parity(p)
    actual=[];private_actual=[]
    for model in ['DRECHANNET','TIMECONV']:
        d,_=neural.data(p,model,1,'GLOBAL');m=neural.make(p,model,1,'GLOBAL');initial=p.own.state_hash(m.state_dict());stats=neural.collect(p,m,d,model,1);m._fit_stats=stats
        assert stats['n']==len(d['train']);ii=neural.groups(d,d['train'])[0];m.train();y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32);pos=torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()),device='cuda')
        if model=='DRECHANNET':
            ll,n=p.own_forward(m,d,ii,1);loss=p.own.MODEL.objective(ll,y,pos);loss.backward()
        else:
            loss,_=p.raw_forward(m,d,ii,True,y,pos)
            _,mask=p.raw.raw_patient(p.raw_store,str(d['patient'][ii[0]]),d['channel'][ii]);n=int(mask.sum())
        assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
        private_actual.append(dict(model=model,complete_patient_channels=len(ii),valid_windows=n))
        actual.append(dict(model=model,fit_channel_count=stats['n'],latent_dimension=len(stats['mean']),complete_patient_backward=True,initial_sha256=initial))
        save(cache/(model+'_INITIAL_STATISTICS.pkl'),dict(initial=initial,statistics=stats))
        del m;gc.collect();torch.cuda.empty_cache()
    p.verify_protected()
    write(ROOT/'PARENT_HASH_BINDING.json',p.parent_protected)
    save(cache/'BACKWARD_COVERAGE_PRIVATE.pkl',private_actual)
    record=dict(status='PASS',source=sources(),source_commit=commit,reused_references=30,source_counts={'hup':36,'lzu':21,'multicenter':15,'pediatric':8},cohort={'patients':80,'channels':7635,'seizures':256,'valid_raw_windows':1471965},reference_replay=rows,feature_parity=feature_parity,gradient_cache=parity,actual_fit_admission=actual,parents_unchanged=True,no_new_test_results=True,free_disk_gib=shutil.disk_usage(ROOT).free/1024**3)
    write(PUB/'ADMISSION.json',record);print('PR_REPR_ADMISSION_PASS',flush=True)
    return record
