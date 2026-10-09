import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from base import seed_all,state_hash,prepare,oof_plan,select_threshold,PRMLP
from model import Core,Temporal
from features import relative_features,extract,pack,normalize,FEATURES
from train import train


def bank():
    seed_all(42); z=torch.randn(5,2,5,18)
    t=torch.tensor([-3.,-1.,1.,4.,10.]).expand(5,2,5).clone()
    mask=torch.ones(5,2,5,dtype=torch.bool)
    return z,t,mask


def record():
    x=np.arange(4*6*9,dtype=float).reshape(4,6,9)/10
    return {'channel_names_norm':['c'+str(i) for i in range(6)],'sfreq':250.,
        'sample':{'window_feature_names':FEATURES,'window_features':x,
        'window_relative_centers_sec':np.array([-1.,1.,3.,10.]),'feature_scale_used_secs':[2.]*4,
        'raw_temporal_sfreq':1000.,'raw_temporal_duration_sec':60.,'raw_valid_start_sample':0,
        'raw_valid_samples':60000,'start_sec':20.,'seizure_onset_sec':50.}}


def test_parameter_initialization_time_and_orientation():
    seed_all(42); a=Temporal('T1'); seed_all(42); b=Temporal('T2')
    assert state_hash(a.state_dict())==state_hash(b.state_dict())
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==5797
    z,t,m=bank(); a.eval(); b.eval()
    assert torch.equal(a(z,t,m),torch.zeros(5)) and torch.equal(b(z,t,m),torch.zeros(5))
    frozen=PRMLP().eval().requires_grad_(False); x=torch.randn(5,88)
    base=frozen(x); assert torch.equal(base+a(z,t,m),base)
    (base+a(z,t,m)).sum().backward(); assert all(p.grad is None for p in frozen.parameters())
    assert torch.sigmoid(-torch.tensor(-1.))>.5
    assert not torch.equal(a.encode(z[:,0],t[:,0],m[:,0]),a.encode(z[:,0],t[:,0]+2,m[:,0]))


def test_official_cotar_parity():
    path=Path(os.environ.get('TECH_LAYER_SOURCE',r'D:\chenyu-iclr\_upstream_TeCh_audit\layers\Transformer_EncDec.py'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='268d8d5486685bdc45a1f8c44056a85d7e327f70e973a8b9e50d63675e352c2d'
    spec=importlib.util.spec_from_file_location('official_tech_parity',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    seed_all(4); off=module.CoTAR(16,4).eval(); new=Core('T2').eval(); new.load_state_dict(off.state_dict())
    x=torch.randn(7,12,16); m=torch.ones(7,12,dtype=torch.bool)
    assert torch.equal(off(x),new(x,m))


def test_local_vs_global_interaction_and_mask():
    seed_all(2); a=Core('T1').eval(); b=Core('T2').eval(); b.load_state_dict(a.state_dict())
    x=torch.randn(3,8,16); m=torch.ones(3,8,dtype=torch.bool); y=x.clone(); y[:,2]+=5
    keep=[0,1,3,4,5,6,7]
    assert torch.equal(a(x,m)[:,keep],a(y,m)[:,keep])
    assert not torch.equal(b(x,m)[:,keep],b(y,m)[:,keep])
    m[:,2]=False; assert torch.equal(b(x,m),b(y,m))
    allmiss=torch.zeros_like(m); assert torch.equal(b(x,allmiss),torch.zeros_like(x))
    z,t,mask=bank(); mask[0]=False; z[0]=float('nan')
    model=Temporal('T2'); model.head[-1].weight.data.fill_(.1)
    out=model(z,t,mask); assert torch.isfinite(out).all() and out[0]==0
    out.sum().backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_stage_cross_seizure_pooling():
    model=Temporal('T2')
    t=torch.tensor([[[-2.,1.,3.,10.],[-2.,1.,3.,10.]]])
    mask=torch.ones(1,2,4,dtype=torch.bool)
    h=torch.tensor([[[1.,2.,4.,8.],[3.,6.,8.,10.]]])[...,None].expand(1,2,4,16)
    emb,early=model.readout(h,mask,t)
    assert early.item()
    assert torch.equal(emb[0,:48],torch.tensor([2.,5.,9.]).repeat_interleave(16))
    assert torch.equal(emb[0,48:],torch.tensor([1.,2.,1.]).repeat_interleave(16))
    mask[:,:,0]=False; emb,_=model.readout(h,mask,t); assert (emb[0,:16]==0).all()
    mask[:,:,[1,2]]=False; _,early=model.readout(h,mask,t); assert not early.item()
    z,t,m=bank(); order=torch.tensor([3,1,4,0,2]); model.eval()
    assert torch.equal(model(z,t,m)[order],model(z[order],t[order],m[order]))


def test_temporal_reference_and_extraction():
    r=record(); names=r['channel_names_norm']; out,t,mask,a=extract(r,names)
    assert out.shape==(6,4,18) and mask.all()
    x=r['sample']['window_features']; med=np.median(x,axis=1,keepdims=True)
    scale=1.4826*np.median(abs(x-med),axis=1,keepdims=True)
    expected=np.clip((x-med)/(scale+1e-8),-8,8)
    np.testing.assert_allclose(out[...,9:],expected.transpose(1,0,2),rtol=1e-6)
    changed=copy.deepcopy(r); changed['labels']=[1]*6
    assert np.array_equal(out,extract(changed,names)[0])
    perm=[4,0,2,1,5,3]; changed['channel_names_norm']=[names[i] for i in perm]
    changed['sample']['window_features']=x[:,perm]
    assert np.array_equal(out,extract(changed,names)[0])
    r['sample']['window_features'][1,2,0]=np.nan
    r['sample']['window_relative_centers_sec'][3]=np.nan
    out,t,m,_=extract(r,names); assert not m[2,1] and np.isfinite(out).all()
    b,details=pack([r,record()],names+['missing']); assert not b['mask'][-1].any()
    const=np.ones((3,5,9)); assert np.isfinite(relative_features(const,np.ones((3,5),bool))).all()


def test_fit_only_normalization_and_oof():
    r=record(); b,_=pack([r],r['channel_names_norm']); banks={'fit':b,'query':copy.deepcopy(b)}
    n,s=normalize(banks,{'fit'}); banks['query']['z']*=10000
    n2,s2=normalize(banks,{'fit'})
    assert np.array_equal(s['mean'],s2['mean']) and np.array_equal(n['fit']['z'],n2['fit']['z'])
    plans=oof_plan({str(i) for i in range(20)},1)
    for p in plans: assert set(p['query']).isdisjoint(p['train']) and set(p['query']).isdisjoint(p['validation'])
    x=np.arange(20*88,dtype=float).reshape(20,88); pid=np.repeat(['fit','query'],10)
    _,pre=prepare(x,pid,np.arange(10)); x[10:]+=np.arange(88)*10000
    _,pre2=prepare(x,pid,np.arange(10)); assert np.array_equal(pre['mean'],pre2['mean'])


def synthetic_development():
    rng=np.random.default_rng(5); patient=np.repeat(['a','b','c','d'],6)
    y=np.tile([0,0,1,1,1,1],4); base=rng.normal(size=24).astype(np.float32)
    banks={}
    for p in ['a','b','c','d']:
        banks[p]={'z':rng.normal(size=(6,2,4,18)).astype(np.float32),
                  'time':np.broadcast_to([-2.,1.,3.,10.],(6,2,4)).copy().astype(np.float32),
                  'mask':np.ones((6,2,4),bool)}
    va=np.arange(12,24); tau=select_threshold(y[va],torch.sigmoid(torch.from_numpy(base[va])).numpy(),patient[va])['threshold']
    return {'banks':banks,'patient':patient,'y':y,'train':np.arange(12),'val':va,
            'base_logits':base,'T0_threshold':tau}


def test_resume_and_epoch0_fallback(tmp_path):
    torch.set_num_threads(2); d=synthetic_development()
    seed_all(52); initial=copy.deepcopy(Temporal('T1').state_dict())
    for mode in ['T1','T2']:
        full,best,h=train(d,mode,tmp_path/(mode+'full'),52,initial,'synthetic','cpu',max_epochs=2)
        train(d,mode,tmp_path/(mode+'resume'),52,initial,'synthetic','cpu',max_epochs=2,stop_after=1)
        resumed,rb,rh=train(d,mode,tmp_path/(mode+'resume'),52,initial,'synthetic','cpu',max_epochs=2)
        assert state_hash(full.state_dict())==state_hash(resumed.state_dict()) and best['epoch']==rb['epoch'] and h==rh
        assert best['macro_f1']>=h[0]['macro_f1']
