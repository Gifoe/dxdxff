"""Unchanged raw storage and fixed-mask feature inputs for neural fits."""
import copy
import gc
from common import *
from model import FeatureHead,M8,Concat

P3CODE=Path(r'D:\d1_small_module_exploration_seed42_v1\phase3\code')
sys.path.insert(0,str(P3CODE))
P3=load('discovery_v2_original_raw_trainer',P3CODE/'run.py')
from compact_delta_classifier import CompactDelta


def restore_cpu_rng(state):
    if torch.is_tensor(state):return state.detach().cpu() if state.dtype==torch.uint8 else state
    if isinstance(state,dict):return {k:restore_cpu_rng(v) for k,v in state.items()}
    if isinstance(state,tuple):return tuple(restore_cpu_rng(v) for v in state)
    if isinstance(state,list):return [restore_cpu_rng(v) for v in state]
    return state


def template(core,fac,records,fold):
    seed=42+1009*fold;core.seed_all(seed)
    base=CompactDelta(fac(seed).network,records) # discarded untrained RNG template only
    return P3.MorphCorrection(base).cuda()


def make(core,fac,records,fold,family,groups,base_state=None):
    seed=42+1009*fold;original=template(core,fac,records,fold)
    if family=='M8':
        base=FeatureHead(seed).cuda()
        if base_state is not None:base.load_state_dict(base_state)
        model=M8(base,original,groups,seed).cuda()
    else:model=Concat(original,groups,seed).cuda()
    assert core.state_hash(model.cnn.state_dict())==core.state_hash(original.cnn.state_dict())
    del original
    return model


class Inputs:
    def __init__(self,reg):
        self.raw=Store(BASE);self.registry=reg;self.delta={}

    def get(self,patient,channels,device):
        pid=str(patient);plan=self.raw.registry['plans'][pid]
        raw,mask,times,perms=self.raw.get(pid,channels,device)
        if pid not in self.delta:
            entry=self.registry['files'][pid];assert sha(entry['path'])==entry['sha256'];q=read(entry['path'])
            assert q['channels']==plan['channels'] and q['raw_bank_sha256']==plan['sha256']
            assert np.array_equal(mask.cpu().numpy(),q['mask']) if channels==plan['channels'] else True
            self.delta[pid]=np.nan_to_num(q['delta'],nan=0.,posinf=0.,neginf=0.)
        ix=[plan['channels'].index(ch) for ch in channels]
        delta=torch.from_numpy(self.delta[pid][:,ix]).to(device)
        assert delta.shape[-1]==32
        return raw,mask,times,delta,perms

    def scale(self,fit_ids):
        values=[]
        for pid in sorted(fit_ids):
            entry=self.registry['files'][pid];assert sha(entry['path'])==entry['sha256'];q=read(entry['path'])
            values.append(np.nanmedian(np.abs(q['delta'][q['mask']]),axis=0))
        typical=np.nanmedian(np.stack(values),axis=0)
        return np.maximum(np.nan_to_num(typical,nan=0.),1e-30).astype(np.float32)


def bank(original,summary,core):
    raw=summary[original['archive_indices']];parts=[];prep=[]
    for sl in [slice(0,14),slice(14,64)]:
        x,pre=core.prepare(raw[:,sl],original['patient'],original['train']);parts.append(x);prep.append(pre)
    x=np.concatenate(parts,1);pre={k:np.concatenate([p[k] for p in prep]) for k in ['imputer_statistics','mean','scale','var']}
    pre['fit_patient_ids']=prep[0]['fit_patient_ids']
    assert set(pre['fit_patient_ids'])==original['roles']['fit']
    return dict(original,x=x,pre=pre)


def masked(d,groups):
    keep=np.zeros(32,np.float32);keep[groups]=1
    return dict(d,x=d['x']*np.repeat(keep,2))


def transform(raw,patients,pre,core):
    z=np.concatenate([core.patient_z(raw[:,:14],patients),core.patient_z(raw[:,14:],patients)],1)
    return ((np.where(np.isnan(z),pre['imputer_statistics'],z)-pre['mean'])/pre['scale']).astype(np.float32)
