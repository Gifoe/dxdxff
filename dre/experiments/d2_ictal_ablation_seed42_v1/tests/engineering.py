"""Pre-training integrity gates, synthetic resume only; no TEST labels/scores."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
import run_study as run
from ablation_core import NEW_ARMS, KEYS, parse_names, mask_for, masked, source_free_factory


def main():
    from types import SimpleNamespace
    a=SimpleNamespace(source=Path(r'E:\DRE-nips\new-pipeline\7-11'),runtime=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'),
        original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'),full=Path(r'C:\d2_ictal_outer_seed42_runtime'),
        noz=Path(r'C:\d2_no_patient_z_outer_seed42_runtime'),
        features=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'),
        protocol=Path(__file__).parents[1]/'PROTOCOL_LOCK.json')
    torch.set_num_threads(2);o=run.setup(a);lock,ref,c,core,model,trainer,helper,bind=o
    pre=json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text());assert pre['binding']==bind
    manifest=json.loads((a.runtime/'public/FEATURE_GROUP_MANIFEST.json').read_text());columns=manifest['columns']
    checks=[]
    def check(description,value):
        assert value,description;checks.append({'check':description,'status':'PASS'})
    check('original full model parameter count',sum(p.numel() for p in model.SourceA0().parameters())==9221)
    sf=source_free_factory(model.SourceA0)
    check('shared-only 8817, no source V u b',sum(p.numel() for p in sf().parameters())==8817 and set(sf().state_dict())==set(core.PRMLP().state_dict()))
    for fold in range(1,6):
        seed=42+1009*fold;core.seed_all(seed);full=model.SourceA0(seed).eval()
        core.seed_all(seed);shared=sf(seed).eval()
        check(f'fold{fold} shared init parity',core.state_hash(full.network.state_dict())==core.state_hash(shared.network.state_dict()))
        z=torch.randn(12,88);g=torch.arange(12)%4
        check(f'fold{fold} shared-only exact forward and source invariance',torch.equal(shared(z,g),full.network(z).squeeze(-1)) and torch.equal(shared(z,g),shared(z,(g+1)%4)))
        for arm in NEW_ARMS:
            d=torch.load(a.runtime/KEYS[arm]/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
            original=torch.load(a.original/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
            check(f'{arm} fold{fold} actual FIT VAL mask',np.all(d['x'][:,~d['mask']]==0) and np.array_equal(d['x'][:,d['mask']],original['x'][:,d['mask']]))
    rng=np.random.default_rng(42);x=rng.normal(size=(32,88));x[2,5]=np.nan
    pid=np.repeat(['FIT_a','FIT_b','VAL_a','VAL_b'],8);tr=np.arange(16);va=np.arange(16,32)
    z,pp=core.prepare(x,pid,tr);changed=x.copy();changed[va]=changed[va]*10+20
    _,pp2=core.prepare(changed,pid,tr)
    check('FIT-only imputation scaling independent of held-out patients',all(np.array_equal(pp[k],pp2[k]) for k in ['mean','scale','var','imputer_statistics']))
    for arm in NEW_ARMS:
        mask=mask_for(columns,arm);out=masked(z,mask)
        reduced,rpre=core.prepare(x[:,mask],pid,tr)
        check(arm+' retained transform unchanged by other columns',np.array_equal(reduced,out[:,mask]))
        for col in columns:
            if 'duplicate_of' in col:assert mask[col['index']]==mask[col['duplicate_of']]
    check('ABS-only has no relative-view main or auxiliary columns',all(col['view']=='abs' for col in columns if mask_for(columns,'ABS Only')[col['index']]))
    y=np.tile([0,1],16);g=np.arange(32)%4
    for arm in NEW_ARMS:
        factory=sf if arm=='w/o Source' else model.SourceA0
        core.seed_all(1051);m=factory(1051).cuda().eval()
        if arm!='w/o Source':m.u.data.fill_(.01)
        tx=torch.tensor(masked(z,mask_for(columns,arm)),device='cuda');tg=torch.tensor(g,device='cuda')
        loss=torch.nn.functional.binary_cross_entropy_with_logits(m(tx,tg),torch.tensor(y,dtype=torch.float32,device='cuda'))
        if arm!='w/o Source':loss=loss+.001*m.regularizer(torch.ones(4,device='cuda'))
        loss.backward()
        check(arm+' valid all-component gradient flow',all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters()))
    values=c.metric(np.array([0,0,1,1]),np.array([.1,.7,.2,.8]),.5)
    check('EZ0 NEZ1 threshold direction', [values[k] for k in ['TP','FP','TN','FN']]==[1,1,1,1])
    # Exercise both changed paths and original source model on the same CUDA runtime.
    root=Path(tempfile.mkdtemp(prefix='synthetic_resume_',dir=a.runtime))
    for key,factory,mask,penalty in [('full',model.SourceA0,np.ones(88,dtype=bool),.001),
        ('masked',model.SourceA0,mask_for(columns,'ABS Only'),.001),('shared',sf,np.ones(88,dtype=bool),0.)]:
        original_factory=trainer.SourceA0;trainer.SourceA0=factory
        try:
            args=(masked(z,mask),y,pid,g,tr,va)
            trainer.train(*args,root/key/'uninterrupted',1051,'synthetic_resume',max_epochs=6,device='cuda',penalty=penalty)
            trainer.train(*args,root/key/'resumed',1051,'synthetic_resume',max_epochs=6,stop_after=2,device='cuda',penalty=penalty)
            trainer.train(*args,root/key/'resumed',1051,'synthetic_resume',max_epochs=6,device='cuda',penalty=penalty)
        finally:trainer.SourceA0=original_factory
        p=torch.load(root/key/'uninterrupted/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
        q=torch.load(root/key/'resumed/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
        check(key+' exact interrupted-resume model history selection',all(torch.equal(v,q['model'][k]) for k,v in p['model'].items()) and p['history']==q['history'] and p['best_key']==q['best_key'])
        check(key+' exact resumed Adam and RNG',all(torch.equal(v,q['optimizer']['state'][k][n]) for k,slot in p['optimizer']['state'].items() for n,v in slot.items()) and torch.equal(p['rng']['torch'],q['rng']['torch']) and all(torch.equal(v,w) for v,w in zip(p['rng']['cuda'],q['rng']['cuda'])))
    c.write_json(a.runtime/'public/ENGINEERING_TESTS.json',{'status':'PASS','binding':bind,'checks':checks,
        'test_count':len(checks),'no_outer_labels_or_scores_read':True,'scientific_training_not_started':True,
        'CUDA_synthetic_resume_paths':3,'formal_reference_files_modified':False})
    print('ALL_ENGINEERING_GATES_PASS',len(checks),flush=True)


if __name__=='__main__':main()
