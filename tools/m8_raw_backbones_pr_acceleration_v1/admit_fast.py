"""Outcome-free legal-FIT numerical parity and alternating wall-time probe."""
import sys
import copy
import time
from pathlib import Path
import numpy as np
import torch
CODE=Path(r'D:\m8_raw_backbones_pr_seed42_v1')
sys.path.insert(0,str(CODE))
from support import *
import engine
import models.omni_archived as archived
from fast_engine import install

def run(out):
    lock,core,rec,banks,binding=setup(RUNTIME)
    store=Store(BASELINE);d=banks[0];tr=d['train'];pid=sorted(set(d['patient'][tr]))[0];ix=tr[d['patient'][tr]==pid]
    raw,mask=raw_patient(store,pid,d['channel'][ix]);y=torch.tensor(d['y'][ix],device='cuda',dtype=torch.float32);pos=torch.tensor(.7,device='cuda')
    original=engine.forward;oldcheck=archived._check
    a=build('TimeConvCNN',True,1,lock,core);b=copy.deepcopy(a)
    core.seed_all(777);a.train();torch.cuda.synchronize();t=time.perf_counter();loss1,ll1=original(a,raw,mask,lock['chunk']['TimeConvCNN'],True,y,pos);torch.cuda.synchronize();oldseconds=time.perf_counter()-t
    buffers={k:v.detach().cpu().clone() for k,v in a.named_buffers()}
    final_rng=engine.rng()
    _,_,audit=install(engine,archived)
    core.seed_all(777);b.train();torch.cuda.synchronize();t=time.perf_counter();loss2,ll2=engine.forward(b,raw,mask,lock['chunk']['TimeConvCNN'],True,y,pos);torch.cuda.synchronize();newseconds=time.perf_counter()-t
    assert torch.equal(ll1,ll2) and loss1==loss2
    maxgrad=0.
    for (n,p),(n2,q) in zip(a.named_parameters(),b.named_parameters()):
        assert n==n2 and (p.grad is None)==(q.grad is None)
        if p.grad is not None:
            maxgrad=max(maxgrad,float((p.grad-q.grad).abs().max()));assert torch.equal(p.grad,q.grad),n
    assert all(torch.equal(buffers[k],v.cpu()) for k,v in b.named_buffers())
    assert all(torch.equal(x,z) for x,z in zip(final_rng,engine.rng()))
    times=[]
    # No optimizer updates, no labels/predictions exported and no selection.
    for fast in [True,False,False,True]:
        m=build('TimeConvCNN',True,1,lock,core).train()
        if fast:install(engine,archived) if engine.forward is original else None
        else:engine.forward=original;archived._check=oldcheck
        core.seed_all(999);torch.cuda.synchronize();t=time.perf_counter()
        engine.forward(m,raw,mask,lock['chunk']['TimeConvCNN'],True,y,pos)
        torch.cuda.synchronize();times.append(dict(deferred=fast,seconds=time.perf_counter()-t));del m;torch.cuda.empty_cache()
    audit.update(status='PASS',complete_legal_FIT_patient=True,valid_channel_windows=int(mask.sum()),logits_loss_all_gradients_BN_buffers_and_RNG_bit_identical=True,max_gradient_error=maxgrad,first_original_seconds=oldseconds,first_deferred_seconds=newseconds,alternating_timings=times,original_source_input_binding_sha256=digest(binding),no_optimizer_step_or_scientific_prediction_changes=True)
    out.mkdir(parents=True,exist_ok=True);write(out/'DEFERRED_FINITE_ADMISSION.json',audit)
    print('DEFERRED_FINITE_BIT_EXACT_PASS',oldseconds,newseconds,times,flush=True)

if __name__=='__main__':run(Path(r'D:\m8_raw_backbones_pr_seed42_v1_execution_v1'))
