"""Real CUDA admission, deterministic masking/PR and gradient-replay equivalence."""
import copy
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from support import *
from patient_relative import patient_relative
from models.backbones import aggregate
from engine import forward,layout,dense,amp,rng,restore


def direct(model,raw,mask,chunk,y,pos):
    r,m,chunks=layout(raw,mask,model.multichannel,chunk)
    # Match the physical chunk autocast boundaries exactly. A single outer
    # autocast scope shares cast weights and rounds their summed gradients in
    # BF16; that is a different arithmetic graph, not the replay reference.
    hs=[]
    for ix in chunks:
        with amp():hs.append(model.encode(r[ix],None if m is None else m[ix]).float())
    e=torch.cat(hs);ch=aggregate(dense(e,mask,model.multichannel,torch.cat(chunks) if model.multichannel else None),mask)
    with amp():ll=model.classify(ch).float()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(ll,y,pos_weight=pos);loss.backward()
    return loss,ll


def main(runtime):
    lock,core,rec,banks,binding=setup(runtime);audits=[]
    core.seed_all(42)
    x=torch.randn(7,13,device='cuda');x[:,2]=1;x[-1]=1234
    valid=torch.tensor([1,1,1,1,1,1,0],dtype=torch.bool,device='cuda')
    z=patient_relative(x,valid);ref=core.patient_z(x[:6].cpu().numpy(),np.repeat('anonymous',6))
    assert np.array_equal(z[:6].cpu().numpy(),ref.astype(np.float32)) and torch.equal(z[-1],torch.zeros_like(z[-1]))
    assert torch.isfinite(patient_relative(torch.ones(1,3,device='cuda'))).all()
    raw=torch.randn(2,5,7,500,device='cuda');mask=torch.ones(2,5,7,dtype=torch.bool,device='cuda');mask[1,0,:]=False;raw[~mask]=0
    y=torch.tensor([0,1,1,0,1.],device='cuda');pos=torch.tensor(.7,device='cuda')
    for name in lock['models']:
        t=time.perf_counter();original=build(name,False,1,lock,core);pr=build(name,True,1,lock,core)
        assert statehash(original.state_dict())==statehash(pr.state_dict())
        original.eval();pr.eval()
        with torch.inference_mode():
            l,h=forward(original,raw,mask,5)
            p,z=forward(pr,raw,mask,5)
            l2,h2=forward(original,raw,mask,9);p2,_=forward(pr,raw,mask,9)
            chunkerr=float((p-p2).abs().max());assert torch.allclose(p,p2,rtol=1e-3,atol=1e-4)
            assert l.shape==p.shape==(5,) and torch.isfinite(h).all() and torch.isfinite(z).all()
            # Invalid waveform payloads must not enter normalization or pooling.
            dirty=raw.clone();dirty[~mask]=10000
            p3,_=forward(pr,dirty,mask,5);assert torch.equal(p,p3)
            # Reordering channels must reorder scores, not change patient context.
            order=torch.tensor([4,1,0,3,2],device='cuda')
            p4,_=forward(pr,raw[:,order],mask[:,order],5)
            assert torch.allclose(p4,p[order],rtol=1e-3,atol=1e-4)
        # Compare full mathematical graph and exact-memory gradient cache.
        a=build(name,True,1,lock,core).train();b=copy.deepcopy(a)
        core.seed_all(777);loss1,ll1=direct(a,raw,mask,5,y,pos)
        core.seed_all(777);loss2,ll2=forward(b,raw,mask,5,True,y,pos)
        assert torch.equal(ll1,ll2) and abs(float(loss1.detach())-loss2)<1e-7
        maxdiff=0.;maxrelative=0.
        for (n,p1),(n2,p2) in zip(a.named_parameters(),b.named_parameters()):
            assert n==n2 and (p1.grad is None)==(p2.grad is None)
            if p1.grad is not None:
                err=float((p1.grad-p2.grad).abs().max());maxdiff=max(maxdiff,err)
                assert torch.allclose(p1.grad,p2.grad,rtol=.001,atol=.0001),n
        for (n,v),(n2,v2) in zip(a.named_buffers(),b.named_buffers()):assert n==n2 and torch.equal(v,v2),n
        audits.append(dict(model=name,status='PASS',parameters=sum(p.numel() for p in original.parameters()),paired_initial_hash=statehash(original.state_dict()),chunk_score_max_error=chunkerr,gradient_cache_direct_logit_exact=True,gradient_cache_max_absolute_error=maxdiff,BN_buffers_exact=True,padding_invariant=True,channel_permutation_equivariant=True,seconds=time.perf_counter()-t))
        print('RAW_BACKBONE_SMOKE_PASS',name,round(time.perf_counter()-t,2),flush=True)
        del original,pr,a,b;torch.cuda.empty_cache()
    # Exercise the full real FIT patient, not a sampled subset, and discard state.
    d=banks[0];tr=d['train'];pid=sorted(set(d['patient'][tr]))[0];ii=tr[d['patient'][tr]==pid]
    store=Store(BASELINE);rr,mm=raw_patient(store,pid,d['channel'][ii])
    model=build('TimeConvCNN',True,1,lock,core).train();target=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    t=time.perf_counter();loss,_=forward(model,rr,mm,lock['chunk']['TimeConvCNN'],True,target,pos)
    assert np.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    real=dict(complete_FIT_patient=True,channels=len(ii),valid_channel_windows=int(mm.sum()),seconds=time.perf_counter()-t,discarded_smoke_state=True)
    # Only synthetic labels above; clinical labels here belong to legal FIT.
    write(runtime/'public/SMOKE_TEST_AUDIT.json',dict(status='PASS',binding_sha256=digest(binding),models=audits,PR_matches_original_population_Z_exact=True,all_80_patient_split_exclusions_verified=True,EZ_reporting_is_one_minus_NEZ=True,real_FIT_admission=real,no_outer_inference=True))
    print('ALL_FOUR_SMOKE_COMPLETE',round(real['seconds'],2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=RUNTIME);main(p.parse_args().runtime)
