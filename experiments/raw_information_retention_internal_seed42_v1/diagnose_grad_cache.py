"""Outcome-free diagnosis of the failed long-CNN gradient admission."""
import copy
from common import *
from model import Model,pooling
from engine import direct,rng,restore,scatter,CHUNK


def cached_ordered(model,raw,mask,times,y,pos,reverse):
    flat=model.normalized_windows(raw,mask,times);chunks=list(flat.split(CHUNK));states=[];parts=[]
    with torch.no_grad():
        for q in chunks:states.append(rng());parts.append(model.cnn(q[:,None]).flatten(1))
    h=torch.cat(parts).detach().requires_grad_(True)
    z=model.aggregate(scatter(h,mask),mask,times);score=model.classify(z)
    loss=pooling.objective(score,y,pos,'ez_rank');loss.backward();gradient=h.grad.detach()
    after=rng();starts=np.r_[0,np.cumsum([len(c) for c in chunks])]
    order=list(reversed(range(len(chunks)))) if reverse else list(range(len(chunks)))
    try:
        for j in order:
            restore(states[j]);q=model.cnn(chunks[j][:,None]).flatten(1)
            q.backward(gradient[int(starts[j]):int(starts[j+1])])
    finally:restore(after)
    return score.detach(),float(loss.detach().cpu())


def compare(a,b):
    maximum=0.;bad=[]
    for (name,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==nn and (p.grad is None)==(q.grad is None)
        if p.grad is None:continue
        error=float((p.grad-q.grad).abs().max().cpu());maximum=max(maximum,error)
        if not torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5):bad.append(dict(parameter=name,error=error))
    return dict(maximum_gradient_error=maximum,failed_parameters=bad,original_gate_passed=not bad)


def main():
    binding=setup()
    actual={
        'reconstruction':sha(io.BASE/'RECONSTRUCTION_PRIVATE.pkl'),
        'raw_registry':sha(io.BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),
        'inner_registry':sha(io.OLD/'ENCODER_REGISTRY_PRIVATE.pkl')}
    assert actual==dict(reconstruction=io.EXPECTED_REC,raw_registry=io.EXPECTED_REG,inner_registry=io.EXPECTED_ENCODERS)
    banks=legal_fit_banks(True);paths=role_paths(banks);d=banks[0];rr=paths[(1,1)]['roles'];tr=np.r_[rr['train'],rr['choice']]
    pid=sorted(set(d['patient'][tr]))[0];ii=tr[d['patient'][tr]==pid]
    raw,mask,times=RawStore().get(pid,d['channel'][ii]);y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    pos=torch.tensor(float((1-d['y'][tr]).sum())/max(float(d['y'][tr].sum()),1),device='cuda')
    results=[]
    for deterministic in [False,True]:
        torch.backends.cudnn.deterministic=deterministic
        for arm in ['C2_WINDOW_LONG','C4_PRE_LONG_CONTEXT']:
            a=Model(arm,1).cuda();state=rng();trace=[];counter=[0]
            def forward_hook(module,args,out):
                j=counter[0];counter[0]+=1
                out.register_hook(lambda grad,j=j:trace.append(j))
            hook=a.cnn.register_forward_hook(forward_hook)
            loss,score,z=direct(a,raw,mask,times,y,pos);loss.backward();hook.remove()
            row=dict(arm=arm,cuDNN_deterministic=deterministic,direct_chunk_backward_order=trace)
            b=Model(arm,1).cuda();restore(state);l,ss,zz=direct(b,raw,mask,times,y,pos);l.backward()
            row['direct_self_replay']=compare(a,b);row['direct_self_logit_error']=float((score-ss).abs().max().cpu());del b,l,ss,zz
            for reverse in [False,True]:
                b=Model(arm,1).cuda();restore(state);ss,value=cached_ordered(b,raw,mask,times,y,pos,reverse)
                row['reverse' if reverse else 'forward']=dict(**compare(a,b),logit_error=float((score-ss).abs().max().cpu()),loss_error=abs(float(loss.detach().cpu())-value))
                del b,ss
            results.append(row);print('CACHE_DIAGNOSIS_CASE',arm,deterministic,row['forward']['original_gate_passed'],row['reverse']['original_gate_passed'],flush=True)
            del a,loss,score,z;__import__('gc').collect();torch.cuda.empty_cache()
    write(PUB/'FAILED_GATE_DIAGNOSIS.json',dict(status='DIAGNOSIS_ONLY_NO_TRAINING',input_binding_sha256=digest(binding),
        actual_source_hashes_verified=actual,original_gradient_tolerances_unchanged=True,formal_epochs=0,results=results))


if __name__=='__main__':main()
