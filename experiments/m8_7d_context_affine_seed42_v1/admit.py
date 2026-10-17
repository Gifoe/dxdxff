"""Initial exact C2 replay, full-model/cache parity and context-only gradients."""
from common import *
import pytest


class Counts:
    def __init__(self):self.collected=0;self.passed=0
    def pytest_collection_finish(self,session):self.collected=len(session.items)
    def pytest_runtest_logreport(self,report):
        if report.when=='call' and report.passed:self.passed+=1


def main():
    binding,core,fac,records,rec,banks,fr=setup();count=Counts()
    assert pytest.main([str(HERE/'test_model.py'),'-q'],plugins=[count])==0 and count.collected==count.passed==5
    assert jr(PUB/'REPRESENTATION_STATUS.json')['status']=='PASS'
    inputs=store();checks=[];largest=None;seen={}
    for fold in range(1,6):
        d=bank(fold,banks);q=cached(fold,d);m,r,state,old=body(fr,fold,core,fac,records)
        for arm in NEW:
            head=make(arm,m.head[2],fold);out=predict(head,q['hidden'],d,d['val']);error=float(np.max(np.abs(out-old['logit'])));assert error<=1e-6
            grid=np.round(np.arange(0,1.0001,.005),3);assert np.array_equal(probability(out)[:,None]<grid,probability(old['logit'])[:,None]<grid)
            checks.append(dict(arm=arm,fold=fold,initial_replay_error=error,all201_decisions_exact=True))
        for pid in sorted(set(d['patient'][d['train']])):
            ii=d['train'][d['patient'][d['train']]==pid];plan=inputs.raw.registry['plans'][str(pid)]
            if str(pid) not in seen:
                assert sha(plan['path'])==plan['sha256'];src=torch.load(plan['path'],map_location='cpu',weights_only=False);seen[str(pid)]=int(src['mask'][0].sum());del src
            if largest is None or seen[str(pid)]>largest['windows']:largest=dict(fold=fold,d=d,ii=ii,pid=pid,windows=seen[str(pid)])
    q=largest;fold=q['fold'];d=q['d'];ii=q['ii'];ss=scale(fold);bankq=cached(fold,d)
    m,_,_,_=body(fr,fold,core,fac,records);raw,mask,times,delta,_=inputs.get(str(q['pid']),d['channel'][ii].astype(str).tolist(),'cuda')
    x=torch.as_tensor(d['x'][ii],device='cuda');h=torch.as_tensor(bankq['hidden'][ii],device='cuda');y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    pos=torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()),device='cuda');records=[]
    for arm in NEW:
        head=make(arm,m.head[2],fold);full=MODEL.Fused(m,head).cuda().eval()
        with torch.inference_mode():
            out=full(x,raw,mask,times,delta,ss);assert torch.equal(out['logit'],head(h)) and torch.equal(out['hidden'],h)
            order=torch.arange(len(x)-1,-1,-1,device='cuda');assert torch.allclose(out['logit'][order],full(x[order],raw[:,order],mask[:,order],times,delta[:,order],ss)['logit'],atol=1e-6,rtol=1e-5)
        before=state_hash(head.readout.state_dict());params=[p for p in head.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.0003,weight_decay=.001);norms=[]
        for step in range(2):
            opt.zero_grad(set_to_none=True);loss=MODEL.objective(head(h),y,pos);loss.backward();assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params) and all(p.grad is None for p in head.readout.parameters())
            g={n:float(p.grad.norm()) for n,p in head.named_parameters() if p.grad is not None}
            if arm=='G1_PATIENT':
                assert g['context.2.weight']>0
                if step:assert g['context.0.weight']>0
            else:assert g['offset']>0
            torch.nn.utils.clip_grad_norm_(params,1.);opt.step();norms.append(g)
        assert state_hash(head.readout.state_dict())==before
        records.append(dict(arm=arm,full_parameters=sum(p.numel() for p in full.parameters()),trainable_parameters=sum(p.numel() for p in params),channels=len(ii),valid_windows=q['windows'],
            exact_full_model_cached_initial_logits=True,full_model_has_one_scalar_readout=True,readout_unchanged=True,discarded_steps=2,gradient_norms=norms))
    write(PUB/'ADMISSION.json',dict(status='PASS',tests_passed=5,input_binding_sha256=digest(binding),initial_checks=checks,actual_FIT_checks=records,formal_epochs_discarded=0,new_Outer_passes=0))
    print('CONTEXT_AFFINE_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
