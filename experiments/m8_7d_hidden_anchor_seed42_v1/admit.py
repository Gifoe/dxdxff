"""Real control replay and complete-patient gradient opening before fitting."""
import gc
from common import *
import pytest


def selected_state(m,prefixes):return {k:v for k,v in m.state_dict().items() if k.startswith(prefixes)}


class Counts:
    def __init__(self):self.collected=0;self.passed=0
    def pytest_collection_finish(self,session):self.collected=len(session.items)
    def pytest_runtest_logreport(self,report):
        if report.when=='call' and report.passed:self.passed+=1


def main():
    binding,core,fac,records,rec,banks,fr=setup();counts=Counts()
    assert pytest.main([str(HERE/'test_model.py'),'-q'],plugins=[counts])==0 and counts.collected==counts.passed==5
    inputs=store();checks=[];largest=None;seen={}
    for fold in range(1,6):
        d=bank(fold,banks);_,_,_,_,error=reference(fr,fold,core,fac,records,d,inputs,scale(fold))
        checks.append(dict(fold=fold,maximum_replay_error=error,all201_decisions_exact=True))
        for pid in sorted(set(d['patient'][d['train']])):
            ii=d['train'][d['patient'][d['train']]==pid];plan=inputs.raw.registry['plans'][str(pid)]
            if str(pid) not in seen:
                assert sha(plan['path'])==plan['sha256'];q=torch.load(plan['path'],map_location='cpu',weights_only=False)
                seen[str(pid)]=int(q['mask'][0].sum());del q
            if largest is None or seen[str(pid)]>largest['windows']:largest=dict(fold=fold,d=d,ii=ii,pid=pid,windows=seen[str(pid)])
    q=largest;fold=q['fold'];d=q['d'];ii=q['ii'];ss=scale(fold)
    raw,mask,times,delta,_=inputs.get(str(q['pid']),d['channel'][ii].astype(str).tolist(),'cuda');assert int(mask.sum())==q['windows']
    x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32);tr=d['train']
    pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda');source=legacy.template(core,fac,records,fold).eval()
    with torch.inference_mode():expected=source.features(raw,mask,times,delta,ss)[0];phys=source.base(x)
    rows=[];initial=[]
    for arm in NEW:
        m=make(core,fac,records,fold,arm);m.eval();initial.append(state_hash(m.state_dict()))
        with torch.inference_mode():
            assert torch.equal(m.normalized(x),source.base.normalized(x))
            out=m(x,raw,mask,times,delta,ss);assert torch.equal(out['raw_summary'],expected) and torch.equal(out['logit'],phys)
            pad=raw.clone();pad[~mask]=123;assert torch.equal(out['logit'],m(x,pad,mask,times,delta,ss)['logit'])
            order=torch.arange(len(x)-1,-1,-1,device='cuda')
            assert torch.allclose(out['raw_summary'][order],m(x[order],raw[:,order],mask[:,order],times,delta[:,order],ss)['raw_summary'],atol=1e-6,rtol=1e-5)
        m.stage('fixed' if arm=='H2_WARM_FIXED' else 'joint');frozen=state_hash(selected_state(m,('phys_first.','readout.')))
        m.train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0003,weight_decay=.001);norms=[]
        for step in range(3):
            opt.zero_grad(set_to_none=True);loss=MODEL.objective(m(x,raw,mask,times,delta,ss)['logit'],y,pos);loss.backward();assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
            g={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.grad is not None};assert g['raw_projection.weight']>0
            if step==0:assert g['cnn.0.weight']==0
            if step>=1:assert g['cnn.0.weight']>0 and g['gate.network.2.weight']>0
            if step>=2:assert g['gate.network.0.weight']>0
            if arm=='H2_WARM_FIXED':assert all(p.grad is None for n,p in m.named_parameters() if n.startswith(('phys_first.','readout.')))
            else:assert g['phys_first.weight']>0 and g['readout.weight']>0
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step();norms.append(g)
        if arm=='H2_WARM_FIXED':assert state_hash(selected_state(m,('phys_first.','readout.')))==frozen
        rows.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),
            channels=len(ii),valid_windows=q['windows'],original_Raw_summary_exact=True,original_physiological_normalization_exact=True,
            initial_single_head_equals_untrained_phys_only=True,all7_coordinates_retained=True,discarded_steps=3,gradient_norms=norms))
        del m,opt;gc.collect();torch.cuda.empty_cache()
    assert len(set(initial))==1
    write(PUB/'ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),tests_passed=5,reference_checks=checks,actual_FIT_gradient_checks=rows,
        all3_initial_states_exact=True,new_Outer_passes=0,formal_epochs_discarded=0))
    print('HIDDEN_ANCHOR_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
