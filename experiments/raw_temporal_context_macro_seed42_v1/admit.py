"""Actual full-patient memory, gradient and pre-body replay admission."""
import gc
import time
import pytest
from common import *
from model import Model,objective


def evaluate(model,d,ix,store):
    model.eval();out=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            loc=np.flatnonzero(d['patient'][ix]==pid);ii=ix[loc]
            raw,mask,times=store.get(pid,d['channel'][ii]);out[loc]=model(raw,mask,times).cpu().numpy()
    assert np.isfinite(out).all();return out


class Counts:
    def __init__(self):self.collected=0;self.passed=0;self.failed=0;self.skipped=0
    def pytest_collection_finish(self,session):self.collected=len(session.items)
    def pytest_runtest_logreport(self,report):
        if report.when=='call':self.passed+=int(report.passed);self.failed+=int(report.failed);self.skipped+=int(report.skipped)


def main():
    binding=setup();count=Counts();code=pytest.main([str(HERE/'test_model.py'),'-q'],plugins=[count])
    assert code==0 and count.collected==count.passed==4 and count.failed==count.skipped==0
    cc=contexts();store=RawStore();training={};phase=jr(WAVE/'PHASE_VIEWS_SEAL_PRIVATE.json');largest=None
    for ctx in cc:
        d=ctx['data'];tr=np.r_[ctx['roles']['train'],ctx['roles']['choice']]
        assert ctx['pre_result']['binding']['phase_seal_sha256']==sha(WAVE/'PHASE_VIEWS_SEAL_PRIVATE.json')
        for pid in sorted(set(d['patient'][tr])):
            ii=tr[d['patient'][tr]==pid];p=wave.path_for(ctx,pid);assert sha(p)==phase['parts'][str(p.relative_to(WAVE))]
            q=read(p);assert q['binding']['archive_indices']==d['archive_indices'][ii].tolist()
            training[str(pid)]=True
            if largest is None or q['valid_windows']>largest['windows']:largest=dict(ctx=ctx,ii=ii,pid=pid,windows=q['valid_windows'])
    ctx=largest['ctx'];d=ctx['data'];ii=largest['ii'];tr=np.r_[ctx['roles']['train'],ctx['roles']['choice']]
    raw,mask,times=store.get(largest['pid'],d['channel'][ii]);assert int(mask.sum())==largest['windows']
    y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32);pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda')
    records=[];reference=None;shared=None
    for arm in TRAINED:
        m=Model(arm,ctx['fold'],ctx['reference_state'],ctx['matrix']).cuda();m.train()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        score=m(raw,mask,times);loss=objective(score,y,pos);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        base={n:p.grad.detach().clone() for n,p in m.named_parameters() if not n.startswith('cnn.4.')}
        if reference is None:reference=score.detach().clone();shared=base
        else:
            assert torch.equal(score,reference)
            for n,v in base.items():assert torch.allclose(v,shared[n],atol=1e-6,rtol=1e-5)
        records.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),complete_channels=len(ii),valid_windows=largest['windows'],
            largest_of_unique_training_patients=len(training),seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            initial_logits_exact=True,shared_gradient_gate_pass=True,all_parameter_gradients_finite=True))
        print('RAW_TEMPORAL_FULL_PATIENT_ADMITTED',arm,records[-1],flush=True)
        del m,score,loss,base;gc.collect();torch.cuda.empty_cache()
    del raw,mask,times,reference,shared;replays=[]
    for ctx in cc:
        d=ctx['data'];ix=ctx['roles']['cal'];m=Model('PRE_TUNE',ctx['fold'],ctx['reference_state'],ctx['matrix']).cuda()
        new=evaluate(m,d,ix,store);old=ctx['pre_result']['cal_logit_nez'];a,b=probability(new),probability(old)
        grid=np.round(np.arange(0,1.0001,.005),3);error=float(np.max(np.abs(a-b)))
        assert error<=1e-6 and np.array_equal(a[:,None]>=grid,b[:,None]>=grid)
        assert calibrate(d,ix,new)['threshold']==ctx['pre_result']['summary']['threshold']
        replays.append(dict(fold=ctx['fold'],screen=ctx['screen'],probability_max_error=error,logit_max_error=float(np.max(np.abs(new-old))),all201_grid_decisions_exact=True))
        del m;gc.collect();torch.cuda.empty_cache()
    write(PUB/'ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),tests_passed=4,complete_patient_backward=records,
        all15_initial_CAL_replays=replays,formal_fits_started=0,original_Development_Outer_used=False))
    print('RAW_TEMPORAL_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
