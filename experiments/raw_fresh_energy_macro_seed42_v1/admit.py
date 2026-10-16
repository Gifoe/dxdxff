"""Outcome-free actual complete-patient admission for four fresh Raw networks."""
import gc
import time
import pytest
from common import *
from model import Model,objective


def evaluate(model,d,ix,store):
    model.eval();out=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            j=np.flatnonzero(d['patient'][ix]==pid);ii=ix[j]
            raw,mask,times=store.get(pid,d['channel'][ii]);out[j]=model(raw,mask,times).cpu().numpy()
    assert np.isfinite(out).all();return out


class Counts:
    def __init__(self):self.collected=0;self.passed=0;self.failed=0;self.skipped=0
    def pytest_collection_finish(self,session):self.collected=len(session.items)
    def pytest_runtest_logreport(self,report):
        if report.when=='call':self.passed+=int(report.passed);self.failed+=int(report.failed);self.skipped+=int(report.skipped)


def main():
    binding=setup();count=Counts();code=pytest.main([str(HERE/'test_model.py'),'-q'],plugins=[count])
    assert code==0 and count.collected==count.passed==4 and count.failed==count.skipped==0
    cc=contexts();store=RawStore();phase=jr(WAVE/'PHASE_VIEWS_SEAL_PRIVATE.json');largest=None;training=set();initial=[]
    for fold in range(1,6):
        hashes=[state_hash(Model(a,fold).state_dict()) for a in TRAINED];assert len(set(hashes))==1
        for ctx in [q for q in cc if q['fold']==fold]:
            assert sha(ctx['reference_checkpoint'])==ctx['reference_checkpoint_sha256']
            old=torch.load(ctx['reference_checkpoint'],map_location='cpu',weights_only=False)['model']
            assert hashes[0]!=state_hash({n:v for n,v in old.items() if n!='keep32'})
        initial.append(dict(fold=fold,shared_initial_sha256=hashes[0],allfour_exact=True,no_trained_reference_state_loaded=True))
    for ctx in cc:
        d=ctx['data'];tr=np.r_[ctx['roles']['train'],ctx['roles']['choice']]
        for pid in sorted(set(d['patient'][tr])):
            ii=tr[d['patient'][tr]==pid];p=WAVE/'phase_views_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/(io.private_key(pid)+'.pkl')
            assert sha(p)==phase['parts'][str(p.relative_to(WAVE))];q=read(p)
            assert q['binding']['archive_indices']==d['archive_indices'][ii].tolist();training.add(str(pid))
            if largest is None or q['valid_windows']>largest['windows']:largest=dict(ctx=ctx,ii=ii,pid=pid,windows=q['valid_windows'])
    ctx=largest['ctx'];d=ctx['data'];ii=largest['ii'];tr=np.r_[ctx['roles']['train'],ctx['roles']['choice']]
    raw,mask,times=store.get(largest['pid'],d['channel'][ii]);assert int(mask.sum())==largest['windows']
    y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32);pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda');records=[]
    for arm in TRAINED:
        model=Model(arm,ctx['fold']).cuda();before=state_hash(model.state_dict())
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        logit=model(raw,mask,times);loss=objective(logit,y,pos);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.cnn[0].weight.grad.abs().sum()>0 and model.head[2].weight.grad.abs().sum()>0
        assert state_hash(model.state_dict())==before
        records.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),channels=len(ii),valid_windows=largest['windows'],
            largest_of_unique_training_patients=len(training),seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            full_parameter_gradients_finite=True,backbone_head_gradients_nonzero=True,discarded_optimizer_updates=0))
        print('FRESH_RAW_COMPLETE_PATIENT_ADMITTED',arm,records[-1],flush=True)
        del model,logit,loss;gc.collect();torch.cuda.empty_cache()
    write(PUB/'ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),tests_passed=4,initial_states=initial,
        largest_complete_patient_backward=records,all15_role_identities_verified=True,random_start_only=True,
        formal_fits_started=0,original_Development_Outer_used=False))
    print('FRESH_RAW_ALL_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
