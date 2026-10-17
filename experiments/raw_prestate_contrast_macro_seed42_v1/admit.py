"""Actual complete legal training-patient backward admission; zero updates."""
import gc
import time
from common import *
import pytest
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
    binding=setup();counts=Counts();code=pytest.main([str(HERE/'test_model.py'),'-q'],plugins=[counts])
    assert code==0 and counts.collected==counts.passed==5 and counts.failed==counts.skipped==0
    cc=contexts();initial=[];store=RawStore();phase=jr(previous.WAVE/'PHASE_VIEWS_SEAL_PRIVATE.json');largest=None;training=set()
    for fold in range(1,6):
        hashes=[state_hash(Model(a,fold).state_dict()) for a in TRAINED];assert len(set(hashes))==1
        initial.append(dict(fold=fold,all3_initial_states_exact=True,initial_state_sha256=hashes[0],random_start=True))
    for ctx in cc:
        d=ctx['data'];tr=np.r_[ctx['roles']['train'],ctx['roles']['choice']]
        for pid in sorted(set(d['patient'][tr])):
            ii=tr[d['patient'][tr]==pid]
            path=previous.WAVE/'phase_views_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/(io.private_key(pid)+'.pkl')
            assert sha(path)==phase['parts'][str(path.relative_to(previous.WAVE))];q=read(path)
            assert q['binding']['archive_indices']==d['archive_indices'][ii].tolist();training.add(str(pid))
            if largest is None or q['valid_windows']>largest['windows']:largest=dict(ctx=ctx,ii=ii,pid=pid,windows=q['valid_windows'])
    ctx=largest['ctx'];d=ctx['data'];ii=largest['ii'];tr=np.r_[ctx['roles']['train'],ctx['roles']['choice']]
    raw,mask,times=store.get(largest['pid'],d['channel'][ii]);assert int(mask.sum())==largest['windows']
    y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda');records=[]
    for arm in TRAINED:
        model=Model(arm,ctx['fold']).cuda();before=state_hash(model.state_dict())
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        logit=model(raw,mask,times);loss=objective(logit,y,pos);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.cnn[0].weight.grad.abs().sum()>0 and model.head[2].weight.grad.abs().sum()>0
        assert state_hash(model.state_dict())==before
        records.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),channels=len(ii),valid_windows=largest['windows'],
            largest_of_unique_training_patients=len(training),seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            all_parameter_gradients_finite=True,backbone_head_gradients_nonzero=True,optimizer_updates=0))
        print('PRESTATE_RAW_ACTUAL_PATIENT_ADMITTED',arm,records[-1],flush=True)
        del model,logit,loss;gc.collect();torch.cuda.empty_cache()
    write(PUB/'ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),tests_passed=5,initial_states=initial,
        largest_complete_patient_backward=records,all15_role_identities_verified=True,strong_firstpath_reference_verified_all5folds=True,
        original_Development_Outer_used=False,formal_updates=0))
    print('PRESTATE_RAW_ALL_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
