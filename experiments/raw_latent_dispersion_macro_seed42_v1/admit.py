"""Full actual-patient memory/gradient and frozen Native calibration admission."""
import copy
import gc
import time
import pytest
from common import *
from model import Model,TRAINED,objective


class Counts:
    def __init__(self):self.collected=0;self.passed=0;self.failed=0;self.skipped=0
    def pytest_collection_finish(self,session):self.collected=len(session.items)
    def pytest_runtest_logreport(self,report):
        if report.when=='call':
            self.passed+=int(report.passed);self.failed+=int(report.failed);self.skipped+=int(report.skipped)


def evaluate(model,d,ix,store):
    model.eval();result=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            loc=np.flatnonzero(d['patient'][ix]==pid);ii=ix[loc]
            raw,mask,times=store.get(pid,d['channel'][ii]);result[loc]=model(raw,mask,times).cpu().numpy()
    assert np.isfinite(result).all();return result


def main():
    binding=setup();count=Counts();code=pytest.main([str(HERE/'test_model.py'),'-q'],plugins=[count])
    assert code==0 and count.collected==count.passed==4 and count.failed==count.skipped==0
    cc=contexts();store=RawStore();training={}
    for ctx in cc:
        d=ctx['data'];tr=np.r_[ctx['roles']['train'],ctx['roles']['choice']]
        for pid in sorted(set(d['patient'][tr])):
            if pid not in training:training[pid]=(ctx,tr[d['patient'][tr]==pid])
    # Largest complete patient in the union of actual training roles, not a
    # convenient short example and not a patient selected from label outcomes.
    sizes=[]
    for pid,(ctx,ii) in sorted(training.items()):
        d=ctx['data'];raw,mask,times=store.get(pid,d['channel'][ii],device='cpu')
        sizes.append((int(mask.sum()),str(pid)))
    windows,pid=max(sizes,key=lambda r:(r[0],r[1]));first,ii=training[pid]
    d=first['data'];rr=first['roles'];tr=np.r_[rr['train'],rr['choice']]
    raw,mask,times=store.get(pid,d['channel'][ii])
    y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda')
    records=[];reference_score=None;shared_grad=None
    for arm in TRAINED:
        model=Model(arm,first['fold'],first['reference_state']).cuda();model.train()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        score=model(raw,mask,times);loss=objective(score,y,pos);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        base={n:p.grad.detach().clone() for n,p in model.named_parameters() if not n.startswith('cnn.4.')}
        if reference_score is None:reference_score=score.detach().clone();shared_grad=base
        else:
            assert torch.equal(score,reference_score)
            for name,v in base.items():assert torch.allclose(v,shared_grad[name],atol=1e-6,rtol=1e-5)
        records.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),
                            actual_complete_patient=True,largest_of_training_patients=len(training),
                            fold=first['fold'],screen=first['screen'],channels=len(ii),valid_windows=windows,
                            direct_forward_backward_seconds=time.perf_counter()-tick,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            all_gradients_finite=True,initial_logit_identity=True))
        print('DISPERSION_FULL_PATIENT_ADMISSION',arm,records[-1],flush=True)
        del model,score,loss,base;gc.collect();torch.cuda.empty_cache()
    del raw,mask,times,reference_score,shared_grad
    replay=[]
    for ctx in cc:
        d=ctx['data'];cal=ctx['roles']['cal'];model=Model('MEAN_TUNE',ctx['fold'],ctx['reference_state']).cuda().eval()
        new=evaluate(model,d,cal,store);old=ctx['stored']['raw_logit_nez'][cal]
        a=probability(new);b=probability(old);grid=np.round(np.arange(0,1.0001,.005),3)
        pe=float(np.max(np.abs(a-b)));le=float(np.max(np.abs(new-old)))
        assert pe<=1e-6 and np.array_equal(a[:,None]>=grid,b[:,None]>=grid)
        assert calibrate(d,cal,new)['threshold']==ctx['reference_threshold']
        replay.append(dict(fold=ctx['fold'],screen=ctx['screen'],probability_max_error=pe,
                           logit_drift_recorded_not_claimed_bit_equivalent=le,all201_grid_decisions_identical=True))
        del model;gc.collect();torch.cuda.empty_cache()
    write(PUB/'ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),tests_collected=count.collected,
          tests_passed=count.passed,complete_patient_direct_backward=records,calibration_replays=replay,
          all15_role_source_checkpoint_identities_verified=True,precision='FP32_TF32_OFF_CUDNN_DETERMINISTIC',
          formal_fits_started=0,new_Outer_access=False))
    print('DISPERSION_ALL_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
