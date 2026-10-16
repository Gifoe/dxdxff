"""All ownFIT phase caches and actual complete-patient Raw/cache admission."""
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


def main():
    binding=setup();counts=Counts();code=pytest.main([str(HERE/'test_model.py'),'-q'],plugins=[counts])
    assert code==0 and counts.collected==counts.passed==4 and counts.failed==counts.skipped==0
    cc=contexts();store=RawStore();parts={};reports=[];normalization=[];started=time.perf_counter();largest=None
    for ctx in cc:
        d=ctx['data'];model=Model('NATIVE',ctx['fold'],ctx['reference_state']).cuda().eval();native=np.empty(len(d['patient']),np.float32)
        training=np.r_[ctx['roles']['train'],ctx['roles']['choice']];training_ids=set(d['patient'][training])
        for pid in sorted(set(d['patient'])):
            ix=np.flatnonzero(d['patient']==pid);path=path_for(ctx,pid)
            pb=dict(input=binding,fold=ctx['fold'],screen=ctx['screen'],archive_indices=d['archive_indices'][ix].tolist(),
                    checkpoint_sha256=ctx['stored']['checkpoint_sha256'],waveform_sha256=store.plans[str(pid)]['sha256'])
            if path.exists():q=read(path);assert q['binding']==pb
            else:
                raw,mask,times=store.get(pid,d['channel'][ix]);cache,audit=model.encode(raw,mask,times)
                with torch.inference_mode():logit=model.forward_stats(cache).cpu().numpy()
                q=dict(binding=pb,**{k:v.cpu().numpy() for k,v in cache.items()},native_logit_nez=logit,
                       audit=audit,valid_windows=int(mask.sum()),complete_patient_channels=True,no_labels_or_manual_features_used=True)
                save(path,q)
            assert q['window'].shape==q['reference'].shape and q['window'].shape[1:]==(len(ix),2,24)
            assert q['available'].shape==q['window'].shape[:-1] and q['available'].any(0).all()
            assert np.isfinite(q['window']).all() and np.isfinite(q['reference']).all()
            native[ix]=q['native_logit_nez'];parts[str(path.relative_to(ROOT))]=sha(path)
            normalization.append(dict(fold=ctx['fold'],screen=ctx['screen'],**q['audit']))
            if pid in training_ids and (largest is None or q['valid_windows']>largest['windows']):
                largest=dict(ctx=ctx,pid=pid,indices=ix,windows=q['valid_windows'])
            write(PUB/'PREPARE_STATUS.json',dict(status='ENCODING_OWNFIT_ONLY',complete_patient_cells=len(parts),
                  total_patient_cells=765,elapsed_seconds=time.perf_counter()-started,original_Development_Outer_used=False))
        cal=ctx['roles']['cal'];reference=ctx['stored']['raw_logit_nez'];error=float(np.max(np.abs(native-reference)))
        assert error<=1e-6,('Native identity',ctx['fold'],ctx['screen'],error)
        grid=np.round(np.arange(0,1.0001,.005),3);a=probability(native[cal]);b=probability(reference[cal])
        assert np.array_equal(a[:,None]>=grid,b[:,None]>=grid)
        assert calibrate(d,cal,native[cal])['threshold']==ctx['reference_threshold']
        reports.append(dict(fold=ctx['fold'],screen=ctx['screen'],patients=len(set(d['patient'])),channels=len(d['patient']),
                            Native_all_ownFIT_logit_max_error=error,all201_CAL_grid_decisions_unchanged=True))
        print('PRECOMPAND_PATH_ENCODED',ctx['fold'],ctx['screen'],len(parts),'of765',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(parts)==765
    ctx=largest['ctx'];d=ctx['data'];ix=largest['indices'];q=read(path_for(ctx,largest['pid']))
    raw,mask,times=store.get(largest['pid'],d['channel'][ix]);cache=payload(q)
    y=torch.tensor(d['y'][ix],device='cuda',dtype=torch.float32);tr=np.r_[ctx['roles']['train'],ctx['roles']['choice']]
    pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda');actual=[]
    for arm in TRAINED:
        a=Model(arm,ctx['fold'],ctx['reference_state']).cuda();b=copy.deepcopy(a)
        with torch.no_grad():a.matrix.copy_(torch.linspace(-.01,.01,576,device='cuda').reshape(24,24));b.matrix.copy_(a.matrix)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        one=a(raw,mask,times);two=b.forward_stats(cache);loss=objective(one,y,pos);other=objective(two,y,pos)
        loss.backward();other.backward();torch.cuda.synchronize()
        assert torch.equal(one,two) and float(loss.detach())==float(other.detach())
        ge=float((a.matrix.grad-b.matrix.grad).abs().max());assert torch.allclose(a.matrix.grad,b.matrix.grad,atol=1e-6,rtol=1e-5)
        assert torch.isfinite(a.matrix.grad).all() and a.matrix.grad.abs().sum()>0
        assert all(p.grad is None for p in a.cnn.parameters()) and all(p.grad is None for p in a.head.parameters())
        actual.append(dict(arm=arm,channels=len(ix),valid_windows=largest['windows'],Raw_cache_logit_loss_error=0.,
                           matrix_gradient_max_error=ge,all_trainable_gradients_finite=True,body_head_frozen=True,
                           seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        del a,b,one,two,loss,other;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(normalization).groupby(['fold','screen']).sum().reset_index().to_csv(PUB/'NORMALIZATION_COUNTS.csv',index=False)
    freeze(ROOT/'PHASE_VIEWS_SEAL_PRIVATE.json',dict(input=binding,parts=parts))
    write(PUB/'ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),tests_passed=counts.passed,
          ownFIT_patient_cells=765,Raw_cache_actual_complete_patient=actual,reports=reports,
          phase_seal_sha256=sha(ROOT/'PHASE_VIEWS_SEAL_PRIVATE.json'),body_head_updates=0,
          normalization_counts_include_repeated_encodings=True,original_Development_Outer_used=False))
    print('PRECOMPAND_ALL765_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
