"""Actual full-patient admission and every reused control checkpoint replay."""
import gc
import time
from common import *
import pytest
from model import Model,ExtendedFirstConv,collapsed_state,objective


def evaluate(model,d,ix,store):
    model.eval();out=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            j=np.flatnonzero(d['patient'][ix]==pid);ii=ix[j]
            raw,mask,times=store.get(pid,d['channel'][ii]);out[j]=model(raw,mask,times).cpu().numpy()
    assert np.isfinite(out).all();return out


def replay_reuse(ctx,store):
    cell=ctx['reused_short_mean'];assert cell is not None
    q=cell['q'];state=torch.load(cell['bp'],map_location='cpu',weights_only=False)
    assert state['binding']==q['binding'] and q['binding']['arm']=='FRESH_BASE' and state['epoch']>=1
    m=Model('SHORT_MEAN',ctx['fold']).cuda();assert state_hash(m.state_dict())==q['binding']['initial_state_sha256']
    m.load_state_dict(state['model']);d,rr=ctx['data'],ctx['roles']
    cl=evaluate(m,d,rr['cal'],store);hh=evaluate(m,d,rr['held'],store)
    error=max(float(np.max(np.abs(cl-q['cal_logit_nez']))),float(np.max(np.abs(hh-q['logit_nez']))));assert error<=1e-6
    grid=np.round(np.arange(0,1.0001,.005),3)
    assert np.array_equal(probability(cl)[:,None]<grid,probability(q['cal_logit_nez'])[:,None]<grid)
    assert calibrate(d,rr['cal'],cl)==state['selection']
    del m;gc.collect();torch.cuda.empty_cache();return cl,hh,error


class Counts:
    def __init__(self):self.collected=0;self.passed=0;self.failed=0;self.skipped=0
    def pytest_collection_finish(self,session):self.collected=len(session.items)
    def pytest_runtest_logreport(self,report):
        if report.when=='call':self.passed+=int(report.passed);self.failed+=int(report.failed);self.skipped+=int(report.skipped)


def main():
    binding=setup();counts=Counts();code=pytest.main([str(HERE/'test_model.py'),'-q'],plugins=[counts])
    assert code==0 and counts.collected==counts.passed==6 and counts.failed==counts.skipped==0
    cc=contexts();store=RawStore();initial=[];reuse=[];phase=jr(previous.WAVE/'PHASE_VIEWS_SEAL_PRIVATE.json');largest=None;training=set()
    for fold in range(1,6):
        hashes=[state_hash(collapsed_state(Model(a,fold))) for a in FAMILY];assert len(set(hashes))==1
        initial.append(dict(fold=fold,all4_collapsed_random_core_states_exact=True,shared_core_sha256=hashes[0],random_start=True))
    for ctx in cc:
        if ctx['reused_short_mean'] is not None:
            _,_,error=replay_reuse(ctx,store);reuse.append(dict(fold=ctx['fold'],screen=ctx['screen'],maximum_replay_error=error,all201_CAL_decisions_exact=True))
        d=ctx['data'];tr=np.r_[ctx['roles']['train'],ctx['roles']['choice']]
        for pid in sorted(set(d['patient'][tr])):
            ii=tr[d['patient'][tr]==pid]
            path=previous.WAVE/'phase_views_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/(io.private_key(pid)+'.pkl')
            assert sha(path)==phase['parts'][str(path.relative_to(previous.WAVE))];q=read(path)
            assert q['binding']['archive_indices']==d['archive_indices'][ii].tolist();training.add(str(pid))
            if largest is None or q['valid_windows']>largest['windows']:largest=dict(ctx=ctx,ii=ii,pid=pid,windows=q['valid_windows'])
    assert len(reuse)==7
    ctx=largest['ctx'];d=ctx['data'];ii=largest['ii'];tr=np.r_[ctx['roles']['train'],ctx['roles']['choice']]
    raw,mask,times=store.get(largest['pid'],d['channel'][ii]);assert int(mask.sum())==largest['windows']
    y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32);pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda')
    records=[];short_outputs={};short_grads={}
    for arm in FAMILY:
        m=Model(arm,ctx['fold']).cuda();before=state_hash(m.state_dict());torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();tick=time.perf_counter();logit=m(raw,mask,times);loss=objective(logit,y,pos);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert m.cnn[0].weight.grad.abs().sum()>0 and m.head[2].weight.grad.abs().sum()>0 and state_hash(m.state_dict())==before
        shared={k:p.grad.detach().clone() for k,p in m.named_parameters() if not k.startswith('cnn.0.')}
        first=m.cnn[0].center if isinstance(m.cnn[0],ExtendedFirstConv) else m.cnn[0]
        shared.update({'cnn.0.weight':first.weight.grad.detach().clone(),'cnn.0.bias':first.bias.grad.detach().clone()})
        error=0.
        if arm.startswith('SHORT'):
            short_outputs[arm]=logit.detach().clone();short_grads[arm]=shared
        else:
            short=arm.replace('LONG','SHORT');assert torch.equal(logit.detach(),short_outputs[short])
            for name,value in shared.items():
                base=short_grads[short][name];error=max(error,float((value-base).abs().max()))
                assert torch.allclose(value,base,atol=1e-6,rtol=1e-5)
            assert m.cnn[0].outer.grad.abs().sum()>0
        records.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),channels=len(ii),valid_windows=largest['windows'],
            largest_of_unique_training_patients=len(training),seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            all_parameter_gradients_finite=True,backbone_head_gradients_nonzero=True,optimizer_updates=0,
            matching_pool_pair_initial_logits_exact=True,maximum_shared_pair_gradient_error=error,
            outer_filter_gradient_nonzero=(True if arm.startswith('LONG') else None)))
        print('EARLY_FIR_ACTUAL_PATIENT_ADMITTED',arm,records[-1],flush=True)
        del m,logit,loss;gc.collect();torch.cuda.empty_cache()
    write(PUB/'ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),tests_passed=6,initial_states=initial,
        reused_control_checks=reuse,reused_control_cells=len(reuse),maximum_reused_control_error=max(r['maximum_replay_error'] for r in reuse),
        largest_complete_patient_backward=records,all15_role_identities_verified=True,original_Development_Outer_used=False,formal_updates=0))
    print('EARLY_FIR_ALL_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
