"""Concurrent source-wrapper parity and complete genuine-TRAIN critical paths."""
import time
import unittest
from common import *

def equal_tree(a,b):
    """Optimizer state is nested, unlike the original flat model-tensor hash API."""
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b

def main():
    import test_model
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_model));assert tests.wasSuccessful()
    binding,core,fac,records,rec,banks=setup(False);inputs=store();bridges=[];roles=[]
    for fold in range(1,6):
        d,scale=data(fold,core,rec,banks,inputs,binding)
        roles.append(dict(fold=fold,train_patients=len(d['roles']['train']),cal_patients=len(d['roles']['cal']),assessment_patients=len(d['roles']['assessment']),
                          train_channels=len(d['train']),cal_channels=len(d['val']),assessment_channels=len(d['assessment'])))
        for seed in SEEDS:
            base=parent.make(core,fac,records,fold,'FULL',seed);full=make(core,fac,records,fold,'FULL',seed)
            assert state_hash(base.state_dict())==state_hash(full.state_dict())
            r=parent.previous_cell(seed,fold);v=read(r['calibration']);s=torch.load(r['checkpoint'],weights_only=False,map_location='cpu')
            state=full.state_dict()
            for key,value in s['model'].items():state[key]=value
            full.load_state_dict(state);out,_=predict(full,d,d['val'],inputs,scale)
            assert np.array_equal(out,v['logit']) and np.array_equal(d['archive_indices'][d['val']],v['archive_indices'])
            bridges.append(dict(seed=seed,fold=fold,initial_parent_state_exact=True,stored_isolated_zero_gain_CAL_bit_exact=True))
            del base,full;gc.collect();torch.cuda.empty_cache()
        print('CORE_PARENT_INITIAL_AND_STORED_BRIDGE_PASS',fold,flush=True)
    d,scale=data(1,core,rec,banks,inputs,binding);counts=[]
    for pid in sorted(d['roles']['train']):
        ii=d['train'][d['patient'][d['train']]==pid];raw,mask,_,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cpu')
        counts.append((int(mask.sum()),str(pid)));del raw,delta
    _,pid=max(counts);ii=d['train'][d['patient'][d['train']]==pid];tr=d['train']
    y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda');pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
    a=parent.make(core,fac,records,1,'FULL',42);b=make(core,fac,records,1,'FULL',42)
    oa=torch.optim.AdamW(a.parameters(),lr=.0003,weight_decay=.001);ob=torch.optim.AdamW(b.parameters(),lr=.0003,weight_decay=.001)
    a.train();b.train();out_a,_=parent.forward(a,d,ii,inputs,scale);out_b,_=forward(b,d,ii,inputs,scale)
    la=parent.MODEL.objective(out_a,y,pos);lb=parent.MODEL.objective(out_b,y,pos);assert torch.equal(out_a,out_b) and torch.equal(la,lb)
    la.backward();lb.backward();pa,pb=dict(a.named_parameters()),dict(b.named_parameters());assert set(pa)==set(pb)
    assert all(pa[n].grad is not None and torch.equal(pa[n].grad,pb[n].grad) for n in pa)
    for m,o in [(a,oa),(b,ob)]:torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
    assert state_hash(a.state_dict())==state_hash(b.state_dict()) and equal_tree(oa.state_dict(),ob.state_dict())
    del a,b,oa,ob,la,lb,out_a,out_b;gc.collect();torch.cuda.empty_cache();audits=[]
    for arm in NEW:
        m=make(core,fac,records,1,arm,42);tick=time.perf_counter();out,stats=forward(m,d,ii,inputs,scale)
        loss=parent.MODEL.objective(out,y,pos);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        audits.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),channels=len(ii),seconds=time.perf_counter()-tick,
                           complete_patient_backward_finite=True,no_formal_update=True,**stats))
        del m,loss,out;gc.collect();torch.cuda.empty_cache();print('CORE_COMPLETE_TRAIN_GRADIENT_PASS',arm,flush=True)
    csv_out('ADMISSION_BRIDGES',bridges);csv_out('ROLE_AUDIT',roles)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    write(PUB/'ADMISSION.json',dict(status='PASS',source=sources(),parent_source=parent.sources(),synthetic_tests=tests.testsRun,initial_bridges=15,
                                   full_logits_loss_all_gradients_oneAdam_exact=True,maximum_wrapper_gradient_error=0,discarded_parity_updates=2,
                                   paths=audits,no_new_Outer=True,no_assessment_used=True))
    print('CORE_CONCURRENT_NUMERIC_AND_PATH_ADMISSION_PASS',flush=True)

if __name__=='__main__':main()
