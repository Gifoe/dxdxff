"""Exact original/pruned bridges, new TRAIN-only anchors and genuine gradients."""
import time
import unittest
from common import *


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def main():
    import test_model
    test=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_model));assert test.wasSuccessful()
    binding,core,fac,records,rec,banks=setup(False);inputs=store();bridges=[];anchors=[];roles=[]
    for fold in range(1,6):
        d,scale=data(fold,core,rec,banks,inputs,binding)
        roles.append(dict(fold=fold,train_patients=len(d['roles']['train']),cal_patients=len(d['roles']['cal']),assessment_patients=len(d['roles']['assessment']),
                          train_channels=len(d['train']),cal_channels=len(d['val']),assessment_channels=len(d['assessment'])))
        for seed in SEEDS:
            anchor=fit_anchor(fold,seed,d,scale,inputs,binding,core,fac,records)
            anchors.append(dict(fold=fold,seed=seed,anchor_sha256=sha(anchor_path(fold,seed)),TRAIN_patients=anchor['patients'],TRAIN_channels=anchor['channels'],
                                TRAIN_valid_windows=anchor['valid_windows'],constant_coordinates=anchor['constant_coordinates'],initial_only_not_current_reference=True))
            r=previous_cell(seed,fold);v=read(r['calibration']);state=torch.load(r['checkpoint'],weights_only=False,map_location='cpu')
            b=make(core,fac,records,fold,'B0',seed);initial=state_hash(b.state_dict())
            assert initial==v['binding']['initial'] and preprocessor(d,scale,'B0',fold,seed)['summary']==r['preprocessor']
            b.load_state_dict(state['model']);out,_=predict(b,d,d['val'],inputs,scale)
            assert np.array_equal(out,v['logit']) and np.array_equal(d['archive_indices'][d['val']],v['archive_indices'])
            original=MODEL.remove_guidance(component.make(core,fac,records,fold,'FULL',seed))
            joint=make(core,fac,records,fold,'JOINT_PRUNED',seed)
            shared=original.state_dict();assert all(torch.equal(joint.state_dict()[k],value) for k,value in shared.items())
            a,_=predict(original,d,d['val'],inputs,scale);z,_=predict(joint,d,d['val'],inputs,scale)
            assert np.array_equal(a,z)
            bridges.append(dict(arm='B0',seed=seed,fold=fold,initial_hash=initial,checkpoint_sha256=r['checkpoint_sha256'],calibration_sha256=r['calibration_sha256'],
                                stored_B0_CAL_bit_exact=True,independent_pruned_joint_initial_CAL_bit_exact=True,selected_epoch=r['epoch'],threshold=r['threshold']))
            del b,original,joint;gc.collect();torch.cuda.empty_cache()
        print('PBF_FOLD_ANCHORS_AND_BRIDGES_PASS',fold,flush=True)
    d,scale=data(1,core,rec,banks,inputs,binding);counts=[]
    for pid in sorted(d['roles']['train']):
        ii=d['train'][d['patient'][d['train']]==pid];raw,mask,_,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cpu')
        counts.append((int(mask.sum()),str(pid)));del raw,delta
    _,pid=max(counts);ii=d['train'][d['patient'][d['train']]==pid];tr=d['train']
    y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda');pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
    original=MODEL.remove_guidance(component.make(core,fac,records,1,'FULL',42));joint=make(core,fac,records,1,'JOINT_PRUNED',42)
    oa=torch.optim.AdamW(original.parameters(),lr=.0003,weight_decay=.001);ob=torch.optim.AdamW(joint.parameters(),lr=.0003,weight_decay=.001)
    original.train();joint.train();a,_=forward(original,d,ii,inputs,scale);b,_=forward(joint,d,ii,inputs,scale)
    la=MODEL.objective(a,y,pos);lb=MODEL.objective(b,y,pos);assert torch.equal(a,b) and torch.equal(la,lb)
    la.backward();lb.backward();pa,pb=dict(original.named_parameters()),dict(joint.named_parameters());assert set(pa)==set(pb)
    assert all(pa[n].grad is not None and torch.equal(pa[n].grad,pb[n].grad) for n in pa)
    for m,o in [(original,oa),(joint,ob)]:torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
    assert all(torch.equal(value,joint.state_dict()[name]) for name,value in original.state_dict().items())
    assert equal_tree(oa.state_dict(),ob.state_dict())
    del original,joint,oa,ob,la,lb,a,b;gc.collect();torch.cuda.empty_cache();audits=[]
    for arm in ARMS[1:]:
        m=make(core,fac,records,1,arm,42);tick=time.perf_counter();out,stats=forward(m,d,ii,inputs,scale,True)
        loss=MODEL.objective(out,y,pos);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert float(m.cnn[0].weight.grad.norm())>0 and float(m.head[0].weight.grad[:,:96].norm())>0 and float(m.head[0].weight.grad[:,96:].norm())>0
        audits.append(dict(arm=arm,parameters=MODEL.PARAMETERS[arm],channels=len(ii),seconds=time.perf_counter()-tick,
                           complete_patient_backward_finite=True,no_formal_update=True,**stats))
        del m,loss,out;gc.collect();torch.cuda.empty_cache();print('PBF_COMPLETE_TRAIN_GRADIENT_PASS',arm,flush=True)
    csv_out('ADMISSION_REUSE',bridges);csv_out('ANCHOR_ADMISSION',anchors);csv_out('ROLE_AUDIT',roles)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    write(PUB/'ADMISSION.json',dict(status='PASS',source=sources(),reuse_cells=15,initial_TRAIN_only_anchors=15,synthetic_tests=test.testsRun,
                                   independent_pruned_joint_logits_loss_all_gradients_oneAdam_exact=True,maximum_wrapper_gradient_error=0,
                                   discarded_parity_updates=2,new_path_admission=audits,no_new_Outer=True,no_assessment_used_for_admission=True,
                                   new_formal_fits=0,completed_studies_unchanged=True))
    print('PBF_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
