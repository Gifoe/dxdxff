"""Three exact references,30reused Rawanchors and unchanged NoPR update parity."""
import time
import unittest
from common import *


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def main():
    import test_preprocessing
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_preprocessing));assert tests.wasSuccessful()
    binding,core,fac,records,rec,banks=setup(False);inputs=store();bridges=[];anchors=[];roles=[];prechecks=[]
    active=np.array([True]*8+[False]*2+[True]*4)
    for fold in range(1,6):
        d,scale=data(fold,core,rec,banks,inputs,binding)
        roles.append(dict(fold=fold,train_patients=len(d['roles']['train']),cal_patients=len(d['roles']['cal']),assessment_patients=len(d['roles']['assessment']),
                          train_channels=len(d['train']),cal_channels=len(d['val']),assessment_channels=len(d['assessment'])))
        for phase,ii in [('TRAIN',d['train']),('CAL',d['val'])]:
            prechecks.append(dict(fold=fold,phase=phase,patients=len(set(d['patient'][ii])),channels=len(ii),width=14,active_width=12,
                                  legacy_active_abs_max=float(abs(d['x_legacy'][ii][:,active]).max()),stable_active_abs_max=float(abs(d['x'][ii][:,active]).max()),
                                  stable_all_finite=bool(np.isfinite(d['x'][ii]).all()),NoFeaturePR_bit_unchanged=True,TRAIN_only_unit_fit=True,
                                  zero_variance_TRAIN_unit_coordinates=int((d['pre']['TRAIN_unit_variance']==0).sum())))
        for seed in SEEDS:
            for kind in ['pre','no_pre']:
                a=read(anchor_path(fold,seed,kind));initial=initialize(core,fac,records,fold,'FULL' if kind=='pre' else 'WO_PRE_REFERENCE',seed)
                assert state_hash(initial.state_dict())==a['binding']['initial']
                assert a['binding']['train_patients']==digest(sorted(d['roles']['train'])) and a['binding']['train_indices']==legacy.array_sha(d['archive_indices'][d['train']])
                del initial
                anchors.append(dict(fold=fold,seed=seed,kind=kind,anchor_sha256=sha(anchor_path(fold,seed,kind)),TRAIN_patients=a['patients'],TRAIN_channels=a['channels'],
                                    TRAIN_valid_windows=a['valid_windows'],constant_coordinates=a['constant_coordinates'],initial_only_not_current_reference=True,reused_parent_anchor=True))
            for arm in REFERENCES:
                r=reference_cell(arm,seed,fold);v=read(r['calibration']);state=torch.load(r['checkpoint'],weights_only=False,map_location='cpu')
                m=make(core,fac,records,fold,arm,seed);initial=state_hash(m.state_dict())
                assert initial==v['binding']['initial'] and preprocessor(d,scale,arm,fold,seed)==r['preprocessor']
                m.load_state_dict(state['model']);out,_=predict(m,d,d['val'],inputs,scale)
                assert np.array_equal(out,v['logit']) and np.array_equal(d['archive_indices'][d['val']],v['archive_indices'])
                bridges.append(dict(arm=arm,origin_arm=REFERENCES[arm],seed=seed,fold=fold,initial_hash=initial,checkpoint_sha256=r['checkpoint_sha256'],calibration_sha256=r['calibration_sha256'],
                                    stored_CAL_bit_exact=True,selected_epoch=r['epoch'],threshold=r['threshold']))
                del m;gc.collect();torch.cuda.empty_cache()
        print('UCP_FOLD30_ANCHORS_AND45_REFERENCE_BRIDGES_PROGRESS',fold,flush=True)
    d,scale=data(1,core,rec,banks,inputs,binding);counts=[]
    for pid in sorted(d['roles']['train']):
        ii=d['train'][d['patient'][d['train']]==pid];raw,mask,_,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cpu')
        counts.append((int(mask.sum()),str(pid)));del raw,delta
    _,pid=max(counts);ii=d['train'][d['patient'][d['train']]==pid];tr=d['train']
    y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda');pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
    a=make(core,fac,records,1,'WO_FEATURE_PR',42);b=parent.make(core,fac,records,1,'WO_FEATURE_PR',42)
    assert state_hash(a.state_dict())==state_hash(b.state_dict())
    oa=torch.optim.AdamW(a.parameters(),lr=.0003,weight_decay=.001);ob=torch.optim.AdamW(b.parameters(),lr=.0003,weight_decay=.001)
    a.train();b.train();va,_=forward(a,d,ii,inputs,scale);vb,_=parent.forward(b,legacy_data(d),ii,inputs,scale)
    la=MODEL.objective(va,y,pos);lb=MODEL.objective(vb,y,pos);assert torch.equal(va,vb) and torch.equal(la,lb)
    la.backward();lb.backward();pa,pb=dict(a.named_parameters()),dict(b.named_parameters());assert set(pa)==set(pb)
    assert all(pa[n].grad is not None and torch.equal(pa[n].grad,pb[n].grad) for n in pa)
    for m,opt in [(a,oa),(b,ob)]:torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    assert state_hash(a.state_dict())==state_hash(b.state_dict()) and equal_tree(oa.state_dict(),ob.state_dict())
    del a,b,oa,ob,la,lb,va,vb;gc.collect();torch.cuda.empty_cache();paths=[]
    for arm in NEW:
        m=make(core,fac,records,1,arm,42);tick=time.perf_counter();out,info=forward(m,d,ii,inputs,scale,True)
        loss=MODEL.objective(out,y,pos);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        for value in [m.cnn[0].weight.grad,m.head[0].weight.grad[:,:96],m.head[0].weight.grad[:,96:192],m.head[0].weight.grad[:,192:]]:assert float(value.norm())>0
        paths.append(dict(arm=arm,parameters=MODEL.PARAMETERS[arm],channels=len(ii),seconds=time.perf_counter()-tick,
                          complete_patient_backward_finite=True,no_formal_update=True,**info))
        del m,out,loss;gc.collect();torch.cuda.empty_cache();print('UCP_COMPLETE_TRAIN_PATIENT_GRADIENT_PASS',arm,flush=True)
    csv_out('ADMISSION_REUSE',bridges);csv_out('ANCHOR_ADMISSION',anchors);csv_out('ROLE_AUDIT',roles);csv_out('PREPROCESSING_ADMISSION',prechecks)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    write(PUB/'ADMISSION.json',dict(status='PASS',source=sources(),parent_source=parent.sources(),reuse_cells=45,initial_TRAIN_only_anchors=30,reused_initial_raw_anchors=True,
                                   synthetic_tests=tests.testsRun,unchanged_NoFeaturePR_logits_loss_all_gradients_oneAdam_exact=True,maximum_control_gradient_error=0,
                                   discarded_parity_updates=2,new_path_admission=paths,no_new_Outer=True,no_assessment_used_for_admission=True,new_formal_fits=0,completed_studies_unchanged=True))
    print('UCP_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
