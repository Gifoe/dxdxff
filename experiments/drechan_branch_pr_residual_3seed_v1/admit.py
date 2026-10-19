"""Old B0 bridges, TRAIN-only initial anchors and complete new-model gradients."""
import time
import unittest
from common import *

def main():
    import test_model,test_relative_math
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in [test_model,test_relative_math]])
    test=unittest.TextTestRunner(verbosity=2).run(suite);assert test.wasSuccessful()
    binding,core,fac,records,rec,banks=setup(False);inputs=store();replays=[];anchors=[];roles=[]
    for fold in range(1,6):
        d,scale=data(fold,core,rec,banks,inputs,binding)
        roles.append(dict(fold=fold,train_patients=len(d['roles']['train']),cal_patients=len(d['roles']['cal']),assessment_patients=len(d['roles']['assessment']),
                          train_channels=len(d['train']),cal_channels=len(d['val']),assessment_channels=len(d['assessment'])))
        for seed in SEEDS:
            anchor=fit_anchor(fold,seed,d,scale,inputs,binding,core,fac,records)
            anchors.append(dict(fold=fold,seed=seed,anchor_sha256=sha(anchor_path(fold,seed)),TRAIN_patients=anchor['patients'],TRAIN_channels=anchor['channels'],
                                TRAIN_valid_windows=anchor['valid_windows'],constant_coordinates=anchor['constant_coordinates'],initial_only_not_current_reference=True))
            r=previous_cell(seed,fold);state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['calibration'])
            b=make(core,fac,records,fold,'B0',seed);initial=state_hash(b.state_dict())
            assert initial==v['binding']['initial'] and preprocessor(d,scale,'B0',fold,seed)['summary']==r['preprocessor']
            b.load_state_dict(state['model']);out,_=predict(b,d,d['val'],inputs,scale)
            assert np.array_equal(out,v['logit']) and np.array_equal(d['archive_indices'][d['val']],v['archive_indices'])
            for arm in ['FULL','CHANNEL_CONTEXT']:
                m=make(core,fac,records,fold,arm,seed);shared=m.state_dict()
                for key,value in state['model'].items():shared[key]=value
                m.load_state_dict(shared);check,_=predict(m,d,d['val'],inputs,scale)
                assert np.array_equal(check,v['logit']) and torch.count_nonzero(m.gain_raw)==torch.count_nonzero(m.gain_feature)==0
                del m;gc.collect();torch.cuda.empty_cache()
            replays.append(dict(arm='B0',fold=fold,seed=seed,initial_hash=initial,checkpoint_sha256=r['checkpoint_sha256'],calibration_sha256=r['calibration_sha256'],
                                original_CAL_and_zero_gain_FULL_and_control_bit_exact=True,selected_epoch=r['epoch'],threshold=r['threshold']))
            del b;gc.collect();torch.cuda.empty_cache()
        print('BPR_FOLD_ANCHOR_AND_BRIDGE_PASS',fold,flush=True)
    d,scale=data(1,core,rec,banks,inputs,binding);counts=[]
    for pid in sorted(d['roles']['train']):
        ii=d['train'][d['patient'][d['train']]==pid];raw,mask,_,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cpu')
        counts.append((int(mask.sum()),str(pid)));del raw,delta
    _,pid=max(counts);ii=d['train'][d['patient'][d['train']]==pid];audits=[]
    for arm in ARMS[1:]:
        m=make(core,fac,records,1,arm,42);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001);tick=time.perf_counter();norms=[]
        with torch.no_grad():_,stats=forward(m,d,ii,inputs,scale,True)
        for step in range(2):
            opt.zero_grad(set_to_none=True);out,_=forward(m,d,ii,inputs,scale);loss=out.square().mean()+.1*out.mean();loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            selected={n:float(dict(m.named_parameters())[n].grad.norm()) for n in ['cnn.stem.0.weight','head.0.weight','gain_raw','gain_feature']}
            assert all(v>0 for v in selected.values());norms.append(selected);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        audits.append(dict(arm=arm,parameters=11553,channels=len(ii),seconds=time.perf_counter()-tick,discarded_admission_updates=2,
                           all_gradients_finite=True,gradient_norms=norms,**stats))
        del m,opt;gc.collect();torch.cuda.empty_cache();print('BPR_COMPLETE_TRAIN_PATIENT_GRADIENT_PASS',arm,flush=True)
    csv_out('ADMISSION_REUSE',replays);csv_out('ANCHOR_ADMISSION',anchors);csv_out('ROLE_AUDIT',roles)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    write(PUB/'ADMISSION.json',dict(status='PASS',source=sources(),reuse_cells=15,initial_TRAIN_only_anchors=15,synthetic_tests=test.testsRun,new_path_admission=audits,
                                   no_new_Outer=True,no_assessment_used_for_admission=True,new_formal_fits=0,completed_studies_unchanged=True))
    print('BPR_ADMISSION_PASS',flush=True)

if __name__=='__main__':main()
