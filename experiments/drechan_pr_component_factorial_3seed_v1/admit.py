"""Fifteen exact completed B0 replays, one real full-patient new-path admission."""
import time
import unittest
from common import *

def main():
    import test_model
    test=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_model));assert test.wasSuccessful()
    binding,core,fac,records,rec,banks=setup(False);inputs=store();replays=[];roles=[]
    for fold in range(1,6):
        d,scale=data(fold,core,rec,banks,inputs,binding)
        roles.append(dict(fold=fold,train_patients=len(d['roles']['train']),cal_patients=len(d['roles']['cal']),assessment_patients=len(d['roles']['assessment']),
                          train_channels=len(d['train']),cal_channels=len(d['val']),assessment_channels=len(d['assessment']),
                          feature_PR_preprocessor_hash=digest(preprocessor(d,scale,'FULL')),no_feature_PR_preprocessor_hash=digest(preprocessor(d,scale,'WO_FEATURE_PR'))))
        for seed in SEEDS:
            r=previous_cell(seed,fold);s=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['calibration'])
            m=make(core,fac,records,fold,'FULL',seed);initial=state_hash(m.state_dict())
            assert initial==v['binding']['initial'] and preprocessor(d,scale,'FULL')==r['preprocessor']
            assert v['binding']['preprocessor']==r['preprocessor'] and v['binding']['input']==jr(prior.ROOT/'INPUT_FREEZE_PRIVATE.json')
            m.load_state_dict(s['model']);out,_=predict(m,d,d['val'],inputs,scale)
            assert np.array_equal(out,v['logit']) and np.array_equal(d['archive_indices'][d['val']],v['archive_indices'])
            sel=core.select_threshold(d['y'][d['val']],probability(out),d['patient'][d['val']])
            assert all(abs(float(value)-float(s[k]))<1e-12 for k,value in sel.items())
            assert all(abs(float(s[k])-float(r[k]))<1e-12 for k in ['epoch','threshold'])
            replays.append(dict(arm='FULL',seed=seed,fold=fold,initial_hash=initial,checkpoint_sha256=r['checkpoint_sha256'],calibration_sha256=r['calibration_sha256'],
                                preprocessing_roles_initialization_and_CAL_bit_exact=True,selected_epoch=r['epoch'],threshold=r['threshold']))
            del m;gc.collect();torch.cuda.empty_cache()
        print('PRF_FOLD_REUSE_ADMITTED',fold,flush=True)
    d,scale=data(1,core,rec,banks,inputs,binding)
    # Choose the largest genuine TRAIN input, independently of labels/outcomes.
    counts=[]
    for pid in sorted(d['roles']['train']):
        ii=d['train'][d['patient'][d['train']]==pid];raw,mask,times,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cpu')
        counts.append((int(mask.sum()),str(pid)));del raw,delta
    _,pid=max(counts);ii=d['train'][d['patient'][d['train']]==pid];audits=[]
    for arm in ARMS[1:]:
        m=make(core,fac,records,1,arm,42);tick=time.perf_counter()
        out,stats=forward(m,d,ii,inputs,scale,True);loss=out.square().mean()+.1*out.mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        for n in ['cnn.stem.0.weight','head.0.weight']:assert dict(m.named_parameters())[n].grad.norm()>0
        audits.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),channels=len(ii),seconds=time.perf_counter()-tick,
                           full_patient_all_gradients_finite=True,no_optimizer_step=True,**stats))
        del m;gc.collect();torch.cuda.empty_cache();print('PRF_NEW_PATH_BACKWARD_PASS',arm,flush=True)
    csv_out('ADMISSION_REUSE',replays);csv_out('ROLE_AUDIT',roles)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    write(PUB/'ADMISSION.json',dict(status='PASS',source=sources(),reuse_cells=15,synthetic_tests=test.testsRun,new_path_admission=audits,
                                   no_new_Outer=True,no_assessment_used_for_admission=True,new_formal_fits=0,original_sources_unchanged=True))
    print('PRF_ADMISSION_PASS',flush=True)

if __name__=='__main__':main()
