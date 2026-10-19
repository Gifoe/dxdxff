"""Before training: exact old-model replay and largest legal FIT backward gates."""
import subprocess
from common import *

def main():
    assert subprocess.run([sys.executable,str(HERE/'test_model.py')],cwd=HERE).returncode==0
    binding,core,fac,records,rec,banks=setup(False);inputs=store();checks=[];counts={};largest=None
    for fold in range(1,6):
        for arm in ['FULL_F0','WO_BOTH_PR']:
            d=bank(fold,banks,arm);raw14=rec['R14'][d['archive_indices']]
            z=raw14 if arm=='WO_BOTH_PR' else core.patient_z(raw14,d['patient']);pp=d['pre']
            expected=((np.where(np.isnan(z),pp['imputer_statistics'],z)-pp['mean'])/pp['scale']).astype(np.float32)
            assert np.array_equal(expected,d['x']) and set(pp['fit_patient_ids'])==d['roles']['fit']
        for arm in REF_NAMES:
            d=bank(fold,banks,arm);r,state,v=reference(arm,fold);m=make(core,fac,records,fold,arm)
            assert state_hash(m.state_dict())==v['binding']['initial'];m.load_state_dict(state['model'])
            out=predict(m,d,d['val'],inputs,scale(fold),fold,binding)
            assert np.array_equal(out,v['logit']) and np.array_equal(v['archive_indices'],d['archive_indices'][d['val']])
            checks.append(dict(arm=arm,fold=fold,initial_state_exact=True,validation_logits_bit_exact=True,parameters=sum(p.numel() for p in m.parameters())))
            del m;gc.collect();torch.cuda.empty_cache();print('NI_REFERENCE_EXACT_PASS',arm,fold,flush=True)
        d=bank(fold,banks,'FULL_F0')
        for pid in sorted(d['roles']['fit']):
            if pid not in counts:
                plan=inputs.raw.registry['plans'][str(pid)];assert sha(plan['path'])==plan['sha256']
                q=torch.load(plan['path'],map_location='cpu',weights_only=False);counts[pid]=int(q['mask'][0].sum());del q
            if largest is None or counts[pid]>largest['windows']:largest=dict(fold=fold,pid=pid,windows=counts[pid])
    audits=[];fold=largest['fold'];ss=scale(fold)
    for arm in NEW:
        d=bank(fold,banks,arm);tr=d['train'];ii=tr[d['patient'][tr]==largest['pid']]
        m=make(core,fac,records,fold,arm);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
        pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda');y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32);norms=[]
        for step in range(3):
            opt.zero_grad(set_to_none=True);logit,info=forward(m,d,ii,inputs,ss,fold,binding);loss=MODEL.objective(logit,y,pos)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            g={n:float(p.grad.norm()) for n,p in m.named_parameters()}
            assert g['head.0.weight']>0 and g['head.2.weight']>0
            assert g['cnn.stem.0.weight' if m.early else 'cnn.0.weight']>0
            if m.early:assert g['cnn.conditioner.2.weight']>0
            if m.late:assert g['gate.network.2.weight']>0
            if step and m.early:assert g['cnn.conditioner.0.weight']>0
            if step and m.late:assert g['gate.network.0.weight']>0
            norms.append(g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        audits.append(dict(arm=arm,fold=fold,channels=len(ii),valid_windows=info['valid_windows'],parameters=sum(p.numel() for p in m.parameters()),discarded_admission_steps=3,gradient_norms=norms))
        del m,opt;gc.collect();torch.cuda.empty_cache();print('NI_REAL_FIT_BACKWARD_PASS',arm,flush=True)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    write(PUB/'ADMISSION.json',dict(status='PASS',input_sha256=sha(ROOT/'INPUT_FREEZE_PRIVATE.json'),source=sources(),reference_replays=checks,
        actual_full_patient_backward=audits,tests_passed=5,new_fits_started=0,all_interactions_absent=True,variance_permanently_masked=True,both_PR_removed_only_in_WO_BOTH_PR=True))
    print('NI_14_GROUP_ADMISSION_PASS',flush=True)

if __name__=='__main__':main()
