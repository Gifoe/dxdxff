"""Repair only an admission role-union assumption; scientific sources unchanged."""
import gc
import time
import final_stage as f
from model import Model,objective
c=f.c


def main():
    binding,_,_=f.setup()
    assert c.sha(f.FINAL/'INPUT_FREEZE_PRIVATE.json')=='cd08eeec8272affa4ac7dac448b47b206a725f4ec0c74abe00faf27c25b0d6f2'
    diagnostic=c.jr(f.PUBLIC/'FINAL_ROLE_ADMISSION_DIAGNOSTIC.json')
    assert diagnostic['role_union_counts']==dict(fit=79,validation=47,test=80) and diagnostic['all_original_role_identities_unchanged']
    meta,banks=c.io.identity_and_banks(labels=True);cohort=set(meta['patient'])
    fit_union=set().union(*(d['roles']['fit'] for d in banks))
    assert len(cohort)==80 and len(meta['patient'])==7635 and len(fit_union)==79
    assert c.digest(sorted(map(str,fit_union)))==diagnostic['FIT_union_identity_sha256']=='574217d8957f84f472b270c325e8092f231eaf2b9804bbdb4f05c8240d29c0bc'
    assert c.digest(sorted(map(str,cohort)))==diagnostic['cohort_identity_sha256']=='51f07e3a9211d5efa311c888c1562269f49b329805a9516dd21dc5b0e7bb85dc'
    store=c.RawStore();largest=None;seen=set()
    for fold,d in enumerate(banks,1):
        roles=d['roles'];assert len(roles['fit'])==[51,51,50,52,51][fold-1] and len(roles['validation'])==13 and len(roles['test'])==[16,16,17,15,16][fold-1]
        assert roles['fit'].isdisjoint(roles['validation']|roles['test']) and roles['validation'].isdisjoint(roles['test'])
        assert set.union(roles['fit'],roles['validation'],roles['test'])==cohort
        assert set(d['patient'][d['train']])==roles['fit'] and set(d['patient'][d['val']])==roles['validation']
        for pid in sorted(set(d['patient'][d['train']])):
            if str(pid) in seen:continue
            ii=d['train'][d['patient'][d['train']]==pid];raw,mask,times=store.get(pid,d['channel'][ii],device='cpu');seen.add(str(pid));n=int(mask.sum())
            if largest is None or n>largest['windows']:largest=dict(fold=fold,ii=ii,pid=pid,windows=n)
    assert seen==set(map(str,fit_union))
    d=banks[largest['fold']-1];ii=largest['ii'];tr=d['train'];assert largest['pid'] in d['roles']['fit']
    raw,mask,times=store.get(largest['pid'],d['channel'][ii]);assert int(mask.sum())==largest['windows']
    y=c.torch.tensor(d['y'][ii],device='cuda',dtype=c.torch.float32);pos=c.torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda');records=[]
    for arm in c.TRAINED:
        model=Model(arm,largest['fold']).cuda();before=c.state_hash(model.state_dict());c.torch.cuda.reset_peak_memory_stats()
        c.torch.cuda.synchronize();tick=time.perf_counter();out=model(raw,mask,times);loss=objective(out,y,pos);loss.backward();c.torch.cuda.synchronize()
        assert c.torch.isfinite(loss) and all(p.grad is not None and c.torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.cnn[0].weight.grad.abs().sum()>0 and model.head[2].weight.grad.abs().sum()>0 and c.state_hash(model.state_dict())==before
        records.append(dict(arm=arm,parameters=14577,channels=len(ii),valid_windows=largest['windows'],largest_of_unique_FIT_patients=len(seen),
            seconds=time.perf_counter()-tick,peak_allocated_bytes=c.torch.cuda.max_memory_allocated(),all_gradients_finite=True,optimizer_updates=0))
        del model,out,loss;gc.collect();c.torch.cuda.empty_cache()
    c.write(f.PUBLIC/'FINAL_ADMISSION.json',dict(status='PASS',input_binding_sha256=c.digest(binding),largest_complete_FIT_patient_backward=records,
        admission_loss_uses_only_ownFIT_labels=True,optimizer_updates=0,original_FIT_union_patients=79,unchanged_Outer_contract_patients=80,
        original_input_freeze_sha256=c.sha(f.FINAL/'INPUT_FREEZE_PRIVATE.json'),original_role_diagnostic_sha256=c.sha(f.PUBLIC/'FINAL_ROLE_ADMISSION_DIAGNOSTIC.json'),
        engineering_admission_source_sha256=c.sha(c.Path(__file__)),no_model_data_loss_split_or_selection_change=True,
        repair='OriginalFITunion is79,not80; original39fittedinternalmodels andallscientificsourcebindings preserved. No cohortorrolechange.'))
    print('JOINT_RAW_EXACT_ORIGINAL_FIT_ADMISSION_PASS',records,flush=True)


if __name__=='__main__':main()
