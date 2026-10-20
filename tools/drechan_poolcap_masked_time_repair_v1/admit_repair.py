"""Outcome-free mask repair admission over every legal FIT support, preserving P1."""
import common as c
from common import *
from repair import attention_mean
binding,core,fac,records,rec,banks=setup(False)
assert binding==jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
diagnosis=jr(ROOT/'GRADIENT_DIAGNOSIS_PRIVATE.json')
assert diagnosis['nonfinite_gradient_names']==['attention_weight'] and diagnosis['masked_nonfinite_time_cells']>0
assert not (ROOT/'models_private/P2_TIME_ATTENTION/seed42/fold1/LAST_PRIVATE.pt').exists()
preserved={}
for name in ['BEST_PRIVATE.pt','VALIDATION_PRIVATE.pkl','LAST_PRIVATE.pt','COMPLETE_PRIVATE.json']:
    rel='models_private/P1_TEMPORAL_STATS/seed42/fold1/'+name;assert (ROOT/rel).is_file();preserved[rel]=sha(ROOT/rel)
assert len(list((ROOT/'models_private').rglob('COMPLETE_PRIVATE.json')))==1
# Four arithmetic cases: valid-only equality including gradients; NaN-padded
# forward equality/finite repaired derivatives; empty/singleton support; time use.
torch.manual_seed(194);latent=torch.randn(3,7,24);times=torch.arange(7).float()*2
mask=torch.tensor([[True,True,True,False,False,False,False],[True]+[False]*6,[False]*7])
weight=torch.randn(26)*.03
def sample(fn,t):
    x=latent.clone().requires_grad_();w=weight.clone().requires_grad_();out,present,weights=fn(x,t,mask,w)
    out.square().sum().backward();return out.detach(),x.grad,w.grad,present,weights.detach()
old=sample(c.ORIGINAL_ATTENTION,times);new=sample(attention_mean,times)
assert all(torch.equal(a,b) for a,b in zip(old,new))
bad_times=times.clone();bad_times[3:]=float('nan')
old_bad=sample(c.ORIGINAL_ATTENTION,bad_times);new_bad=sample(attention_mean,bad_times)
assert torch.equal(old_bad[0],new_bad[0]) and not torch.isfinite(old_bad[2]).all()
assert all(torch.isfinite(a).all() for a in new_bad[:3])
assert new_bad[3].tolist()==[True,True,False] and (new_bad[4][~mask]==0).all()
assert all(torch.equal(a,b) for a,b in zip(new[:3],new_bad[:3]))
seen=set();seen_fit=set();checks=0;fit_checks=0;validation_checks=0;masked_nonfinite=0;inputs=store()
for fold in range(1,6):
    d=bank(fold,banks,'P2_TIME_ATTENTION');ss=scale(fold);m=make(core,fac,records,fold,'P2_TIME_ATTENTION',42)
    for phase,indices in [('fit',d['train']),('validation',d['val'])]:
        for pid in sorted(set(d['patient'][indices])):
            ii=indices[d['patient'][indices]==pid]
            raw,mask,times,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
            masked_nonfinite+=int(((~torch.isfinite(times))[:,None,:]&~mask).sum())
            m.zero_grad(set_to_none=True);logit,_=forward(m,d,ii,inputs,ss,fold,binding)
            assert torch.isfinite(logit).all();logit.square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            seen.add(str(pid));checks+=1
            if phase=='fit':seen_fit.add(str(pid));fit_checks+=1
            else:validation_checks+=1
    del m;gc.collect();torch.cuda.empty_cache();print('PCA_REPAIR_ALL_FIT_SUPPORT_PASS',fold,flush=True)
assert len(seen)==80 and len(seen_fit)==79 and fit_checks==255 and validation_checks==65 and checks==320 and masked_nonfinite>0
for rel,h in preserved.items():assert sha(ROOT/rel)==h
freeze(ROOT/'EXECUTION_OVERLAY_PRIVATE.json',dict(execution_manifest_sha256=c.EXECUTION_MANIFEST_SHA256,
    scientific_input=digest(binding),scientific_source=sources(),preserved_completed_files=preserved))
write(PUB/'ENGINEERING_REPAIR_ADMISSION.json',dict(status='PASS',execution_manifest_sha256=c.EXECUTION_MANIFEST_SHA256,
    scientific_source=sources(),original13_sources_unchanged=True,arithmetic_cases=4,
    finite_support_forward_and_all_derivatives_bit_exact=True,NaN_padded_forward_bit_exact=True,
    padding_excluded_before_matmul=True,all5_FIT_folds_checked=True,complete_patient_support_checks=checks,
    unique_patients_on_legal_FIT_or_validation_roles=len(seen),unique_patients_on_legal_FIT_roles=len(seen_fit),
    complete_patient_FIT_checks=fit_checks,complete_patient_validation_checks=validation_checks,
    label_free_gradient_checks=True,zero_optimizer_updates=True,masked_nonfinite_time_cells_checked=masked_nonfinite,
    original_failure_only_attention_gradient=True,preserved_completed_P1_fold1_files=preserved,
    completed_epochs_discarded=0,P2_completed_epochs_before_repair=0,no_Outer_access=True,no_performance_tuning=True))
print('PCA_ENGINEERING_MASK_REPAIR_ADMISSION_PASS',flush=True)
