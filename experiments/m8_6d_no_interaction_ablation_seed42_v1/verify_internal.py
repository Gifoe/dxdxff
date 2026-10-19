"""Independently replay every legal validation selection before any new Outer."""
from common import *

def main():
    binding,core,fac,records,rec,banks=setup();fr=jr(ROOT/'MODEL_FREEZE_PRIVATE.json')
    assert fr['input']==binding and len(fr['cells'])==70 and jr(PUB/'TRAIN_STATUS.json')['actual_new_fits']==55
    inputs=store();checks=[];pre=[]
    for r in fr['cells']:
        arm,fold=r['arm'],r['fold'];d=bank(fold,banks,arm);ss=scale(fold)
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['validation']);m=make(core,fac,records,fold,arm)
        assert state_hash(m.state_dict())==v['binding']['initial'];m.load_state_dict(state['model']);before=state_hash(m.state_dict())
        out=predict(m,d,d['val'],inputs,ss,fold,binding)
        assert np.array_equal(out,v['logit']) and np.array_equal(v['archive_indices'],d['archive_indices'][d['val']])
        assert state_hash(m.state_dict())==before and not m.interaction and not hasattr(m,'interaction_gain')
        assert sum(isinstance(q,torch.nn.Linear) and q.out_features==1 for q in m.modules())==1
        selected=core.select_threshold(d['y'][d['val']],probability(out),d['patient'][d['val']])
        for k,val in selected.items():assert abs(float(val)-float(state[k]))<1e-12
        assert state['epoch']==r['epoch'] and state['threshold']==r['threshold'] and preprocessor(d,ss)==r['preprocessor']
        if not r['reused']:
            assert v['binding']['input']==binding and v['binding']['preprocessor']==r['preprocessor']
            h=v['history'];winner=max(h,key=lambda z:(z['macro_f1'],z['ez_f1'],-z['epoch']))
            assert winner['epoch']==state['epoch'] and 6<=len(h)<=30
        checks.append(dict(arm=arm,fold=fold,validation_logits_bit_exact=True,selected_epoch=r['epoch'],threshold=r['threshold'],no_head_interaction=True))
        pre.append(dict(arm=arm,fold=fold,preprocessor=r['preprocessor']));del m;gc.collect();torch.cuda.empty_cache()
        print('NI_INTERNAL_CELL_PASS',arm,fold,flush=True)
    csv_out('VALIDATION_REPLAY_CHECKS',checks)
    write(PUB/'INTERNAL_VERIFICATION.json',dict(status='PASS',checkpoint_checks=70,validation_logits_bit_exact=True,selected_epochs_and_thresholds_verified=True,
        all_neural_models_single_head=True,all_interactions_absent=True,no_Development_eligibility_gate=True,actual_new_fits=55,reused_cells=15))
    freeze(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json',dict(input=binding,source=sources(),cells=fr['cells'],preprocessing=pre,
        internal_verification=sha(PUB/'INTERNAL_VERIFICATION.json'),model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),scientific_status='EXPLORATORY_REPEATED_TEST'))
    write(PUB/'FREEZE_STATUS.json',dict(status='ALL70_FROZEN_BEFORE_NEW_OUTER',freeze_sha256=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),models=70,new_fits=55,reused=15))
    print('NI_ALL70_MODELS_FROZEN_BEFORE_OUTER',flush=True)

if __name__=='__main__':main()
