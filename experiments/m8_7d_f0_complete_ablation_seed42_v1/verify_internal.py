"""Independently replay all selected legal validation scores; freeze all90 cells."""
from common import *

def main():
    binding,core,fac,records,rec,banks=setup();fr=jr(ROOT/'MODEL_FREEZE_PRIVATE.json');assert fr['input']==binding and len(fr['cells'])==90
    assert jr(PUB/'TRAIN_STATUS.json')['actual_new_fits']==80
    inputs=store();checks=[];maximum=0.;initials={};pre=[]
    for r in fr['cells']:
        arm,fold=r['arm'],r['fold'];d=bank(fold,banks,arm);ss=scale(fold)
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['validation'])
        m=make(core,fac,records,fold,arm);assert state_hash(m.state_dict())==v['binding']['initial'];m.load_state_dict(state['model'])
        before=state_hash(m.state_dict());out=predict(m,d,d['val'],inputs,ss,fold,binding);error=float(np.max(np.abs(out-v['logit'])))
        assert error<=1e-6 and np.array_equal(v['archive_indices'],d['archive_indices'][d['val']])
        assert state_hash(m.state_dict())==before and sum(isinstance(q,torch.nn.Linear) and q.out_features==1 for q in m.modules())==1
        assert np.array_equal(probability(out)[:,None]<np.round(np.arange(0,1.0001,.005),3),probability(v['logit'])[:,None]<np.round(np.arange(0,1.0001,.005),3))
        selected=core.select_threshold(d['y'][d['val']],probability(out),d['patient'][d['val']])
        for k,val in selected.items():assert abs(float(val)-float(state[k]))<1e-12
        assert state['epoch']==r['epoch'] and state['threshold']==r['threshold'] and preprocessor(d,ss)==r['preprocessor']
        if not r['reused']:
            assert v['binding']['input']==binding and v['binding']['preprocessor']==r['preprocessor']
            h=v['history'];winner=max(h,key=lambda z:(z['macro_f1'],z['ez_f1'],-z['epoch']));assert winner['epoch']==state['epoch'] and 6<=len(h)<=30
        checks.append(dict(arm=arm,fold=fold,maximum_replay_error=error,all201_decisions_exact=True));maximum=max(maximum,error)
        pre.append(dict(arm=arm,fold=fold,preprocessor=r['preprocessor']));del m;gc.collect();torch.cuda.empty_cache()
        print('INTERNAL_FROZEN_CELL_PASS',arm,fold,flush=True)
    write(PUB/'INTERNAL_VERIFICATION.json',dict(status='PASS',checkpoint_checks=90,maximum_replay_error=maximum,all201_threshold_decisions_exact=True,selected_epochs_and_thresholds_verified=True,
        no_Development_eligibility_gate=True,actual_new_fits=80,reused_cells=10))
    csv_out('VALIDATION_REPLAY_CHECKS',checks)
    freeze(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json',dict(input=binding,source=sources(),cells=fr['cells'],preprocessing=pre,internal_verification=sha(PUB/'INTERNAL_VERIFICATION.json'),
        model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),old_score_seal=sha(OLDROOT/'OUTER_SCORE_SEAL_PRIVATE.json'),all90_selections_frozen=True,scientific_status='EXPLORATORY_REPEATED_TEST'))
    write(PUB/'FREEZE_STATUS.json',dict(status='ALL90_FROZEN_BEFORE_NEW_OUTER',freeze_sha256=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),models=90,new_fits=80,reused=10))
    print('ALL90_MODELS_THRESHOLDS_PREPROCESSORS_FROZEN_BEFORE_OUTER',flush=True)

if __name__=='__main__':main()
