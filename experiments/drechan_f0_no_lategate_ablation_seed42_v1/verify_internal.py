"""Replay all90 legal selections and independent FIT preprocessing before Outer."""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from common import *


def independent_preprocessing(d, rec, arm):
    raw = np.asarray(rec['R14'][d['archive_indices']], np.float64)
    fit = d['train']
    if arm in MODEL.DROP:
        j = MODEL.DROP[arm]; raw[:, 2*j:2*j+2] = 0.
    imp0 = SimpleImputer().fit(raw[fit])
    units = StandardScaler().fit(imp0.transform(raw[fit]))
    z = np.full_like(raw, np.nan)
    for pid in sorted(set(d['patient'])):
        jj = np.flatnonzero(d['patient'] == pid)
        v = raw[jj]
        mu, sd = np.nanmean(v, 0), np.nanstd(v, 0)
        for j in range(14):
            z[jj, j] = ((v[:, j]-mu[j])/sd[j] if np.isfinite(sd[j]) and sd[j] > 1e-8*units.scale_[j]
                        else np.where(np.isnan(v[:, j]), np.nan, 0.))
    imp = SimpleImputer().fit(z[fit])
    scaler = StandardScaler().fit(imp.transform(z[fit]))
    expected = scaler.transform(imp.transform(z)).astype(np.float32)
    assert np.array_equal(expected, d['x'])
    for key, value in [('imputer_statistics', imp.statistics_), ('mean', scaler.mean_),
                       ('scale', scaler.scale_), ('var', scaler.var_),
                       ('FIT_unit_scale', units.scale_), ('FIT_unit_variance', units.var_)]:
        assert np.array_equal(value, d['pre'][key]), key
    assert set(d['pre']['fit_patient_ids']) == d['roles']['fit']


def main():
    binding, core, fac, records, rec, banks = setup()
    fr = jr(ROOT/'MODEL_FREEZE_PRIVATE.json')
    assert fr['input'] == binding and {(r['arm'],r['seed'],r['fold']) for r in fr['cells']} == expected_cells()
    admission = jr(PUB/'ADMISSION.json')
    assert sum(not r['reused'] for r in fr['cells']) == admission['expected_fresh_fits']
    inputs, checks, prep, preprocessing = store(), [], [], []
    for fold in range(1,6):
        for arm in ['FULL']+list(MODEL.DROP):
            d = bank(fold,banks,arm); independent_preprocessing(d,rec,arm)
        d=bank(fold,banks,'WO_BOTH_PR'); old=current.bank(fold,banks,'WO_BOTH_PR')
        assert np.array_equal(d['x'],old['x']) and preprocessor(d,scale(fold))==current.preprocessor(old,current.scale(fold))
        preprocessing.append(dict(fold=fold,FIT_only_reconstruction_bit_exact=True,masked_feature_banks_checked=6,
                                  no_PR_original_input_bit_exact=True,own_Outer_excluded=True))
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];d,ss=bank(fold,banks,arm),scale(fold)
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['validation'])
        m=make(core,fac,records,fold,arm,seed);assert state_hash(m.state_dict())==v['binding']['initial']
        m.load_state_dict(state['model']);before=state_hash(m.state_dict())
        out=predict(m,d,d['val'],inputs,ss,fold,binding)
        assert np.array_equal(out,v['logit']) and np.array_equal(v['archive_indices'],d['archive_indices'][d['val']])
        assert state_hash(m.state_dict())==before and not m.interaction and not hasattr(m,'interaction_gain')
        assert sum(isinstance(q,torch.nn.Linear) and q.out_features==1 for q in m.modules())==1
        assert sum(p.numel() for p in m.parameters())==MODEL.PARAMETERS[arm]==r['parameters']
        assert arm in MODEL.REFERENCES or (not m.late and not hasattr(m,'gate'))
        selected=core.select_threshold(d['y'][d['val']],probability(out),d['patient'][d['val']])
        assert all(abs(float(z)-float(state[k]))<1e-12 for k,z in selected.items())
        assert state['epoch']==r['epoch'] and state['threshold']==r['threshold'] and preprocessor(d,ss)==r['preprocessor']
        winner=max(v['history'],key=lambda z:(z['macro_f1'],z['ez_f1'],-z['epoch']))
        assert winner['epoch']==state['epoch'] and len(v['history'])==r['epochs'] and 6<=r['epochs']<=30
        if r['reused']:
            ref,oldstate,oldv=reference(arm,seed,fold)
            assert ref['checkpoint_sha256']==r['checkpoint_sha256'] and ref['validation_sha256']==r['validation_sha256']
            reference_score(r)
        else:
            assert v['binding']['input']==binding and v['binding']['preprocessor']==r['preprocessor']
            assert v['binding']['seed']==seed
        checks.append(dict(arm=arm,seed=seed,fold=fold,validation_logits_bit_exact=True,selected_epoch=r['epoch'],
                           threshold=r['threshold'],parameters=r['parameters'],reused=r['reused']))
        prep.append(dict(arm=arm,seed=seed,fold=fold,preprocessor=r['preprocessor']))
        del m,state,v;gc.collect();torch.cuda.empty_cache()
        print('NLG_INTERNAL_CELL_PASS',arm,seed,fold,flush=True)
    assert len(checks)==90
    csv_out('VALIDATION_REPLAY_CHECKS',checks);csv_out('PREPROCESSING_REPLAY',preprocessing)
    write(PUB/'INTERNAL_VERIFICATION.json',dict(status='PASS',checkpoint_checks=90,validation_logits_bit_exact=True,
        FIT_only_preprocessing_independently_reconstructed=35,all_models_single_head=True,all_interactions_absent=True,
        actual_new_fits=admission['expected_fresh_fits'],reused_cells=admission['expected_reused_cells']))
    freeze(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json',dict(input=binding,source=sources(),cells=fr['cells'],preprocessing=prep,
        internal_verification=sha(PUB/'INTERNAL_VERIFICATION.json'),model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),
        scientific_status='EXPLORATORY_REPEATED_TEST'))
    write(PUB/'FREEZE_STATUS.json',dict(status='ALL90_FROZEN_BEFORE_NEW_OUTER',models=90,
        freeze_sha256=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),new_fits=admission['expected_fresh_fits'],
        reused=admission['expected_reused_cells']))
    print('NLG_ALL90_MODELS_FROZEN_BEFORE_OUTER',flush=True)

if __name__=='__main__':main()
