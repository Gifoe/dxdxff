"""Replay all25 legal selections and independent FIT preprocessing before Outer."""
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
    binding,core,fac,records,rec,banks=setup();fr=jr(ROOT/'MODEL_FREEZE_PRIVATE.json')
    assert fr['input']==binding and {(r['arm'],r['seed'],r['fold']) for r in fr['cells']}==expected_cells()
    assert len(fr['cells'])==10 and all(not r['reused'] for r in fr['cells'])
    inputs,checks,prep,preprocessing=store(),[],[],[]
    for fold in range(1,6):
        d=bank(fold,banks,ARMS[0]);independent_preprocessing(d,rec,ARMS[0])
        preprocessing.append(dict(fold=fold,FIT_only_reconstruction_bit_exact=True,own_Outer_excluded=True))
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];d,ss=bank(fold,banks,arm),scale(fold)
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['validation'])
        m=make(core,fac,records,fold,arm,seed);assert state_hash(m.state_dict())==v['binding']['initial']
        m.load_state_dict(state['model']);before=state_hash(m.state_dict())
        out=predict(m,d,d['val'],inputs,ss,fold,binding)
        assert np.array_equal(out,v['logit']) and np.array_equal(v['archive_indices'],d['archive_indices'][d['val']])
        assert state_hash(m.state_dict())==before and not m.interaction and not m.early and not m.late
        assert m.feature_pr and m.raw_pr and m.pre_reference and not m.guidance_pr and not hasattr(m,'gate')
        assert sum(p.numel() for p in m.parameters())==r['parameters']==9329
        selected=core.select_threshold(d['y'][d['val']],probability(out),d['patient'][d['val']])
        assert all(abs(float(z)-float(state[k]))<1e-12 for k,z in selected.items())
        assert state['epoch']==r['epoch'] and state['threshold']==r['threshold'] and preprocessor(d,ss)==r['preprocessor']
        winner=max(v['history'],key=lambda z:(z['macro_f1'],z['ez_f1'],-z['epoch']))
        assert winner['epoch']==state['epoch'] and len(v['history'])==r['epochs'] and 6<=r['epochs']<=30
        assert v['binding']['input']==binding and v['binding']['preprocessor']==r['preprocessor'] and v['binding']['seed']==seed
        checks.append(dict(arm=arm,seed=seed,fold=fold,validation_logits_bit_exact=True,selected_epoch=r['epoch'],
                           threshold=r['threshold'],parameters=r['parameters'],reused=False))
        prep.append(dict(arm=arm,seed=seed,fold=fold,preprocessor=r['preprocessor']))
        del m,state,v;gc.collect();torch.cuda.empty_cache();print('NEL2_INTERNAL_CELL_PASS',seed,fold,flush=True)
    assert len(checks)==10
    csv_out('VALIDATION_REPLAY_CHECKS',checks);csv_out('PREPROCESSING_REPLAY',preprocessing)
    write(PUB/'INTERNAL_VERIFICATION.json',dict(status='PASS',checkpoint_checks=10,validation_logits_bit_exact=True,
        FIT_only_preprocessing_independently_reconstructed=5,all_models_no_early_no_late=True,actual_new_fits=10,reused_cells=0))
    freeze(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json',dict(input=binding,source=sources(),cells=fr['cells'],preprocessing=prep,
        internal_verification=sha(PUB/'INTERNAL_VERIFICATION.json'),model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),
        scientific_status='EXPLORATORY_REPEATED_TEST'))
    write(PUB/'FREEZE_STATUS.json',dict(status='ALL10_FROZEN_BEFORE_NEW_OUTER',models=10,
        freeze_sha256=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),new_fits=10,reused=0))
    print('NEL2_ALL10_MODELS_FROZEN_BEFORE_OUTER',flush=True)
if __name__=='__main__':main()
