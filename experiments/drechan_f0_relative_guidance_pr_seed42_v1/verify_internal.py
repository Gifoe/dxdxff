"""Replay all45 legal selections and independent FIT preprocessing before Outer."""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from common import *


def independent_preprocessing(d, rec):
    raw = np.asarray(rec['R14'][d['archive_indices']], np.float64)
    fit = d['train']
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
    assert fr['input'] == binding and len(fr['cells']) == 45
    assert jr(PUB/'TRAIN_STATUS.json')['actual_new_fits'] == 40
    inputs = store()
    checks, pre, preprocessing, null = [], [], [], []
    old = parent.frozen()
    for fold in range(1, 6):
        d = bank(fold, banks, 'FULL')
        independent_preprocessing(d, rec)
        no_pr = bank(fold, banks, 'WO_BOTH_PR')
        original = parent.bank(fold, banks, 'WO_BOTH_PR')
        assert np.array_equal(no_pr['x'], original['x'])
        assert preprocessor(no_pr, scale(fold)) == parent.preprocessor(original, parent.scale(fold))
        preprocessing.append(dict(fold=fold, FIT_only_reconstruction_bit_exact=True,
                                  no_PR_original_input_bit_exact=True, own_Outer_excluded=True))
    for r in fr['cells']:
        arm, fold = r['arm'], r['fold']
        d, ss = bank(fold, banks, arm), scale(fold)
        assert sha(r['checkpoint']) == r['checkpoint_sha256'] and sha(r['validation']) == r['validation_sha256']
        state = torch.load(r['checkpoint'], map_location='cpu', weights_only=False)
        v = read(r['validation'])
        m = make(core, fac, records, fold, arm)
        assert state_hash(m.state_dict()) == v['binding']['initial']
        m.load_state_dict(state['model'])
        before = state_hash(m.state_dict())
        out = predict(m, d, d['val'], inputs, ss, fold, binding)
        assert np.array_equal(out, v['logit']) and np.array_equal(v['archive_indices'], d['archive_indices'][d['val']])
        assert state_hash(m.state_dict()) == before and not m.interaction and not hasattr(m, 'interaction_gain')
        assert sum(isinstance(q, torch.nn.Linear) and q.out_features == 1 for q in m.modules()) == 1
        assert sum(p.numel() for p in m.parameters()) == MODEL.PARAMETERS[arm] == r['parameters']
        selected = core.select_threshold(d['y'][d['val']], probability(out), d['patient'][d['val']])
        for k, val in selected.items():
            assert abs(float(val)-float(state[k])) < 1e-12
        assert state['epoch'] == r['epoch'] and state['threshold'] == r['threshold']
        assert preprocessor(d, ss) == r['preprocessor']
        h = v['history']
        winner = max(h, key=lambda z: (z['macro_f1'], z['ez_f1'], -z['epoch']))
        assert winner['epoch'] == state['epoch'] and len(h) == r['epochs'] and 6 <= len(h) <= 30
        if not r['reused']:
            assert v['binding']['input'] == binding and v['binding']['preprocessor'] == r['preprocessor']
        if arm == 'WO_BOTH_PR':
            ref = next(x for x in old['cells'] if x['arm'] == 'WO_BOTH_PR' and x['fold'] == fold)
            previous_state = torch.load(ref['checkpoint'], map_location='cpu', weights_only=False)
            previous_validation = read(ref['validation'])
            assert set(state['model']) == set(previous_state['model'])
            assert all(torch.equal(state['model'][k], previous_state['model'][k]) for k in state['model'])
            assert np.array_equal(out, previous_validation['logit'])
            assert state['epoch'] == previous_state['epoch'] and state['threshold'] == previous_state['threshold']
            assert len(h) == len(previous_validation['history'])
            null.append(dict(fold=fold, fresh_training=True, selected_tensors_bit_exact=True,
                             validation_bit_exact=True, epoch_and_threshold_bit_exact=True,
                             total_epochs_bit_exact=True))
        checks.append(dict(arm=arm, fold=fold, validation_logits_bit_exact=True,
                           selected_epoch=r['epoch'], threshold=r['threshold'], parameters=r['parameters'],
                           no_head_interaction=True))
        pre.append(dict(arm=arm, fold=fold, preprocessor=r['preprocessor']))
        del m
        gc.collect()
        torch.cuda.empty_cache()
        print('RGF0_INTERNAL_CELL_PASS', arm, fold, flush=True)
    assert len(checks) == 45 and len(null) == len(preprocessing) == 5
    csv_out('VALIDATION_REPLAY_CHECKS', checks)
    csv_out('PREPROCESSING_REPLAY', preprocessing)
    csv_out('NULL_CONTROL_RETRAIN_PARITY', null)
    write(PUB/'INTERNAL_VERIFICATION.json', dict(status='PASS', checkpoint_checks=45,
          validation_logits_bit_exact=True, selected_epochs_and_thresholds_verified=True,
          FIT_only_preprocessing_independently_reconstructed=5, exact_fresh_no_PR_retrains=5,
          all_neural_models_single_head=True, all_interactions_absent=True,
          no_Development_eligibility_gate=True, actual_new_fits=40, reused_cells=5))
    freeze(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json', dict(input=binding, source=sources(), cells=fr['cells'],
           preprocessing=pre, internal_verification=sha(PUB/'INTERNAL_VERIFICATION.json'),
           model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'), scientific_status='EXPLORATORY_REPEATED_TEST'))
    write(PUB/'FREEZE_STATUS.json', dict(status='ALL45_FROZEN_BEFORE_NEW_OUTER',
          freeze_sha256=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'), models=45, new_fits=40, reused=5))
    print('RGF0_ALL45_MODELS_FROZEN_BEFORE_OUTER', flush=True)


if __name__ == '__main__':
    main()
