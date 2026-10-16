"""Complete the registered final equal-k controls, on legal choice patients only."""
from scipy.special import expit
from sklearn.linear_model import Ridge
from conditional_io import *

DEST = Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_dense_reference_runtime')


def main():
    common, _, _, _, primitives = old_modules()
    _, core, _, _, _, _, banks, _ = common.legal()
    pub = DEST / 'public'
    assert jr(pub / 'INTERNAL_REPLAY_VERIFICATION.json')['status'] == 'PASS'
    binding = jr(DEST / 'INPUT_FREEZE_PRIVATE.json')
    assert binding['original_scientific_binding'] == bindings()
    pool = read(OLDROOT / 'POOL_PRIVATE.pkl')
    features = jr(DEST / 'SELECTED_FEATURES_PRIVATE.json')
    rows, actual_fits = [], 0
    metrics = ['accuracy', 'auroc', 'auprc', 'macro_f1', 'f1', 'nez_f1', 'mrr', 'hit_at_1', 'ez_fraction_bias']
    for fold, bank in enumerate(banks, 1):
        q = read(DEST / f'fold{fold}_SCREEN_PRIVATE.pkl')['paths'][0]
        assert sha(Path(q['raw_entry']['path'])) == q['raw_entry']['sha256']
        raw = read(Path(q['raw_entry']['path']))
        fit = bank['train']; pid = bank['patient'][fit]; y = (bank['y'][fit] == 0).astype(int)
        x = np.c_[raw['raw'], core.patient_z(pool['summary'][bank['archive_indices'][fit]], pid)]
        pre = q['pre']; filled = np.where(np.isfinite(x), x, pre['imputer'])
        z = (filled - pre['mean']) / pre['scale']
        train, cal, choice = [q['roles'][k] for k in ['train', 'cal', 'choice']]
        w = primitives.weights(y[train], pid[train]); w = w * len(w) / w.sum()
        cache = {}
        for recipe in ['RAW_ONLY', 'SELECTED', 'BEST_SWAP', 'RANDOM_A', 'RANDOM_B', 'OLD7']:
            groups = features['folds'][str(fold)][recipe]; key = tuple(groups)
            if key not in cache:
                columns = np.r_[np.arange(192), 192 + np.array([2*g+j for g in groups for j in [0, 1]], int)]
                model = Ridge(alpha=1., fit_intercept=True, solver='cholesky')
                model.fit(z[train][:, columns], 2*y[train]-1, sample_weight=w)
                probability = expit(model.predict(z[:, columns]))
                threshold = primitives.threshold(y[cal], probability[cal], pid[cal])
                pt = primitives.metric_rows(y[choice], probability[choice], pid[choice], bank['channel'][fit][choice], bank['center'][fit][choice], threshold)
                cache[key] = dict(threshold=threshold, **{m:float(np.mean([r[m] for r in pt])) for m in metrics})
                actual_fits += 1
            summary = cache[key]
            utility = float(np.mean([summary[m] for m in ['auroc', 'auprc', 'macro_f1', 'f1']]))
            if recipe == 'SELECTED':
                assert abs(utility - q['path'][len(groups)]['utility']) < 1e-12
            rows.append(dict(fold=fold, recipe=recipe, k=len(groups),
                features=';'.join(pool['names'][g] for g in groups),
                choice_patients=len(np.unique(pid[choice])), choice_utility=utility,
                selection_conditioned=True, independent_assessment=False, **summary))
    assert sha(OLDROOT / 'OUTER_PREDICTIONS_SEAL_PRIVATE.json') == binding['original_scientific_binding']['parent_Outer_seal_protection']
    pd.DataFrame(rows).to_csv(pub / 'CHOICE_CONTROL_COMPARISON.csv', index=False)
    write(pub / 'CHOICE_CONTROL_STATUS.json', dict(status='COMPLETE_REGISTERED_CHOICE_CONTROLS',
        recipe_rows=len(rows), actual_ridge_fits=actual_fits, duplicate_subsets_fit_once=True,
        only_preexisting_frozen_feature_maps=True, choice_not_independent_assessment=True,
        no_new_selection_training_or_Outer=True, source_sha256=sha(Path(__file__))))
    print('REGISTERED_CHOICE_CONTROLS_COMPLETE', len(rows), actual_fits, flush=True)


if __name__ == '__main__':
    main()
