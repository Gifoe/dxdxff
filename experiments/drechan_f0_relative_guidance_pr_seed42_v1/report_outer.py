"""Report all nine fixed groups after every score is sealed; never select from Outer."""
from common import *


def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def load_rows():
    fr = frozen()
    fh = sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    seal = jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert seal['freeze'] == fh and len(seal['files']) == 45 and seal['all45_cells_sealed_before_metric_reporting']
    for key, h in seal['files'].items():
        assert sha(ROOT/'scores_private'/(key+'.pkl')) == h
    ctx = read(ROOT/'METADATA_PRIVATE.pkl')
    meta = ctx['meta']
    archive = Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive) == ctx['archive_sha256']
    with np.load(archive, allow_pickle=True) as q:
        y = 1-q['y']
    assert len(y) == 7635 and int(y.sum()) == 1743
    rows, pool = [], {a: dict(y=[], p=[], v=[]) for a in ARMS}
    for r in fr['cells']:
        arm, fold = r['arm'], r['fold']
        q = read(ROOT/'scores_private'/f'{arm}_{fold}.pkl')
        ii = q['indices']
        p, v = 1-q['score_nez'].astype(float), q['score_nez'] < r['threshold']
        assert q['freeze'] == fh and q['checkpoint'] == r['checkpoint_sha256'] and q['threshold'] == r['threshold']
        for pid in sorted(set(meta['patient'][ii])):
            j = meta['patient'][ii] == pid
            values = dict(**EV.classification(y[ii][j], p[j], v[j]), **EV.ranking(y[ii][j], p[j], meta['channel'][ii][j]))
            assert all(np.isfinite(values[m]) for m in METRICS)
            rows.append(dict(arm=arm, fold=fold, patient=str(pid), center=str(meta['center'][ii][j][0]),
                             channels=int(j.sum()), **{m: float(values[m]) for m in METRICS}))
        for k, a in [('y', y[ii]), ('p', p), ('v', v)]:
            pool[arm][k].append(a)
    assert len(rows) == 720
    return rows, pool


def contrast_definitions():
    return [('absolute', a, {a: 1}) for a in ARMS]+[('paired', name, w) for name, w in CONTRASTS.items()]


def component_evidence(rows, matrix, family):
    result = []
    for name in PRIMARY:
        weights = CONTRASTS[name]
        differences = sum(matrix[a]*w for a, w in weights.items())
        means = differences.mean(0)
        directions = []
        for fold in range(1, 6):
            point = sum(vector([r for r in rows if r['arm'] == a and r['fold'] == fold]).mean(0)*w for a, w in weights.items())
            directions.append(bool(point[0] > 0 and point[1] > 0))
        lower = {m: next(float(r['low']) for r in family if r['contrast'] == name and r['metric'] == m)
                 for m in ['macro_f1', 'auroc']}
        mean_positive = bool(means[0] > 0 and means[1] > 0)
        family_positive = bool(min(lower.values()) > 0)
        fold_positive = sum(directions)
        result.append(dict(contrast=name, macro_f1_difference=float(means[0]), auroc_difference=float(means[1]),
                           both_point_estimates_positive=mean_positive, macro_family_low=lower['macro_f1'],
                           auroc_family_low=lower['auroc'], both_family_lower_bounds_positive=family_positive,
                           folds_both_positive=fold_positive, folds=5,
                           stable_within_this_seed_protocol=mean_positive and family_positive and fold_positive >= 4))
    return result


def main():
    rows, pool = load_rows()
    ids = sorted({r['patient'] for r in rows})
    matrix, mainrows, folds, sd, source, coverage = {}, [], [], [], [], []
    for arm in ARMS:
        rr = sorted([r for r in rows if r['arm'] == arm], key=lambda r: r['patient'])
        assert [r['patient'] for r in rr] == ids and len(ids) == 80 and sum(r['channels'] for r in rr) == 7635
        vv = vector(rr)
        matrix[arm] = vv
        mainrows.append(dict(arm=arm, patients=80, **dict(zip(METRICS, vv.mean(0).tolist()))))
        ff = []
        for fold in range(1, 6):
            q = [r for r in rr if r['fold'] == fold]
            values = vector(q).mean(0)
            ff.append(values)
            folds.append(dict(arm=arm, fold=fold, patients=len(q), **dict(zip(METRICS, values.tolist()))))
        sd.append(dict(arm=arm, **dict(zip(METRICS, np.std(ff, axis=0, ddof=1).tolist()))))
        for center in sorted({r['center'] for r in rr}):
            q = [r for r in rr if r['center'] == center]
            source.append(dict(arm=arm, center=center, patients=len(q), suppressed=len(q) < 5,
                               **dict(zip(METRICS, vector(q).mean(0).tolist() if len(q) >= 5 else [float('nan')]*9))))
        coverage.extend(dict(arm=arm, metric=m, finite_patients=int(np.isfinite(vv[:, j]).sum()), patients=80)
                        for j, m in enumerate(METRICS))
    for name, data in [('MAIN_RESULTS', mainrows), ('FOLD_RESULTS', folds), ('FOLD_SD', sd),
                       ('SOURCE_RESULTS', source), ('METRIC_COVERAGE', coverage)]:
        csv_out(name, data)
    draws = np.random.default_rng(42).integers(0, 80, (10000, 80))
    ci, family = [], []
    for kind, label, weights in contrast_definitions():
        v = sum(matrix[a]*w for a, w in weights.items())
        mean, lim = interval(v, draws)
        ci.extend(dict(kind=kind, contrast=label, metric=m, mean=float(mean[j]), low=float(lim[0, j]),
                       high=float(lim[1, j]), patients=80, draws=10000) for j, m in enumerate(METRICS))
        if kind == 'paired' and label in PRIMARY:
            tail = .05/28
            _, limits = interval(v, draws, (tail, 1-tail))
            for m in ['macro_f1', 'auroc']:
                j = METRICS.index(m)
                family.append(dict(kind=kind, contrast=label, metric=m, mean=float(mean[j]), low=float(limits[0, j]),
                                   high=float(limits[1, j]), comparisons=14, draws=10000, tail=tail))
    assert len(ci) == 162 and len(family) == 14
    csv_out('INTERVALS', ci)
    csv_out('PRIMARY_FAMILY_INTERVALS', family)
    components = component_evidence(rows, matrix, family)
    csv_out('COMPONENT_EVIDENCE', components)
    pooled = []
    for arm in ARMS:
        q = pool[arm]
        y, p, v = [np.concatenate(q[k]) for k in ['y', 'p', 'v']]
        s = EV.classification(y, p, v)
        pooled.append(dict(arm=arm, channels=len(y), aggregation='POOLED_CHANNELS_NOT_PRIMARY',
                           **{m: float(s[m]) for m in METRICS if m not in ['mrr', 'hit_at_1']},
                           tp=int((v & (y == 1)).sum()), tn=int((~v & (y == 0)).sum()),
                           fp=int((v & (y == 0)).sum()), fn=int((~v & (y == 1)).sum())))
    csv_out('POOLED_RESULTS', pooled)
    save(ROOT/'PATIENT_METRICS_PRIVATE.pkl', rows)
    path = PREVROOT/'public/MAIN_RESULTS.csv'
    with path.open(encoding='utf-8', newline='') as f:
        oldrows = list(csv.DictReader(f))
    old = next(r for r in oldrows if r['arm'] == 'FULL_F0')
    for j, m in enumerate(METRICS):
        assert abs(float(old[m])-matrix['B0_ORIGINAL'][:, j].mean()) < 1e-12
    b0, full = matrix['B0_ORIGINAL'].mean(0), matrix['FULL'].mean(0)
    conditions = dict(main_macro_f1_above_0p64=bool(full[0] > .64),
                      macro_f1_within_1pp_of_original=bool(full[0]-b0[0] >= -.01),
                      main_auroc_above_original=bool(full[1] > b0[1]),
                      all_retained_components_have_stable_two_metric_evidence=all(
                          r['stable_within_this_seed_protocol'] for r in components if r['contrast'] != 'FULL_MINUS_B0'))
    write(PUB/'GOAL_AUDIT.json', dict(status='PASS' if all(conditions.values()) else 'FAIL', conditions=conditions,
          full_macro_f1=float(full[0]), full_auroc=float(full[1]), original_macro_f1=float(b0[0]), original_auroc=float(b0[1]),
          selected_arm=None, success_was_not_guaranteed=True, outer_is_exploratory_repeated_test=True,
          seed_robustness_established=False, unsupported_components_not_promoted=True))
    write(PUB/'REPORT_STATUS.json', dict(status='ALL9_FIXED_GROUPS_REPORTED', patient_rows=720,
          nominal_intervals=162, primary_family_intervals=14, selected_arm=None,
          scientific_status='EXPLORATORY_REPEATED_TEST', independent_confirmation=False, no_post_Outer_tuning=True))
    print('RGF0_ALL9_FULL80_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
