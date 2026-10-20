"""Pure diagnostic arithmetic. No learned decoder or inference policy is added."""
import numpy as np

GRID = np.round(np.arange(0., 1.0001, .005), 3)


def curves(y, score, patient):
    y, score, patient = np.asarray(y), np.asarray(score), np.asarray(patient)
    assert len(y) == len(score) == len(patient) and len(y) > 0
    assert np.isfinite(score).all() and set(y.tolist()) <= {0, 1}
    predicted = score[:, None] >= GRID[None, :]
    macro, ez, counts = [], [], []
    for pid in np.unique(patient):
        ix = patient == pid; a = y[ix, None]; b = predicted[ix]
        tp = ((a == 1) & b).sum(0); tn = ((a == 0) & ~b).sum(0)
        fp = ((a == 0) & b).sum(0); fn = ((a == 1) & ~b).sum(0)
        f1 = np.divide(2.*tp, 2*tp+fp+fn, out=np.zeros(len(GRID)), where=2*tp+fp+fn > 0)
        f0 = np.divide(2.*tn, 2*tn+fp+fn, out=np.zeros(len(GRID)), where=2*tn+fp+fn > 0)
        macro.append((f0+f1)/2); ez.append(f0); counts.append((tp, tn, fp, fn))
    tp, tn, fp, fn = np.asarray(counts).sum(0)
    recalls = []
    if (y == 1).any(): recalls.append(tp/(tp+fn))
    if (y == 0).any(): recalls.append(tn/(tn+fp))
    return np.asarray(macro).mean(0), np.asarray(ez).mean(0), np.mean(recalls, axis=0)


def choose(y, score, patient):
    macro, ez, ba = curves(y, score, patient)
    k = max(range(len(GRID)), key=lambda j: (float(macro[j]), float(ez[j]), float(ba[j]),
                                            -abs(float(GRID[j])-.5), -float(GRID[j])))
    return dict(threshold=float(GRID[k]), macro_f1=float(macro[k]), ez_f1=float(ez[k]), pooled_ba=float(ba[k]))


def binary_utility(y_nez, score, threshold):
    y, pred = np.asarray(y_nez), np.asarray(score) >= threshold
    tp = int(((y == 1) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    a = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    b = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    return (a+b)/2, b


def leave_one(y, score, patient, threshold):
    y, score, patient = np.asarray(y), np.asarray(score), np.asarray(patient)
    result = []
    for pid in np.unique(patient):
        held = patient == pid; fitted = choose(y[~held], score[~held], patient[~held])
        macro, ez = binary_utility(y[held], score[held], fitted['threshold'])
        original, _ = binary_utility(y[held], score[held], threshold)
        result.append(dict(patient=pid, threshold=fitted['threshold'], macro_f1=macro, ez_f1=ez,
                           original_macro_f1=original,
                           changed=float(np.mean((score[held] >= fitted['threshold']) != (score[held] >= threshold)))))
    return result


def stability(y, score, patient, threshold):
    rows = leave_one(y, score, patient, threshold)
    tau = np.array([r['threshold'] for r in rows]); original = np.mean([r['original_macro_f1'] for r in rows])
    loo = np.mean([r['macro_f1'] for r in rows]); q = np.quantile(tau, [.25, .5, .75])
    return dict(cal_patients=len(rows), cal_channels=len(y), selected_threshold=float(threshold),
                cal_macro_f1=float(original), leave_one_CAL_threshold_macro_f1=float(loo),
                difference=float(loo-original), loo_threshold_min=float(tau.min()),
                loo_threshold_q25=float(q[0]), loo_threshold_median=float(q[1]), loo_threshold_q75=float(q[2]),
                loo_threshold_max=float(tau.max()), loo_threshold_sd=float(tau.std(ddof=1)),
                mean_abs_threshold_change=float(np.mean(np.abs(tau-threshold))),
                mean_fraction_held_decisions_changed=float(np.mean([r['changed'] for r in rows])),
                checkpoint_selected_on_all_CAL=True, not_independent_validation=True)


def raw_stats(v, mean, scale):
    v, mean, scale = np.asarray(v, np.float64), np.asarray(mean, np.float64), np.asarray(scale, np.float64)
    assert v.ndim == 2 and v.shape[1] == 96 and mean.shape == scale.shape == (96,)
    assert np.isfinite(v).all() and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()
    g = (v-mean)/scale; q = np.quantile(np.abs(g), [.5, .95, .99])
    return dict(channels=len(v), mean_coordinate_rms=float(np.sqrt(np.mean(g.mean(0)**2))),
                population_sd_coordinate_rms=float(np.sqrt(np.mean(g.std(0, ddof=0)**2))),
                abs_q50=float(q[0]), abs_q95=float(q[1]), abs_q99=float(q[2]), abs_max=float(np.max(np.abs(g))),
                fraction_abs_above5=float(np.mean(np.abs(g)>5)), fraction_abs_above10=float(np.mean(np.abs(g)>10)),
                anchor_scale_min=float(scale.min()), anchor_scale_median=float(np.median(scale)), anchor_scale_max=float(scale.max()))


def score_stats(logit, probability, patient):
    logit, probability, patient = np.asarray(logit, np.float64), np.asarray(probability), np.asarray(patient)
    assert np.isfinite(logit).all() and np.isfinite(probability).all() and len(logit) == len(patient)
    locations = np.array([np.median(logit[patient == pid]) for pid in np.unique(patient)])
    q = np.quantile(locations, [.25, .5, .75])
    return dict(channels=len(logit), patients=len(locations), logit_mean=float(logit.mean()), logit_population_sd=float(logit.std(ddof=0)),
                patient_logit_median_q25=float(q[0]), patient_logit_median_median=float(q[1]), patient_logit_median_q75=float(q[2]),
                patient_logit_median_sd=float(locations.std(ddof=1)) if len(locations)>1 else 0.,
                fraction_abs_logit_above20=float(np.mean(np.abs(logit)>20)),
                probability_saturation_fraction=float(np.mean((probability == 0)|(probability == 1))),
                unique_logit_fraction=len(np.unique(logit))/len(logit), unique_probability_fraction=len(np.unique(probability))/len(logit))


def legal_scope(patient, indices, roles, phase):
    assert phase in ['TRAIN', 'CAL']
    indices = np.asarray(indices); wanted = roles['train' if phase == 'TRAIN' else 'cal']
    assert len(indices) == len(set(indices.tolist())) > 0
    assert set(np.asarray(patient)[indices]) == wanted
    assert set(np.asarray(patient)[indices]).isdisjoint(roles['assessment']|roles['test'])
