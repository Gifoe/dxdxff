"""Small exact binary metrics, fixed calibration and patient-equal LR probe."""
import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def metric_rows(y, probability, patients, channels, centers, threshold):
    result = []
    for pid in sorted(set(patients)):
        ix = np.flatnonzero(patients == pid)
        yy = y[ix].astype(int); score = probability[ix]; pred = score >= threshold
        positive = int(yy.sum()); negative = len(yy) - positive
        tp = int(((yy == 1) & pred).sum()); tn = int(((yy == 0) & ~pred).sum())
        fp = negative - tn; fn = positive - tp
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        f0 = 2 * tn / max(2 * tn + fp + fn, 1)
        auc = (rankdata(score)[yy == 1].sum() - positive * (positive + 1) / 2) / (positive * negative) if positive and negative else np.nan
        order = np.argsort(-score, kind='stable'); ys = yy[order]; ps = score[order]; cum = ys.cumsum()
        ends = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])
        ap = float((np.diff(np.r_[0, cum[ends]]) / positive * cum[ends] / (ends + 1)).sum()) if positive else np.nan
        ranking = yy[np.lexsort((channels[ix].astype(str), -score))]
        mrr = 1 / (np.flatnonzero(ranking)[0] + 1) if positive else np.nan
        result.append(dict(patient=str(pid), center=str(centers[ix[0]]), n=len(ix), positives=positive, negatives=negative,
            TP=tp, TN=tn, FP=fp, FN=fn, accuracy=(tp+tn)/len(ix), auroc=float(auc), auprc=ap,
            macro_f1=(f1+f0)/2, f1=f1 if positive else np.nan, nez_f1=f0 if negative else np.nan,
            mrr=float(mrr), hit_at_1=float(ranking[0]) if positive else np.nan,
            ez_fraction_bias=float(pred.mean()-yy.mean())))
    return result


def threshold(y, probability, patients):
    grid = np.round(np.arange(0, 1.0001, .005), 3)
    predicted = probability[:, None] >= grid[None]
    macro, ez = [], []
    for pid in sorted(set(patients)):
        ix = np.flatnonzero(patients == pid); a = y[ix, None]; b = predicted[ix]
        tp = ((a == 1) & b).sum(0); tn = ((a == 0) & ~b).sum(0)
        fp = ((a == 0) & b).sum(0); fn = ((a == 1) & ~b).sum(0)
        f1 = np.divide(2.*tp, 2.*tp+fp+fn, out=np.zeros(len(grid)), where=(2*tp+fp+fn)>0)
        f0 = np.divide(2.*tn, 2.*tn+fp+fn, out=np.zeros(len(grid)), where=(2*tn+fp+fn)>0)
        macro.append((f1+f0)/2); ez.append(f1)
    macro = np.mean(macro, 0); ez = np.mean(ez, 0)
    order = np.lexsort((grid, np.abs(grid-.5), -ez, -macro))
    return float(grid[order[0]])


def weights(y, patients):
    ids, counts = np.unique(patients, return_counts=True); sizes = dict(zip(ids, counts))
    w = np.array([len(y)/(len(ids)*sizes[p]) for p in patients])
    classes = np.bincount(y, minlength=2); assert (classes>0).all()
    return w*len(y)/(2*classes[y])


def preprocess(z, fit):
    good = np.isfinite(z[fit]); den = good.sum(0)
    mean = np.divide(np.nansum(z[fit], 0), den, out=np.zeros(z.shape[1]), where=den>0)
    filled = np.where(np.isfinite(z), z, mean)
    scaler = StandardScaler().fit(filled[fit])
    return scaler.transform(filled).astype(np.float32), dict(imputer_statistics=mean, mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)


def fit_probe(x, y, pid, channel, center, train, cal, held, columns, seed):
    model = LogisticRegression(C=1, solver='lbfgs', max_iter=1000, tol=1e-6, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x[train][:, columns], y[train], sample_weight=weights(y[train], pid[train]))
    pc = model.predict_proba(x[cal][:, columns])[:, 1]
    th = threshold(y[cal], pc, pid[cal])
    ph = model.predict_proba(x[held][:, columns])[:, 1]
    rows = metric_rows(y[held], ph, pid[held], channel[held], center[held], th)
    return dict(model=model, threshold=th, calibration_probability=pc, probability=ph, rows=rows,
                warnings=sorted(set(type(w.message).__name__ for w in caught)), iterations=int(model.n_iter_.max()))
