"""Read-only independent sklearn/NumPy verification of already sealed predictions.

Does not instantiate a model, change scores, refit preprocessing, or select thresholds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             recall_score, accuracy_score, confusion_matrix)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, default=Path(r'C:\d2_ictal_outer_seed42_runtime'))
    a = parser.parse_args()
    freeze = json.loads((a.runtime / 'FROZEN_BEFORE_OUTER.json').read_text())
    seal = json.loads((a.runtime / 'OUTER_PREDICTIONS_SEAL.json').read_text())
    assert seal['binding'] == sha(a.runtime / 'FROZEN_BEFORE_OUTER.json')
    status = json.loads((a.runtime / 'public/OUTER_STATUS.json').read_text())
    assert status['status'] == 'ONE_FROZEN_D2_OUTER_PASS_COMPLETE'
    features = Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(features) == freeze['hashes']['features']
    with np.load(features, allow_pickle=True) as f:
        y, patient, channel = f['y'], f['patient'], f['channel']
    checked, drift, pooled_pred = [], 0., np.empty(7635, dtype=int)
    metrics = pd.DataFrame(torch.load(a.runtime/'PATIENT_METRICS_PRIVATE.pt', map_location='cpu', weights_only=False))
    for fold in range(1,6):
        path = a.runtime / f'fold{fold}_OUTER_PREDICTIONS_PRIVATE.pt'
        assert sha(path) == seal['files'][str(fold)]
        cell = torch.load(path, map_location='cpu', weights_only=False)
        assert cell['binding'] == seal['binding']
        ix, scores = cell['indices'], cell['scores']
        pooled_pred[ix] = scores >= cell['threshold']
        for pat in sorted(set(patient[ix])):
            k = np.flatnonzero(patient[ix] == pat)
            k = k[np.argsort(channel[ix][k].astype(str),kind='stable')]
            true, score = y[ix[k]], scores[k].astype(float)
            pred = (score >= cell['threshold']).astype(int)
            rec0 = recall_score(true,pred,pos_label=0,zero_division=0)
            rec1 = recall_score(true,pred,pos_label=1,zero_division=0)
            order = np.argsort(score,kind='stable')
            positives = np.flatnonzero(true[order] == 0)
            ez = 1-true
            independent = {'macro_f1':f1_score(true,pred,labels=[0,1],average='macro',zero_division=0),
                'ez_f1':f1_score(true,pred,pos_label=0,zero_division=0),
                'nez_f1':f1_score(true,pred,pos_label=1,zero_division=0),
                'balanced_accuracy':(rec0+rec1)/2,'sensitivity':rec0,'specificity':rec1,
                'accuracy':accuracy_score(true,pred),
                'ez_auroc':roc_auc_score(ez,1-score) if len(set(true))==2 else np.nan,
                'ez_auprc':average_precision_score(ez,1-score) if ez.sum() else 0.,
                'ez_mrr':1/(positives[0]+1) if len(positives) else 0.,
                'top1_is_ez':float(true[order[0]] == 0)}
            # This actual cohort has both classes for all 80 patients.
            assert len(set(true)) == 2
            old = metrics.loc[metrics.patient == pat].iloc[0]
            for name, value in independent.items():
                difference = abs(value-float(old[name])); drift=max(drift,difference)
                assert difference < 1e-12, (fold,name)
            checked.append(pat)
    assert len(checked)==len(set(checked))==80
    published = pd.read_csv(a.runtime/'public/OUTER_PATIENT_EQUAL_METRICS.csv')
    keys = published.metric.tolist()
    vals = metrics.sort_values('patient')[keys].to_numpy()
    indices = np.random.default_rng(42).integers(0,80,size=(10000,80))
    draws = np.nanmean(vals[indices],axis=1)
    for j, row in published.iterrows():
        assert abs(vals[:,j].mean()-row.estimate) < 1e-12
        assert abs(np.nanpercentile(draws[:,j],2.5)-row.ci_low) < 1e-12
        assert abs(np.nanpercentile(draws[:,j],97.5)-row.ci_high) < 1e-12
    pooled = pd.read_csv(a.runtime/'public/OUTER_POOLED_CHANNEL_METRICS.csv').iloc[0]
    assert abs(f1_score(y,pooled_pred,labels=[0,1],average='macro')-pooled.macro_f1)<1e-12
    cm = confusion_matrix(y,pooled_pred,labels=[0,1])
    assert [int(pooled[k]) for k in ['TP','FN','FP','TN']] == cm.reshape(-1).tolist()
    out = a.runtime/'public/INDEPENDENT_AGGREGATE_VERIFICATION.json'
    assert not out.exists()
    out.write_text(json.dumps({'status':'PASS','patients':80,'metrics_per_patient':11,
        'maximum_metric_absolute_drift':drift,'bootstrap_draws':10000,'bootstrap_seed':42,
        'independent_sklearn_metrics':True,'independent_numpy_percentile_intervals':True,
        'prediction_hashes_unchanged':True,'model_replayed':False,'threshold_reoptimized':False,
        'source_sha256':sha(__file__)},indent=2),encoding='utf-8')
    print('INDEPENDENT_AGGREGATE_VERIFICATION_PASS',drift,flush=True)


if __name__ == '__main__':
    main()
