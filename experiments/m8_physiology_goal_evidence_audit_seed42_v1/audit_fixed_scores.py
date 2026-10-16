"""Post hoc completion evidence, existing private metrics only; no prediction."""
import hashlib
import itertools
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(r'D:\m8_physiology_discovery_selection_seed42_v2_runtime')
OUT=Path(r'D:\m8_physiology_goal_evidence_audit_seed42_v1_runtime\public')
METRICS=['auroc','auprc','macro_f1','f1','mrr','hit_at_1','accuracy','nez_f1']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def intervals(delta):
    n=len(delta);draws=np.random.default_rng(42).integers(n,size=(10000,n))
    distribution=np.mean(delta[draws],axis=1)
    return np.mean(delta,axis=0),distribution


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'public/INDEPENDENT_RESULTS_VERIFICATION.json')['status']=='PASS'
    path=ROOT/'OUTER_PATIENT_METRICS_PRIVATE.pkl'
    assert sha(path)==read(path.with_suffix('.sha256.json'))['sha256']
    with path.open('rb') as stream:frame=pickle.load(stream)
    assert len(frame)==1680 and frame.model.nunique()==21 and frame.patient.nunique()==80
    pairs=[('Concat__SELECTED','Concat__RAW_ONLY'),('Concat__SELECTED','Concat__RAW_SUPPORT30'),
           ('M8__SELECTED','Concat__RAW_ONLY'),('M8__SELECTED','M8__OLD7'),('D1__SELECTED','D1__BEST_SWAP')]
    prior=pd.read_csv(ROOT/'public/outer/PAIRED_COMPARISONS.csv');rows=[]
    for name,reference in pairs:
        a=frame[frame.model==name].sort_values('patient');b=frame[frame.model==reference].sort_values('patient')
        assert list(a.patient)==list(b.patient)
        delta=a[METRICS].to_numpy(float)-b[METRICS].to_numpy(float);mean,dist=intervals(delta)
        ordinary=np.percentile(dist,[2.5,97.5],axis=0)
        for j,metric in enumerate(METRICS):
            old=prior[(prior.model==name)&(prior.reference==reference)&(prior.metric==metric)].iloc[0]
            assert np.allclose([old.delta,old.ci_low,old.ci_high],[mean[j],*ordinary[:,j]],rtol=0,atol=1e-12)
        for family in [6,8]:
            lo,hi=np.percentile(dist,[100*.05/(2*family),100*(1-.05/(2*family))],axis=0)
            for j,metric in enumerate(METRICS[:family]):
                rows.append(dict(model=name,reference=reference,metric=metric,paired_mean_delta=mean[j],
                    ordinary95_low=ordinary[0,j],ordinary95_high=ordinary[1,j],endpoint_family=family,
                    bonferroni_family95_low=lo[j],bonferroni_family95_high=hi[j],patients=80,draws=10000,
                    posthoc_fixed_score_diagnostic=True,selection_retrained=False,independent_confirmation=False))
    pd.DataFrame(rows).to_csv(OUT/'MULTIPLICITY_SENSITIVITY.csv',index=False)
    subsets=pd.read_csv(ROOT/'public/FROZEN_FEATURE_SELECTION.csv');subsets=subsets[subsets.arm=='SELECTED']
    stability=[]
    for (_,a),(_,b) in itertools.combinations(subsets.iterrows(),2):
        x=set(str(a.features).split(';'));y=set(str(b.features).split(';'))
        stability.append(dict(fold_a=int(a.outer_fold),fold_b=int(b.outer_fold),groups_a=len(x),groups_b=len(y),
            intersection=len(x&y),union=len(x|y),jaccard=len(x&y)/len(x|y),independent_replicates=False))
    pd.DataFrame(stability).to_csv(OUT/'FINAL_FOLD_SET_COMPARISON.csv',index=False)
    result=dict(goal_completion_proven=False,execution_complete=True,final_global_minimum_identified=False,
        observed_final_fold_set_mean_jaccard=float(np.mean([r['jaccard'] for r in stability])),
        no_selection_retraining_uncertainty_estimated=True,no_new_model_or_threshold_or_prediction=True,
        original_protocol_and_test_seal_unchanged=True,source_sha256=sha(Path(__file__)),
        prior_seal_sha256=sha(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),metric_artifact_sha256=sha(path),
        status='EXECUTION_COMPLETE_SCIENTIFIC_MINIMUM_AND_STABLE_M8_INCREMENT_NOT_ESTABLISHED')
    (OUT/'GOAL_EVIDENCE_STATUS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('GOAL_EVIDENCE_AUDIT_COMPLETE_NO_NEW_TRAINING_OR_PREDICTION',flush=True)


if __name__=='__main__':main()
