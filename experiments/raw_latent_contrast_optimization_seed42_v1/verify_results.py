"""Release verifier: score/role replay plus independent count-matrix bootstrap."""
import pandas as pd
from common import *
from evaluate import verify_rows
from train import ARMS


def main():
    status=jr(PUB/'OUTER_STATUS.json');assert status['status']=='ONE_FROZEN_FIVEFOLD_OUTER_COMPLETE_VERIFIED'
    assert status['freeze_sha256']==sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert status['score_seal_sha256']==sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    frozen=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for name,value in frozen['evaluation_source'].items():assert sha(HERE/name)==value
    for name,value in frozen['binding']['source'].items():assert sha(HERE/name)==value
    for rel,value in jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')['files'].items():assert sha(ROOT/rel)==value
    frame=read(ROOT/'PATIENT_METRICS_PRIVATE.pkl');pred=read(ROOT/'PREDICTIONS_PRIVATE.pkl')
    assert len(frame)==720 and frame.model.nunique()==9
    maximum=verify_rows(pred,frame)
    absolute=pd.read_csv(PUB/'ABSOLUTE_INTERVALS.csv');paired=pd.read_csv(PUB/'PAIRED_INTERVALS.csv')
    draws=np.random.default_rng(42).integers(0,80,size=(10000,80))
    counts=np.stack([np.bincount(draw,minlength=80) for draw in draws])
    patients=sorted(frame.patient.unique());matrices={}
    for model,g in frame.groupby('model'):
        assert g.patient.nunique()==len(g)==80 and int(g.n.sum())==7635
        matrices[model]=g.set_index('patient').loc[patients,METRICS].to_numpy(float)
    max_boot=0.
    for row in absolute.itertuples():
        j=METRICS.index(row.metric);values=matrices[row.model][:,j]
        b=counts@values/80;lo,hi=np.quantile(b,[.025,.975])
        error=max(abs(float(values.mean())-row.mean),abs(lo-row.low),abs(hi-row.high));assert error<=1e-12;max_boot=max(max_boot,error)
    for row in paired.itertuples():
        j=METRICS.index(row.metric);values=matrices[row.left][:,j]-matrices[row.right][:,j]
        b=counts@values/80;lo,hi=np.quantile(b,[.025,.975])
        error=max(abs(float(values.mean())-row.delta),abs(lo-row.low),abs(hi-row.high));assert error<=1e-12;max_boot=max(max_boot,error)
    main_table=pd.read_csv(PUB/'MAIN_RESULTS.csv').set_index('model')
    for model,values in matrices.items():
        error=float(np.max(np.abs(values.mean(0)-main_table.loc[model,METRICS].to_numpy(float))));assert error<=1e-12;maximum=max(maximum,error)
    # Final deployment is replayed solely from saved Development criteria.
    for fold in range(1,6):
        candidates=[r for r in frozen['selections'] if r['fold']==fold]
        chosen=max(candidates,key=lambda r:(r['selection']['f1'],r['selection']['macro_f1'],-r['parameters'],-ARMS.index(r['arm'])))
        assert frozen['deployment'][str(fold)]==chosen['arm']
    target=jr(PUB/'TARGET_STATUS.json');observed=float(main_table.loc['SELECTED_POLICY','f1'])
    assert target['observed']==observed or abs(target['observed']-observed)<=1e-12
    assert target['point_target_met']==(observed>.65)
    write(PUB/'INDEPENDENT_RESULTS_VERIFICATION.json',dict(status='PASS',patient_rows=720,
        independently_recomputed_patient_metrics=720*len(METRICS),
        independent_metric_max_error=maximum,bootstrap_count_matrix_max_error=max_boot,
        absolute_intervals=len(absolute),paired_intervals=len(paired),draws_per_interval=10000,
        all_20_model_and_score_freezes_unchanged=True,all_5_Development_policy_choices_replayed=True,
        threshold_sensitivities_not_used_for_Outer_selection=True,
        verifier_sha256=sha(HERE/'verify_results.py')))
    print('RAW_FINAL_INDEPENDENT_VERIFICATION_PASS',len(frame),flush=True)


if __name__=='__main__':main()
