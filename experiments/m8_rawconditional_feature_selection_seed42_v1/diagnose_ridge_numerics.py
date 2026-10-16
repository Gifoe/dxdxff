"""Numerical discrepancies only. No classification-performance metric reader."""
from sklearn.linear_model import Ridge
from scipy.special import expit
from conditional_io import *
from conditional_ridge import ConditionalRidge,prepare


def main():
    common,data,training,support,p=old_modules();_,core,_,_,_,_,banks,_=common.legal()
    pool=read(OLDROOT/'POOL_PRIVATE.pkl');rows=[]
    assert sha(OLD/'probes.py')==jr(OLDROOT/'public/SCREEN_STATUS.json')['source']['probes.py']
    for fold,d in enumerate(banks,1):
        fit=d['train'];pid=d['patient'][fit]
        for screen in range(4):
            q=read(ROOT/'encoders_private'/f'fold{fold}_screen{screen}'/'REPRESENTATIONS_PRIVATE.pkl');tr=q['roles']['train'];cal=q['roles']['cal']
            x=np.concatenate([q['raw'],core.patient_z(pool['summary'][d['archive_indices'][fit]],pid)],1);xx,pre=prepare(x,tr)
            y=np.full(len(fit),np.nan);use=np.r_[tr,cal];y[use]=(d['y'][fit][use]==0).astype(int)
            m=ConditionalRidge(xx,2*y-1,p.weights(y[tr].astype(int),pid[tr]),tr)
            cell=[]
            for groups in [[],*[[g] for g in range(32)],list(range(7)),list(range(9)),list(range(32))]:
                z=m.fit(groups);cols=z['columns'];ref=Ridge(alpha=1.,fit_intercept=True,solver='cholesky')
                ref.fit(xx[tr][:,cols],2*y[tr]-1,sample_weight=m.weights);oracle=ref.predict(xx[:,cols])
                error=np.abs(z['score']-oracle);probdiff=np.abs(z['probability']-expit(oracle));grid=np.round(np.arange(0,1.0001,.005),3)
                cell.append(dict(fold=fold,screen=screen,groups=','.join(map(str,groups)),raw_score_error=float(error.max()),
                    calibration_score_error=float(error[cal].max()),probability_error=float(probdiff.max()),
                    calibration_grid_decision_disagreements=int(np.sum((z['probability'][cal,None]>=grid)!=(expit(oracle[cal,None])>=grid))),
                    calibration_threshold_identical=p.threshold(y[cal].astype(int),z['probability'][cal],pid[cal])==p.threshold(y[cal].astype(int),expit(oracle[cal]),pid[cal]),
                    max_abs_reference_score=float(np.max(np.abs(oracle))),max_abs_scaled_input=float(np.max(np.abs(xx[:,cols]))),
                    min_training_scale=float(np.min(pre['scale'][cols])),maximum_coefficient_error=float(np.max(np.abs(z['coef']-ref.coef_))),
                    passed_original_absolute_score_gate=bool(error.max()<=2e-10),classification_performance_not_computed=True))
            rows+=cell
            print('NUMERIC_DIAGNOSTIC_CELL',fold,screen,'failed',sum(not r['passed_original_absolute_score_gate'] for r in cell),
                'maxerror',max(r['raw_score_error'] for r in cell),'maxinput',max(r['max_abs_scaled_input'] for r in cell),flush=True)
    table=pd.DataFrame(rows);table.to_csv(PUB/'NUMERICAL_DIAGNOSTIC.csv',index=False)
    write(PUB/'NUMERICAL_ADMISSION_STATUS.json',dict(status='FAIL' if not table.passed_original_absolute_score_gate.all() else 'PASS',
        cases=len(table),failed_absolute_score_cases=int((~table.passed_original_absolute_score_gate).sum()),
        maximum_score_error=float(table.raw_score_error.max()),maximum_probability_error=float(table.probability_error.max()),
        threshold_disagreement_cases=int((~table.calibration_threshold_identical).sum()),
        original_tolerance_unchanged=2e-10,new_screen_results_not_accepted=True,no_classification_performance_computed=True,
        no_hyperparameter_preprocessing_or_threshold_changes=True,source_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
