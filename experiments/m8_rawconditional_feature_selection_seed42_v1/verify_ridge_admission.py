"""Actual inner-training/calibration exact-solver/threshold admission, no held y."""
import argparse
from sklearn.linear_model import Ridge
from scipy.special import expit
from conditional_io import *
from conditional_ridge import ConditionalRidge,prepare


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--first',action='store_true');args=parser.parse_args()
    common,data,training,support,p=old_modules();_,core,_,_,_,_,banks,_=common.legal()
    assert sha(OLD/'probes.py')==jr(OLDROOT/'public/SCREEN_STATUS.json')['source']['probes.py']
    pool=read(OLDROOT/'POOL_PRIVATE.pkl');checks=[];maximum=0.
    cells=[(1,0)] if args.first else [(fold,screen) for fold in range(1,6) for screen in range(4)]
    for fold,screen in cells:
        d=banks[fold-1];fit=d['train'];pid=d['patient'][fit]
        q=read(ROOT/'encoders_private'/f'fold{fold}_screen{screen}'/'REPRESENTATIONS_PRIVATE.pkl');tr=q['roles']['train'];cal=q['roles']['cal']
        assert np.array_equal(q['archive_indices'],d['archive_indices'][fit])
        x=np.concatenate([q['raw'],core.patient_z(pool['summary'][d['archive_indices'][fit]],pid)],1);xx,_=prepare(x,tr)
        y=np.full(len(fit),np.nan);use=np.r_[tr,cal];y[use]=(d['y'][fit][use]==0).astype(int)
        m=ConditionalRidge(xx,2*y-1,p.weights(y[tr].astype(int),pid[tr]),tr)
        count=0;local_maximum=0.
        for groups in [[],*[ [g] for g in range(32)],list(range(7)),list(range(9)),list(range(32))]:
            z=m.fit(groups);cols=z['columns'];ref=Ridge(alpha=1.,fit_intercept=True,solver='cholesky')
            ref.fit(xx[tr][:,cols],2*y[tr]-1,sample_weight=m.weights);oracle=ref.predict(xx[:,cols])
            error=float(np.max(np.abs(z['score']-oracle)));assert error<=2e-10;maximum=max(maximum,error);local_maximum=max(local_maximum,error)
            theta=p.threshold(y[cal].astype(int),z['probability'][cal],pid[cal]);other=p.threshold(y[cal].astype(int),expit(oracle[cal]),pid[cal])
            assert theta==other
            np.testing.assert_array_equal(z['probability'][cal,None]>=np.arange(201)[None,:]*.005,expit(oracle[cal,None])>=np.arange(201)[None,:]*.005)
            count+=1
        checks.append(dict(fold=fold,screen=screen,actual_subset_solves=count,all_calibration_grid_predictions_and_thresholds_identical=True,maximum_score_error=local_maximum))
    write(PUB/('FIRST_ACTUAL_RIDGE_ADMISSION.json' if args.first else 'ALL20_ACTUAL_RIDGE_ADMISSION.json'),dict(status='PASS',cells=checks,
        maximum_score_error=maximum,tolerance=2e-10,old_primitives_sha256=sha(OLD/'probes.py'),assessment_choice_original_validation_Outer_labels_not_used=True,
        no_neural_training_or_prediction_rerun=True,only_algebra_admission_ridge_refits=True,
        only_existing_frozen_representations_and_train_calibration_labels=True,source_sha256=sha(Path(__file__))))
    print('ACTUAL_RIDGE_AND_CALIBRATION_ADMISSION_PASS',len(checks),maximum,flush=True)


if __name__=='__main__':main()
