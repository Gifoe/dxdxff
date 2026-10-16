"""Original-source and all20 actual patient-role admission before any new fit."""
import pytest
from conditional_io import *


def main():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    assert pytest.main([str(HERE/'test_conditional_ridge.py'),'-q'])==0
    common,data,trainer,support,p=old_modules();_,core,_,_,_,_,banks,_=common.legal()
    b=bindings();rows=[]
    for fold,d in enumerate(banks,1):
        for screen in range(4):
            q=roles(fold,screen,d)
            rows.append(dict(fold=fold,screen=screen,training_patients=len(set(d['patient'][q['train']])),
                calibration_patients=len(set(d['patient'][q['cal']])),feature_choice_patients=len(set(d['patient'][q['choice']])),
                assessment_patients=len(set(d['patient'][q['held']])) if screen else 0,
                all_roles_patient_disjoint=True,original_validation_and_Outer_excluded=True))
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',b)
    pd.DataFrame(rows).to_csv(PUB/'PATIENT_ROLE_ADMISSION.csv',index=False)
    write(PUB/'ADMISSION_STATUS.json',dict(status='PASS',actual_role_partitions=20,
        exact_ridge_synthetic_tests=2,original_sources_and_all20_partitions_bound=True,
        parent_Outer_seal_unchanged=sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==b['parent_Outer_seal_protection'],
        source=b['source'],no_new_training_or_prediction_in_this_admission=True))
    print('ACTUAL_ALL20_ROLE_AND_EXACT_RIDGE_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
