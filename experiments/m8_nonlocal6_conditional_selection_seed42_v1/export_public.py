"""Explicit anonymous whitelist, verified before download."""
from study_io import *

FILES=['INPUT_ADMISSION.json','OWN_TRAIN_ELIGIBILITY.csv','CHOICE_STATUS.json','EXHAUSTIVE_CHOICE_RECORD.csv',
       'FROZEN_SELECTION_PLAN.csv','INNER_ROLE_ISOLATION.csv','REFIT_COUNTS.csv','SEAL_STATUS.json',
       'MAIN_RESULTS.csv','FOLD_RESULTS.csv','SOURCE_RESULTS.csv','IDENTITY_EQUAL_RESULTS.csv',
       'SELECTION_FREQUENCY.csv','SELECTION_OVERLAP_PAIRS.csv','SELECTION_STABILITY.csv',
       'PAIRED_INTERVALS.csv','UTILITY_INTERVALS.csv','SIX_ENDPOINT_SIMULTANEOUS_SENSITIVITY.csv',
       'ALLOCATION_GATE.json','REPORT_STATUS.json','INDEPENDENT_EXPERIMENT_VERIFICATION.json']


if __name__=='__main__':
    assert jr(PUB/'INDEPENDENT_EXPERIMENT_VERIFICATION.json')['status']=='PASS'
    assert jr(PUB/'REPORT_STATUS.json')['status']=='COMPLETE_SEALED_INTERNAL_SIX_GROUP_SCREEN'
    for name in FILES:
        if name.endswith('.csv'):
            columns=set(pd.read_csv(PUB/name,nrows=1).columns)
            assert not columns&{'patient','patient_id','channel','channel_id','probability','prediction','y','path'}
    write(PUB/'PUBLIC_TRANSFER_MANIFEST.json',dict(status='PASS',files={n:dict(bytes=(PUB/n).stat().st_size,sha256=sha(PUB/n)) for n in FILES},
        source=source_binding(),additional_source={n:sha(HERE/n) for n in ['evaluate_sealed.py','verify_experiment.py','export_public.py']},
        no_private_arrays_models_scalers_or_identifiers=True))
    print('VERIFIED_ANONYMOUS_EXPORT_READY',len(FILES),flush=True)
