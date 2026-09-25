import pandas as pd
from neuroez_c.task2.mosaic.pilot_cohort import CENTERS,select_pilot16,build_pilot_outer_folds

def test_fixed_pilot_and_folds():
    rows=[]
    for c in CENTERS:
        for y in (0,1):
            for i in range(4): rows.append(dict(patient_key=f"{c}_{y}_{i}",center=c,outcome_true=y,original_outer_fold=i%5+1,n_seizures=i+1,n_valid_channels=10,raw_feature_match_rate=.99,target_source="clinical",valid_channel_ratio=.9,raw_duration_sec=20,missing_field_count=0))
    pilot=select_pilot16(pd.DataFrame(rows),resume=False); folds=build_pilot_outer_folds(pilot)
    assert len(pilot)==16 and pilot.groupby(["center","outcome_true"]).size().eq(2).all()
    assert folds.groupby("outer_fold").center.nunique().eq(4).all()
    assert folds.groupby("outer_fold").outcome_true.sum().eq(2).all()
