import numpy as np

from neuroez_c.task2.ngbr.patient_aggregation import CORE_FEATURES,HFO_FEATURES,aggregate_patients
from neuroez_c.task2.ngbr.schema import P2NEZRecord


def _seizure(index):
    return {"patient_key":"p","center":"c","seizure_id":str(index),"fragility_outside_top10":.8-index*.1,"ei_outside_top10":.7,"low_entropy_outside_top10":.6,"target_to_outside_delay":1+index,"outside_recruited_3s_fraction":.5,"bio_top10":.7,"biomarker_target_alignment":.2,"top_biomarker_channels":{"A2","A3"},"joint_top10":.5,"joint_high_risk_fraction":.4,"biomarker_nez_top10_overlap":.5,"top_joint_channels":{"A3"},"hfo_outside_top10":np.nan,"hfo_target_alignment":np.nan,"hfo_hub_outside_burden":np.nan,"fragility_valid":True,"ei_valid":True,"entropy_valid":True,"hfo_valid":False}


def test_patient_representation_is_fixed_19_and_recurrence():
    p2=P2NEZRecord("p","c",["A1","A2","A3","A4"],np.array([.9,.2,.3,.4]),None,np.ones(4,bool),np.array([1,0,0,0],bool))
    table,definition,recurrence=aggregate_patients([_seizure(0),_seizure(1)],{"p":p2})
    assert len(CORE_FEATURES)==16 and len(HFO_FEATURES)==3 and len(definition)==19
    assert table.loc[0,"biomarker_channel_recurrence"]==1 and recurrence.loc[0,"recurrence_valid"]
