import torch
from neuroez_c.task2.tech_outcome_c1.schema import validate_patient
def patient(labels=None):
 s={'seizure_id':'s','windows':torch.randn(3,3,500),'window_mask':torch.ones(3,3,dtype=torch.bool),'phase_ids':torch.tensor([0,1,2]),'relative_times_sec':torch.tensor([-2.,1.,8.]),'channel_names':['a','b','c']}
 if labels is not None:s['channel_labels_nez']=torch.as_tensor(labels)
 return {'patient_key':'p','center':'x','outcome_success':1,'seizures':[s]}
def test_schema_accepts_variable_channel_patient():validate_patient(patient())
