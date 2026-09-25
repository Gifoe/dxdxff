from test_trace_aux_stable_model import seizure
from neuroez_c.task2.trace_aux_stable.model import TraceAuxStable
def test_ez_nez_routing():
 o=TraceAuxStable().eval()([{'patient_key':'p','center':'x','outcome_success':1,'view_id':0,'seizures':[seizure([0,1,0,1])]}])[0];assert o['seizure_audit'][0]['n_ez']==2 and o['seizure_audit'][0]['n_nez']==2
