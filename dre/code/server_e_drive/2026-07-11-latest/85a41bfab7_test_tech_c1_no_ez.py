import torch
from test_tech_c1_schema import patient
from neuroez_c.task2.tech_outcome_c1.data import TeChC1PatientViewBuilder
from neuroez_c.task2.tech_outcome_c1.model import TeChOutcomeC1
def test_ez_values_never_enter_view_or_change_output():
 a=patient([0,0,0]);b=patient([1,1,1]);b['seizures'][0]['windows']=a['seizures'][0]['windows'].clone();builder=TeChC1PatientViewBuilder();va,vb=builder.build(a),builder.build(b);assert 'channel_labels_nez' not in str(va);m=TeChOutcomeC1().eval()
 with torch.inference_mode():assert torch.equal(m([va])[0]['logit'],m([vb])[0]['logit'])
