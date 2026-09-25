import torch
from test_tech_c1_schema import patient
from neuroez_c.task2.tech_outcome_c1.data import TeChC1PatientViewBuilder
from neuroez_c.task2.tech_outcome_c1.model import TeChOutcomeC1
def test_channel_permutation_does_not_change_logit_or_seizure_embedding():
 v=TeChC1PatientViewBuilder().build(patient());p=torch.tensor([2,0,1]);s=v['seizures'][0];vp={'patient_key':'p','center':'x','seizures':[{**s,'windows':s['windows'][p],'window_mask':s['window_mask'][p],'phase_ids':s['phase_ids'][p],'relative_times_sec':s['relative_times_sec'][p],'channel_names':[s['channel_names'][i] for i in p]}]};m=TeChOutcomeC1().eval()
 with torch.inference_mode():a=m([v])[0];b=m([vp])[0]
 assert torch.allclose(a['logit'],b['logit'],atol=1e-6) and torch.allclose(a['seizure_embeddings'],b['seizure_embeddings'],atol=1e-6)
def test_seizure_permutation_does_not_change_patient_logit():
 v=TeChC1PatientViewBuilder().build(patient());v['seizures']=v['seizures']*2;m=TeChOutcomeC1().eval()
 with torch.inference_mode():a=m([v])[0]['logit'];v['seizures'].reverse();b=m([v])[0]['logit']
 assert torch.allclose(a,b,atol=1e-6)
def test_flattened_patient_batch_matches_individual_forward():
 v=TeChC1PatientViewBuilder().build(patient());m=TeChOutcomeC1().eval()
 with torch.inference_mode():batch=m([v,v]);single=m([v])[0]
 assert torch.allclose(batch[0]['logit'],single['logit'],atol=1e-6) and torch.allclose(batch[1]['logit'],single['logit'],atol=1e-6)
def test_cpu_bfloat16_autocast_is_finite():
 v=TeChC1PatientViewBuilder().build(patient());m=TeChOutcomeC1().eval()
 with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):value=m([v])[0]['logit']
 assert torch.isfinite(value)
