import torch
from neuroez_c.task2.native_ssl_n2.data import Task2NativeViewBuilder,ViewConfig
def test_native_view_never_contains_channel_labels():
 p={'patient_key':'p','center':'x','outcome_success':1,'seizures':[{'seizure_id':'s','windows':torch.randn(3,3,500),'side_features':torch.randn(3,3,12),'window_mask':torch.ones(3,3,dtype=torch.bool),'phase_ids':torch.tensor([0,1,2]),'relative_times_sec':torch.arange(3.),'channel_names':['a','b','c'],'channel_labels_nez':torch.tensor([0,1,0])}]};v=Task2NativeViewBuilder(42,ViewConfig()).view(p,0);assert 'channel_labels_nez' not in str(v) and 'outcome_success' not in v
