import inspect,torch,pytest
from neuroez_c.task2.trace_dre_lite.cache_builder import build_patient
from neuroez_c.task2.trace_dre_lite.model import TRACEDRELiteV2
from neuroez_c.task2.trace_dre_lite.propagation import PropagationEscape
def test_zero_seizure_patient_is_rejected(tmp_path):
 with pytest.raises(RuntimeError,match='no valid seizure'):build_patient('p',[],1,tmp_path,False)
def test_side_ablation_disables_encoder_and_propagation():
 m=TRACEDRELiteV2(disable_side_features=True);assert not m.encoder.use_side and not m.prop.use_side_features
 c,t=3,8;r={'pre':torch.randn(c,32),'onset':torch.randn(c,32),'spread':torch.randn(c,32),'curve':torch.randn(c,t),'tau':torch.arange(c).float()};labels=torch.tensor([0,1,1]);side=torch.randn(c,t,12);mask=torch.ones(c,t,dtype=torch.bool);p=m.prop(r,labels,side,mask);assert torch.all(p['ez_descriptor'][-4:]==0) and torch.all(p['nez_descriptor'][:,-4:]==0)
def test_runner_has_no_inner_split_or_stratified_import():
 text=open('P2_Q10_NPAM/scripts/task2/run_trace_dre_lite_v2.py',encoding='utf8').read();assert 'StratifiedShuffleSplit' not in text and 'trace_dre_inner_validation_manifest' not in text and 'validation_selection' not in text
