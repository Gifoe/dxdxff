import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader
from task1_clean_nez.model import CleanNEZModel
from task1_clean_nez.trainer import dry_initialize_model
def batch():
    b,s,t,c=1,2,4,3
    return {"b0_features":torch.randn(b,s,t,c,36),"physics_features":torch.randn(b,s,t,c,12),"window_centers":torch.arange(t).float().expand(b,s,t),"window_mask":torch.ones(b,s,t,dtype=torch.bool),"seizure_channel_mask":torch.ones(b,s,c,dtype=torch.bool),"seizure_mask":torch.ones(b,s,dtype=torch.bool),"channel_mask":torch.ones(b,c,dtype=torch.bool),"labels_nez":torch.ones(b,c),"labels_ez":torch.zeros(b,c)}
def test_dry_init_materializes_lazy_parameters():
    model=CleanNEZModel(); loader=DataLoader([batch()],batch_size=None); dry_initialize_model(model,loader,torch.device("cpu")); assert not any(isinstance(p,UninitializedParameter) for p in model.parameters())
