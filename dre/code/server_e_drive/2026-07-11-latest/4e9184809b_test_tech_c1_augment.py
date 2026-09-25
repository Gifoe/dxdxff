import torch
from neuroez_c.task2.tech_outcome_c1.augment import augment_seizure,NAMES
def test_allowed_augmentations_preserve_shapes_and_no_flip():
 w=torch.randn(8,5,500);m=torch.ones(8,5,dtype=torch.bool)
 for name in NAMES:a,b,_=augment_seizure(w,m,torch.Generator().manual_seed(1),name);assert a.shape==w.shape and b.shape==m.shape and b.any(1).sum()>=2
 assert 'flip' not in NAMES
