import torch
from neuroez_c.task2.native_ssl_n2.augment import augment
def test_augmentation_preserves_shape_and_masks():
 w=torch.randn(2,8,500); s=torch.randn(2,8,12); m=torch.ones(2,8,dtype=torch.bool); a,b,c=augment(w,s,m,torch.Generator().manual_seed(1)); assert a.shape==w.shape and b.shape==s.shape and c.shape==m.shape and (~c).any()
