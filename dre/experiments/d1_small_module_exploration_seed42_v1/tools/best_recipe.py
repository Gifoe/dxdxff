"""Reusable winning recipe: frozen D1 + zero-output morphology correction + softMF.

This factory has no patient lookup, data reader, credentials or new threshold rule.
The user supplies the exact frozen D1 and aligned original inputs.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).parents[1]
sys.path.insert(0,os.environ.get('DGM_COMPLETED_CODE',r'D:\dgm_net_seed42_v1\code'))
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'phase2/code'))
sys.path.insert(0,str(ROOT/'phase3/code'))
from direct_correction import MorphCorrection
from decision_modules import objective

def build(frozen_d1):
    """11,018 total /9,481 trainable; step0 final logits exactly frozen D1."""
    model=MorphCorrection(frozen_d1)
    assert sum(p.numel() for p in model.parameters())==11018
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==9481
    return model

def training_loss(final_nez_logit,nez_labels,fit_pos_weight):
    """One patient per update; class weight from original FIT only."""
    fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(fit_pos_weight,device=final_nez_logit.device,dtype=final_nez_logit.dtype))
    return objective(final_nez_logit,nez_labels,fn,'M8_MACRO_AWARE')
