"""Pinned original Development dependencies; never load an Outer output."""
import os
import sys
from pathlib import Path
import numpy as np  # Required before conda torch: do not use KMP_DUPLICATE_LIB_OK.
import pandas as pd
import torch

BASE_CODE = Path(os.environ.get('DGM_BASE_CODE', r'E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code'))
sys.path.insert(0, str(BASE_CODE))
from io_utils import sha, write, read, save, load
from dependencies import context
from compact_delta_classifier import CompactDelta
from raw_store import Store
from raw_physiological_features import Extractor, FEATURES
from construct_delta import window_delta, aggregate, historical_record_mean

def original_modules():
    metrics = load('dgm_original_evaluate', BASE_CODE/'evaluate.py')
    boot = load('dgm_original_bootstrap', BASE_CODE/'bootstrap.py')
    return metrics, boot

def make_base(core, fac, records, checkpoint, seed):
    core.seed_all(seed)
    base = CompactDelta(fac(seed).network, records)
    base.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    return base

def model_fingerprint(core, model):
    return core.state_hash(model.state_dict())
