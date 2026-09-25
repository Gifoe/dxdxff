from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


CONTROLS=("none","COP_TARGET_PERMUTATION","COP_RAW_CHANNEL_PERMUTATION","COP_OUTCOME_PERMUTATION")


def permute_target_lookup(lookup:Mapping[str,Mapping[str,Any]],seed:int)->dict[str,dict[str,Any]]:
    output={}
    for patient,entry in lookup.items():
        local=dict(entry);key="clinical_target_mask" if "clinical_target_mask" in entry else "mask";values=np.asarray(entry[key],dtype=bool);digest=int(hashlib.sha256(f"{seed}|{patient}".encode()).hexdigest()[:8],16);local[key]=np.random.default_rng(digest).permutation(values);output[str(patient)]=local
    return output


def permute_training_outcomes(y:np.ndarray,seed:int)->np.ndarray:return np.random.default_rng(seed).permutation(np.asarray(y,dtype=int))


__all__=["CONTROLS","permute_target_lookup","permute_training_outcomes"]
