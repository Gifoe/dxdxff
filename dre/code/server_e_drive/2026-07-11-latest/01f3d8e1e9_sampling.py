from __future__ import annotations
import numpy as np
def sampled_seizures(patient,rng,max_seizures=3,max_ez=4,max_nez=12,max_windows=2):
 value=list(patient.seizures); rng.shuffle(value); return value[:max_seizures]
