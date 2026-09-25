from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass
class RawWindow:
    patient_key:str; center:str; seizure_id:str; channel_name:str; signal:np.ndarray; sampling_rate:float; relative_start_sec:float; relative_end_sec:float; channel_label_nez:int|None; outcome_success:int; valid:bool
@dataclass
class PatientSample:
    patient_key:str; center:str; outcome_success:int; seizures:list
def masks(labels):
    x=np.asarray(labels,float); return x==0,x==1,np.isin(x,(0,1))
