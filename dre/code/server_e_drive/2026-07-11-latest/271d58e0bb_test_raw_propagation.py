import numpy as np
from neuroez_c.task2.cop.raw_propagation import build_raw_propagation,recruitment_times
from neuroez_c.task2.cop.schema import RawRunRecord


def test_raw_recruitment_detects_persistent_post_onset_change():
    fs=250.;rng=np.random.default_rng(4);signal=rng.normal(0,.1,(4,2500));onset=1250;signal[0,onset:onset+500]+=4*np.sign(np.sin(np.arange(500)*np.pi/2));record=RawRunRecord("p","c","s",signal,fs,onset,["A","B","C","D"],np.ones(4,dtype=bool),np.array([1,1,0,0],dtype=bool))
    times,audit=recruitment_times(record);assert np.isfinite(times[0]) and times[0]>=0 and audit["gamma_available"]
    table,channels,seizures=build_raw_propagation([record]);assert len([c for c in table if c.startswith("raw__")])==12 and len(channels)==4 and len(seizures)==1
