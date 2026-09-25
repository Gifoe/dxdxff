import numpy as np

from neuroez_c.task2.ngbr.epileptogenicity_index import compute_ei,detect_persistent
from neuroez_c.task2.ngbr.schema import RawRunRecord


def _record(fs=250.):
    rng=np.random.default_rng(4);onset=int(11*fs);time=np.arange(int(22*fs))/fs;signal=.15*rng.normal(size=(4,len(time)))
    signal[0,onset:]+=3*np.sin(2*np.pi*45*time[:len(time)-onset])
    return RawRunRecord("p","c","s",signal,fs,onset,[f"A{i}" for i in range(4)],np.ones(4,bool),np.array([1,0,0,0],bool),0,11)


def test_early_high_frequency_channel_has_higher_ei():
    score,table,audit=compute_ei(_record())
    assert audit["ei_valid"] and score[0]>=np.nanmax(score[1:])


def test_undetected_channel_has_zero_latency_contribution():
    z=np.zeros((2,20));times=np.linspace(-1,1,20)
    assert np.isnan(detect_persistent(z,times)).all()


def test_low_sampling_rate_ei_invalid():
    _,_,audit=compute_ei(_record(60.))
    assert not audit["ei_valid"]
