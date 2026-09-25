import numpy as np

from neuroez_c.task2.ngbr.low_entropy import compute_low_entropy,normalized_spectral_entropy
from neuroez_c.task2.ngbr.schema import RawRunRecord


def test_periodic_signal_has_lower_entropy_than_white_noise():
    fs=250.;time=np.arange(250)/fs;rng=np.random.default_rng(3)
    assert normalized_spectral_entropy(np.sin(2*np.pi*10*time),fs)<normalized_spectral_entropy(rng.normal(size=250),fs)


def test_artifact_subwindow_is_excluded():
    fs=250.;onset=2500;time=np.arange(3000)/fs;rng=np.random.default_rng(8);signal=np.vstack([np.sin(2*np.pi*(8+i)*time)+.02*rng.normal(size=len(time)) for i in range(4)]);signal[0,500:750]=1e6
    record=RawRunRecord("p","c","s",signal,fs,onset,[f"A{i}" for i in range(4)],np.ones(4,bool),np.array([1,0,0,0],bool),0,10)
    _,table,_=compute_low_entropy(record)
    assert table.loc[table.channel=="A0","artifact_fraction"].iloc[0]>0
