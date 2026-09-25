import numpy as np
from neuroez_c.task2.mosaic.schema import MosaicRunRecord
from neuroez_c.task2.mosaic.spectral_state import compute_spectral_state

def test_spectral_band_ranking():
    fs=250.; t=np.arange(0,12,1/fs); signal=np.vstack([np.sin(2*np.pi*10*t),np.sin(2*np.pi*20*t),np.sin(2*np.pi*40*t),np.sin(2*np.pi*80*t)])
    r=MosaicRunRecord("p","hup","s",signal,fs,int(10*fs),list("ABCD"),np.ones(4,bool),np.array([1,1,0,0],bool)); state,_=compute_spectral_state(r)
    assert np.nanmedian(state.maps["alpha"][:,0])>np.nanmedian(state.maps["alpha"][:,2])
    assert np.nanmedian(state.maps["high_gamma"][:,3])>np.nanmedian(state.maps["high_gamma"][:,0])
