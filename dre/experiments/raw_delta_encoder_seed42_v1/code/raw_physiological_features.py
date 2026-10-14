"""Pinned exact physiology formulas; original samples, no additional filtering."""
import numpy as np
from io_utils import load,sha

FEATURES=['log_bp_beta','log_bp_low_gamma','log_bp_high_gamma','rms','variance','line_length_per_sec','spectral_entropy']

class Extractor:
 def __init__(self,source,expected):
  path=source/'ez_features.py';assert sha(path)==expected;self.module=load('rawdelta_pinned_physiology',path)
  self.indices=[list(self.module.BASE_SPECTRAL_FEATURE_NAMES).index(n) for n in FEATURES]
 def __call__(self,wave):
  return self.all(wave)[...,self.indices]
 def all(self,wave):
  assert wave.shape[-1]==500
  out=self.module.compute_spectral_channel_features(wave.reshape(-1,500),250,spectral_min_freq=10,spectral_max_freq=150)
  return out.reshape(*wave.shape[:-1],len(self.module.BASE_SPECTRAL_FEATURE_NAMES))

def support_manifest():
 f=np.fft.rfftfreq(500,1/250);allowed=f[(f>=10)&(f<=150)]
 return dict(sfreq=250,Nyquist=125,bin_step=.5,nperseg=500,noverlap=250,detrend=False,spectral_restricted_bins=len(allowed),entropy_support=[float(allowed.min()),float(allowed.max())],band_support={n:dict(first_bin=float(allowed[(allowed>=lo)&(allowed<hi)].min()),last_bin=float(allowed[(allowed>=lo)&(allowed<hi)].max()),bins=int(((allowed>=lo)&(allowed<hi)).sum())) for n,lo,hi in [('beta',13,30),('low_gamma',30,80),('raw_partial_high_gamma_80_125',80,150)]},high_gamma_full80_150_reconstructable=False,extra_filters_or_upsampling=False)
