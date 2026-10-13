"""Deterministic Welch, frequency, valid-window and no-training contracts."""
import sys
import tempfile
import warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from audit import bands,lock,header,quiet_edf,native_data


def synthetic_edf_reader_contract():
 """Small synthetic file: name handling and SI calibration vs original MNE."""
 import mne
 def field(value,width):return str(value).encode('latin1').ljust(width,b' ')
 labels=['A1','A1','B2','EDF Annotations'];ns=4;fs=250;nr=3
 base=b''.join(field(v,w) for v,w in [('0',8),('SYNTHETIC',80),('READER TEST',80),('01.01.01',8),('00.00.00',8),(256*(ns+1),8),('',44),(nr,8),(1,8),(ns,4)])
 spec=[(labels,16),(['']*ns,80),(['uV','mV','V',''],8),([-100,-1,-.01,-1],8),([100,1,.01,1],8),([-32768]*ns,8),([32767]*ns,8),(['HP:0 LP:100']*ns,80),([fs]*ns,8),(['']*ns,32)]
 h=b''.join(field(v,w) for vals,w in spec for v in vals)
 digital=(np.arange(nr*fs*ns,dtype=np.int32).reshape(nr,ns,fs)*17%40001-20000).astype('<i2');digital[:,3]=0
 with tempfile.TemporaryDirectory(prefix='fcpa_synthetic_edf_') as temp:
  path=Path(temp)/'synthetic.edf';path.write_bytes(base+h+digital.tobytes())
  parsed=header(path);e=quiet_edf(path)
  assert parsed['n_records']==nr and e.ch_names==['A1-0','A1-1','B2'] and e.header_indices==[0,1,2]
  ours,rate=native_data(path,[2,0,1],87,613);assert rate==fs
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   raw=mne.io.read_raw_edf(path,preload=False,verbose='ERROR')
  assert raw.ch_names==e.ch_names
  expected=raw.get_data(picks=[2,0,1],start=87,stop=613);raw.close()
  assert np.allclose(ours,expected,rtol=1e-12,atol=1e-18),np.max(np.abs(ours-expected))


def main():
 p=lock();fs=250;t=np.arange(500)/fs
 for band,freq in [('delta',2),('theta',6),('alpha',11),('beta',20)]:
  powers=bands(np.sin(2*np.pi*freq*t)[None,None,:],fs,p)
  assert np.isclose(powers[band][0,0],.5,atol=1e-8)
  assert np.isfinite(powers['total']).all()
 # Padding is never a valid full window, even if it has huge finite values.
 starts=np.arange(0,14501,250);valid=(starts>=4750)&(starts+500<=15000)
 assert (~valid).sum()==19 and starts[valid].min()==4750
 for name in ['delta','theta','alpha']:
  assert p['bands'][name][1]<fs/2
 assert p['no_high_gamma_reextraction'] and p['A0_features_preprocessing_and_sealed_tests_unchanged']
 text=(Path(__file__).parents[1]/'code/audit.py').read_text()
 assert 'train(' not in text and 'roc_auc' not in text and 'predict(' not in text
 assert 'f[\'y\']' not in text and 'f[\'x\']' not in text
 synthetic_edf_reader_contract()
 print('FCPA_SIGNAL_ONLY_ENGINEERING_TESTS_PASS',flush=True)


if __name__=='__main__':main()
