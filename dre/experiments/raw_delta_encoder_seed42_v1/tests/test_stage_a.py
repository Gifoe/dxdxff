from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
import numpy as np
import pandas as pd
from raw_alignment import support,exact_slices
from construct_delta import window_delta,seizure_mean,aggregate
from raw_physiological_features import support_manifest
from feature_fidelity import rows,gate
from prepare import parity

def test_exact_samples_names_and_padding():
 t=np.array([-29,-28,0,29]);starts,valid=support(t,250,15000,500,14000,np.full(4,2));assert valid.tolist()==[False,False,True,False]
 wave=np.arange(30000,dtype=np.float32).reshape(2,15000);aligned=np.zeros((4,2,500),np.float32)
 for w in np.flatnonzero(valid):aligned[w]=wave[::-1,starts[w]:starts[w]+500]
 assert exact_slices(wave,['B','A'],['A','B'],starts,valid,aligned)==2  # channel-windows, not sample count

def test_all_negative_baseline_not_cotar_subset():
 t=np.array([-20,-15,-5,5,15]);x=np.arange(5,dtype=np.float32)[:,None,None]*np.ones((1,2,7),np.float32);mask=np.ones((5,2),bool);d,b,n=window_delta(x,t,mask)
 assert np.array_equal(b,np.ones((2,7))) and np.all(n==3) and np.array_equal(d[:,0,0],np.array([-1,0,1,2,3],np.float32))

def test_missing_baseline_and_aggregation_order():
 x=np.ones((3,2,7),np.float32);m=np.ones((3,2),bool);m[0,1]=False;d,b,n=window_delta(x,[-1,1,2],m);assert np.isnan(d[:,1]).all() and n[1]==0
 a=[np.array([[1,2]],np.float32),np.array([[3,6]],np.float32)];out,count=aggregate(a);assert np.array_equal(out,[[2,1,4,2]]) and np.all(count==2)
 one,_=aggregate(a[:1]);assert np.array_equal(one,[[1,0,2,0]])

def test_frequency_support_not_full_gamma():
 q=support_manifest();assert q['Nyquist']==125 and not q['high_gamma_full80_150_reconstructable'];assert q['band_support']['raw_partial_high_gamma_80_125']['last_bin']==125
 assert q['band_support']['beta']['last_bin']==29.5 and q['entropy_support']==[10,125]

def test_fidelity_no_constant_cases_hidden():
 patient=np.repeat(['FIT_A','FIT_B'],5);center=np.repeat(['synthetic'],10);E=np.tile(np.arange(5.)[:,None],(2,14));R=E.copy();df=rows(E,R,patient,center,{'FIT_A','FIT_B'});assert gate(df)['pass_gate']
 E[:,:4]=0;df=rows(E,R,patient,center,{'FIT_A','FIT_B'});g=gate(df);assert not g['pass_gate'] and g['passing_families']==3 and (~df.estimable).sum()==8

def test_E14_parity_gate():
 a=np.arange(14,dtype=np.float32).reshape(1,14);assert parity(a,a)['pass_gate'];b=a.copy();b[0,3]+=.01;assert not parity(a,b)['pass_gate']

def test_original_per_coordinate_float32_seizure_arithmetic():
 a=np.random.default_rng(42).normal(1e-5,1e-8,size=(5,3,44)).astype(np.float32);out,_=aggregate(a)
 for c in range(3):
  for f in range(44):assert out[c,2*f]==np.nanmean(a[:,c,f]) and out[c,2*f+1]==np.nanstd(a[:,c,f],ddof=0)
