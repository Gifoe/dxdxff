import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
import numpy as np
from construct_patient_context import construct
from synchronization_controls import shift
from summarize_temporal_features import summarize,across_seizures,schema
from validated_alignment import extract,FEATURES
from fit_probe import plans,fit_preprocessor,transform,threshold
from common import metric,patient_weights
from bootstrap import paired

def fixture():
 t=np.r_[np.arange(-4,0),np.arange(8),np.arange(8,20)];x=np.random.default_rng(42).normal(size=(len(t),7,7));m=np.ones(x.shape[:2],bool);return t,x,m

def test_query_exclusion():
 t,x,m=fixture();a=construct(x,t,m);y=x.copy();y[t>=0,0]+=10000;b=construct(y,t,m);assert np.array_equal(a['m'][t>=0,0],b['m'][t>=0,0])
 for ti in range(len(t)):
  for c in range(7):assert np.allclose(a['m'][ti,c],np.median(np.delete(a['d'][ti],c,axis=0),axis=0))

def test_permutation():
 t,x,m=fixture();p=np.array([5,1,0,3,6,2,4]);a=construct(x,t,m);b=construct(x[:,p],t,m[:,p]);assert np.allclose(a['u'][:,p],b['u'])
 names=[str(i) for i in range(7)];aa,au,_=shift(a['d'],a['m'],t,names,['p','s','r']);bb,bu,_=shift(b['d'],b['m'],t,[names[i] for i in p],['p','s','r']);assert np.allclose(aa[:,p],bb,equal_nan=True)

def test_missing_exclusion_and_minimum():
 t,x,m=fixture();m[:,6]=False;a=construct(x,t,m);y=x.copy();y[:,6]=1e20;b=construct(y,t,m);assert np.allclose(a['u'],b['u'],equal_nan=True)
 assert np.isnan(a['u'][:,6]).all();m[:,4:]=False;a=construct(x,t,m);assert np.isnan(a['m']).all()

def test_same_time_and_seizure():
 t,x,m=fixture();a=construct(x,t,m);y=x.copy();ti=np.flatnonzero(t==1)[0];y[ti,1:]+=100;b=construct(y,t,m);assert np.allclose(a['m'][np.arange(len(t))!=ti,0],b['m'][np.arange(len(t))!=ti,0]);assert np.allclose(b['m'][ti,0]-a['m'][ti,0],100)
 other=construct(x+77,t,m);assert np.allclose(other['u'],a['u']);assert np.array_equal(a['m'],construct(x,t,m)['m'])

def record():
 rng=np.random.default_rng(42);t=np.arange(-29,30,dtype=float);x=rng.normal(size=(59,7,9));return dict(channel_names_norm=[str(j) for j in range(7)],sfreq=250,sample=dict(window_feature_names=FEATURES,window_features=x,window_relative_centers_sec=t,raw_temporal_sfreq=250,raw_temporal_duration_sec=60,seizure_onset_sec=100,start_sec=92,raw_valid_start_sample=5500,raw_valid_samples=9500,feature_scale_used_secs=np.full(59,2.)))

def test_padding_and_channel_names():
 r=record();a,t,m,detail=extract(r,['3','0','ABSENT']);assert not m[2].any();assert detail['missing_channels']==1 and detail['invalid_boundary_windows']>0
 rr=record();rr['sample']['window_features'][rr['sample']['window_relative_centers_sec']<-7]=1e20;_,tt,mm,_=extract(rr,['3','0','ABSENT']);assert np.array_equal(m,mm)
 assert not m[:2,t[0]<-7].any()

def test_baseline_unavailable():
 t,x,m=fixture();m[t<0,0]=False;a=construct(x,t,m);assert np.isnan(a['b'][0]).all() and np.isnan(a['d'][:,0]).all() and (a['baseline_counts'][0]==0).all()

def test_shift_determinism_masks_stages():
 t,x,m=fixture();a=construct(x,t,m);a['m'][t==2,0]=np.nan;u,v,r=shift(a['d'],a['m'],t,list(map(str,range(7))),['p','s','r']);uu,vv,rr=shift(a['d'],a['m'],t,list(map(str,range(7))),['p','s','r'])
 assert np.array_equal(u,uu,equal_nan=True);assert np.array_equal(np.isfinite(u),np.isfinite(a['m']));assert np.array_equal(u[t<0],a['m'][t<0]);assert all(q['offset']>0 for q in r if q['eligible'])
 for lo,hi in [(0,8),(8,20)]:
  for c in range(7):
   for k in range(7):
    ix=(t>=lo)&(t<hi)&np.isfinite(a['m'][:,c,k]);assert np.array_equal(np.sort(u[ix,c,k]),np.sort(a['m'][ix,c,k]))

def test_summary_counts_actual_time():
 t=np.array([-1,0,2,8,19,20],float);v=np.broadcast_to((3*t+2)[:,None,None],(6,2,7)).copy();s,c=summarize(v,t);assert np.allclose(s[:,2::3],3);assert np.array_equal(c,np.full((2,2,7),2))
 v[2]=np.nan;s,c=summarize(v,t);assert (s[:,:21].reshape(2,7,3)[:,:,2]==0).all() and (c[:,0]==1).all()
 a,n=across_seizures([s]);assert np.nanmax(a[:,42:])==0;assert len(schema([str(i) for i in range(7)]))==182

def test_summary_mean_shift_identity():
 t,x,m=fixture();a=construct(x,t,m);_,v,_=shift(a['d'],a['m'],t,list(map(str,range(7))),['p','s','r']);s,_=summarize(a['u'],t);ss,_=summarize(v,t);assert np.allclose(s[:,::3],ss[:,::3],atol=1e-12,rtol=0)

def test_preprocessor_train_only_width_weights():
 rng=np.random.default_rng(42);x=rng.normal(size=(12,182));pid=np.array(['a']*2+['b']*5+['held']*5);pre=fit_preprocessor(x[:7],pid[:7]);changed=x.copy();changed[7:]=1e20;other=fit_preprocessor(changed[:7],pid[:7]);assert np.array_equal(pre['median'],other['median']);assert transform(pre,x).shape==(12,182)
 w=patient_weights(pid[:7]);assert np.isclose(w[:2].sum(),1) and np.isclose(w[2:].sum(),1)
 neutral=x.copy();neutral[:,98:]=0;pre=fit_preprocessor(neutral[:7],pid[:7]);assert not transform(pre,neutral)[:,98:].any()

def test_strict_plans():
 ps=plans([str(i) for i in range(51)],1);seen=[]
 for q in ps:assert not set(q['held'])&(set(q['train'])|set(q['select']));seen+=q['held']
 assert len(seen)==len(set(seen))==51 and plans([str(i) for i in range(51)],1)==ps

def test_threshold_and_patient_metric():
 y=np.array([0,1,1,0,1,0]);p=np.array([.8,.2,.4,.3,.7,.6]);pid=np.array(['a','a','a','b','b','b']);s=threshold(y,p,pid);candidates=[]
 for t in np.round(np.arange(0,1.0001,.005),3):
  q=[metric(y[pid==q],p[pid==q],t) for q in ['a','b']];candidates.append((np.mean([a['macro_f1'] for a in q]),np.mean([a['ez_f1'] for a in q]),np.mean([a['balanced_accuracy'] for a in q]),-abs(t-.5),-t,t))
 assert s['threshold']==max(candidates)[-1]

def test_bootstrap_repeated_patient():
 import pandas as pd
 rows=[]
 for fold,pid in [(1,'a'),(2,'a'),(1,'b'),(2,'c')]:
  for arm,value in [('C0',.5),('C1',.6 if pid=='a' else .55)]:rows.append(dict(fold=fold,inner=0,patient=pid,arm=arm,ez_auroc=value))
 df=pd.DataFrame(rows);a=paired(df,'C1','C0',['ez_auroc']);b=paired(df,'C1','C0',['ez_auroc']);assert a==b and a[0]['unique_patient_clusters']==3 and a[0]['appearances']==4
