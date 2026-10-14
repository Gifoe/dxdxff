from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
import numpy as np
from evaluate_outer import frozen_transform,role_ids,checked_cached
from io_utils import save

class Core:
 @staticmethod
 def patient_z(x,patient):
  a=np.asarray(x,dtype=float).copy()
  for p in np.unique(patient):
   ix=patient==p;v=a[ix];s=np.nanstd(v,axis=0);a[ix]=(v-np.nanmean(v,axis=0))/np.where(s>1e-8,s,1.)
  return a
def test_frozen_scaler_applied_without_refit():
 x=np.array([[1.,2.],[3.,4.],[10.,8.],[12.,10.]]);pid=np.array(['A','A','B','B']);pre=dict(imputer_statistics=np.array([1.,1.]),mean=np.array([.3,.5]),scale=np.array([2.,4.]));out=frozen_transform(x,pid,pre,Core)
 assert np.array_equal(out,((np.array([[-1,-1],[1,1],[-1,-1],[1,1]])-pre['mean'])/pre['scale']).astype(np.float32));assert np.array_equal(pre['mean'],[.3,.5])
def test_roles_and_overlap_fail_closed():
 q=dict(split_fold=np.array([1,1,1]),split_role=np.array(['fit','validation','test']),split_patient=np.array(['A','B','C']));assert role_ids(q,1)['test']=={'C'};q['split_patient'][2]='A'
 try:role_ids(q,1)
 except AssertionError:pass
 else:raise AssertionError('role overlap accepted')
def test_private_prediction_resume_requires_identical_binding(tmp_path):
 p=tmp_path/'scores.pkl';save(p,dict(binding='frozen',checkpoint_sha256='weights',indices=np.array([1,2]),logit=np.array([.1,.2])))
 checked_cached(p,'frozen','weights',np.array([1,2]))
 for bind,cp,ix in [('changed','weights',[1,2]),('frozen','other',[1,2]),('frozen','weights',[2,1])]:
  try:checked_cached(p,bind,cp,np.array(ix))
  except AssertionError:pass
  else:raise AssertionError('changed frozen prediction cache accepted')
