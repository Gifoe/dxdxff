"""Read-only access to pinned original families, preserving their logical paths."""
import sys
from pathlib import Path
import numpy as np
import torch
from task_io import module,jr,read,sha,HERE

NGA=Path(r'D:\drechan_no_guidance_ablation_42_44_62_v1')
NGA_ROOT=Path(r'D:\drechan_no_guidance_ablation_42_44_62_v1_runtime')
SVM=Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1')
SVM_ROOT=Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1_runtime')
RAW=Path(r'D:\m8_raw_backbones_pr_seed42_v1')
RAW_ROOT=Path(r'D:\m8_raw_backbones_pr_seed42_v1_runtime')

class Parents:
    def __init__(self):
        self.own=module('repr_parent_nga',NGA/'common.py')
        self.binding,self.core,self.fac,self.records,self.rec,self.banks=self.own.setup(False)
        self.own_freeze=jr(NGA_ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
        self.svm_freeze=read(SVM_ROOT/'TABULAR_FREEZE_PRIVATE.pkl')
        self.svm_config=jr(SVM/'BASELINE_CONFIGS.json')['configs']['Linear_SVM']
        self.md=module('repr_tabular_original',Path(r'D:\m8_feature_pr_baselines_seed42_v1')/'models.py')
        self.bl=module('repr_linear_original',Path(r'D:\m8_14d_pr_linear_svm_adaboost_seed42_v1')/'run.py')
        sys.path.insert(0,str(RAW))
        self.raw=module('repr_raw_original_support',RAW/'support.py')
        self.raw_lock=jr(RAW/'configs/PROTOCOL_LOCK.json')
        self.raw_freeze=jr(RAW_ROOT/'TimeConvCNN/MODEL_FREEZE_PRIVATE.json')
        self.engine=module('repr_raw_original_engine',RAW/'engine.py')
        self.ev=self.raw.EV
        self.inputs=self.own.store()
        self.raw_store=self.raw.Store(self.raw.BASELINE)
        self.meta=self.rec['meta']
        assert len(self.meta['patient'])==7635 and len(set(self.meta['patient']))==80
        self.parent_protected={str(p):sha(p) for p in [NGA_ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json',NGA_ROOT/'OUTER_SCORE_SEAL_PRIVATE.json',SVM_ROOT/'TABULAR_FREEZE_PRIVATE.pkl',SVM_ROOT/'OUTER_SCORE_SEAL_PRIVATE.json',RAW_ROOT/'TimeConvCNN/MODEL_FREEZE_PRIVATE.json',RAW_ROOT/'TimeConvCNN/OUTER_PREDICTIONS_SEAL_PRIVATE.json']}
        self.refs={}
        for fold in range(1,6):
            for arm,old in [('NO_PR','WO_BOTH_PR'),('FULL_PR','FULL')]:
                r=next(x for x in self.own_freeze['cells'] if x['arm']==old and x['seed']==42 and x['fold']==fold)
                self.refs['DRECHANNET',arm,fold]=dict(r)
            for arm,old in [('NO_PR','Linear_SVM_NO_PR'),('FULL_PR','Linear_SVM_PR')]:
                r=next(x for x in self.svm_freeze['cells'] if x['arm']==old and x['fold']==fold)
                self.refs['LINEAR_SVM',arm,fold]=dict(r)
            for arm,old in [('NO_PR','Original'),('FULL_PR','PR')]:
                r=next(x for x in self.raw_freeze['selections'] if x['version']==old and x['fold']==fold)
                self.refs['TIMECONV',arm,fold]=dict(r)
        sys.path.insert(0,str(HERE))

    def bank(self,fold,full=False):return self.own.bank(fold,self.banks,'FULL' if full else 'WO_BOTH_PR')
    def ref(self,model,arm,fold):return self.refs[model,arm,fold]
    def own_model(self,fold,arm='WO_BOTH_PR'):
        return self.own.make(self.core,self.fac,self.records,fold,arm,42)
    def raw_model(self,fold,pr=False):return self.raw.build('TimeConvCNN',pr,fold,self.raw_lock,self.core)
    def own_raw(self,model,d,ix,fold,bypass=False):
        patient=str(d['patient'][ix[0]]);channels=d['channel'][ix].astype(str).tolist()
        raw,mask,times,delta,_=self.inputs.get(patient,channels,'cuda')
        fn=model._base_features if bypass else model.features
        z,_=fn(raw,mask,times,delta,self.own.scale(fold))
        return z,int(mask.sum())
    def own_forward(self,model,d,ix,fold):
        x=torch.as_tensor(d['x'][ix],device='cuda');z,n=self.own_raw(model,d,ix,fold)
        return model.classify(z,x),n
    def raw_forward(self,model,d,ix,train=False,target=None,pos=None):
        patient=str(d['patient'][ix[0]]);raw,mask=self.raw.raw_patient(self.raw_store,patient,d['channel'][ix])
        return self.engine.forward(model,raw,mask,128,train,target,pos)
    def verify_protected(self):
        for p,h in self.parent_protected.items():assert sha(p)==h,p

