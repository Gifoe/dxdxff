"""Regenerate the three locked explanatory R4 PCA panels."""
import argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit_core',HERE/'run_audit.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
p=argparse.ArgumentParser();p.add_argument('--train-repr',type=Path,required=True);p.add_argument('--test-repr',type=Path,required=True);p.add_argument('--train-pred',type=Path,required=True);p.add_argument('--test-pred',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
tr,xt,_,_,_=core.load_repr(a.train_repr,'TRAIN');te,xe,_,_,_=core.load_repr(a.test_repr,'TEST')
tp=pd.read_csv(a.train_pred).rename(columns={'patient':'patient_id','edf':'edf_name','channel':'channel_name','y_true':'label'});ep=pd.read_csv(a.test_pred).rename(columns={'patient':'patient_id','edf':'edf_name','channel':'channel_name','y_true':'label'})
keys=['patient_id','edf_name','channel_name'];tr=tr.merge(tp[keys+['label']],on=keys,how='left');te=te.merge(ep[keys+['label']],on=keys,how='left');tr.label=tr.label.where(tr.label.isin([0,1]));te.label=te.label.where(te.label.isin([0,1]))
df=pd.concat([tr,te],ignore_index=True);X=np.vstack([xt['R4'],xe['R4']])[:,:32];ia=np.random.default_rng(42).choice(len(tr),min(5000,len(tr)),False);ib=np.random.default_rng(43).choice(len(te),min(5000,len(te)),False);ix=np.r_[ia,len(tr)+ib]
pc=PCA(2,random_state=42).fit_transform(StandardScaler().fit_transform(X[ix]));df=df.iloc[ix].reset_index(drop=True);df['label_display']=df.label.map({0.0:'normal',1.0:'pathological'}).fillna('unlabeled')
for col,name in [('split','pca_train_test.pdf'),('center','pca_center.pdf'),('label_display','pca_label.pdf')]:
    fig,ax=plt.subplots(figsize=(7,5));vals=df[col].astype(str)
    for value in sorted(vals.unique()):
        q=vals==value;ax.scatter(pc[q,0],pc[q,1],s=5,alpha=.28,label=value,rasterized=True)
    ax.set(xlabel='PC1',ylabel='PC2');ax.legend(markerscale=3,fontsize=7);fig.tight_layout();fig.savefig(a.output/name);plt.close(fig)
