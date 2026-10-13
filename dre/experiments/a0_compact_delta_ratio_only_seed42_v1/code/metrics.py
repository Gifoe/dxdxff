"""Independent sklearn metric calculation, no model inference."""
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,confusion_matrix

def independent(y,score,tau):
 z=np.asarray(y)==0;p=np.asarray(score)<tau;tn,fp,fn,tp=confusion_matrix(z,p,labels=[False,True]).ravel()
 f=f1_score(z,p,labels=[False,True],average=None,zero_division=0);se=tp/(tp+fn) if tp+fn else np.nan;sp=tn/(tn+fp) if tn+fp else np.nan
 order=np.argsort(score,kind='stable');pos=np.flatnonzero(z[order])
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');ap=average_precision_score(z,1.0-np.asarray(score,dtype=float)) if z.any() else 0.
 return dict(macro_f1=f.mean(),ez_f1=f[1],nez_f1=f[0],balanced_accuracy=np.nanmean([se,sp]),
  ez_auprc=ap,ez_auroc=roc_auc_score(z,1.0-np.asarray(score,dtype=float)) if z.any() and (~z).any() else np.nan,
  ez_mrr=1/(pos[0]+1) if len(pos) else 0.,top1_is_ez=float(z[order[0]]),sensitivity=se,specificity=sp,
  accuracy=(tp+tn)/len(y),predicted_ez_fraction=p.mean(),TP=tp,FP=fp,TN=tn,FN=fn,channels=len(y),observed_ez_fraction=z.mean())
