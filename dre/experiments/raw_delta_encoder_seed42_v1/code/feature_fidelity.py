"""Label-free, patient-local Raw/engineered comparisons; all denominators explicit."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from raw_physiological_features import FEATURES

def rows(E,R,patient,center,fit_ids):
 out=[]
 for p in sorted(fit_ids):
  take=patient==p;ce=center[take][0]
  for j,feature in enumerate(FEATURES):
   category='A' if j<2 else ('C' if j in [2,6] else 'B')
   for stat,k in [('mean',2*j),('std',2*j+1)]:
    x=E[take,k].astype(float);y=R[take,k].astype(float);ok=np.isfinite(x)&np.isfinite(y);xx=x[ok];yy=y[ok];estimable=len(xx)>=3 and len(np.unique(xx))>1 and len(np.unique(yy))>1 and xx.std()>0 and yy.std()>0
    rho=float(spearmanr(xx,yy).statistic) if estimable else np.nan;pear=float(pearsonr(xx,yy).statistic) if estimable else np.nan;mad=float(np.median(np.abs(xx-yy))) if len(xx) else np.nan
    i,h=np.triu_indices(len(xx),1);sx=np.sign(xx[i]-xx[h]);sy=np.sign(yy[i]-yy[h]);pairs=(sx!=0)&(sy!=0)
    out.append(dict(patient=p,center=ce,feature=feature,category=category,stat=stat,total_channels=int(take.sum()),paired_channels=int(ok.sum()),estimable=estimable,spearman=rho,pearson=pear,median_absolute_difference=mad,normalized_MAE=float(np.mean(np.abs(xx-yy))/max(np.median(np.abs(xx)),1e-30)) if len(xx) else np.nan,engineered_mean=float(xx.mean()) if len(xx) else np.nan,raw_mean=float(yy.mean()) if len(xx) else np.nan,engineered_variance=float(xx.var()) if len(xx) else np.nan,raw_variance=float(yy.var()) if len(xx) else np.nan,ranking_concordance=float((sx[pairs]==sy[pairs]).mean()) if pairs.any() else np.nan,estimable_channel_pairs=int(pairs.sum())))
 return pd.DataFrame(out)

def summarize(df,by):
 out=[]
 fields=['spearman','pearson','median_absolute_difference','normalized_MAE','engineered_mean','raw_mean','engineered_variance','raw_variance','ranking_concordance']
 for key,g in df.groupby(by):
  key=key if isinstance(key,tuple) else (key,);out.append(dict(zip(by,key),patients=g.patient.nunique(),patient_cases=len(g),estimable_cases=int(g.estimable.sum()),nonestimable_cases=int((~g.estimable).sum()),paired_channels=int(g.paired_channels.sum()),**{f:float(g[f].median()) if g[f].notna().any() else None for f in fields}))
 return out

def gate(df):
 checks={};medians={}
 for feature in ['log_bp_beta','log_bp_low_gamma','rms','variance','line_length_per_sec']:
  q=df[df.feature==feature];vals={s:float(q[q.stat==s].spearman.median()) if q[q.stat==s].spearman.notna().any() else None for s in ['mean','std']};medians[feature]=vals;checks[feature]=all(v is not None and v>=.8 for v in vals.values())
 return dict(required_families=4,total_AB_families=5,passing_families=int(sum(checks.values())),families=checks,patient_median_spearman=medians,all_nonestimable_cases_reported=True,median_uses_estimable_cases_only=True,pass_gate=sum(checks.values())>=4)
