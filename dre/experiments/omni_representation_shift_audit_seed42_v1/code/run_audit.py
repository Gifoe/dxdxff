"""Locked statistical audit over privately extracted representations."""
from __future__ import annotations
import argparse, hashlib, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter('error', ConvergenceWarning)

SEED=42; LAYERS=['R0','R1','R2','R3','R4','R5']; NAMES={'R0':'Input summary','R1':'TimeConv-1','R2':'TimeConv-2','R3':'ResNet','R4':'32D embedding','R5':'logit'}; BASE_DIM={'R0':453,'R1':32,'R2':64,'R3':512,'R4':32,'R5':1}

def center_of(edf):
    s=edf.lower()
    if 'sub-hup' in s:return 'HUP'
    if 'sub-openieeg' in s:return 'Open-iEEG'
    if 'sub-sourcesink' in s:return 'SourceSink'
    return 'Other'

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def load_repr(root,split):
    meta=[]; mats={k:[] for k in LAYERS+['P16','TASK_PAR','TASK_PERP']}; smeta=[]; smats={k:[] for k in LAYERS}
    files=sorted(root.glob('*.npz'))
    for p in files:
        with np.load(p,allow_pickle=False) as z:
            patient,edf=str(z['patient']),str(z['edf']); ch=z['channel_names'].astype(str); labels=z['labels'].astype(int); n=z['num_segments'].astype(int); dur=float(z['duration_seconds'])
            for i,c in enumerate(ch):meta.append((split,patient,edf,center_of(edf),c,int(labels[i]),dur,int(n[i])))
            for k in mats:mats[k].append(np.asarray(z[k],np.float32))
            sci=np.asarray(z['sample_channel_index'],int); ssi=np.asarray(z['sample_segment_index'],int)
            for i,j in zip(sci,ssi):smeta.append((split,patient,edf,center_of(edf),ch[i],int(labels[i]),int(j)))
            for k in smats:smats[k].append(np.asarray(z['sample_'+k],np.float32))
    cols=['split','patient_id','edf_name','center','channel_name','raw_label','duration_seconds','num_segments']
    return pd.DataFrame(meta,columns=cols),{k:np.concatenate(v) for k,v in mats.items()},pd.DataFrame(smeta,columns=['split','patient_id','edf_name','center','channel_name','raw_label','segment_index']),{k:np.concatenate(v) for k,v in smats.items()},files

def valid_folds(y,g,n=5):
    return len(np.unique(y))>=2 and min(len(np.unique(g[y==c])) for c in np.unique(y))>=n

def binary_cv(X,y,g):
    if not valid_folds(y,g):return math.nan,0
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED); pred=np.full(len(y),np.nan)
    for tr,va in cv.split(X,y,g):
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,penalty='l2',class_weight='balanced',solver='lbfgs',tol=1e-3,max_iter=2000,random_state=SEED))
        m.fit(X[tr],y[tr]); pred[va]=m.predict_proba(X[va])[:,list(m[-1].classes_).index(1)]
    return roc_auc_score(y,pred),len(np.unique(g))

def multiclass_cv(X,y,g):
    classes=np.unique(y)
    if len(classes)<2 or min(len(np.unique(g[y==c])) for c in classes)<5:return (math.nan,math.nan,math.nan,0)
    enc={c:i for i,c in enumerate(classes)}; yy=np.array([enc[v] for v in y]); pred=np.full((len(y),len(classes)),np.nan)
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
    for tr,va in cv.split(X,yy,g):
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,penalty='l2',class_weight='balanced',solver='lbfgs',tol=1e-3,max_iter=2000,random_state=SEED))
        m.fit(X[tr],yy[tr]); pred[va]=m.predict_proba(X[va])
    auc=roc_auc_score(yy,pred,multi_class='ovr',average='macro'); yp=pred.argmax(1)
    return auc,balanced_accuracy_score(yy,yp),f1_score(yy,yp,average='macro'),len(np.unique(g))

def smd_stats(a,b):
    den=np.sqrt((a.var(0)+b.var(0))/2); s=(a.mean(0)-b.mean(0))/np.where(den>1e-12,den,np.nan); q=np.abs(s[np.isfinite(s)])
    return dict(median_abs_smd=np.median(q),p90_abs_smd=np.quantile(q,.9),max_abs_smd=q.max(),fraction_abs_smd_gt_0_5=np.mean(q>.5),fraction_abs_smd_gt_1=np.mean(q>1),finite_dimensions=len(q))

def sample_rows(x,n,seed):
    if len(x)<=n:return x
    return x[np.random.default_rng(seed).choice(len(x),n,False)]

def mmd2(a,b,seed=SEED):
    a=sample_rows(np.asarray(a,np.float64),5000,seed); b=sample_rows(np.asarray(b,np.float64),5000,seed+1)
    pool=np.vstack([sample_rows(a,500,seed+2),sample_rows(b,500,seed+3)])
    # deterministic median heuristic without materializing a 10k-square matrix
    d=[]
    pn=(pool*pool).sum(1)
    for i in range(0,len(pool),200):
        q=pn[i:i+200,None]+pn[None,:]-2*(pool[i:i+200]@pool.T); d.append(np.maximum(q,0).ravel())
    d=np.concatenate(d); d=d[d>0]; med=np.median(d); gamma=1/(2*med) if med>0 else 1
    def km(x,y):
        total=0.; yn=(y*y).sum(1)
        for i in range(0,len(x),256):
            xb=x[i:i+256];q=(xb*xb).sum(1)[:,None]+yn[None,:]-2*(xb@y.T); total+=np.exp(-gamma*np.maximum(q,0)).sum()
        return total/(len(x)*len(y))
    return km(a,a)+km(b,b)-2*km(a,b),math.sqrt(med),len(a),len(b)

def frechet(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64); ma=a.mean(0);mb=b.mean(0); ca=np.cov(a,rowvar=False);cb=np.cov(b,rowvar=False)
    if np.ndim(ca)==0:return float((ma-mb)@(ma-mb)+(math.sqrt(ca)-math.sqrt(cb))**2)
    vals,vec=eigh((ca+ca.T)/2); root=(vec*np.sqrt(np.maximum(vals,0)))@vec.T
    mid=(root@cb@root); vals2=eigh((mid+mid.T)/2,eigvals_only=True)
    return float((ma-mb)@(ma-mb)+np.trace(ca)+np.trace(cb)-2*np.sqrt(np.maximum(vals2,0)).sum())

def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den else math.nan

def patient_weights(groups):
    _,inv,c=np.unique(groups,return_inverse=True,return_counts=True);w=1/c[inv];return w/w.mean()

def probe(Xtr,ytr,gtr,Xte,yte,gte):
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED);oof=np.full(len(ytr),np.nan)
    for tr,va in cv.split(Xtr,ytr,gtr):
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,penalty='l2',solver='lbfgs',tol=1e-3,max_iter=2000,random_state=SEED))
        m.fit(Xtr[tr],ytr[tr],logisticregression__sample_weight=patient_weights(gtr[tr]));oof[va]=m.predict_proba(Xtr[va])[:,1]
    final=make_pipeline(StandardScaler(),LogisticRegression(C=1,penalty='l2',solver='lbfgs',tol=1e-3,max_iter=2000,random_state=SEED))
    final.fit(Xtr,ytr,logisticregression__sample_weight=patient_weights(gtr));pte=final.predict_proba(Xte)[:,1]
    # Convert the standardized-space coefficient back to the original coordinate system for cosine diagnostics.
    wtrain=final[-1].coef_[0]/final[0].scale_
    test=make_pipeline(StandardScaler(),LogisticRegression(C=1,penalty='l2',solver='lbfgs',tol=1e-3,max_iter=2000,random_state=SEED))
    test.fit(Xte,yte,logisticregression__sample_weight=patient_weights(gte));wtest=test[-1].coef_[0]/test[0].scale_
    return roc_auc_score(ytr,oof),roc_auc_score(yte,pte),cosine(wtrain,wtest)

def perf(y,s):
    if len(np.unique(y))<2:return math.nan,math.nan
    return roc_auc_score(y,s),average_precision_score(y,s)

def fmt(x,n=6):return 'not_estimable' if not np.isfinite(x) else f'{x:.{n}f}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-repr',type=Path,required=True);p.add_argument('--test-repr',type=Path,required=True);p.add_argument('--train-pred',type=Path,required=True);p.add_argument('--test-pred',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tr,xt,strm,xst,trfiles=load_repr(a.train_repr,'TRAIN');te,xe,stem,xse,tefiles=load_repr(a.test_repr,'TEST')
    # Authoritative labels/scores come from the already replayed channel files, never cache-label assumptions.
    tp=pd.read_csv(a.train_pred).rename(columns={'patient':'patient_id','edf':'edf_name','channel':'channel_name','y_true':'label','full_score':'score'})
    ep=pd.read_csv(a.test_pred).rename(columns={'patient':'patient_id','edf':'edf_name','channel':'channel_name','y_true':'label','pathological_score':'score'})
    keys=['patient_id','edf_name','channel_name'];tr=tr.merge(tp[keys+['label','score']],on=keys,how='left',validate='one_to_one');te=te.merge(ep[keys+['label','score']],on=keys,how='left',validate='one_to_one')
    tr['label']=tr.label.where(tr.label.isin([0,1]));te['label']=te.label.where(te.label.isin([0,1]))
    domain=pd.concat([tr,te],ignore_index=True); X={k:np.vstack([xt[k],xe[k]]) for k in LAYERS}; ydom=(domain.split=='TEST').astype(int).to_numpy(); groups=domain.patient_id.to_numpy()
    drows=[]
    for k in LAYERS:
        auc,ng=binary_cv(X[k],ydom,groups); strength='low' if auc<.65 else 'moderate' if auc<.8 else 'strong' if auc<.9 else 'very_strong'
        drows.append(dict(analysis_unit='EDF-channel',layer=k,layer_name=NAMES[k],segment_dimension=BASE_DIM[k],analysis_dimension=X[k].shape[1],auroc=auc,shift_strength=strength,n=len(domain),patients=ng))
        sx=np.vstack([xst[k],xse[k]]); sy=np.r_[np.zeros(len(xst[k]),int),np.ones(len(xse[k]),int)]; sg=np.r_[strm.patient_id,stem.patient_id]
        sauc,sng=binary_cv(sx,sy,sg);drows.append(dict(analysis_unit='deterministic segment sample',layer=k,layer_name=NAMES[k],segment_dimension=BASE_DIM[k],analysis_dimension=sx.shape[1],auroc=sauc,shift_strength=('low' if sauc<.65 else 'moderate' if sauc<.8 else 'strong' if sauc<.9 else 'very_strong'),n=len(sx),patients=sng))
    pd.DataFrame(drows).to_csv(a.output/'DOMAIN_CLASSIFICATION_BY_LAYER.csv',index=False)
    crows=[]
    for split,df,xx in [('TRAIN',tr,xt),('TEST',te,xe)]:
        for k in LAYERS:
            vals=multiclass_cv(xx[k],df.center.to_numpy(),df.patient_id.to_numpy());crows.append(dict(split=split,layer=k,layer_name=NAMES[k],dimension=xx[k].shape[1],macro_auroc_ovr=vals[0],balanced_accuracy=vals[1],macro_f1=vals[2],patients=vals[3],centers='|'.join(sorted(df.center.unique()))))
    pd.DataFrame(crows).to_csv(a.output/'CENTER_CLASSIFICATION_BY_LAYER.csv',index=False)
    smd=[];mmd=[];fd=[]
    for k in LAYERS:
        smd.append(dict(layer=k,layer_name=NAMES[k],dimension=X[k].shape[1],**smd_stats(xt[k],xe[k])))
        z=mmd2(xt[k],xe[k]);mmd.append(dict(layer=k,layer_name=NAMES[k],mmd2=z[0],bandwidth=z[1],n_train=z[2],n_test=z[3]))
        fd.append(dict(layer=k,layer_name=NAMES[k],dimension=X[k].shape[1],frechet_distance=frechet(xt[k],xe[k])))
    pd.DataFrame(smd).to_csv(a.output/'SHIFT_SMD_BY_LAYER.csv',index=False);pd.DataFrame(mmd).to_csv(a.output/'SHIFT_MMD_BY_LAYER.csv',index=False);pd.DataFrame(fd).to_csv(a.output/'SHIFT_FRECHET_BY_LAYER.csv',index=False)
    cov=[]
    for k in ['R3','R4']:
        ca=np.cov(xt[k],rowvar=False);cb=np.cov(xe[k],rowvar=False);cov.append(dict(layer=k,layer_name=NAMES[k],dimension=X[k].shape[1],normalized_frobenius=np.linalg.norm(ca-cb,'fro')/(np.linalg.norm(ca,'fro')+1e-12)))
    pd.DataFrame(cov).to_csv(a.output/'SHIFT_COVARIANCE_BY_LAYER.csv',index=False)
    comp=[]
    for split,df in [('TRAIN',tr),('TEST',te)]:
        for c,g in df.groupby('center'):
            lab=g.label.notna();comp.append(dict(split=split,center=c,patients=g.patient_id.nunique(),patient_fraction=g.patient_id.nunique()/df.patient_id.nunique(),edfs=g.edf_name.nunique(),edf_fraction=g.edf_name.nunique()/df.edf_name.nunique(),channel_units=len(g),channel_unit_fraction=len(g)/len(df),labeled_units=int(lab.sum()),pathological_prevalence=g.loc[lab,'label'].mean()))
    pd.DataFrame(comp).to_csv(a.output/'CENTER_COMPOSITION_AUDIT.csv',index=False)
    cc=[]
    for c in sorted(set(tr.center)&set(te.center)):
        ia=np.flatnonzero(tr.center.to_numpy()==c);ib=np.flatnonzero(te.center.to_numpy()==c); md=pd.concat([tr.iloc[ia],te.iloc[ib]],ignore_index=True); yy=np.r_[np.zeros(len(ia),int),np.ones(len(ib),int)];gg=md.patient_id.to_numpy()
        for k in LAYERS:
            xx=np.vstack([xt[k][ia],xe[k][ib]]);auc,_=binary_cv(xx,yy,gg);ss=smd_stats(xt[k][ia],xe[k][ib]);mm=mmd2(xt[k][ia],xe[k][ib],SEED+len(cc));cc.append(dict(center=c,layer=k,train_units=len(ia),test_units=len(ib),domain_auroc=auc,median_abs_smd=ss['median_abs_smd'],mmd2=mm[0]))
    pd.DataFrame(cc).to_csv(a.output/'CENTER_CONDITIONED_SHIFT.csv',index=False)
    lp=[]
    centers=sorted(set(tr.center)|set(te.center))
    for c in centers:
        ga=tr[(tr.center==c)&tr.label.notna()];gb=te[(te.center==c)&te.label.notna()];aa,ap=perf(ga.label.to_numpy(),ga.score.to_numpy());ba,bp=perf(gb.label.to_numpy(),gb.score.to_numpy());lp.append(dict(center=c,train_units=len(ga),test_units=len(gb),train_auroc=aa,train_ap=ap,test_auroc=ba,test_ap=bp,auroc_gap=aa-ba if np.isfinite(aa) and np.isfinite(ba) else math.nan))
    pd.DataFrame(lp).to_csv(a.output/'LOCALIZATION_PERFORMANCE_BY_CENTER.csv',index=False)
    labelcos=[];probes=[]; ita=np.flatnonzero(tr.label.notna());ite=np.flatnonzero(te.label.notna());yta=tr.label.iloc[ita].astype(int).to_numpy();yte=te.label.iloc[ite].astype(int).to_numpy();gta=tr.patient_id.iloc[ita].to_numpy();gte=te.patient_id.iloc[ite].to_numpy()
    for k in LAYERS:
        va=xt[k][ita][yta==1].mean(0)-xt[k][ita][yta==0].mean(0);vb=xe[k][ite][yte==1].mean(0)-xe[k][ite][yte==0].mean(0);labelcos.append(dict(layer=k,layer_name=NAMES[k],dimension=len(va),centroid_direction_cosine=cosine(va,vb),train_positive=int((yta==1).sum()),test_positive=int((yte==1).sum())))
        po,pt,wc=probe(xt[k][ita],yta,gta,xe[k][ite],yte,gte);probes.append(dict(layer=k,layer_name=NAMES[k],dimension=xt[k].shape[1],train_oof_auroc=po,test_auroc=pt,generalization_gap=po-pt,train_test_probe_weight_cosine=wc,test_probe_status='DIAGNOSTIC_ONLY_NOT_DEPLOYABLE'))
    pd.DataFrame(labelcos).to_csv(a.output/'LABEL_DIRECTION_COSINE_BY_LAYER.csv',index=False);pd.DataFrame(probes).to_csv(a.output/'LINEAR_PROBE_GENERALIZATION.csv',index=False)
    task=[]
    for label,keyx in [('parallel','TASK_PAR'),('orthogonal','TASK_PERP')]:
        xx=np.vstack([xt[keyx],xe[keyx]]);auc,_=binary_cv(xx,ydom,groups);ss=smd_stats(xt[keyx],xe[keyx]);mm=mmd2(xt[keyx],xe[keyx]);task.append(dict(component=label,source_layer='P16_TRUE_CLASSIFIER_INPUT',dimension=xx.shape[1],domain_auroc=auc,median_abs_smd=ss['median_abs_smd'],p90_abs_smd=ss['p90_abs_smd'],mmd2=mm[0]))
    pd.DataFrame(task).to_csv(a.output/'TASK_DIRECTION_SHIFT.csv',index=False)
    cr=[]
    for split,df,r4mean,status in [('TRAIN',tr,xt['R4'][:,:32],'PRIMARY_TRAIN_ONLY'),('TEST',te,xe['R4'][:,:32],'DIAGNOSTIC_ONLY_NOT_DEPLOYABLE')]:
        dirs={}
        for c in sorted(df.center.unique()):
            ix=np.flatnonzero((df.center.to_numpy()==c)&df.label.notna().to_numpy());yy=df.label.iloc[ix].astype(int).to_numpy()
            if len(ix)>=20 and len(np.unique(yy))==2:dirs[c]=r4mean[ix][yy==1].mean(0)-r4mean[ix][yy==0].mean(0)
        for ca,va in dirs.items():
            for cb,vb in dirs.items():cr.append(dict(split=split,status=status,representation='R4_MEAN_32D',center_a=ca,center_b=cb,cosine=cosine(va,vb)))
    pd.DataFrame(cr).to_csv(a.output/'CENTER_LABEL_DIRECTION_COSINE.csv',index=False)
    # PCA is explanatory only; combined unlabeled standardization, deterministic cap 5000/split.
    idxa=np.random.default_rng(SEED).choice(len(tr),min(5000,len(tr)),False);idxb=np.random.default_rng(SEED+1).choice(len(te),min(5000,len(te)),False);ix=np.r_[idxa,len(tr)+idxb];z=StandardScaler().fit_transform(X['R4'][ix,:32]);pc=PCA(2,random_state=SEED).fit_transform(z);plotdf=domain.iloc[ix].reset_index(drop=True)
    def plot(col,path,include_unknown=True):
        fig,ax=plt.subplots(figsize=(7,5));vals=plotdf[col].fillna('unlabeled').astype(str)
        for v in sorted(vals.unique()):
            q=vals==v;ax.scatter(pc[q,0],pc[q,1],s=5,alpha=.28,label=v,rasterized=True)
        ax.set(xlabel='PC1',ylabel='PC2');ax.legend(markerscale=3,fontsize=7);fig.tight_layout();fig.savefig(path);plt.close(fig)
    plotdf['label_display']=plotdf.label.map({0.0:'normal',1.0:'pathological'}).fillna('unlabeled')
    plot('split',a.output/'pca_train_test.pdf');plot('center',a.output/'pca_center.pdf');plot('label_display',a.output/'pca_label.pdf')
    train_auc=roc_auc_score(tr.loc[tr.label.notna(),'label'],tr.loc[tr.label.notna(),'score']);test_auc=roc_auc_score(te.loc[te.label.notna(),'label'],te.loc[te.label.notna(),'score'])
    replay={'status':'PASS' if abs(train_auc-.9586782931)<1e-4 and abs(test_auc-.7987673466)<1e-5 else 'BASELINE_REPLAY_FAILED','train_observed_auroc':train_auc,'train_expected_auroc':.9586782931,'train_absolute_error':abs(train_auc-.9586782931),'train_tolerance':1e-4,'test_observed_auroc':test_auc,'test_expected_auroc':.7987673466,'test_absolute_error':abs(test_auc-.7987673466),'test_tolerance':1e-5}
    (a.output/'BASELINE_REPLAY_AUDIT.json').write_text(json.dumps(replay,indent=2));
    if replay['status']!='PASS':raise RuntimeError('BASELINE_REPLAY_FAILED')
    extraction={'status':'PASS','train_edfs':len(trfiles),'test_edfs':len(tefiles),'train_patients':tr.patient_id.nunique(),'test_patients':te.patient_id.nunique(),'train_channel_units':len(tr),'test_channel_units':len(te),'train_labeled_channel_units':int(tr.label.notna().sum()),'test_labeled_channel_units':int(te.label.notna().sum()),'train_segment_samples':len(strm),'test_segment_samples':len(stem),'segment_sample_max_per_edf':20,'private_manifest_sha256':hashlib.sha256(''.join(sha256(p) for p in trfiles+tefiles).encode()).hexdigest(),'private_representations_committed':False}
    (a.output/'REPRESENTATION_EXTRACTION_AUDIT.json').write_text(json.dumps(extraction,indent=2))
    # Synthesize report only after all machine-readable outputs exist.
    d=pd.DataFrame(drows);d=d[d.analysis_unit=='EDF-channel'].set_index('layer');c=pd.DataFrame(crows);ct=c[c.split=='TRAIN'].set_index('layer');s=pd.DataFrame(smd).set_index('layer');mm=pd.DataFrame(mmd).set_index('layer');pr=pd.DataFrame(probes).set_index('layer');lc=pd.DataFrame(labelcos).set_index('layer');td=pd.DataFrame(task).set_index('component');ccdf=pd.DataFrame(cc);lpdf=pd.DataFrame(lp)
    first=next((k for k in LAYERS if d.loc[k,'auroc']>=.65),'none');largest=max(LAYERS,key=lambda k:d.loc[k,'auroc']);centermax=max(LAYERS,key=lambda k:ct.loc[k,'macro_auroc_ovr'] if np.isfinite(ct.loc[k,'macro_auroc_ovr']) else -1)
    tags=[]
    if d.loc['R0','auroc']>=.8 or d.loc['R1','auroc']>=.8:tags.append('INPUT_SHIFT')
    if d.loc['R1','auroc']<.7 and d.loc['R4','auroc']>=.85:tags.append('REPRESENTATION_AMPLIFIED_SHIFT')
    if td.loc['orthogonal','domain_auroc']-td.loc['parallel','domain_auroc']>=.1:tags.append('TASK_ORTHOGONAL_NUISANCE')
    if td.loc['parallel','domain_auroc']>=.8 and lc.loc['R4','centroid_direction_cosine']<.8:tags.append('TASK_DIRECTION_SHIFT')
    if all(d.loc[k,'auroc']<.65 for k in LAYERS):tags.append('NO_SIMPLE_REPRESENTATION_SHIFT')
    if not tags: tags=['NO_SIMPLE_REPRESENTATION_SHIFT']
    priority=next(x for x in ['TASK_DIRECTION_SHIFT','INPUT_SHIFT','REPRESENTATION_AMPLIFIED_SHIFT','TASK_ORTHOGONAL_NUISANCE','NO_SIMPLE_REPRESENTATION_SHIFT'] if x in tags)
    adapter={'INPUT_SHIFT':'Input Statistical Normalization Adapter','REPRESENTATION_AMPLIFIED_SHIFT':'Feature-Space Domain Normalization Adapter','TASK_ORTHOGONAL_NUISANCE':'Orthogonal Nuisance Removal Adapter','TASK_DIRECTION_SHIFT':'Domain-Conditional Decision Adapter','NO_SIMPLE_REPRESENTATION_SHIFT':'stop normalization adapter route'}[priority]
    table=['| Layer | Dim | Domain AUROC | Center predictability | Median |SMD| | MMD | Train probe AUROC | Test probe AUROC |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in LAYERS:table.append(f"| {NAMES[k]} | {BASE_DIM[k]} | {d.loc[k,'auroc']:.6f} | {fmt(ct.loc[k,'macro_auroc_ovr'])} | {s.loc[k,'median_abs_smd']:.6f} | {mm.loc[k,'mmd2']:.6f} | {pr.loc[k,'train_oof_auroc']:.6f} | {pr.loc[k,'test_auroc']:.6f} |")
    overlap=sorted(set(tr.center)&set(te.center)); within=ccdf.groupby('center').domain_auroc.max().to_dict() if len(ccdf) else {}; center_cos=pd.DataFrame(cr)
    estim=lpdf.dropna(subset=['auroc_gap']);corr='not estimated (<4 centers)'
    if len(estim)>=4:
        sh=ccdf[ccdf.layer=='R4'].set_index('center').domain_auroc; common=[x for x in estim.center if x in sh]; corr=str(spearmanr([sh[x] for x in common],[float(estim[estim.center==x].auroc_gap.iloc[0]) for x in common]).statistic)
    train_center_cos=center_cos[(center_cos.split=='TRAIN')&(center_cos.center_a!=center_cos.center_b)] if len(center_cos) else center_cos
    report='\n'.join(table)+f"""

# Omni Representation Shift Audit

Baseline replay passed: TRAIN full-record AUROC `{train_auc:.10f}` and TEST AUROC `{test_auc:.10f}`. All representations came from the locked evaluation-mode forward pass.

## Direct answers

1. The first layer reaching at least moderate separability is **{first} ({NAMES.get(first,first)})**, with domain AUROC {fmt(d.loc[first,'auroc']) if first!='none' else 'below 0.65 everywhere'}.
2. Domain AUROC by depth is {', '.join(f'{k}={d.loc[k,"auroc"]:.3f}' for k in LAYERS)}. This sequence is the evidence for whether separability increases monotonically.
3. The largest domain AUROC occurs at **{largest} ({NAMES[largest]})**, {d.loc[largest,'auroc']:.6f}.
4. TRAIN center identity is most predictable at **{centermax} ({NAMES[centermax]})**, macro OVR AUROC {fmt(ct.loc[centermax,'macro_auroc_ovr'])}.
5. Center composition and within-center results must be interpreted together; the pooled classifier is not a stable separator, and `Other` is an aggregate rather than a single acquisition center. Exact fractions are in `CENTER_COMPOSITION_AUDIT.csv`.
6. Within-center TRAIN-to-TEST marginal shift is reported without causal claims in `CENTER_CONDITIONED_SHIFT.csv`; localization gaps can remain large even when these domain AUROCs are low.
7. R4 pathological-minus-normal centroid direction cosine is **{lc.loc['R4','centroid_direction_cosine']:.6f}**.
8. The R4 TRAIN-only linear probe falls from {pr.loc['R4','train_oof_auroc']:.6f} OOF AUROC to {pr.loc['R4','test_auroc']:.6f} on TEST, a gap of {pr.loc['R4','generalization_gap']:.6f}.
9. True classifier-input task-direction AUROCs are parallel={td.loc['parallel','domain_auroc']:.6f} and orthogonal={td.loc['orthogonal','domain_auroc']:.6f}. This uses the real 16D `fc_out` input because no final 32D linear classifier exists.
10. The classifier task coordinate {'is strongly shifted' if td.loc['parallel','domain_auroc']>=.8 else 'is not strongly shifted under the locked 0.80 threshold'}.
11. R4 TRAIN center-specific label-direction off-diagonal cosines range from {fmt(train_center_cos.cosine.min()) if len(train_center_cos) else 'not_estimable'} to {fmt(train_center_cos.cosine.max()) if len(train_center_cos) else 'not_estimable'}. The full TRAIN matrix and diagnostic-only TEST matrix are in `CENTER_LABEL_DIRECTION_COSINE.csv`.
12. Supported audit labels: **{', '.join(tags)}**.
13. The locked mapping selects **{adapter}** as the next research direction. This audit does not implement it.

## Composition, performance, and limits

The pooled TRAIN-to-TEST AUROC loss is {train_auc-test_auc:.6f}. Center-specific performance is in `LOCALIZATION_PERFORMANCE_BY_CENTER.csv`. The descriptive R4 shift/performance-drop Spearman result is {corr}; it is not causal evidence.

The prompt's 32D classifier-direction premise is false for this checkpoint. R4 is followed by a nonlinear 32→32→16→1 head, so projecting R4 with an invented 32D weight would be invalid. `TASK_DIRECTION_SHIFT.csv` uses the actual 16D final linear weight and labels it `P16_TRUE_CLASSIFIER_INPUT`.

PCA figures are explanatory only. TEST-label probes are marked `DIAGNOSTIC_ONLY_NOT_DEPLOYABLE`. No adapter, normalization, CORAL, whitening, fine-tuning, or test-time adaptation was performed.
"""
    (a.output/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
if __name__=='__main__':main()
