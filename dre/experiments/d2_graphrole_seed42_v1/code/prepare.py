"""Frozen D2 replay, GraphRole quality audits and FIT-only graph preprocessing."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from common import sha,digest,write_json,metric,METRICS
from inspect_source import ROOT,EXPORT
from graph_metrics import NAMES,BASE
from model import SourceA0
from runtime import torch_write,patient_z

PRIOR=Path(r'C:\a0_source_posterior_decoder_seed42_runtime')
SPLIT=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')

def permutation(signatures,patient,seed=42):
    h=int.from_bytes(__import__('hashlib').sha256(f'{seed}|{patient}|graph_shuffle'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(h); out=np.arange(len(signatures)); groups={}
    for c,sig in enumerate(signatures): groups.setdefault(tuple(sig),[]).append(c)
    for group in groups.values():
        if len(group)>1:
            order=rng.permutation(group); out[order]=np.roll(order,1)
    return out

def graph_preprocess(values,patients,train):
    z=patient_z(values,patients); assert np.isfinite(z[train]).any(axis=0).all(),'Unidentifiable FIT graph dimension'
    imputer=SimpleImputer(strategy='mean'); scaler=StandardScaler()
    imputer.fit(z[train]); scaler.fit(imputer.transform(z[train])); x=scaler.transform(imputer.transform(z)).astype('float32')
    assert x.shape==values.shape and x.shape[1]==16 and np.isfinite(x).all()
    return x,{'imputer_statistics':imputer.statistics_,'mean':scaler.mean_,'scale':scaler.scale_,'var':scaler.var_,
              'fit_patient_ids':sorted(set(patients[train]))}

def main():
    a=argparse.ArgumentParser(); a.add_argument('--protocol',type=Path,required=True); a.add_argument('--manifest',type=Path,required=True); args=a.parse_args()
    lock=json.loads(args.protocol.read_text()); frozen=json.loads(args.manifest.read_text()); torch.set_num_threads(2)
    assert sha(EXPORT)==lock['input_sha256']['feature_export'] and sha(SPLIT)==lock['input_sha256']['split']
    for rel,target in frozen.items(): assert sha(PRIOR/rel)==target,'Historical frozen input changed'
    with np.load(EXPORT,allow_pickle=True) as d:
        pid=d['patient'].astype(str); channel=d['channel'].astype(str); source=d['center'].astype(str); original_names=d['features'].astype(str)
    assert __import__('hashlib').sha256(json.dumps(original_names.tolist(),separators=(',',':')).encode()).hexdigest()==lock['input_sha256']['feature_order']
    with np.load(ROOT/'GRAPH16_PRIVATE.npz') as d:
        graph=d['graph']; assert np.array_equal(pid,d['patient']) and np.array_equal(channel,d['channel']) and d['feature_names'].tolist()==NAMES
    assert graph.shape==(7635,16)
    align=json.loads((ROOT/'public/GRAPH_ALIGNMENT_AUDIT.json').read_text()); assert align['status']=='PASS' and align['graph_bank_sha256']==sha(ROOT/'GRAPH16_PRIVATE.npz')
    quality=[]; missing=[]; problems=[]
    within=np.array([[np.nanstd(graph[pid==p,j]) if np.isfinite(graph[pid==p,j]).any() else np.nan for j in range(16)] for p in sorted(set(pid))])
    means=np.array([[np.nanmean(graph[pid==p,j]) if np.isfinite(graph[pid==p,j]).any() else np.nan for j in range(16)] for p in sorted(set(pid))])
    for j,name in enumerate(NAMES):
        unidentifiable=int(((~np.isfinite(within[:,j]))|(within[:,j]<=1e-8)).sum()); problem=unidentifiable>40 or np.isfinite(graph[:,j]).mean()<.05
        if problem: problems.append(name)
        for center in ['ALL']+sorted(set(source)):
            take=np.ones(len(pid),bool) if center=='ALL' else source==center; v=graph[take,j]; ok=np.isfinite(v)
            quality.append({'feature':name,'center':center,'rows':int(take.sum()),'finite_fraction':float(ok.mean()),
                'mean':float(np.mean(v[ok])) if ok.any() else np.nan,'std':float(np.std(v[ok])) if ok.any() else np.nan,
                'q05':float(np.quantile(v[ok],.05)) if ok.any() else np.nan,'q95':float(np.quantile(v[ok],.95)) if ok.any() else np.nan,
                'between_patient_mean_std':float(np.nanstd(means[:,j])) if center=='ALL' else np.nan,
                'mean_within_patient_std':float(np.nanmean(within[:,j])) if center=='ALL' else np.nan,
                'unidentifiable_patients':unidentifiable if center=='ALL' else np.nan,
                'structural_problem':problem if center=='ALL' else False})
            missing.append({'feature':name,'center':center,'missing_rows':int((~ok).sum()),'total_rows':len(v),'missing_fraction':float((~ok).mean())})
    pd.DataFrame(quality).to_csv(ROOT/'public/GRAPH_FEATURE_QUALITY.csv',index=False)
    pd.DataFrame(missing).to_csv(ROOT/'public/GRAPH_MISSINGNESS_AUDIT.csv',index=False)
    write_json(ROOT/'public/GRAPH_FEATURE_SCHEMA.json',{'dimensions':16,'names':NAMES,'base_metrics':BASE,'means_then_population_stds':True,
       'equal_seizure_weight':True,'temporal_variation_denominator':'max(actual seconds,1.0)','rank_ties':'average; (rank-1)/(C-1)',
       'stability_missing_when_less_than_two_seizures':True,'existing88_names':original_names.tolist(),'existing_graph_name_overlap':[],
       'structural_problem_dimensions':problems,'degenerate':len(problems)>8})
    if len(problems)>8:
        write_json(ROOT/'public/RUN_STATUS.json',{'status':'BLOCKED','terminal':'GRAPH_INFORMATION_DEGENERATE','trained_runs':0}); raise RuntimeError('Graph feature degeneracy gate failed')
    with (ROOT/'SIGNATURES_PRIVATE.pkl').open('rb') as f: signatures=dict(pickle.load(f))
    perm=np.arange(len(pid)); stats=[]
    for p in sorted(set(pid)):
        ix=np.flatnonzero(pid==p); sig=signatures[p]; assert len(sig)==len(ix)
        order=permutation(sig,p); perm[ix]=ix[order]; assert np.array_equal(sig,sig[order])
        stats.append({'channels':len(ix),'shuffled':int((order!=np.arange(len(ix))).sum())})
    shuffled=graph[perm]; assert all(np.array_equal(np.sort(graph[pid==p],axis=0),np.sort(shuffled[pid==p],axis=0),equal_nan=True) for p in set(pid))
    write_json(ROOT/'public/GRAPH_SHUFFLE_AUDIT.json',{'status':'PASS','within_patient':True,'labels_used':False,'availability_signatures_preserved':True,
       'channels':7635,'shuffled_channels':sum(s['shuffled'] for s in stats),'unshufflable_singletons':sum(s['channels']-s['shuffled'] for s in stats),
       'shuffled_fraction':sum(s['shuffled'] for s in stats)/7635,'complete_vectors_preserved':True})
    assert sum(s['shuffled'] for s in stats)/7635>=.95,'Insufficient correspondence-control coverage'
    control=[]; replay=[]; preprocessing=[]; redundancy=[]; channel_count=[]
    for p in sorted(set(pid)): channel_count.extend([int((pid==p).sum())]*int((pid==p).sum()))
    # Count relationship is descriptive only; actual aligned count array follows export order.
    channel_count=np.array([int((pid==p).sum()) for p in pid])
    for fold in range(1,6):
        bank=PRIOR/f'fold{fold}/BANK_PRIVATE.pt'; ck=PRIOR/f'fold{fold}/D2/BEST_PRIVATE.pt'; pred=PRIOR/f'fold{fold}/D2_FROZEN_PRIVATE.pt'
        d=torch.load(bank,map_location='cpu',weights_only=False); st=torch.load(ck,map_location='cpu',weights_only=False); old=torch.load(pred,map_location='cpu',weights_only=False)
        assert old['checkpoint_sha256']==sha(ck); assert not set(d['patient'][d['train']])&set(d['patient'][d['val']])
        model=SourceA0(dimension=88).eval(); model.load_state_dict(st['model'])
        with torch.no_grad(): logits=model(torch.from_numpy(d['x']),torch.from_numpy(d['g'])).numpy()
        drift=float(np.abs(logits[d['val']]-old['logits'][d['val']]).max()); assert drift<2e-6
        lookup={(p,c):i for i,(p,c) in enumerate(zip(pid,channel))}; indices=np.array([lookup[(str(p),str(c))] for p,c in zip(d['patient'],d['channel'])]); values=graph[indices]; fake=shuffled[indices]
        for k in ['patient','channel','center']: assert np.array_equal(d[k],{'patient':pid,'channel':channel,'center':source}[k][indices])
        gx,pre=graph_preprocess(values,d['patient'],d['train']); sx,spre=graph_preprocess(fake,d['patient'],d['train'])
        for k in ['imputer_statistics','mean','scale','var']: assert np.allclose(pre[k],spre[k],rtol=0,atol=1e-12),'Shuffle FIT distribution changed'
        inputs={'G1':np.concatenate([d['x'],gx],1),'G2':np.concatenate([d['x'],sx],1),'G3':np.concatenate([d['x'],np.zeros_like(gx)],1)}
        for arm,x in inputs.items(): assert x.shape[1]==104 and np.array_equal(x[:,:88],d['x'])
        assert not np.any(inputs['G3'][:,88:]); tr=d['train']; rows=[]
        for p in sorted(set(d['patient'][d['val']])):
            ix=d['val'][d['patient'][d['val']]==p]; ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
            rows.append({'fold':fold,'patient':p,'center':str(d['center'][ix[0]]),'method':'D2',**metric(d['y'][ix],old['scores'][ix],old['threshold'])})
        control.extend(rows)
        binding={'protocol':sha(args.protocol),'graph_bank':sha(ROOT/'GRAPH16_PRIVATE.npz'),'bank':sha(bank),'D2_checkpoint':sha(ck),'original88_bitwise':True}
        torch_write(ROOT/f'fold{fold}/BANK_PRIVATE.pt',{'original':d,'inputs':inputs,'pre':pre,'shuffled_pre':spre,'binding':binding})
        replay.append({'fold':fold,'bank_sha256':sha(bank),'D2_checkpoint_sha256':sha(ck),'max_logit_drift':drift,'threshold':old['threshold'],'parameters':9221})
        preprocessing.append({'fold':fold,'FIT_patients':len(set(d['patient'][tr])),'VAL_patients':len(set(d['patient'][d['val']])),'train_only_fitted':True,
            'original88_bitwise':True,'feature_order_fixed':True,'graph_imputer_scaler_hash':digest({k:pre[k].tolist() for k in ['imputer_statistics','mean','scale','var']})})
        for j,name in enumerate(NAMES):
            corr=np.array([np.corrcoef(gx[tr,j],d['x'][tr,k])[0,1] if np.std(gx[tr,j])>0 and np.std(d['x'][tr,k])>0 else np.nan for k in range(88)])
            best=int(np.nanargmax(np.abs(corr))) if np.isfinite(corr).any() else 0
            redundancy.append({'fold':fold,'graph_feature':name,'most_correlated_88D_feature':str(original_names[best]),'pearson':corr[best],
                'absolute_pearson':abs(corr[best]),'existing_features_abs_r_ge080':int((np.abs(corr)>=.8).sum()),'FIT_channels':len(tr),'VAL_labels_used':False,
                'pearson_available_channel_count':float(np.corrcoef(gx[tr,j],channel_count[indices][tr])[0,1]) if np.std(gx[tr,j])>0 else np.nan})
    frame=pd.DataFrame(control); assert len(frame)==65 and frame.patient.nunique()==47 and frame.channels.sum()==6273
    assert abs(frame.macro_f1.mean()-.6479260773776448)<1e-12
    torch_write(ROOT/'CONTROL_METRICS_PRIVATE.pt',control)
    write_json(ROOT/'public/D2_REPRODUCTION.json',{'status':'PASS','controls_retrained':False,'fresh_logits_then_frozen_exact_scores':True,'folds':replay,'metrics':frame[METRICS].mean().to_dict(),'verified_historical_files':len(frozen)})
    write_json(ROOT/'public/GRAPH_PREPROCESSING_AUDIT.json',{'status':'PASS','folds':preprocessing,'no_population_VAL_fit':True,'no_outer_labels_loaded':True})
    pd.DataFrame(redundancy).to_csv(ROOT/'public/GRAPH_REDUNDANCY_AUDIT.csv',index=False)
    write_json(ROOT/'public/RUN_STATUS.json',{'status':'GRAPH_PREPARATION_AND_D2_REPLAY_COMPLETE','outer_test_accessed':False})
    print('GRAPH_PREPARATION_AND_D2_REPLAY_PASS',flush=True)

if __name__=='__main__': main()
