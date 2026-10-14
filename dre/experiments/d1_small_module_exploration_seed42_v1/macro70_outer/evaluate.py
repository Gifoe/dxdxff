"""Frozen final Macro70 investigation Outer; no training/threshold fitting."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[1]
for f in ['code','phase7/code','phase8/code','phase11/code']:sys.path.insert(0,str(ROOT/f))
from spectral_model import SpectralCorrection
from calibrate import apply_map
from dispersion import transform_scaled,cutoff
from explore import context,make_base,Inputs,EV,sha,read,save,write,original_modules

def transform(x,patient,pre,core):
    z=core.patient_z(x,patient);out=((np.where(np.isnan(z),pre['imputer_statistics'],z)-pre['mean'])/pre['scale']).astype(np.float32);assert np.isfinite(out).all();return out

def prepare(a):
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);out=a.out;pub=out/'public';pub.mkdir(parents=True,exist_ok=True);lock=json.loads((ROOT/'macro70_outer/PROTOCOL_LOCK.json').read_text(encoding='utf-8'));selection=json.loads((a.audit/'FINAL_DEVELOPMENT_SELECTION.json').read_text());assert selection['status']=='PASS' and selection['selected_phase']==14;fr=json.loads((a.calibration/'CALIBRATION_FREEZE_PRIVATE.json').read_text());assert fr['selected']==selection['selected_arm'];assert json.loads((a.calibration/'results/INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
    assert fr['binding']['code']=={p.name:sha(p) for p in (ROOT/'phase14/code').glob('*.py')};p7=json.loads((a.phase7/'INPUT_FREEZE_PRIVATE.json').read_text());assert p7['code']=={p.name:sha(p) for p in (ROOT/'phase7/code').glob('*.py')};df=json.loads((a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json').read_text());m8fr=json.loads((a.m8/'MODEL_FREEZE_PRIVATE.json').read_text());p7fr=json.loads((a.phase7/'MODEL_FREEZE_PRIVATE.json').read_text());sels=[];maps=[];visited=[];meta=rec['meta']
    assert len(meta['patient'])==7635 and len(set(meta['patient']))==80
    for fold in range(1,6):
        test=banks[fold-1]['roles']['test'];ix=np.flatnonzero(np.isin(meta['patient'],list(test)));assert len(test)==lock['patients_by_fold'][fold-1] and len(ix)==lock['channels_by_fold'][fold-1];visited.extend(ix.tolist());bp=a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt';d=torch.load(bp,map_location='cpu',weights_only=False);assert set(d['patient']).isdisjoint(test) and set(d['pre']['fit_patient_ids'])==banks[fold-1]['roles']['fit'];sp=a.phase1/f'scale_fold{fold}_PRIVATE.pkl';sq=read(sp);assert set(sq['fit_patients'])==banks[fold-1]['roles']['fit'];assert np.array_equal(transform(rec['R14'][banks[fold-1]['archive_indices']],d['patient'],d['pre'],core),d['x'])
        mp=a.calibration/f"calibration_private/{fr['selected']}_fold{fold}.pkl";q=read(mp);r=next(v for v in fr['selections'] if v['arm']==fr['selected'] and v['fold']==fold);assert sha(mp)==r['calibration_sha256'] and q['base']=='M18_TIME_CONTROL' and q['coefficient']==0. and q['exponent']==.5 and q['D1_bank_sha256']==sha(bp);assert set(q['fit_source_coverage'])==set(d['center'][d['train']].astype(str));maps.append(q)
        for arm,rt,freeze in [('D1',a.baseline,df),('M8_MACRO_AWARE',a.m8,m8fr),('M18_TIME_CONTROL',a.phase7,p7fr)]:
            r=next(v for v in freeze['selections'] if v['arm']==arm and v['fold']==fold);cp=rt/arm/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';assert sha(cp)==r['checkpoint_sha256'];ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold'];done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt');sels.append(dict(arm=arm,fold=fold,checkpoint=str(cp),checkpoint_sha256=sha(cp),bank=str(bp),bank_sha256=sha(bp),scale_sha256=sha(sp),threshold=r['threshold'],calibration_sha256=sha(mp) if arm=='M18_TIME_CONTROL' else None))
    assert sorted(visited)==list(range(7635));old=a.previous_outer;old_fr=old/'FROZEN_BEFORE_OUTER_PRIVATE.json';old_seal=old/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';oldstatus=json.loads((old/'public/OUTER_STATUS.json').read_text());assert oldstatus['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE' and oldstatus['freeze_sha256']==sha(old_fr) and oldstatus['seal_sha256']==sha(old_seal)
    binding=dict(protocol=sha(ROOT/'macro70_outer/PROTOCOL_LOCK.json'),source={str(p.relative_to(ROOT)):sha(p) for folder in ['code','phase7/code','phase8/code','phase11/code','macro70_outer'] for p in (ROOT/folder).glob('*.py')},dependencies=deps,reconstruction=sha(a.baseline/'RECONSTRUCTION_PRIVATE.pkl'),raw_registry=sha(a.baseline/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),conditioning_registry=sha(a.data/'CONDITIONING_REGISTRY_PRIVATE.pkl'),calibration_freeze=sha(a.calibration/'CALIBRATION_FREEZE_PRIVATE.json'),global_selection=sha(a.audit/'FINAL_DEVELOPMENT_SELECTION.json'),previous_outer_freeze=sha(old_fr),previous_outer_seal=sha(old_seal),torch=torch.__version__,numpy=np.__version__)
    return c,core,fac,records,rec,banks,sels,maps,binding,pub

def freeze(a):
    c,core,fac,records,rec,banks,sels,maps,binding,pub=prepare(a);fp=a.out/'FROZEN_BEFORE_OUTER_PRIVATE.json';q=dict(binding=binding,selections=sels,calibration_maps=maps,full_test_labels_not_loaded=True)
    if fp.exists():assert json.loads(fp.read_text())==q
    else:write(fp,q)
    write(pub/'OUTER_FREEZE_AUDIT.json',dict(status='PASS',freeze_sha256=sha(fp),binding=binding,models15_locked=True,own_fold_test_excluded=True,source_cutoffs_and_rule_frozen=True,no_training=True,no_refitting=True,no_new_extraction=True,selections=[{k:v for k,v in r.items() if k not in ['checkpoint','bank']} for r in sels]));print('MACRO70_FINAL_OUTER_FROZEN_BEFORE_TEST',flush=True)

def evaluate(a):
    c,core,fac,records,rec,banks,sels,maps,binding,pub=prepare(a);fp=a.out/'FROZEN_BEFORE_OUTER_PRIVATE.json';fh=sha(fp);fr=json.loads(fp.read_text());assert fr['binding']==binding and fr['selections']==sels and fr['calibration_maps']==maps
    if (pub/'OUTER_STATUS.json').exists():assert json.loads((pub/'OUTER_STATUS.json').read_text())['freeze_sha256']==fh;print('OUTER_ALREADY_COMPLETE_NO_RERUN',flush=True);return
    oldseal=json.loads((a.previous_outer/'OUTER_PREDICTIONS_SEAL_PRIVATE.json').read_text());store=Inputs(a.baseline,a.data,a.phase7);store.arm='M1_SCALED_DGM';meta=rec['meta'];cells={};files={}
    for r in sels:
        arm,fold=r['arm'],r['fold'];ix=np.flatnonzero(np.isin(meta['patient'],list(banks[fold-1]['roles']['test'])));pp=a.out/f'scores_private/{arm}_fold{fold}.pkl'
        if pp.exists():q=read(pp);assert q['freeze_sha256']==fh and q['checkpoint_sha256']==r['checkpoint_sha256'] and np.array_equal(q['indices'],ix) and 'y' not in q
        elif arm in ['D1','M8_MACRO_AWARE']:
            old=a.previous_outer/f'scores_private/{arm}_fold{fold}.pkl';assert sha(old)==oldseal['files'][f'{arm}_{fold}'];prior=read(old);assert prior['checkpoint_sha256']==r['checkpoint_sha256'] and np.array_equal(prior['indices'],ix) and 'y' not in prior;q=dict(logit=prior['logit'],indices=ix,freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256'],reused_previous_scores_sha256=sha(old));save(pp,q)
        else:
            d=torch.load(r['bank'],map_location='cpu',weights_only=False);bank=dict(patient=meta['patient'][ix],channel=meta['channel'][ix],x=transform(rec['R14'][ix],meta['patient'][ix],d['pre'],core));cp=Path(r['checkpoint']);model=SpectralCorrection(make_base(core,fac,records,a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',42+1009*fold),arm).cuda().eval();ck=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);scale=torch.tensor(read(a.phase1/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda');result={k:np.empty(len(ix),np.float32) for k in ['logit','base_logit','residual']}
            for n,pid in enumerate(sorted(set(bank['patient'])),1):
                jj=np.flatnonzero(bank['patient']==pid);gix=ix[jj];import hashlib
                token=hashlib.sha256(str(pid).encode()).hexdigest()[:24];pf=a.out/f'patients_private/fold{fold}/{token}.pkl'
                if pf.exists():v=read(pf);assert v['freeze_sha256']==fh and v['checkpoint_sha256']==r['checkpoint_sha256'] and np.array_equal(v['indices'],gix) and 'y' not in v
                else:v=EV.predict(model,bank,jj,store,scale);v={k:v[k] for k in result};v.update(indices=gix,freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256']);save(pf,v)
                for k in result:result[k][jj]=v[k]
                store.raw.cache.clear();store.cache.clear();print('FINAL_OUTER_PATIENT_SEALED',fold,n,'of',len(set(bank['patient'])),flush=True)
            assert core.state_hash(model.state_dict())==core.state_hash(ck['model']);q=dict(**result,indices=ix,freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256']);save(pp,q);del model,ck,d;torch.cuda.empty_cache()
        assert np.isfinite(q['logit']).all();cells[arm,fold]=q;files[f'{arm}_{fold}']=sha(pp)
    decisions={};identity=[]
    for fold in range(1,6):
        q=cells['M18_TIME_CONTROL',fold];base=cells['D1',fold];assert np.max(np.abs(q['base_logit']-base['logit']))<=1e-6;ix=q['indices'];p=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();params=maps[fold-1];shift,median,scale=transform_scaled(p,meta['patient'][ix],params['coefficient'],params['exponent']);tau=apply_map(meta['center'][ix],params['default'],params['mapping']);cut=cutoff(tau,median,scale,params['coefficient']);assert np.array_equal(p>=cut,shift>=tau);dp=a.out/f'decisions_private/FINAL_DECODER_fold{fold}.pkl';v=dict(indices=ix,score_sha256=files[f'M18_TIME_CONTROL_{fold}'],cutoff=cut,prediction_nez=p>=cut,freeze_sha256=fh,unknown_source_channels=int(sum(str(z) not in params['mapping'] for z in meta['center'][ix])));save(dp,v);decisions[fold]=v;files[f'FINAL_DECODER_{fold}']=sha(dp);identity.append(dict(fold=fold,embedded_D1_max_difference=float(np.abs(q['base_logit']-base['logit']).max()),ranking_exact_M18=True,transformed_and_original_decisions_equal=True,unknown_source_channels=v['unknown_source_channels']))
    seal=a.out/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';obj=dict(freeze_sha256=fh,files=files,all15_scores_and5_decoders_sealed_before_full_test_labels=True)
    if seal.exists():assert json.loads(seal.read_text())==obj
    else:write(seal,obj)
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=z['y']
    assert len(y)==7635 and (y==0).sum()==1743 and (y==1).sum()==5892;m,b=original_modules();rows=[];pooled=[]
    for arm in ['D1','M8_MACRO_AWARE','M18_TIME_CONTROL','FINAL_DECODER']:
        allscore=np.empty(7635);allpred=np.zeros(7635,bool)
        for fold in range(1,6):
            q=cells['M18_TIME_CONTROL' if arm=='FINAL_DECODER' else arm,fold];ix=q['indices'];score=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();r=next(v for v in sels if v['arm']==('M18_TIME_CONTROL' if arm=='FINAL_DECODER' else arm) and v['fold']==fold);cut=decisions[fold]['cutoff'] if arm=='FINAL_DECODER' else np.full(len(ix),r['threshold']);allscore[ix]=score;allpred[ix]=score>=cut
            for pid in sorted(set(meta['patient'][ix])):
                k=np.flatnonzero(meta['patient'][ix]==pid);k=k[np.argsort(meta['channel'][ix][k].astype(str),kind='stable')];ii=ix[k];assert np.ptp(cut[k])==0;metric=c.metric(y[ii],score[k],float(cut[k][0]));ind=m.independent(y[ii],score[k],float(cut[k][0]))
                for key in ind:assert np.isclose(metric[key],ind[key],atol=1e-12,rtol=0,equal_nan=True)
                rows.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(meta['center'][ii[0]]),**metric))
        order=np.lexsort((meta['channel'],meta['patient']));v=c.metric(y[order],allscore[order],0,pred=allpred[order]);v.pop('ez_mrr');v.pop('top1_is_ez');pooled.append(dict(arm=arm,unit='secondary_pooled_channels',**v))
    df=pd.DataFrame(rows)
    for arm,g in df.groupby('arm'):assert len(g)==g.patient.nunique()==80 and g.channels.sum()==7635
    save(a.out/'OUTER_METRICS_PRIVATE.pkl',rows);df.groupby('arm')[m.METRICS].mean().reset_index().to_csv(pub/'OUTER_SUMMARY.csv',index=False)
    for field in ['fold','center']:pd.DataFrame([dict(arm=arm,**{field:v},patients=len(g),channels=int(g.channels.sum()),**g[m.METRICS].mean().to_dict()) for (arm,v),g in df.groupby(['arm',field])]).to_csv(pub/f'OUTER_BY_{field.upper()}.csv',index=False)
    boots=[]
    for left,right in json.loads((ROOT/'macro70_outer/PROTOCOL_LOCK.json').read_text(encoding='utf-8'))['bootstrap']['comparisons']:boots.extend(dict(comparison=left+' minus '+right,**q) for q in b.compare(c,df[df.arm==left],df[df.arm==right],m.METRICS))
    pd.DataFrame(boots).to_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv',index=False);pd.DataFrame(pooled).to_csv(pub/'OUTER_POOLED_CHANNELS.csv',index=False);write(pub/'OUTER_IDENTITY_AUDIT.json',dict(status='PASS',folds=identity));write(pub/'OUTER_STATUS.json',dict(status='ONE_FROZEN_OUTER_PASS_COMPLETE',freeze_sha256=fh,seal_sha256=sha(seal),patients_per_arm=80,channels_per_arm=7635,no_training_or_refit_or_extraction=True,source_map_unchanged=True,scientific_status='USER_AUTHORIZED_POSTHOC_OUTER_EXPLORATORY_REPEATED_TEST'));print('MACRO70_FINAL_OUTER_COMPLETE',flush=True)

def arguments():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','evaluate']);p.add_argument('--out',type=Path,default=Path(r'D:\d1_macro70_final_outer_runtime'));p.add_argument('--calibration',type=Path,default=Path(r'D:\d1_macro70_phase14_runtime'));p.add_argument('--audit',type=Path,default=Path(r'D:\d1_macro70_calibration_audit'));p.add_argument('--phase7',type=Path,default=Path(r'D:\d1_macro70_phase7_runtime'));p.add_argument('--m8',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'));p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--data',type=Path,default=Path(r'D:\dgm_net_seed42_v1_runtime'));p.add_argument('--previous-outer',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));return p
if __name__=='__main__':a=arguments().parse_args();(freeze if a.phase=='freeze' else evaluate)(a)
