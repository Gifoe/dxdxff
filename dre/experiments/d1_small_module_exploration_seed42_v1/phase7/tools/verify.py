"""Read-only fresh selected-checkpoint inference and independent metrics."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[2];sys.path.insert(0,str(ROOT/'phase7/code'));sys.path.insert(0,str(ROOT/'code'))
from spectral_model import SpectralCorrection,ARMS
from explore import context,make_base,Inputs,EV,read,sha,write,original_modules

def main():
    rt=Path(r'D:\d1_macro70_phase7_runtime');baseline=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime');p1=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime');data=Path(r'D:\dgm_net_seed42_v1_runtime');fr=json.loads((rt/'MODEL_FREEZE_PRIVATE.json').read_text());binding=json.loads((rt/'INPUT_FREEZE_PRIVATE.json').read_text());assert binding['code']=={p.name:sha(p) for p in (ROOT/'phase7/code').glob('*.py')} and fr['binding']==binding and len(fr['selections'])==15
    c,core,fac,trainer,records,rec,banks,hist,deps=context(Path(r'E:\DRE-nips\new-pipeline\7-11'),baseline);m,b=original_modules();store=Inputs(baseline,data,rt);store.arm='M1_SCALED_DGM';rows=[];audits=[]
    for r in fr['selections']:
        arm,fold=r['arm'],r['fold'];cp=rt/arm/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';assert sha(cp)==r['checkpoint_sha256'];ck=torch.load(cp,map_location='cpu',weights_only=False);done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt') and ck['epoch']==r['epoch'] and ck['threshold']==r['threshold'];basecp=baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt';assert sha(basecp)==r['D1_checkpoint_sha256'];baseck=torch.load(basecp,map_location='cpu',weights_only=False);assert all(torch.equal(ck['model']['base.'+k],v) for k,v in baseck['model'].items());bp=basecp.parent.parent/'BANK_PRIVATE.pt';assert sha(bp)==r['D1_bank_sha256'];d=torch.load(bp,map_location='cpu',weights_only=False);assert set(d['patient']).isdisjoint(banks[fold-1]['roles']['test'])
        model=SpectralCorrection(make_base(core,fac,records,basecp,42+1009*fold),arm).cuda().eval();model.load_state_dict(ck['model'],strict=True);scale=torch.tensor(read(p1/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda');pred=EV.predict(model,d,d['val'],store,scale);q=read(rt/f'predictions_private/{arm}_fold{fold}.pkl');assert q['binding']==binding and q['checkpoint_sha256']==sha(cp);drift=float(np.abs(pred['logit']-q['logit']).max());assert drift<=1e-6 and np.array_equal(pred['base_logit'],q['base_logit']);assert np.max(np.abs(q['logit']-q['base_logit']-q['residual']))<=1e-6 and np.max(np.abs(q['residual']))<=1.5+1e-6;score=torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
        for pid in sorted(set(d['patient'][d['val']])):
            k=np.flatnonzero(d['patient'][d['val']]==pid);k=k[np.argsort(d['channel'][d['val']][k].astype(str),kind='stable')];ii=d['val'][k];rows.append(dict(arm=arm,fold=fold,patient=str(pid),**m.independent(d['y'][ii],score[k],r['threshold'])))
        audits.append(dict(arm=arm,fold=fold,selected_checkpoint_replay_max_difference=drift,base_exact_frozen=True,checkpoint_sha256=sha(cp)));del model,d;store.raw.cache.clear();torch.cuda.empty_cache()
    df=pd.DataFrame(rows);saved=pd.DataFrame(read(rt/'DEVELOPMENT_METRICS_PRIVATE.pkl'));keys=['arm','fold','patient'];l=df.set_index(keys).sort_index();r=saved[saved.arm!='D1'].set_index(keys).sort_index();assert l.index.equals(r.index) and np.allclose(l[m.METRICS],r[m.METRICS],rtol=0,atol=1e-12,equal_nan=True);means=df.groupby('arm')[m.METRICS].mean();summary=pd.read_csv(rt/'results/DEVELOPMENT_SUMMARY.csv').set_index('arm');assert np.allclose(summary.loc[means.index,m.METRICS],means,rtol=0,atol=1e-12,equal_nan=True)
    bs=pd.read_csv(rt/'results/PAIRED_BOOTSTRAP.csv')
    for arm in ARMS:
        assert len(df[df.arm==arm])==65 and df[df.arm==arm].patient.nunique()==47
        for row in b.compare(c,saved[saved.arm==arm],saved[saved.arm=='D1'],m.METRICS):
            q=bs[(bs.comparison==arm+' minus D1')&(bs.metric==row['metric'])].iloc[0];assert np.allclose(q[['delta','ci_low','ci_high']].to_numpy(float),[row['delta'],row['ci_low'],row['ci_high']],atol=1e-12,rtol=0,equal_nan=True)
    write(rt/'results/INDEPENDENT_VERIFICATION.json',dict(status='PASS',all15_actual_training_cells=True,all15_strict_factory_checkpoint_replays=True,independent_sklearn_cells=195,bootstrap_independent_multiplicity=True,all_frozen_D1_weights_exact=True,original_roles_preserved=True,no_TEST_access=True,folds=audits));print('PHASE7_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
