"""Fresh read-only winning-factory replay of already selected Development, never Outer."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(Path(__file__).parent))
from explore import context,make_base,original_modules,Inputs,EV,sha,read,write
from best_recipe import build,training_loss

def main(a):
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);metrics,_=original_modules();store=Inputs(a.baseline,a.data,a.runtime);store.arm='M1_SCALED_DGM'
    freeze=json.loads((a.runtime/'MODEL_FREEZE_PRIVATE.json').read_text());source=json.loads((a.runtime/'INPUT_FREEZE_PRIVATE.json').read_text())
    assert source['code']=={p.name:sha(p) for p in (ROOT/'phase3/code').glob('*.py')}
    rs=[];audits=[]
    for fold in range(1,6):
        base=a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt';d=torch.load(base.parent.parent/'BANK_PRIVATE.pt',map_location='cpu',weights_only=False);row=next(q for q in freeze['selections'] if q['arm']=='M8_MACRO_AWARE' and q['fold']==fold)
        cp=a.runtime/f'M8_MACRO_AWARE/fold{fold}/TRAIN/BEST_PRIVATE.pt';assert sha(cp)==row['checkpoint_sha256'];ck=torch.load(cp,map_location='cpu',weights_only=False)
        model=build(make_base(core,fac,records,base,42+1009*fold)).cuda();model.load_state_dict(ck['model'],strict=True);model.eval()
        scale=torch.tensor(read(a.phase1/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda');q=EV.predict(model,d,d['val'],store,scale)
        saved=read(a.runtime/f'predictions_private/M8_MACRO_AWARE_fold{fold}.pkl');ref=saved['modes']['full'];drift=float(np.max(np.abs(q['logit']-ref['logit'])))
        s=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();old=torch.sigmoid(torch.from_numpy(ref['logit'])).numpy();assert drift<=1e-6 and np.array_equal(s<row['threshold'],old<row['threshold'])
        assert np.array_equal(q['base_logit'],ref['base_logit'])
        rs.extend(metrics.patient_rows(c,d,s,row['threshold'],'M8_FACTORY_REPLAY',fold));audits.append(dict(fold=fold,logit_max_difference=drift,decisions_exact=True,checkpoint_sha256=sha(cp),base_logit_exact=True))
        del model,d
    mean=pd.DataFrame(rs)[metrics.METRICS].mean();original=pd.read_csv(a.runtime/'results/DEVELOPMENT_SUMMARY.csv').set_index('arm').loc['M8_MACRO_AWARE']
    assert np.allclose(mean.to_numpy(),original[metrics.METRICS].to_numpy(dtype=float),rtol=0,atol=1e-12)
    z=torch.tensor([-2.,0.,1.],requires_grad=True);y=torch.tensor([0.,1.,1.]);p=.5;from decision_modules import soft_macro_loss
    expected=.75*torch.nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(p))+.25*soft_macro_loss(z,y)
    assert torch.equal(training_loss(z,y,p),expected)
    write(a.out,dict(status='PASS',best_recipe_factory_strict_checkpoint_compatible=True,all5_development_folds_freshly_replayed=True,metrics=mean.to_dict(),folds=audits,exact_declared_loss=True,no_new_Outer=True,no_training_or_checkpoint_modification=True,only_aggregate_output=True));print('BEST_RECIPE_FACTORY_FRESH_DEVELOPMENT_REPLAY_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'));p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--data',type=Path,default=Path(r'D:\dgm_net_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));p.add_argument('--out',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_final_audit\BEST_RECIPE_REPLAY.json'));main(p.parse_args())
