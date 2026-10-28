"""All15 new adapters and5 output-only choices freeze before new A Outer."""
import numpy as np
import torch
from scipy.special import expit
from runtime import *
from pr2 import PR2
from svm_adapter import crossfit,logits,probabilities
from training import fit
from metrics import validation

ARMS=['S0_NO_PR','S1_LEGACY_PR','S2_OUTPUT_Z','S3_EXPANDED_ABS','S4_PR2_RANK','S5_PR2_FULL']

def output_z(logit,pid):
    out=np.empty_like(logit,dtype=np.float64)
    for patient in sorted(set(pid)):
        ix=pid==patient;sd=logit[ix].std(ddof=0);out[ix]=(logit[ix]-logit[ix].mean())/(sd if sd>0 else 1.)
    return out

def make(arm,fold):
    stats=torch.load(ROOT/'references_private'/f'fold{fold}.pt',weights_only=False)['D12']
    return PR2(12,stats,mode='abs' if arm=='S3_EXPANDED_ABS' else 'pr2',
               calibration=arm!='S4_PR2_RANK',seed=42+1009*fold+270007)

def inputs(fold):
    path=ROOT/'svm_inputs_private'/f'fold{fold}.pkl'
    return load(path)

def main():
    admission=check_admission();c=context();config=read_json(Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1\BASELINE_CONFIGS.json'))
    binding=dict(input=admission['input'],source=source_hashes());cells=[]
    for fold in range(1,6):
        data=bank_data(fold);tr,va=data['train'],data['val'];path=ROOT/'svm_inputs_private'/f'fold{fold}.pkl'
        if path.exists():
            prepared=load(path);assert prepared['binding']==binding
        else:
            base=load(SVMROOT/'tabular_models_private/Linear_SVM_NO_PR'/f'fold{fold}.pkl')
            oof,audit=crossfit(data['raw12'].astype(float),data['y'],data['patient'],tr,
                               config['configs']['Linear_SVM'][str(fold)]['C'],42+1009*fold)
            base_logit=logits(base['bundle'],data['raw12'].astype(float));base_logit[tr]=oof[tr]
            assert np.isfinite(base_logit).all()
            prepared=dict(binding=binding,base=base_logit,crossfit_audit=audit,reference=sha(ROOT/'references_private'/f'fold{fold}.pt'))
            save(path,prepared)
        data['base']=prepared['base']
        for arm in ARMS:
            if arm in ['S0_NO_PR','S1_LEGACY_PR']:
                oldarm='Linear_SVM_'+('PR' if arm=='S1_LEGACY_PR' else 'NO_PR')
                cp=SVMROOT/'tabular_models_private'/oldarm/f'fold{fold}.pkl';q=load(cp)
                assert sha(cp)==admission['input']['baselines'][f'{oldarm}_{fold}']
                cells.append(dict(arm=arm,fold=fold,stage='A',checkpoint=str(cp),checkpoint_sha256=sha(cp),
                       threshold=float(q['selected']['threshold']),threshold_kind='EZ_GE',added_parameters=0,
                       trainable_parameters=0,reused=True,epoch=0))
            elif arm=='S2_OUTPUT_Z':
                z=output_z(data['base'][va],data['patient'][va]);o=dict(final=z,rank_score=z)
                sel=validation(subset(data,va),o)
                cells.append(dict(arm=arm,fold=fold,stage='A',checkpoint='',checkpoint_sha256='',
                       threshold=sel['threshold'],threshold_kind='EZ_GE',added_parameters=0,trainable_parameters=0,reused=False,epoch=0))
            else:
                model=make(arm,fold);cells.append(fit(model,data,fold,arm,'A',binding));del model
            write(PUBLIC/'A_TRAIN_CELLS.json',dict(completed=len(cells),total=30))
            print('PR2_A_CELL',arm,fold,len(cells),30,flush=True)
    assert len(cells)==30
    own_freeze('A',cells)
    write(PUBLIC/'A_TRAIN_STATUS.json',dict(status='ALL30_CHOICES_FROZEN',fresh_plugin_fits=15,
         historical_reuses=10,output_only_thresholds=5,nested_base_SVC_fits=60))
    print('PR2_A_ALL_SELECTIONS_FROZEN',flush=True)

if __name__=='__main__':main()
