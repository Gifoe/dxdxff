"""Matched joint fits, then a validation-only gate, then one B model freeze."""
import torch
import numpy as np
from runtime import *
from pr2 import PR2
from drechan_adapter import DREChanAdapter
from training import fit

ARMS=['D0_RANK_LOSS','D1_EXPANDED_ABS','D2_PR2_FULL']

def make(arm,fold):
    base=base_model(fold)
    assert sum(p.numel() for p in base.parameters())==11425
    stats=torch.load(ROOT/'references_private'/f'fold{fold}.pt',weights_only=False)
    assert state_hash(base.state_dict())==stats['initial_state']
    if arm=='D0_RANK_LOSS':return DREChanAdapter(base)
    plugin=PR2(108,stats['D108'],mode='abs' if arm=='D1_EXPANDED_ABS' else 'pr2',
               calibration=arm!='D3_PR2_NO_CAL',seed=42+1009*fold+270007).cuda()
    if arm=='D4_PR2_NO_RANK':plugin.ranking.requires_grad_(False)
    return DREChanAdapter(base,plugin,ranking=arm!='D4_PR2_NO_RANK',calibration=arm!='D3_PR2_NO_CAL')

def main():
    admission=check_admission();assert read_json(PUBLIC/'A_VERIFIED.json')['status']=='PASS'
    c=context();parent=c['parent'];binding=dict(input=admission['input'],source=source_hashes());cells=[]
    for fold in range(1,6):
        data=bank_data(fold)
        for arm in ARMS:
            model=make(arm,fold);cells.append(fit(model,data,fold,arm,'B',binding))
            del model;gc.collect();torch.cuda.empty_cache()
            write(PUBLIC/'B_TRAIN_CELLS.json',dict(completed=len(cells),minimum_new_cells=15))
            print('PR2_B_CELL',arm,fold,len(cells),flush=True)
    metric=lambda arm,name:float(np.mean([r['validation_'+name] for r in cells if r['arm']==arm]))
    delta_auc=metric('D2_PR2_FULL','auroc')-metric('D1_EXPANDED_ABS','auroc')
    delta_macro=metric('D2_PR2_FULL','macro_f1')-metric('D1_EXPANDED_ABS','macro_f1')
    gate=dict(status='PASS' if delta_auc>0 and delta_macro>=0 else 'NOT_TRIGGERED',
              validation_delta_auroc=delta_auc,validation_delta_macro_f1=delta_macro,outer_access=False)
    path=PUBLIC/'B_ABLATION_GATE.json'
    if path.exists():assert read_json(path)==gate
    else:write(path,gate)
    if gate['status']=='PASS':
        for fold in range(1,6):
            data=bank_data(fold)
            for arm in ['D3_PR2_NO_CAL','D4_PR2_NO_RANK']:
                model=make(arm,fold);cells.append(fit(model,data,fold,arm,'B',binding))
                del model;gc.collect();torch.cuda.empty_cache()
                print('PR2_B_MATCHED_ABLATION_CELL',arm,fold,flush=True)
    for fold in range(1,6):
        old=next(r for r in parent.frozen()['cells'] if r['arm']=='FULL' and r['seed']==42 and r['fold']==fold)
        assert sha(old['checkpoint'])==admission['input']['baselines'][f'DRECHAN_FULL_{fold}']
        cells.append(dict(arm='D_HISTORICAL_FULL',fold=fold,stage='B',checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
                          threshold=old['threshold'],threshold_kind='NEZ_LT',added_parameters=0,trainable_parameters=0,
                          reused=True,epoch=old.get('epoch',0),historical_cell=old))
    for fold in range(1,6):
        a=next(r for r in cells if r['fold']==fold and r['arm']=='D1_EXPANDED_ABS')
        b=next(r for r in cells if r['fold']==fold and r['arm']=='D2_PR2_FULL')
        assert a['trainable_parameters']==b['trainable_parameters']==13283
        assert a['added_parameters']==b['added_parameters']==1858
    own_freeze('B',cells)
    write(PUBLIC/'B_TRAIN_STATUS.json',dict(status='ALL_B_MODELS_FROZEN',fresh_joint_fits=len(cells)-5,historical_reuses=5,
                                         validation_gate=gate,outer_new_predictions_viewed=False))
    print('PR2_B_ALL_SELECTIONS_FROZEN',flush=True)

if __name__=='__main__':main()
