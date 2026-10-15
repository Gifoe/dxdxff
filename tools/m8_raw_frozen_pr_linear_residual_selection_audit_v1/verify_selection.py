"""Independent validation-only selection/history audit; no Outer inference or fitting."""
import sys
from pathlib import Path

CODE=Path(r'D:\m8_raw_backbones_pr_seed42_v1')
TASK=Path(r'D:\m8_raw_frozen_pr_linear_residual_seed42_v1')
RUNTIME_NEW=Path(r'D:\m8_raw_frozen_pr_linear_residual_seed42_v1_runtime')
sys.path.insert(0,str(CODE))
from support import *


def main():
    module=load('frozen_pr_linear_residual_audit_target',TASK/'run.py')
    old,core,rec,banks,parent_binding=setup(module.PARENT_RUNTIME)
    binding=jr(RUNTIME_NEW/'INPUT_FREEZE_PRIVATE.json')
    for name,h in binding['source'].items():assert sha(TASK/name)==h
    freeze=jr(RUNTIME_NEW/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert freeze['source_sha256']==digest(binding) and len(freeze['selections'])==20
    rows=[];source=digest(binding)
    for r in freeze['selections']:
        name,fold=r['model'],r['fold'];d=banks[fold-1]
        fit=module.roles(rec['meta'],fold)['fit'];val=module.roles(rec['meta'],fold)['validation'];test=module.roles(rec['meta'],fold)['test']
        assert fit.isdisjoint(val) and fit.isdisjoint(test) and val.isdisjoint(test)
        va=module.data(name,fold,'validation',rec,d)
        assert set(va['patient'])==val
        net=torch.nn.Linear(va['z'].shape[1],1)
        torch.nn.init.zeros_(net.weight);torch.nn.init.zeros_(net.bias)
        epoch0,_=module.validate(net,va)
        cp=torch.load(r['checkpoint'],weights_only=False)
        last=torch.load(Path(r['checkpoint']).parent/'HEAD_LAST_PRIVATE.pt',weights_only=False)
        assert cp['source_sha256']==last['source_sha256']==source
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        candidates=[dict(epoch=0,**epoch0)]+last['history']
        chosen=max(candidates,key=lambda q:(q['validation_macro_f1'],q['validation_positive_f1'],-q['epoch']))
        assert chosen['epoch']==r['epoch']==cp['epoch'] and chosen['threshold']==r['threshold']==cp['selection']['threshold']
        net.load_state_dict(cp['state']);selected,_=module.validate(net,va)
        assert selected==cp['selection']
        assert np.isclose(float(net.weight.norm()),float(torch.linalg.vector_norm(cp['state']['weight'])),atol=0,rtol=0)
        rows.append(dict(model=name,fold=fold,selected_epoch=r['epoch'],epochs_run=r['epochs_run'],threshold=r['threshold'],
            parameters=r['parameters'],weight_l2=float(net.weight.norm().detach()),bias=float(net.bias.detach()[0]),
            zero_candidate_validation_macro_f1=epoch0['validation_macro_f1'],selected_validation_macro_f1=selected['validation_macro_f1']))
    for p,h in binding['protected'].items():assert sha(p)==h
    out=RUNTIME_NEW/'public'
    pd.DataFrame(rows).to_csv(out/'HEAD_PARAMETER_DIAGNOSTICS.csv',index=False)
    write(out/'SELECTION_HISTORY_VERIFICATION.json',dict(status='PASS',heads_verified=20,
        epoch_zero_candidate_included=True,all_history_argmax_and_thresholds_recomputed=True,
        patient_disjoint_fit_validation_outer=True,parent_artifact_hashes_unchanged=True,
        new_training_or_Outer_inference=False,source_sha256=sha(Path(__file__))))
    write(out/'PUBLIC_SHA256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='PUBLIC_SHA256.json'})
    print('ALL20_VALIDATION_HISTORY_SELECTIONS_INDEPENDENTLY_VERIFIED',flush=True)


if __name__=='__main__':main()
