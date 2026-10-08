"""Export the EXACT source FIT/VAL/TEST membership of the frozen A1 80-cohort.
Run on the original server with the historical R1_HLV_* environment variables.
This does not train or read outer-test prediction outcomes.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    for k in ('R1_HLV_SOURCE_ROOT','R1_HLV_FIXED_MANIFEST'):
        if not os.environ.get(k):raise RuntimeError(f'missing {k}; run in frozen A1 source environment')
    source=Path(os.environ['R1_HLV_SOURCE_ROOT'])
    sys.path.insert(0,str(source))
    sys.path.insert(0,str(source/'neuroez_c'))
    sys.path.insert(0,str(source/'r1_hlv_ictal_dynamics_seed42_v1'/'code'))
    import exp_ez_hybrid as core
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args
    assert_sources(); install_interleaved_hlv_view()
    args=make_args('R0',Path(a.output).parent/'private_fold_audit_scratch')
    exp=core.Exp_EZHybridLocalization(args)
    if len(exp.patient_index)!=80 or len(exp.outer_splits)!=5:
        raise ValueError('wrong A1 cohort/fold count')
    result=[]
    for fold in exp.outer_splits:
        roles={name:list(map(str,fold[name])) for name in ('fit_subjects','validation_subjects','test_subjects')}
        result.append({'fold_idx':int(fold['fold_idx']),**roles})
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'source':'frozen_a1_r1_hlv_outer_splits',
                                  'folds':result},indent=2,ensure_ascii=False),encoding='utf-8')
    print('EXPORTED_ROLE_COUNTS',[(r['fold_idx'],len(r['fit_subjects']),len(r['validation_subjects']),len(r['test_subjects'])) for r in result])

if __name__=='__main__':main()
