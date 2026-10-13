"""Verify an honest mandatory-gate stop; refuses efficacy claims or outer work."""
import argparse
import json
from pathlib import Path
import pandas as pd
from audit_source import sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 g=json.loads((a.output/'DATA_GATE.json').read_text())
 t=json.loads((a.output/'SCIENTIFIC_TERMINAL.json').read_text())
 r=json.loads((a.output/'RUN_STATUS.json').read_text())
 assert not g['pass_gate'] and t['terminal']=='LF_DATA_FIDELITY_GATE_FAILED'
 assert not any(g[k] for k in ['training_performed','LF24_extracted','baseline_development_replayed','new_outer_accessed'])
 assert r['models_trained']==r['new_outer_evaluations']==0 and not r['stages_B_to_F_executed']
 assert g['input_hashes']['protocol']==sha(Path(__file__).parents[1]/'PROTOCOL_LOCK.json')
 assert g['input_hashes']['stage_A_code']==sha(Path(__file__).with_name('audit_source.py'))
 table=pd.read_csv(a.output/'LF_FEATURE_COVERAGE.csv')
 assert table.iloc[0].canonical_channels==7635 and not table.LF24_feature_bank_extracted.any()
 print('FCPA_LF_NEGATIVE_AUDIT_INTEGRITY_PASS',flush=True)


if __name__=='__main__':main()
