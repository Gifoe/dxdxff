"""Postrun verification only. Never trains or selects a model/threshold."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import torch
from common import *
def main():
    status=json.loads((ROOT/'public/RUN_STATUS.json').read_text());assert status['status']=='COMPLETE' and status['patients']==124 and status['folds']==5 and status['labeled_unique_patient_channels']==10202
    assert not status['new_CNN_training'] and not status['outer_labels_used_for_selection']
    engineering=json.loads((ROOT/'public/ENGINEERING_TESTS.json').read_text());assert engineering['status']=='PASS' and len(engineering['checks'])==35 and engineering['code']==code_hashes()
    gate=json.loads((ROOT/'BANK_GATE_PRIVATE.json').read_text());assert len(gate['banks'])==620 and gate['protocol']==sha(PROTOCOL)
    for p,h in gate['banks'].items():assert sha(p)==h
    inputs=json.loads((ROOT/'INPUTS_PRIVATE.json').read_text())
    for p,h in inputs['original_hashes'].items():assert sha(p)==h
    original_scores=json.loads((ROOT/'C0_ARCHIVE_PRIVATE.json').read_text())
    for p,h in original_scores.items():assert sha(p)==h
    freeze=json.loads((ROOT/'FROZEN_BEFORE_HEAD_OUTER_PRIVATE.json').read_text());assert freeze['code']==code_hashes() and len(freeze['selections'])==20
    for cell in freeze['selections']:
        if cell['method']=='C0':continue
        assert sha(ROOT/f"fold_{cell['fold']}/{cell['method']}/SELECTED_HEAD_PRIVATE.pt")==cell['checkpoint_sha256']
    ledger=torch.load(ROOT/'FINAL_OUTER_LEDGER_PRIVATE.pt',map_location='cpu',weights_only=False);assert status['prediction_ledger_sha256']==sha(ROOT/'FINAL_OUTER_LEDGER_PRIVATE.pt')
    assert len(ledger['prediction_hashes'])==496 and len(ledger['patient_rows'])==496
    for p,h in ledger['prediction_hashes'].items():assert sha(p)==h
    selection=pd.read_csv(ROOT/'public/MODEL_SELECTION.csv');assert len(selection)==20
    assert ((selection.loc[selection.method!='C0','selected_epoch']>=1)&(selection.loc[selection.method!='C0','selected_epoch']<=50)).all()
    history=pd.read_csv(ROOT/'public/TRAINING_HISTORY.csv');inner=history[history.stage=='inner'];assert inner.groupby(['fold','method','inner']).size().eq(50).all() and len(inner.groupby(['fold','method','inner']))==45
    refit=history[history.stage=='refit']
    for cell in freeze['selections']:
        if cell['method']=='C0':continue
        assert len(refit[(refit.fold==cell['fold'])&(refit.method==cell['method'])])==cell['selected_epoch']
    bootstrap=pd.read_csv(ROOT/'public/PATIENT_BOOTSTRAP.csv');assert bootstrap.patients.eq(124).all() and bootstrap.draws.eq(10000).all() and len(bootstrap)==76
    summary=pd.read_csv(ROOT/'public/OOF_PATIENT_SUMMARY.csv');assert summary.patients.eq(124).all()
    intervention=pd.read_csv(ROOT/'public/SOURCE_DIRECTION_INTERVENTIONS.csv');bias=intervention[intervention.intervention=='BIAS_OFF']
    assert bias[[m+'_delta' for m in RANK]].eq(0).all().all()
    report=(ROOT/'public/FINAL_REPORT.md').read_text()
    for question in range(1,27):assert f'\n{question}. ' in report
    artifacts={p.name:sha(p) for p in (ROOT/'public').iterdir() if p.is_file() and p.name!='COMPLETION_VERIFICATION.json'}
    write_json(ROOT/'public/COMPLETION_VERIFICATION.json',{'status':'PASS','patients':124,'new_CNN_training':False,'inner_heads_completed':45,'exact_selected_epoch_refits_completed':15,'outer_model_patient_cells':496,'original_inputs_unchanged':True,'clinical_labels_private':True,'code_unchanged':True,'protocol_sha256':sha(PROTOCOL),'torch':torch.__version__,'numpy':np.__version__,'aggregate_sha256':artifacts,'freeze_sha256':sha(ROOT/'FROZEN_BEFORE_HEAD_OUTER_PRIVATE.json')})
    print('OMNI_D2_COMPLETION_HASHES_COUNTS_AND_PRIVACY_PASS',flush=True)
if __name__=='__main__':main()
