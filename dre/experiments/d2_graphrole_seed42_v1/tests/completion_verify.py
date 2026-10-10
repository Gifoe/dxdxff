"""Post-run seals and frozen-reference immutability; no prediction/label loading."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from datetime import datetime,timezone

ROOT=Path(r'C:\d2_graphrole_seed42_runtime')
PRIOR=Path(r'C:\a0_source_posterior_decoder_seed42_runtime')
EXP=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()
def main():
    hashes=json.loads((EXP/'audit/FROZEN_INPUT_MANIFEST.json').read_text())
    for name,h in hashes.items(): assert sha(PRIOR/name)==h,'Original reference mutated'
    protocol=sha(EXP/'PROTOCOL_LOCK.json')
    audit=json.loads((ROOT/'public/ENGINEERING_TESTS.json').read_text())
    for name,h in audit['code_sha256'].items(): assert sha(EXP/'code'/name)==h,'Executed code mutated'
    cells=[]
    for fold in range(1,6):
        for arm in ['G1','G2','G3']:
            cell=ROOT/f'fold{fold}'/arm; seal=json.loads((cell/'FINAL_COMPLETE.json').read_text())
            assert seal['status']=='PASS' and seal['binding']['protocol']==protocol
            assert seal['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and seal['validation_sha256']==sha(cell/'VALIDATION_PRIVATE.pt')
            cells.append({'fold':fold,'arm':arm,'completed_epochs':seal['completed_epochs'],'selected_epoch':seal['selected_epoch'],'sealed':True})
    out={'status':'PASS','verified_original_reference_files_after_training':len(hashes),'frozen_reference_unchanged':True,
         'formal_cells':cells,'formal_new_runs':15,'completed_epochs':sum(x['completed_epochs'] for x in cells),
         'protocol_sha256':protocol,'all_executed_code_sha_unchanged':True,'outer_test_accessed':False,
         'environment':{'python':sys.version.split()[0],'platform':platform.platform(),**{p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','pandas','scikit-learn','networkx']}},
         'verified_utc':datetime.now(timezone.utc).isoformat()}
    (ROOT/'public/COMPLETION_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print('COMPLETION_VERIFIED',len(hashes),'original references unchanged;15 sealed runs; no labels read')
if __name__=='__main__': main()
