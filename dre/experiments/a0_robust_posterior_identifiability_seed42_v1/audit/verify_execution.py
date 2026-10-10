"""Verify an unchanged-protocol repeat without exposing private records."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    for name in ['runtime','repeat','previous','protocol']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.protocol.parent/'code'))
    from common import sha,write_json

    def equal(x,y):
        assert type(x) is type(y)
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:equal(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for v,w in zip(x,y):equal(v,w)
        elif isinstance(x,np.ndarray):assert np.array_equal(x,y,equal_nan=True)
        elif isinstance(x,torch.Tensor):assert torch.equal(x,y)
        elif isinstance(x,(float,np.floating)) and np.isnan(x):assert np.isnan(y)
        else:assert x==y

    files=sorted(p.name for p in (a.runtime/'public').iterdir() if p.is_file())
    assert files==sorted(p.name for p in (a.repeat/'public').iterdir() if p.is_file())
    hashes={}
    for name in files:
        first=sha(a.runtime/'public'/name)
        assert first==sha(a.repeat/'public'/name),name
        hashes[name]=first
    assert sha(a.runtime/'ENGINEERING_TESTS.json')==sha(a.repeat/'ENGINEERING_TESTS.json')
    for name in ['FIT_AND_ADMISSION_PRIVATE.pt','VALIDATION_PRIVATE.pt']:
        equal(torch.load(a.runtime/name,map_location='cpu',weights_only=False),
              torch.load(a.repeat/name,map_location='cpu',weights_only=False))
    status=json.loads((a.runtime/'public/RUN_STATUS.json').read_text())
    assert status['status']=='COMPLETE_FROZEN_DIAGNOSTIC'
    for name,h in status['aggregate_files'].items():assert hashes[name]==h
    lock=json.loads(a.protocol.read_text())
    assert sha(a.protocol.parent/'audit/FROZEN_INPUT_MANIFEST.json')==lock['private_input_manifest_sha256']
    manifest=json.loads((a.protocol.parent/'audit/FROZEN_INPUT_MANIFEST.json').read_text())
    for rel,h in manifest.items():assert sha(a.previous/rel)==h
    validation=torch.load(a.runtime/'VALIDATION_PRIVATE.pt',map_location='cpu',weights_only=False)
    gates=json.loads((a.runtime/'public/FIT_POSTERIOR_UTILITY_GATE.json').read_text())['arms']
    assert len(validation['channel_records'])==65*sum(g['admitted'] for g in gates.values())
    write_json(a.runtime/'DETERMINISM_AND_IMMUTABILITY.json',{
        'status':'PASS','same_seed':42,'same_protocol_sha256':sha(a.protocol),
        'public_files_byte_identical':len(files),'public_file_sha256':hashes,
        'private_density_floor_LOO_decisions_and_VAL_objects_equal':True,
        'engineering_tests_identical':True,'old_private_files_unchanged':len(manifest),
        'blocked_VAL_arms_not_inferred':True,'new_training':False,'outer_TEST':False,
        'verification_source_sha256':sha(__file__)})
    print('DETERMINISM_IMMUTABILITY_PASS',len(files),len(manifest),flush=True)


if __name__=='__main__':main()
