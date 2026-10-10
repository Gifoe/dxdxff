"""Finite deterministic CPU-only audit of immutable stored outputs."""
import argparse
import json
from pathlib import Path
import torch
from common import sha,write_json,digest
from inputs import load_inputs
from analysis import fit_phase,validation_phase,transport
from report import complete

def seal(path,value):
    tmp=path.with_suffix(path.suffix+'.partial'); torch.save(value,tmp); tmp.replace(path)

def main():
    p=argparse.ArgumentParser()
    for name in ['runtime','previous','source','protocol']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); torch.set_num_threads(2); torch.manual_seed(42)
    pub=a.runtime/'public'; pub.mkdir(parents=True,exist_ok=True)
    lock=json.loads(a.protocol.read_text(encoding='utf-8'))
    tests=json.loads((a.runtime/'ENGINEERING_TESTS.json').read_text())
    code={q.name:sha(q) for q in Path(__file__).parent.glob('*.py')}
    assert tests['status']=='PASS' and tests['protocol_sha256']==sha(a.protocol) and tests['code']==code
    binding={'protocol':sha(a.protocol),'code':code,'input_manifest':sha(a.protocol.parent/'audit/FROZEN_INPUT_MANIFEST.json')}
    assert binding['input_manifest']==lock['private_input_manifest_sha256']
    write_json(a.runtime/'RUN_BINDING_PRIVATE.json',binding)
    banks,controls=load_inputs(a.previous,a.source,a.protocol,pub)
    fits,gates,loo=fit_phase(banks,lock,pub)
    seal(a.runtime/'FIT_AND_ADMISSION_PRIVATE.pt',{'binding':binding,'fits':fits,'gates':gates,'LOO':loo})
    admission_hash=sha(pub/'FIT_POSTERIOR_UTILITY_GATE.json')
    rows,diag,errors,private=validation_phase(banks,fits,gates,controls,pub)
    seal(a.runtime/'VALIDATION_PRIVATE.pt',{'binding':binding,'admission_sha256':admission_hash,'rows':rows,'diagnostics':diag,'errors':errors,'channel_records':private})
    assert sha(pub/'FIT_POSTERIOR_UTILITY_GATE.json')==admission_hash
    shifts=transport(banks,fits,gates,pub)
    # Input immutability is rechecked after analysis; never write the old runtime.
    manifest=json.loads((a.protocol.parent/'audit/FROZEN_INPUT_MANIFEST.json').read_text())
    for rel,h in manifest.items():assert sha(a.previous/rel)==h
    complete(rows,diag,errors,gates,fits,shifts,lock,pub,a.protocol)

if __name__=='__main__':main()
