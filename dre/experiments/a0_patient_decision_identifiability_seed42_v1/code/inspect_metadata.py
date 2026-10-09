"""Read-only source inventory; never emits patient identities or field values."""
import argparse
import sys
from collections import Counter
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--cache',type=Path,required=True); p.add_argument('--export',type=Path,required=True); a=p.parse_args()
    sys.path.insert(0,str(a.source))
    from outcome_hifos.cache_schema import load_cache_contract,file_sha256
    assert file_sha256(a.cache)=='9b5bb58a0175aba494ad79e30c5a65c7cee33c7d464273f9bcf62b9b280b0087'
    assert file_sha256(a.export)=='1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd'
    cache=load_cache_contract(a.cache)
    with np.load(a.export,allow_pickle=True) as z:ids=set(z['patient'].astype(str))
    metas=[cache.patient_index[i] for i in ids]
    print('TOP_LEVEL_KEYS',cache.top_level_keys)
    print('PATIENT_FIELD_COUNTS',dict(Counter(k for m in metas for k in m)))
    print('CHANNEL_FIELD_COUNTS',dict(Counter(k for m in metas for c in m.get('channel_meta',[]) for k in c)))
    for k in ['config','args','provenance','source_hashes','source','metadata']:
        v=cache.payload.get(k)
        print('SOURCE_CONTAINER',k,type(v).__name__,sorted(v) if isinstance(v,dict) else 'no mapping')

if __name__=='__main__':main()
