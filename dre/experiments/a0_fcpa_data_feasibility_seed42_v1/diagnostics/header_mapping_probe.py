"""Bounded EDF header-only normalizer probe; no MNE annotation scan or PSD."""
import sys,json,pickle
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from audit import resolve,header
from discover import ROOT,SRC,norm
sys.path.insert(0,str(SRC))
from module2_preprocessing import normalize_channel_name as richer


def main():
 with (ROOT/'EDF_PLAN_PRIVATE.pkl').open('rb') as f:plan=pickle.load(f)
 disc=json.loads((ROOT/'DISCOVERY_PRIVATE.json').read_text());raw={tuple(r['key']):r for r in disc['raw_primary']['records']}
 counts=defaultdict(Counter)
 for q in plan:
  if not q.get('failure'):continue
  r=raw[tuple(q['key'])];p,method=resolve(r['source_paths'][0]);c=counts[q['center']];c['records']+=1
  try:
   h=header(p);names=h['label'];clean=[n.rstrip('\x00 ') for n in names]
   for name,cm in zip(r['channel_names'],r['source_channel_meta']):
    c['channels']+=1
    c['direct_header_match']+=any(cm.get(k) in names for k in ['raw_channel_name','channel_name_orig'])
    c['trimmed_direct_header_match']+=any(cm.get(k) in clean for k in ['raw_channel_name','channel_name_orig'])
    c['simple_unique_match']+=sum(norm(n)==name for n in clean)==1
    c['original_EDF_normalizer_unique_match']+=sum(norm(richer(n))==name for n in clean)==1
    c['header_NUL_padding']+=any('\x00' in n for n in names)
  except Exception as ex:c['exception_'+type(ex).__name__]+=1
 result={k:dict(v) for k,v in counts.items()};(ROOT/'public/HEADER_MAPPING_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
