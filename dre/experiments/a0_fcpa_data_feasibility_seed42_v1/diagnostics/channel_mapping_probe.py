"""Aggregate-only source-normalizer diagnosis; no signal or label scoring."""
import sys,json,pickle
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from audit import quiet_edf,resolve,header
from discover import ROOT,SRC,norm
sys.path.insert(0,str(SRC))
from module2_preprocessing import normalize_channel_name as richer


def main():
 with (ROOT/'EDF_PLAN_PRIVATE.pkl').open('rb') as f:plan=pickle.load(f)
 disc=json.loads((ROOT/'DISCOVERY_PRIVATE.json').read_text());raw={tuple(r['key']):r for r in disc['raw_primary']['records']}
 counts=defaultdict(Counter)
 for q in plan:
  if not q.get('failure'):continue
  r=raw[tuple(q['key'])];p,method=resolve(r['source_paths'][0]);c=counts[q['center']]
  c['records']+=1;c['stored_failure_'+q['failure']]+=1
  try:
   e=quiet_edf(p);names=e.ch_names;h=header(p)
   for name,cm in zip(r['channel_names'],r['source_channel_meta']):
    exact=any(cm.get(k) in names for k in ['raw_channel_name','channel_name_orig'])
    simple=sum(norm(n)==name for n in names);rich=sum(norm(richer(n))==name for n in names)
    c['channels']+=1;c['direct_metadata_match']+=exact;c['simple_normalizer_unique_match']+=simple==1;c['original_EDF_normalizer_unique_match']+=rich==1
    c['duplicate_EDF_labels']+=len(h['label'])-len(set(h['label']))>0
   e.close()
  except Exception as ex:c['probe_exception_'+type(ex).__name__]+=1
 result={k:dict(v) for k,v in counts.items()}
 (ROOT/'public/CHANNEL_MAPPING_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result),flush=True)


if __name__=='__main__':main()
