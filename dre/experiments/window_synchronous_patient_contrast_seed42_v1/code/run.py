"""Fixed full trusted-server pipeline. A failed control stops before labels/probes."""
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_data import options,setup,data
from common import read,write_json

def main():
 a=options();lock,paths,binding=setup(a)
 if a.phase in ['data','all']:data(a,lock,paths,binding)
 if a.phase=='data':return
 bank=read(a.runtime/'CONTEXT_TABLES_PRIVATE.pkl')
 if bank['control']['status']!='PASS':
  terminal='SYNCHRONIZATION_CONTROL_NOT_IDENTIFIABLE'
  for filename in ['INNER_EVALUATION_SUMMARY.csv','INNER_EVALUATION_BY_ORIGINAL_FOLD.csv','INNER_EVALUATION_BY_CENTER.csv','UNIQUE_PATIENT_SUMMARY.csv','PAIRED_BOOTSTRAP.csv','SENSITIVITY_SPECIFICITY_AUDIT.csv','RANKING_CHANGES.csv']:
   pd.DataFrame([dict(arm=arm,status='NOT_RUN_CONTROL_GATE',**{m:None for m in lock['metrics']}) for arm in ['C0','C1','C2']]).to_csv(a.pub/filename,index=False)
  write_json(a.pub/'SCIENTIFIC_GATE.json',dict(terminal=terminal,predictive_gates='NOT_EVALUATED',control=bank['control'],no_labels_read_for_probe=True,no_TIDE_or_outer=True));write_json(a.pub/'RUN_STATUS.json',dict(status='COMPLETE_STOPPED_BEFORE_PROBES',terminal=terminal,probe_models_fitted=0));return
 with threadpool_limits(limits=2):
  if a.phase in ['probe','all']:
   from fit_probe import fit;fit(a,lock,binding)
  if a.phase in ['evaluate','all']:
   from evaluate import evaluate;evaluate(a,lock,binding)
   from verify import verify;verify(a,lock,binding)

if __name__=='__main__':main()
