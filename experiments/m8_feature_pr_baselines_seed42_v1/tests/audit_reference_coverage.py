"""Post-inference reporting only: no model/prediction/scaler access or mutation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from support import arguments,read,np,pd,EV,sha,write

def run(a):
    patients=read(a.runtime/'REFERENCE_PATIENT_METRICS_PRIVATE.pkl');rows=[]
    for (model,version),q in patients.groupby(['model','version']):
        for m in EV.CLASSIFICATION+EV.RANKING+['predicted_ez_fraction']:
            valid=int(np.isfinite(q[m]).sum());rows.append(dict(model=model,version=version,metric=m,total_patients=len(q),valid_patients=valid,undefined_patients=len(q)-valid))
    out=a.runtime/'public/reference_metric_coverage.csv';pd.DataFrame(rows).to_csv(out,index=False)
    write(a.runtime/'public/REFERENCE_COVERAGE_AUDIT.json',dict(status='PASS',phase='read-only post-inference reporting',source_sha256=sha(Path(__file__)),aggregate_sha256=sha(out),no_model_or_prediction_changes=True))
    print('REFERENCE_COVERAGE_AUDIT_PASS')

if __name__=='__main__':run(arguments())
