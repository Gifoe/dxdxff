"""Post-failure observation only; frozen algorithms and tolerance unchanged."""
import json
import numpy as np
import qualify as q


def main():
    target=q.ROOT/'UNIT_GATE_INPUT_PRIVATE.pkl'
    if not target.exists():
        original=q.patient_z
        def capture(x,patients,legacy=False):
            if not legacy and not target.exists():q.save(target,dict(values=x,patients=patients))
            return original(x,patients,legacy)
        q.patient_z=capture
        try:q.main()
        except AssertionError:pass
    data=q.read(target);x=data['values'];patients=data['patients'];original=q.patient_z
    from measurement import patient_z,GROUPS
    z=patient_z(x,patients);rows=[]
    for scale in [1e-12,1e12]:
        scaled=x*scale;zs=patient_z(scaled,patients);bad=np.abs(zs-z)>1e-9
        for j,name in enumerate(GROUPS):
            b=bad[:,2*j:2*j+2];issues=[]
            for pid in np.unique(patients):
                ix=patients==pid
                for c in range(2*j,2*j+2):
                    if not bad[ix,c].any():continue
                    a=x[ix,c];a=a[np.isfinite(a)];s=scaled[ix,c];s=s[np.isfinite(s)]
                    issues.append(dict(exact_constant=bool(np.ptp(a)==0),input_range=float(np.ptp(a)),
                        input_abs_max=float(np.max(np.abs(a))),sd=float(np.std(a)),scaled_sd=float(np.std(s)),
                        unique_values=len(np.unique(a))))
            rows.append(dict(feature=name,coordinate_scale=scale,violating_coordinate_cells=int(b.sum()),
                max_abs_error=float(np.nanmax(np.abs(zs[:,2*j:2*j+2]-z[:,2*j:2*j+2]))),
                affected_patient_coordinates=len(issues),exact_constant_patient_coordinates=sum(r['exact_constant'] for r in issues),
                nonconstant_patient_coordinate_cases=[r for r in issues if not r['exact_constant']]))
    q.write(q.PUB/'UNIT_GATE_FAILURE_DIAGNOSIS.json',dict(status='FROZEN_UNIT_GATE_FAIL',tolerance_unchanged=1e-9,
        rows=rows,labels_indexed=False,classifier_fits=0,new_Outer_access=False,
        input_artifact_sha256=q.sha(target),frozen_measurement_code_sha256=q.sha(q.HERE/'measurement.py')))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
