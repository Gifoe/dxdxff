"""Full source metadata audit; no label use, raw waveform load, or test scoring."""
import argparse
from collections import defaultdict
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from base import sha,json_write,torch_write
from features import pack,FEATURES


def main():
    p=argparse.ArgumentParser()
    for n in ['runtime','protocol','source','cache','split','prior']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args(); root=a.runtime; pub=root/'public'; pub.mkdir(parents=True,exist_ok=True)
    lock=json.loads(a.protocol.read_text()); assert sha(a.cache)==lock['feature_cache_sha256']
    assert sha(a.split)==lock['split_sha256']
    export=a.prior/'gate/FEATURES_PRIVATE.npz'; assert sha(export)==lock['private_88D_export_sha256']
    # Audit loads only identity/center arrays from the prior validated export.
    with np.load(export,allow_pickle=True) as d:
        pid=d['patient'].astype(str); ch=d['channel'].astype(str); center=d['center'].astype(str)
    assert len(pid)==7635 and len(set(pid))==80 and len(set(zip(pid,ch)))==7635
    sys.path.insert(0,str(a.source))
    with a.cache.open('rb') as f: payload=pickle.load(f)
    bypatient=defaultdict(list); keys=set()
    for rec in payload['run_records']:
        if str(rec['subject_id']) not in set(pid): continue
        identity=(str(rec['subject_id']),str(rec['run_id']),str(rec['sample']['sample_id']))
        assert identity not in keys; keys.add(identity); bypatient[identity[0]].append(rec)
    assert len(keys)==256 and set(bypatient)==set(pid)
    banks={}; rows=[]; private_records=[]; center_rows=[]
    for patient in sorted(bypatient):
        ix=np.flatnonzero(pid==patient); canonical=ch[ix].tolist()
        recs=sorted(bypatient[patient],key=lambda r:(str(r['run_id']),str(r['sample']['sample_id'])))
        b,details=pack(recs,canonical); b['channels']=canonical; b['center']=str(center[ix[0]])
        banks[patient]=b
        stage=np.stack([((b['time']>=lo)&(b['time']<hi)&b['mask']).sum(2) for lo,hi in [(-10,0),(0,8),(8,20)]],-1)
        for i in range(len(canonical)):
            rows.append({'center':b['center'],'pre_any':bool((stage[i,:,0]>=1).any()),
                         'early_any':bool((stage[i,:,1]>=2).any()),'later_any':bool((stage[i,:,2]>=1).any())})
        for k,d in enumerate(details):
            st=d['stage_counts']; info={key:v for key,v in d.items() if key!='stage_counts'}
            info.update({'center':b['center'],'channels':len(canonical),
                'pre_sufficient':int((st[:,0]>=1).sum()),'early_sufficient':int((st[:,1]>=2).sum()),
                'later_sufficient':int((st[:,2]>=1).sum()),
                'pre_valid_tokens':int(st[:,0].sum()),'early_valid_tokens':int(st[:,1].sum()),'later_valid_tokens':int(st[:,2].sum())})
            center_rows.append(info)
            private_records.append({'patient':patient,'run':str(recs[k]['run_id']),**info})
        print('TEMPORAL_PATIENT_COMPLETE',len(banks),'/80',flush=True)
    frame=pd.DataFrame(rows); rf=pd.DataFrame(center_rows)
    early=float(frame.early_any.mean()); status='PASS' if early>=.9 else 'TEMPORAL_FEATURE_COVERAGE_BLOCKED'
    report={'status':status,'pass':early>=.9,'patients':80,'canonical_channels':7635,'seizures':256,
        'record_channel_instances':int(rf.channels.sum()),'feature_order':FEATURES,
        'unique_patient_channel_stage_coverage':{k:float(frame[k].mean()) for k in ['pre_any','early_any','later_any']},
        'record_channel_stage_coverage':{k:float(rf[k+'_sufficient'].sum()/rf.channels.sum()) for k in ['pre','early','later']},
        'all_records_early_stage_supported':bool((rf.early_sufficient>0).all()),
        'source_window_count_histogram':{str(k):int(v) for k,v in rf.source_windows.value_counts().items()},
        'source_window_duration_seconds':sorted(set(rf.width_seconds)),
        'sampling_interval_min_max_seconds':[float(rf.interval_min.min()),float(rf.interval_max.max())],
        'monotonic_all_records':bool(rf.source_monotonic.all()),'duplicate_times':0,
        'invalid_time_windows':int(rf.invalid_time_windows.sum()),
        'nonfinite_feature_channel_windows':int(rf.nonfinite_feature_windows.sum()),
        'boundary_invalid_source_windows':int(rf.invalid_boundary_windows.sum()),
        'seizure_missing_canonical_channels':int(rf.missing_channels.sum()),
        'left_boundary_records':int((rf.left_padding_seconds>0).sum()),
        'clinical_onset_provenance':'explicit feature relative centers + source annotations and valid-range metadata; NOT independent EDF replay',
        'no_raw_waveform_read':True,'no_labels_used_in_temporal_audit':True,'no_outer_prediction_evaluation':True,
        'source_feature_SHA256':sha(a.cache),'split_SHA256':sha(a.split),'protocol_SHA256':sha(a.protocol)}
    # Public counts by center and coverage histograms; identifiable seizure
    # coverage ledger remains private, not a public per-record medical table.
    aggregate=[]
    for ce,g in rf.groupby('center'):
        u=frame[frame.center==ce]
        aggregate.append({'center':ce,'seizures':len(g),'patient_channels':len(u),'record_channel_instances':int(g.channels.sum()),
            **{k+'_unique_fraction':float(u[k+'_any'].mean()) for k in ['pre','early','later']},
            **{k+'_record_fraction':float(g[k+'_sufficient'].sum()/g.channels.sum()) for k in ['pre','early','later']},
            **{k+'_tokens':int(g[k+'_valid_tokens'].sum()) for k in ['pre','early','later']}})
    pd.DataFrame(aggregate).to_csv(pub/'TEMPORAL_COVERAGE_BY_CENTER.csv',index=False)
    hist=rf.groupby(['center','source_windows']).agg(seizures=('channels','size'),channel_instances=('channels','sum')).reset_index()
    hist.to_csv(pub/'TEMPORAL_RECORD_COVERAGE_HISTOGRAM.csv',index=False)
    torch_write(root/'TEMPORAL_PRIVATE.pt',{'banks':banks,'source_binding':report,'record_coverage_private':private_records})
    report['private_temporal_bank_sha256']=sha(root/'TEMPORAL_PRIVATE.pt')
    json_write(pub/'TEMPORAL_COVERAGE_AUDIT.json',report)
    json_write(pub/'RUN_STATUS.json',{'status':'TEMPORAL_AUDIT_'+status,'trained_arms':0,'outer_evaluation':False})
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
