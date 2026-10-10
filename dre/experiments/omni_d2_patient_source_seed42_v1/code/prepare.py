"""Audit immutable original inputs, vault clinical labels away from extraction."""
import json
from collections import defaultdict
import numpy as np
from common import *

def main():
    cfg=lock();ROOT.mkdir(exist_ok=True);(ROOT/'public').mkdir(exist_ok=True)
    assert sha(OFFICIAL)==cfg['official_cnn_sha256']
    assert sha(OLD/'OOF_FROZEN_LOCK.json')==cfg['historical_freeze_sha256']
    assert sha(BASE/'outputs_r2/MODEL_PROTOCOL_LOCK_V2.json')==cfg['historical_model_protocol_sha256']
    for name,s in cfg['backbone_source_hashes'].items():assert sha(BASE/'code'/name)==s
    mf=manifest();assert len(mf)==124 and mf.fold.value_counts().to_dict()=={1:25,3:25,4:25,5:25,2:24}
    assert mf.center.value_counts().to_dict()==cfg['cohort']['centers']
    records=sorted((OLD/'records_unicode_v1').glob('edf_*.npz'));assert len(records)==137
    record_index=defaultdict(list);labels=defaultdict(dict);edfs=set();unknown=0;archive={}
    for path in records:
        archive[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as d:
            p=str(d['patient_name'].item());names=d['channel_names'].astype(str);y=d['labels'].astype(np.int8)
            assert p in set(mf.patient_name) and len(set(names))==len(names)
            assert int(d['fixed_start_sample'])==1000 and float(d['sampling_rate_hz'])==1000
            assert d['waveforms'].shape==(len(names),60000) and set(y)<= {-1,0,1}
            edf=str(d['edf_name'].item());assert edf not in edfs;edfs.add(edf)
            record_index[p].append(str(path));unknown+=int((y<0).sum())
            for c,t in zip(names,y):
                previous=labels[p].get(c,-1)
                if previous>=0 and t>=0:assert previous==t
                labels[p][c]=max(previous,int(t))
    totals=np.zeros(3,dtype=int)
    for row in mf.itertuples():
        p=str(row.patient_name);names=sorted(labels[p]);y=np.array([labels[p][c] for c in names],np.int8)
        assert len(record_index[p])==row.n_eligible_edfs
        assert int((y==1).sum())==row.n_pathological_channels and int((y==0).sum())==row.n_normal_channels
        assert len(names)==row.n_unique_good_context_channels
        save_npz(ROOT/'labels'/f'{key(p)}.npz',channels=np.array(names),labels=y)
        totals+=np.array([(y>=0).sum(),(y==1).sum(),(y==0).sum()])
    assert totals.tolist()==[10202,1113,9089] and unknown==2339
    for fold in range(1,6):
        cell=OLD/f'checkpoints/fold_{fold}/baseline';ck=cell/'last.pt';complete=json.loads((cell/'TRAINING_COMPLETE.json').read_text())
        assert sha(ck)==cfg['backbone_checkpoints'][str(fold)]==complete['checkpoint_sha256'] and complete['epochs']==30
        archive[str(ck)]=sha(ck);archive[str(cell/'TRAINING_COMPLETE.json')]=sha(cell/'TRAINING_COMPLETE.json')
    write_json(ROOT/'INPUTS_PRIVATE.json',{'records':record_index,'original_hashes':archive,'protocol_sha256':sha(PROTOCOL)})
    # Exact cached C0 replay, with validated cohort and checkpoint provenance.
    baseline=[];scorehashes={}
    for row in mf.itertuples():
        cache=OLD/f'oof_patient_cache/fold_{row.fold}/baseline/{key(str(row.patient_name))}.npz'
        scorehashes[str(cache)]=sha(cache)
        with np.load(cache,allow_pickle=False) as d:
            assert str(d['checkpoint_sha256'].item())==cfg['backbone_checkpoints'][str(row.fold)]
            assert str(d['freeze_sha256'].item())==cfg['historical_freeze_sha256']
            assert str(d['protocol_sha256'].item())==cfg['historical_model_protocol_sha256']
            names=d['channels'].astype(str);y=d['labels'];p=d['scores'];assert names.tolist()==sorted(labels[str(row.patient_name)])
            assert np.array_equal(y,np.array([labels[str(row.patient_name)][c] for c in names]))
            v=y>=0;baseline.append(metric(y[v],p[v]))
    reproduced={m:float(np.mean([r[m] for r in baseline])) for m in cfg['baseline_gate'] if m in METRICS}
    assert all(abs(reproduced[m]-cfg['baseline_gate'][m])<1e-12 for m in reproduced)
    write_json(ROOT/'C0_ARCHIVE_PRIVATE.json',scorehashes)
    write_json(ROOT/'public/BASELINE_REPRODUCTION.json',{'status':'ARCHIVED_REPLAY_PASS_FRESH_EMBEDDING_PENDING','historical_metrics':reproduced,'fresh_extraction_gate_required':True})
    write_json(ROOT/'public/COHORT_AND_SPLIT_AUDIT.json',{'status':'PASS','patients':124,'folds':5,'records':137,'labeled_units':10202,'pathological':1113,'normal':9089,'source_counts':cfg['cohort']['centers'],'unknown_good_occurrences':unknown,'original_manifest_sha256':sha(OLD/'PATIENT_5FOLD_MANIFEST_PRIVATE.csv'),'CNN_checkpoints':cfg['backbone_checkpoints'],'private_label_vault_separate_from_labelblind_extractor':True})
    write_json(ROOT/'public/RUN_STATUS.json',{'status':'PREPARED_EXTRACTION_REQUIRED','new_CNN_training':False})
    print('ORIGINAL124_137_10202_CHECKPOINTS_AND_C0_ARCHIVE_PASS',flush=True)

if __name__=='__main__':main()
