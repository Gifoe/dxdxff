"""Real-cache audit + export for E0-E3.

Only load trusted local pickle files. No synthetic fallback in formal execution.
The 60s waveform midpoint is a SOURCE-CODE hypothesis until EDF provenance is verified.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import pickle
import re
from pathlib import Path
import numpy as np

EXPECTED_PATIENTS=80
EXPECTED_SEIZURES=256
EXPECTED_CHANNELS=7635
EXPECTED_RAW_RATE=250
EXPECTED_RAW_SAMPLES=15000
EXPECTED_FOLD_COUNTS=sorted([15,16,16,16,17])


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
    return h.hexdigest()


def normalize_channel_name(value):
    s=str(value or '').strip().upper()
    if s.startswith('EEG'):s=s[3:].strip()
    s=re.sub(r'[\s_\-]+','',s)
    m=re.match(r'^([A-Z]+)0*([0-9]+)$',s)
    return f'{m.group(1)}{int(m.group(2))}' if m else s


def value(record,name,default=None):
    if name in record:return record[name]
    return (record.get('sample') or {}).get(name,default)


def identity(record):
    sid=str(record.get('subject_id',''));run=str(record.get('run_id',''))
    sample=record.get('sample') or {}
    sample_id=str(sample.get('sample_id',record.get('sample_id',run)))
    onset=value(record,'seizure_onset_sec')
    source=value(record,'source_seizure_id')
    start=value(record,'start_sec');end=value(record,'end_sec')
    identity=str(source) if source not in (None,'') else f'onset={onset}|start={start}|end={end}'
    if not sid or not run or not sample_id or (onset is None and source in (None,'')):
        raise ValueError('missing compound seizure identity')
    return(sid,run,sample_id,identity)


def names(record):
    n=[normalize_channel_name(c) for c in value(record,'channel_names_norm',[]) or []]
    if not n or len(n)!=len(set(n)):raise ValueError(f'missing/duplicate channels in {identity(record)}')
    return n


def frozen_folds(path,patients):
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    folds=raw.get('folds',raw) if isinstance(raw,dict) else raw
    if not isinstance(folds,list) or len(folds)!=5:raise ValueError('need exact frozen five-fold FIT/VAL/TEST JSON')
    total_test=[]; normalized=[]; patient_set=set(patients)
    for entry in folds:
        f=int(entry.get('fold_idx',entry.get('fold',0)))
        roles={name:list(map(str,entry.get(name,[]))) for name in ('fit_subjects','validation_subjects','test_subjects')}
        sets=[set(roles[name]) for name in roles]
        if not sets[0] or not sets[1] or not sets[2] or set.union(*sets)!=patient_set or any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):
            raise ValueError(f'fold {f} leaks or does not cover 80 patients')
        total_test.extend(roles['test_subjects']);normalized.append({'fold_idx':f,**roles})
    if sorted(x['fold_idx'] for x in normalized)!=[1,2,3,4,5]:raise ValueError('expected fold IDs 1..5')
    if sorted(Counter(total_test).values()) != [1]*len(patient_set):raise ValueError('test patients not unique across folds')
    if sorted(len(x['test_subjects']) for x in normalized)!=EXPECTED_FOLD_COUNTS:
        raise ValueError('original frozen 80-patient outer-test sizes mismatch')
    if sorted(len(x['validation_subjects']) for x in normalized)!=[13]*5:
        raise ValueError('original validation counts mismatch')
    return normalized


def extract(feature_cache,raw_cache,folds_path,output,allow_source_only=False,expected_sha=None,
            approved_exclusion_private=None,amendment_json=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    paths={'feature_cache':Path(feature_cache),'raw_cache':Path(raw_cache),'frozen_folds':Path(folds_path)}
    hashes={k:digest(v) for k,v in paths.items()}
    if expected_sha:
        checks=json.loads(Path(expected_sha).read_text())
        for key,sha in checks.items():
            if key in hashes and hashes[key]!=sha:raise ValueError(f'{key} SHA-256 mismatch')
    # Never unpickle an untrusted artifact.
    with paths['feature_cache'].open('rb') as f:feature=pickle.load(f)
    with paths['raw_cache'].open('rb') as f:raw=pickle.load(f)
    # The source feature and raw caches may be larger than the frozen A1 cohort
    # (e.g. 90-person feature and 155-person raw caches). NEVER auto-select
    # patients from the full cache: derive the target cohort from frozen TEST IDs.
    raw_fold_obj=json.loads(Path(folds_path).read_text(encoding='utf-8'))
    raw_folds=raw_fold_obj.get('folds',raw_fold_obj) if isinstance(raw_fold_obj,dict) else raw_fold_obj
    subjects=sorted({str(sid) for fold in raw_folds for sid in fold['test_subjects']})
    if len(subjects)!=EXPECTED_PATIENTS:raise ValueError(f'expected {EXPECTED_PATIENTS} frozen TEST patients, got {len(subjects)}')
    full_index=feature['patient_index']
    if not set(subjects).issubset(full_index):raise ValueError('frozen patients missing from feature patient_index')
    pindex={sid:full_index[sid] for sid in subjects}
    feature_records=[r for r in feature['run_records'] if str(r.get('subject_id')) in pindex]
    raw_records=raw['run_records']
    if len(feature_records)!=EXPECTED_SEIZURES:raise ValueError(f'expected {EXPECTED_SEIZURES} selected feature seizure records, got {len(feature_records)}')
    folds=frozen_folds(folds_path,subjects)
    def index(records):
        result={}
        for rec in records:
            k=identity(rec)
            if k in result:raise ValueError(f'duplicate run key: {k}')
            result[k]=rec
        return result
    fx=index(feature_records);rx=index(raw_records)
    if set(fx)-set(rx):raise ValueError(f'missing {len(set(fx)-set(rx))} required feature/raw compound run keys')
    excluded=set()
    if approved_exclusion_private is not None:
        if amendment_json is None:raise ValueError('exclusion requires explicit protocol amendment')
        amendment=json.loads(Path(amendment_json).read_text(encoding='utf-8'))
        if amendment.get('authorized_exclusion_records')!=1 or amendment.get('authorized_exclusion_channels')!=104 or amendment.get('approval')!='USER_APPROVED':
            raise ValueError('unapproved exclusion amendment')
        excluded_rows=json.loads(Path(approved_exclusion_private).read_text(encoding='utf-8'))
        excluded={tuple(row['key']) for row in excluded_rows}
        if len(excluded)!=1 or not excluded.issubset(set(fx)&set(rx)):
            raise ValueError('expected exactly the one previously audited invalid run')
        hashes['exclusion_private']=digest(approved_exclusion_private)
        hashes['protocol_amendment']=digest(amendment_json)
    excluded_channels=excluded_windows=0
    grouped=defaultdict(list);raw_total=0;valid_total=0;nonfinite=0;onset_flags=Counter();label_support=Counter();center_counts=Counter();valid_channel_pairs=0
    for key,feat in fx.items():
        raw_rec=rx[key];fn=names(feat);rn=names(raw_rec)
        if set(fn)!=set(rn):raise ValueError(f'channel mismatch for {key}')
        sample=raw_rec.get('sample') or {}
        sig=np.asarray(sample.get('raw_waveform'))
        fs=float(sample.get('raw_temporal_sfreq',0) or 0)
        duration=float(sample.get('raw_temporal_duration_sec',0) or 0)
        if sig.ndim!=2 or sig.shape!=(len(rn),EXPECTED_RAW_SAMPLES) or fs!=250 or abs(duration-60)>1e-5:
            raise ValueError(f'waveform contract mismatch for {key}: {sig.shape}, {fs}, {duration}')
        if not np.isfinite(sig).all():raise ValueError(f'raw nonfinite: {key}')
        centers=np.asarray(value(feat,'window_relative_centers_sec',[]),dtype=float)
        if centers.ndim!=1 or len(centers)<2 or not np.isfinite(centers).all() or len(centers)!=len(set(np.round(centers,8))):
            raise ValueError(f'window-center invalid for {key}')
        valid_start=int(sample.get('raw_valid_start_sample',0))
        valid_samples=int(sample.get('raw_valid_samples',EXPECTED_RAW_SAMPLES))
        valid_end=valid_start+valid_samples
        if not 0<=valid_start<valid_end<=EXPECTED_RAW_SAMPLES:raise ValueError(f'raw valid interval invalid {key}')
        # Window-center geometry cross-check; crop times are relative to reported midpoint.
        if not (-30<=min(centers) and max(centers)<=30):raise ValueError(f'feature time centers exceed raw 60s duration {key}')
        status=str(sample.get('raw_temporal_onset_centered',sample.get('onset_centered','MISSING')))
        onset_flags[status]+=1
        # [-15,-5]s and [0,10]s relative to NOMINAL onset (raw center=30s).
        begins=[int((30-15)*fs),int(30*fs)]
        ends=[int((30-5)*fs),int((30+10)*fs)]
        invalid_crop=any(a<valid_start or b>valid_end for a,b in zip(begins,ends))
        if key in excluded:
            if not invalid_crop or len(fn)!=104:
                raise ValueError('approved run no longer matches audited invalid crop')
            if valid_start!=4750 or valid_end<10000 or max(0,valid_start-begins[0])!=1000:
                raise ValueError('approved run validity interval changed')
            excluded_channels+=len(fn);excluded_windows+=len(centers)*len(fn)
            continue
        if invalid_crop:
            raise ValueError(f'pre/ictal regions include padding for {key}')
        maps={name:i for i,name in enumerate(rn)}
        # Keep stored channel order until canonical patient alignment.
        slices=np.stack([sig[:,a:b] for a,b in zip(begins,ends)],axis=1).astype('float32',copy=False)
        if slices.shape!=(len(rn),2,2500):raise ValueError('extracted segment length mismatch')
        grouped[key[0]].append((key,fn,maps,slices))
        raw_total+=len(fn);valid_total+=len(centers)*len(fn)
    export={};nr_channels=0; patients_nonempty=0;missing_count=0
    patients_dir=output/'patients';patients_dir.mkdir(exist_ok=True)
    for sid in subjects:
        meta=pindex[sid]
        canonical=[normalize_channel_name(x) for x in meta['canonical_channels']]
        y=np.asarray(meta['labels'],dtype=np.float32)
        if len(canonical)!=len(y) or len(canonical)!=len(set(canonical)) or not np.isin(y,[0,1]).all():
            raise ValueError(f'canonical names / label contract invalid for {sid}')
        if len(grouped[sid])==0:raise ValueError(f'patient without runs: {sid}')
        if not (y==1).any() or not (y==0).any():label_support['single_class_patients']+=1
        else:label_support['both_class_patients']+=1
        nr_channels+=len(canonical)
        center=str(meta.get('source_center',meta.get('center',sid.split(':')[0]))).lower()
        center_counts[center]+=1
        index={name:i for i,name in enumerate(canonical)}
        bag=np.zeros((len(grouped[sid]),len(canonical),2,2500),dtype=np.float32)
        present=np.zeros((len(grouped[sid]),len(canonical)),dtype=np.bool_)
        for j,(key,fn,maps,slices) in enumerate(grouped[sid]):
            for name in fn:
                if name not in index:raise ValueError(f'run electrode not in patient canonical {sid}: {name}')
                c=index[name];bag[j,c]=slices[maps[name]];present[j,c]=True
        if not present.any():raise ValueError(f'no pairs for {sid}')
        if not present.any(axis=0).all():raise ValueError(f'patient-channel without any usable seizure pair: {sid}')
        missing_count+=int((~present).sum());valid_channel_pairs+=int(present.sum())
        patients_nonempty+=1
        anon=hashlib.sha256(sid.encode()).hexdigest()[:18]
        export[sid]={'file':f'patients/{anon}.npz','center':center,'n_seizures':len(grouped[sid]),'n_channels':len(canonical)}
        np.savez_compressed(patients_dir/f'{anon}.npz',pair=bag,present=present,labels_ez=y,channel_names=np.array(canonical,dtype='U80'))
    if nr_channels!=EXPECTED_CHANNELS:raise ValueError(f'expected {EXPECTED_CHANNELS} unique channels, got {nr_channels}')
    if EXPECTED_PATIENTS==80 and (raw_total+excluded_channels!=24995 or valid_total+excluded_windows!=1471965):
        raise ValueError(f'historical raw alignment footprint changed: run_channels={raw_total}, windows={valid_total}')
    if excluded and (excluded_channels!=104 or sum(len(grouped[s]) for s in subjects)!=EXPECTED_SEIZURES-len(excluded)):
        raise ValueError('amended recording count mismatch')
    if valid_channel_pairs!=raw_total:raise ValueError('unique run-channel count audit disagrees')
    expected_center_counts={'hup':36,'lzu':21,'multicenter':15}
    if EXPECTED_PATIENTS==80:
        def center_family(name):
            key=name.replace('_','').replace('-','').replace(' ','')
            if 'hup' in key:return 'hup'
            if 'lzu' in key or 'lanzhou' in key:return 'lzu'
            if 'multi' in key:return 'multicenter'
            if 'pediatric' in key or 'fudan' in key:return 'pediatric'
            return key
        mapped=Counter()
        for c,n in center_counts.items():mapped[center_family(c)]+=n
        if dict(mapped)!={'hup':36,'lzu':21,'multicenter':15,'pediatric':8}:
            raise ValueError(f'80-patient center-count audit failed: {dict(mapped)}')
    stats={
      'status':'PASS_SOURCE_CODE_ALIGNMENT_ONLY' if allow_source_only else 'BLOCKED_ONSET_PROVENANCE',
      'raw_cache_total_records':len(raw_records),'feature_cache_total_records':len(feature['run_records']),
      'raw_extra_unselected_records':len(set(rx)-set(fx)),
      'reused_historical_cohort':True,'n_patients':len(subjects),'n_seizures':len(feature_records)-len(excluded),
      'source_n_seizures':len(feature_records),'authorized_excluded_records':len(excluded),
      'authorized_excluded_run_channel_pairs':excluded_channels,'authorized_excluded_source_windows':excluded_windows,
      'n_unique_channels':nr_channels,'run_channel_incidences':raw_total,'source_window_incidences':valid_total,
      'valid_pre_ictal_pairs':valid_channel_pairs,'missing_seizure_channel_slots':missing_count,
      'fold_test_counts':{str(x['fold_idx']):len(x['test_subjects']) for x in folds},
      'center_counts':dict(center_counts),'class_support':dict(label_support),
      'source_flag_counts':dict(onset_flags),'independent_onset_provenance_verified_records':0,
      'onset_provenance_note':'At raw midpoint per legacy constructor only; original EDF-to-cache trace remains unverified',
      'data_sha256':hashes,
      'raw_preictal_bounds_sec':[-15,-5],'raw_early_ictal_bounds_sec':[0,10],
      'raw_250hz_60s':True, 'allow_source_only':bool(allow_source_only),
    }
    (output/'audit.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
    mf={'cohort':export,'folds':folds,'audit':'audit.json','protocol':'seed42_locked_80',
        'model_labels_in_npz':'EZ=1, NEZ=0','historical_nez_logits':'NEZ=1, EZ=0'}
    mf['patient_file_sha256']={sid:digest(output/meta['file']) for sid,meta in export.items()}
    (output/'manifest.json').write_text(json.dumps(mf,indent=2,ensure_ascii=False),encoding='utf-8')
    if not allow_source_only:
        raise RuntimeError('ONSET_PROVENANCE_NOT_INDEPENDENTLY_VERIFIED; no real-model run authorized. Add original EDF provenance audit or pass --allow-source-only for exploratory development only.')
    return stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--feature-cache',required=True);p.add_argument('--raw-cache',required=True)
    p.add_argument('--frozen-folds',required=True);p.add_argument('--output',required=True)
    p.add_argument('--expected-sha-file');p.add_argument('--allow-source-only',action='store_true')
    p.add_argument('--approved-exclusion-private');p.add_argument('--amendment-json')
    a=p.parse_args()
    print(json.dumps(extract(a.feature_cache,a.raw_cache,a.frozen_folds,a.output,
          allow_source_only=a.allow_source_only,expected_sha=a.expected_sha_file,
          approved_exclusion_private=a.approved_exclusion_private,amendment_json=a.amendment_json),indent=2))


if __name__=='__main__':main()
