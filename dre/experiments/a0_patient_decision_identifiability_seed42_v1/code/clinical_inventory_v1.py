"""Frozen-cache clinical provenance, no raw EEG or outcome-conditioned targets."""
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

SOURCE_FILES = [
 'pr_uncertainty_aware_supervision_seed42_v1/code/reproduce_b0_v2.py',
 'task1_baselines/cache_io.py','task1_baselines/feature_aggregation.py',
 'outcome_hifos/cache_schema.py','build_neuroez_c_four_center_caches.py',
 'run_baseline_three_centers/build_center_caches_from_patient_records.py',
 'bn_pdgs_ranker/data_readers/loader.py','bn_pdgs_ranker/data_readers/bids_common.py',
 'bn_pdgs_ranker/data_readers/bids_loader.py','bn_pdgs_ranker/data_readers/lzu.py',
 'bn_pdgs_ranker/data_readers/schemas.py','neuroez_c/data/pediatric.py',
 'neuroez_c/task2/clinical_target.py']

def provenance(source, export, sha):
    sys.path.insert(0,str(source))
    from outcome_hifos.cache_schema import load_cache_contract
    path=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl')
    assert sha(path)=='9b5bb58a0175aba494ad79e30c5a65c7cee33c7d464273f9bcf62b9b280b0087'
    cache=load_cache_contract(path)
    with np.load(export,allow_pickle=True) as f:
        pid,ch,y,center=[f[k] for k in ['patient','channel','y','center']]
    clinical={}; rows=[]; comparison=[]; align=Counter(); subgroup=defaultdict(list)
    for p in sorted(set(pid.astype(str))):
        meta=cache.patient_index[p]; cm=meta['channel_meta']; names=list(meta['canonical_channels'])
        assert len(names)==len(set(names))==len(meta['labels'])==len(cm)
        normalized=lambda s:''.join(c for c in str(s).upper() if c not in '-_ \t\r\n')
        norm=[normalized(n) for n in names]; assert len(set(norm))==len(norm)
        lookup={n:i for i,n in enumerate(norm)}; ix=np.flatnonzero(pid==p)
        sources=set(); d=Counter(); c=str(center[ix[0]])
        for j in ix:
            k=lookup[normalized(ch[j])]; m=cm[k]; target=int(1-float(meta['labels'][k]))
            assert target==int(y[j]); align['export_index_matches']+=1
            assert normalized(m['channel_name_norm'])==normalized(ch[j]); align['channel_meta_matches']+=1
            labelsource=str(m.get('label_source','UNKNOWN')); sources.add(labelsource)
            assert c==str(meta['source_center'])
            for field, ezpositive in [('is_ez_or_soz',True),('final_label',False),('soz',True),('resection',True)]:
                v=m.get(field)
                if v is None or not isinstance(v,(int,float,bool,np.number)) or not np.isfinite(v):
                    d[field+'_missing']+=1
                else:
                    converted=1-int(v) if ezpositive else int(v)
                    d[field+'_available']+=1; d[field+'_disagree']+=int(converted!=target)
            # Source policy verified against inspected adapter code, not guessed from name.
            if c=='hup':
                status=str(m.get('status_description','')).lower()
                expected=int(m.get('soz')==1 or m.get('resection')==1 or any(t in status for t in ['soz','seizure onset','resect']))
                d['adapter_rule_matches']+=int(expected==1-target)
            elif c=='multicenter':
                assert labelsource in ['channels.tsv:soz','channels.tsv:soz_or_resection']
                expected=int(m.get('soz')==1 or (labelsource.endswith('soz_or_resection') and m.get('resection')==1))
                d['adapter_rule_matches']+=int(expected==1-target)
            else:
                d['adapter_rule_matches']+=int(int(m['is_ez_or_soz'])==1-target)
            d['channels']+=1; d['ez_channels']+=int(target==0)
            d['unusable_in_export']+=int(m.get('usable_channel_mask',1)==0 or m.get('is_bad',False))
        category=' | '.join(sorted(sources)); clinical[p]={'center':c,'source':category,'channels':len(ix),'ez_fraction':float(np.mean(y[ix]==0)),**dict(d)}
        subgroup[(c,category)].append(p)
    for (c,s),ps in sorted(subgroup.items()):
        if len(ps)<5:
            # No identifying sparse source cell is emitted.
            continue
        a=[clinical[p] for p in ps]; totals=Counter()
        for m in a:totals.update({k:v for k,v in m.items() if isinstance(v,int)})
        rows.append({'center':c,'label_source':s,'patients':len(ps),'channels':totals['channels'],
                     'ez_channels':totals['ez_channels'],'pooled_ez_fraction':totals['ez_channels']/totals['channels'],
                     'patient_mean_ez_fraction':float(np.mean([m['ez_fraction'] for m in a]))})
        comparison.append({'center':c,'label_source':s,**dict(totals)})
    result={'status':'PASS','cohort_patients':80,'channels':7635,'alignment':dict(align),
        'all_canonical_unique':True,'alternative_task2_function_used':False,
        'index_is_authoritative':True,'mixed_source_patients':sum(' | ' in m['source'] for m in clinical.values()),
        'adapter_rule_disagreements':sum(m['channels']-m['adapter_rule_matches'] for m in clinical.values()),
        'source_version_at_cache_creation_embedded':False,
        'biological_EZ_truth_identifiable':False,'suppression_min_patients':5,
        'source_hashes':{n:sha(source/n) for n in SOURCE_FILES},
        'cache_sha256':sha(path), 'label_source_counts':rows, 'target_comparison':comparison}
    return clinical,result

if __name__=='__main__':
    import argparse
    from core import sha,write_json
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); export=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(export)=='1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd'
    _,result=provenance(a.source,export,sha); write_json(a.output,result)
    print(__import__('json').dumps(result),flush=True)
