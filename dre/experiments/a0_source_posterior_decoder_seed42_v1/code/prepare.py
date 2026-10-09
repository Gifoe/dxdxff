"""Original contracts and deterministic legal A0 teachers, no NPZ label read."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import sha,digest,write_json,metric,METRICS
from model import source_index,SOURCES

def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True); a=p.parse_args()
    lock=json.loads(a.protocol.read_text()); core=a.source/'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py'
    assert sha(core)==lock['a0_core_sha256']; sys.path.insert(0,str(core.parent))
    from uas_core import PRMLP,prepare,oof_plan,validate_exclusion,torch_write
    torch.set_num_threads(2); pub=a.runtime/'public'; pub.mkdir(parents=True,exist_ok=True)
    prior=Path(r'C:\pr_uncertainty_aware_supervision_seed42_runtime')
    banks=Path(r'C:\a0_tabm_ensemble_seed42_runtime')
    export=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(export)==lock['private_feature_sha256']==sha(prior/'gate/FEATURES_PRIVATE.npz')
    split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')
    assert sha(split)==lock['split_sha256']
    with np.load(export,allow_pickle=True) as f:
        raw={k:f[k] for k in f.files if k!='y'} # Do not materialize all-cohort clinical labels.
    assert raw['x'].shape==(7635,88) and len(set(raw['patient']))==80
    assert len(set(zip(raw['patient'],raw['channel'])))==7635
    assert __import__('hashlib').sha256(json.dumps(raw['features'].tolist(),separators=(',',':')).encode()).hexdigest()==lock['feature_order_sha256']
    g=source_index(raw['center']); assert (g>=0).all()
    counts=[(len(set(raw['patient'][g==i])),int((g==i).sum())) for i in range(4)]
    assert counts==[(36,3667),(21,2141),(15,925),(8,902)]
    ta=json.loads((prior/'OOF_TEACHER_AUDIT.json').read_text()); assert ta['status']=='PASS'
    replay=[]; teachers=[]; d0rows=[]
    binding={'protocol':sha(a.protocol),'code':{q.name:sha(q) for q in Path(__file__).parent.glob('*.py')},'core':sha(core)}
    for fold in range(1,6):
        path=banks/f'fold{fold}/DEVELOPMENT_PRIVATE.pt'; assert sha(path)==lock['bank_sha256'][fold-1]
        d=torch.load(path,map_location='cpu',weights_only=False)
        s=raw['split_fold']==fold
        ids={r:set(raw['split_patient'][s&(raw['split_role']==r)]) for r in ['fit','validation','test']}
        assert all(ids[r].isdisjoint(ids[t]) for r,t in [('fit','validation'),('fit','test'),('validation','test')])
        di=np.flatnonzero(np.isin(raw['patient'],list(ids['fit']|ids['validation'])))
        for k in ['patient','channel','center']:assert np.array_equal(raw[k][di],d[k])
        assert set(d['patient'][d['train']])==ids['fit'] and set(d['patient'][d['val']])==ids['validation']
        assert set(d['patient']).isdisjoint(ids['test'])
        x,pre=prepare(raw['x'][di],d['patient'],d['train'])
        assert np.array_equal(x,d['x']) and np.array_equal(raw['features'],d['features'])
        for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pre[k],d['pre'][k])
        assert set(pre['fit_patient_ids'])==ids['fit']
        ck=Path(d['A0_checkpoint']); assert sha(ck)==lock['checkpoint_sha256'][fold-1]
        saved=torch.load(ck,map_location='cpu',weights_only=False)
        assert saved['epoch']==lock['epochs'][fold-1] and saved['threshold']==lock['thresholds'][fold-1]==d['A0_threshold']
        model=PRMLP().eval(); model.load_state_dict(saved['model'])
        with torch.no_grad():logits=model(torch.from_numpy(x)).numpy(); probabilities=torch.sigmoid(torch.from_numpy(logits)).numpy()
        ref=pd.read_csv(banks/f'fold{fold}/A0_CHANNEL_PRIVATE.csv')
        va=d['val']; new=pd.DataFrame({'patient':d['patient'][va],'channel':d['channel'][va],'y_nez':d['y'][va],'score':probabilities[va]})
        joined=new.merge(ref,on=['patient','channel','y_nez'],validate='one_to_one'); assert len(joined)==len(ref)==len(va)
        drift=float(np.abs(joined.score-joined.score_nez).max()); assert drift<=2e-7
        # Exact historical serialized prediction reuse after fresh checkpoint gate.
        old={(str(r.patient),str(r.channel)):float(r.score_nez) for r in ref.itertuples()}
        scores=np.full(len(x),np.nan); scores[va]=[old[(str(d['patient'][i]),str(d['channel'][i]))] for i in va]
        op=pd.read_csv(banks/f'fold{fold}/A0_PATIENT_PRIVATE.csv').set_index('patient')
        for pat in sorted(ids['validation']):
            ix=va[d['patient'][va]==pat]; ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
            m=metric(d['y'][ix],scores[ix],saved['threshold'])
            for k in METRICS:assert np.isclose(m[k],op.loc[pat,k],rtol=0,atol=1e-12,equal_nan=True)
            d0rows.append({'fold':fold,'patient':pat,'center':str(d['center'][ix[0]]),**m})
        tr=d['train']; fx=raw['x'][di][tr]; fy=d['y'][tr]; fp=d['patient'][tr]; fc=d['channel'][tr]
        of=np.full(len(tr),np.nan); seen=np.zeros(len(tr),int); plans=oof_plan(ids['fit'],fold)
        frozen=torch.load(prior/f'fold{fold}/OOF_FROZEN_PRIVATE.pt',map_location='cpu',weights_only=False)
        oldbind=json.loads(frozen['binding']); assert oldbind['data']==sha(export) and oldbind['uas_core.py']==sha(core)
        assert np.array_equal(fp,frozen['patient']) and np.array_equal(fc,frozen['channel']) and np.array_equal(fy,frozen['y_nez'])
        for plan in plans:
            k=plan['group']; cell=prior/f'fold{fold}/teacher{k}'
            rec=torch.load(cell/'OOF_PRIVATE.pt',map_location='cpu',weights_only=False)
            aud=next(r for r in ta['folds'] if r['fold']==fold and r['teacher_group']==k)
            assert sha(cell/'OOF_PRIVATE.pt')==aud['OOF_file_sha256']==frozen['teacher_hashes'][k]
            assert sha(cell/'BEST_PRIVATE.pt')==aud['checkpoint_sha256']==rec['source']['checkpoint']
            assert rec['source']['plan']==plan and rec['source']['binding']==frozen['binding']
            validate_exclusion(plan['train'],plan['validation'],plan['query'],ids['fit'])
            assert not set(plan['train']+plan['validation']+plan['query'])&(ids['validation']|ids['test'])
            ti=np.flatnonzero(np.isin(fp,plan['train'])); qi=np.flatnonzero(np.isin(fp,plan['query']))
            xx,pp=prepare(fx,fp,ti)
            assert np.array_equal(rec['patient'],fp[qi]) and np.array_equal(rec['channel'],fc[qi])
            for key in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pp[key],rec['preprocessor'][key])
            assert pp['fit_patient_ids']==rec['preprocessor']['fit_patient_ids']==plan['train']
            ts=torch.load(cell/'BEST_PRIVATE.pt',map_location='cpu',weights_only=False); assert ts['epoch']==aud['selected_epoch']
            tm=PRMLP().eval(); tm.load_state_dict(ts['model'])
            with torch.no_grad():of[qi]=tm(torch.from_numpy(xx[qi])).numpy()
            seen[qi]+=1
            teachers.append({'fold':fold,'family':'A0','group':k,'checkpoint_sha256':sha(cell/'BEST_PRIVATE.pt'),
                'preprocessor_hash':digest({q:pp[q].tolist() for q in ['mean','scale','var','imputer_statistics']}),
                'query_identity_hash':digest(list(zip(fp[qi],fc[qi]))),'query_channels':len(qi),
                'train_patients':len(plan['train']),'selection_patients':len(plan['validation']),'query_patients':len(plan['query']),
                'query_train_overlap':0,'query_selection_overlap':0,'outer_overlap':0,'deterministic_eval_not_MC':True})
        assert (seen==1).all() and np.isfinite(of).all()
        seal={**d,'binding':binding,'raw_x':raw['x'][di], 'g':source_index(d['center']),
              'a0_logits':logits,'a0_scores':scores,'a0_oof':of,'plans':plans}
        torch_write(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',seal)
        replay.append({'fold':fold,'probability_drift':drift,'checkpoint_sha256':sha(ck),'bank_sha256':sha(path),
            'epoch':saved['epoch'],'threshold':saved['threshold'],'identities_preprocessor_exact':True,
            'sealed_hash':sha(a.runtime/f'fold{fold}/BANK_PRIVATE.pt')})
        print('A0_REPLAY_OOF_PASS',fold,flush=True)
    df=pd.DataFrame(d0rows); assert len(df)==65 and df.patient.nunique()==47
    expected={'macro_f1':.6380797828499001,'ez_f1':.431198131752259,'ez_auprc':.518235100110259,'ez_auroc':.7106669304311658}
    for m,v in expected.items():assert abs(df[m].mean()-v)<1e-12
    torch_write(a.runtime/'D0_METRICS_PRIVATE.pt',d0rows)
    write_json(pub/'A0_REPRODUCTION.json',{'status':'PASS','folds':replay,'parameters':8817,'metrics':df[METRICS].mean().to_dict(),'retrained':False,'fresh_replay_then_exact_frozen_probabilities':True})
    write_json(pub/'DATA_AND_SPLIT_AUDIT.json',{'status':'PASS','patients':80,'canonical_pairs':7635,'features':88,
        'validation_appearances':65,'validation_unique_ids':47,'validation_channel_appearances':int(df.channels.sum()),
        'feature_sha256':sha(export),'split_sha256':sha(split),'feature_order_sha256':lock['feature_order_sha256'],
        'all_cohort_y_member_loaded':False,'outer_evaluation':False,'binding':binding})
    write_json(pub/'SOURCE_CATEGORY_AUDIT.json',{'status':'PASS','categories':[{'source':s,'patients':c[0],'channels':c[1]} for s,c in zip(SOURCES,counts)],'source_not_patient_identifier':True})
    write_json(pub/'A0_OOF_PROVENANCE.json',{'status':'PASS','teachers':teachers,'once_per_fit_channel':True})
    write_json(pub/'RUN_STATUS.json',{'status':'A0_REPLAY_AND_LEGAL_OOF_COMPLETE','binding':binding,'outer_evaluation':False})

if __name__=='__main__':main()
