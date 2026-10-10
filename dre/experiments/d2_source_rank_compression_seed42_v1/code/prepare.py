"""Hash-verified exact88 preprocessing and frozen R4; never load NPZ y."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import sha,write_json,metric,METRICS
from model import SourceA0,source_index,SOURCES
from runtime import prepare,torch_write

ROOT=Path(r'C:\d2_source_rank_compression_seed42_runtime')
PRIOR=Path(r'C:\a0_source_posterior_decoder_seed42_runtime')
EXPORT=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
SPLIT=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')
OLD_MODEL=Path(r'E:\DRE-nips\new-pipeline\7-11\a0_source_posterior_decoder_seed42_v1\code\model.py')

def main():
    a=argparse.ArgumentParser(); a.add_argument('--protocol',type=Path,required=True); args=a.parse_args()
    lock=json.loads(args.protocol.read_text()); manifest=json.loads((args.protocol.parent/'audit/FROZEN_INPUT_MANIFEST.json').read_text())
    assert sha(EXPORT)==lock['input_sha256']['feature_export'] and sha(SPLIT)==lock['input_sha256']['split']
    for rel,h in manifest.items(): assert sha(PRIOR/rel)==h,'Archived reference changed'
    torch.set_num_threads(2); pub=ROOT/'public'; pub.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('historical_d2_model',OLD_MODEL); historical=importlib.util.module_from_spec(spec); spec.loader.exec_module(historical)
    with np.load(EXPORT,allow_pickle=True) as f:
        raw={k:f[k] for k in ['x','patient','channel','center','features','split_fold','split_role','split_patient']}
    assert raw['x'].shape==(7635,88) and len(set(raw['patient']))==80 and len(set(zip(raw['patient'],raw['channel'])))==7635
    assert __import__('hashlib').sha256(json.dumps(raw['features'].tolist(),separators=(',',':')).encode()).hexdigest()==lock['input_sha256']['feature_order']
    control=[]; audits=[]; folds=[]; sourcepop=[]
    for c in SOURCES:
        mask=np.char.lower(raw['center'].astype(str))==c.lower(); sourcepop.append({'source':c,'patients':len(set(raw['patient'][mask])),'channels':int(mask.sum())})
    assert [x['patients'] for x in sourcepop]==[36,21,15,8]
    for fold in range(1,6):
        bank=PRIOR/f'fold{fold}/BANK_PRIVATE.pt'; ck=PRIOR/f'fold{fold}/D2/BEST_PRIVATE.pt'; frozen=PRIOR/f'fold{fold}/D2_FROZEN_PRIVATE.pt'
        d=torch.load(bank,map_location='cpu',weights_only=False); st=torch.load(ck,map_location='cpu',weights_only=False); old=torch.load(frozen,map_location='cpu',weights_only=False)
        sf=raw['split_fold']==fold; ids={r:set(raw['split_patient'][sf&(raw['split_role']==r)]) for r in ['fit','validation','test']}
        assert all(ids[a].isdisjoint(ids[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')])
        di=np.flatnonzero(np.isin(raw['patient'],list(ids['fit']|ids['validation'])))
        for k in ['patient','channel','center']: assert np.array_equal(raw[k][di],d[k])
        assert set(d['patient'][d['train']])==ids['fit'] and set(d['patient'][d['val']])==ids['validation']
        assert set(d['patient']).isdisjoint(ids['test']) and set(d['y'])<={0,1}
        assert np.array_equal(d['g'],source_index(d['center'])) and np.array_equal(d['features'],raw['features'])
        x,pre=prepare(raw['x'][di],d['patient'],d['train']); assert np.array_equal(x,d['x'])
        for key in ['mean','scale','var','imputer_statistics']: assert np.array_equal(pre[key],d['pre'][key])
        assert set(pre['fit_patient_ids'])==ids['fit']
        assert old['checkpoint_sha256']==sha(ck) and old['threshold']==st['threshold']
        legacy=historical.SourceA0().eval(); current=SourceA0(rank=4).eval()
        legacy.load_state_dict(st['model'],strict=True); current.load_state_dict(st['model'],strict=True)
        assert set(current.state_dict())==set(st['model'])
        with torch.no_grad():
            tx=torch.from_numpy(x); tg=torch.from_numpy(d['g']); baseline=legacy(tx,tg).numpy(); logit=current(tx,tg).numpy()
        assert np.array_equal(baseline,logit),'Historical R4 arithmetic mapping changed'
        drift=float(np.abs(logit[d['val']]-old['logits'][d['val']]).max()); assert drift<2e-6
        rows=[]
        for p in sorted(ids['validation']):
            ix=d['val'][d['patient'][d['val']]==p]; ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
            rows.append({'fold':fold,'patient':p,'center':str(d['center'][ix[0]]),'method':'R4',**metric(d['y'][ix],old['scores'][ix],old['threshold'])})
        control.extend(rows)
        binding={'protocol':sha(args.protocol),'original_bank':sha(bank),'original_checkpoint':sha(ck),'original_frozen_scores':sha(frozen),'original88_bitwise':True}
        torch_write(ROOT/f'fold{fold}/BANK_PRIVATE.pt',{'original':d,'binding':binding,'R4_checkpoint':str(ck),'R4_frozen':str(frozen)})
        audits.append({'fold':fold,'checkpoint_sha256':sha(ck),'bank_sha256':sha(bank),'max_logit_drift_from_archived':drift,'legacy_generalized_bitwise_forward':True,'threshold':old['threshold'],'parameters':9221})
        folds.append({'fold':fold,'FIT_patients':len(ids['fit']),'VAL_patients':len(ids['validation']),'FIT_channels':len(d['train']),'VAL_channels':len(d['val']),'source_patient_counts_FIT':[len(set(d['patient'][d['train']][d['g'][d['train']]==g])) for g in range(4)],'source_patient_counts_VAL':[len(set(d['patient'][d['val']][d['g'][d['val']]==g])) for g in range(4)],'FIT_only_preprocessor':True,'features88_bitwise':True,'split_and_identity_match':True,'outer_excluded':True})
        print('R4_REPLAY_PREPROCESSING_PASS',fold,flush=True)
    frame=pd.DataFrame(control); assert len(frame)==65 and frame.patient.nunique()==47 and frame.channels.sum()==6273
    expected={'macro_f1':.6479260773776448,'ez_f1':.44319946896815243,'ez_auprc':.5199357062538446,'ez_auroc':.7182207455375411}
    for m,v in expected.items(): assert abs(frame[m].mean()-v)<1e-12
    torch_write(ROOT/'CONTROL_METRICS_PRIVATE.pt',control)
    write_json(pub/'R4_REPRODUCTION.json',{'status':'PASS','folds':audits,'metrics':frame[METRICS].mean().to_dict(),'retrained':False,'fresh_forward_then_exact_archived_scores':True,'verified_historical_files':len(manifest),'historical_model_sha256':sha(OLD_MODEL)})
    write_json(pub/'DATA_SPLIT_AUDIT.json',{'status':'PASS','patients':80,'canonical_pairs':7635,'features':88,'validation_appearances':65,'validation_unique_ids':47,'validation_channel_appearances':6273,'source_categories':sourcepop,'folds':folds,'feature_order_sha256':lock['input_sha256']['feature_order'],'all_cohort_y_member_loaded':False,'outer_TEST_labels_scores_metrics_read':False,'labels_from_hash_verified_original_FIT_VAL_banks':True,'protocol_sha256':sha(args.protocol)})
    write_json(pub/'RUN_STATUS.json',{'status':'R4_REPLAY_AND_PREPROCESSING_COMPLETE','formal_new_runs':0,'outer_TEST_accessed':False})
    print('EXACT88_AND_R4_REPLAY_COMPLETE',flush=True)

if __name__=='__main__':main()
