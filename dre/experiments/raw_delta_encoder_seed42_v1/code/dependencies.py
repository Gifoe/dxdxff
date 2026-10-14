"""Load only original FIT/VAL banks and historical Development checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from io_utils import sha,load,read,write
from compact_delta_classifier import records_from_manifest

def context(source,runtime):
 root=Path(__file__).parents[1];source=Path(source);runtime=Path(runtime)
 gate=json.loads((runtime/'results/STAGE_A_GATE.json').read_text());assert gate['pass_gate']
 assert gate['binding']['protocol']==sha(root/'PROTOCOL_LOCK.json')
 rec=read(runtime/'RECONSTRUCTION_PRIVATE.pkl');assert rec['binding']==gate['binding']
 oc=source/'a0_source_posterior_decoder_seed42_v1/code';c=load('common',oc/'common.py')
 corepath=source/'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py';assert sha(corepath)=='1b1ccb0a361e6245cb2e6089347e147036aac90437c0992194f6db7bebf1503a'
 core=load('uas_core',corepath);original=load('model',oc/'model.py');trainer=load('rd_original_trainer',oc/'train.py')
 origroot=Path(r'C:\a0_source_posterior_decoder_seed42_runtime');original_hashes={p.name:sha(p) for p in oc.glob('*.py')}
 assert original_hashes==json.loads((origroot/'public/DECODER_UNIT_TEST_AUDIT.json').read_text())['code_sha256']
 amendment=source/'d2_ictal_outer_seed42_v1/PROTOCOL_AMENDMENT.json';assert sha(amendment)=='452a4f6957d18125a9ea1b32d827c5f50fbc81c22d9d41100173bddcd2e22503';bank_hashes=json.loads(amendment.read_text())['bank_sha256']  # Provenance metadata only, no Outer outputs.
 hp=source/'d2_ictal_ablation_seed42_v1/code/ablation_core.py';assert sha(hp)=='c567196692be1149795d93586fe25e2b474f071511bd2fbd0f561db82e00b7d5'
 helper=load('rd_ablation_core',hp);fac=helper.source_free_factory(original.SourceA0)
 mp=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime\public\FEATURE_GROUP_MANIFEST.json');assert sha(mp)=='f7869e994e8494397761e4efa0a2a2057c2a13541331e5933e94c6f7b82a065f'
 records=records_from_manifest(json.loads(mp.read_text()));meta=rec['meta'];indices=[q['index'] for q in records]
 assert np.array_equal(rec['E14'],rec['archived_E14'])
 assert [meta['features'][i] for i in indices]==[f'b0_delta__{f}__seizure_{s}' for f in json.loads((root/'PROTOCOL_LOCK.json').read_text())['features'] for s in ['mean','std']]
 # Read the historical DEVELOPMENT selection metadata only. No Outer seal/output.
 histroot=Path(r'D:\a0_compact_delta_ratio_only_seed42_v1_runtime');fr=json.loads((histroot/'DEVELOPMENT_FREEZE.json').read_text())
 banks=[];hist=[];hashes={'core':sha(corepath),'factory':sha(hp),'manifest':sha(mp),'original_code':original_hashes,'historical_development_freeze':sha(histroot/'DEVELOPMENT_FREEZE.json')}
 for fold in range(1,6):
  bp=origroot/f'fold{fold}/BANK_PRIVATE.pt';assert sha(bp)==bank_hashes[fold-1];d=torch.load(bp,map_location='cpu',weights_only=False)
  roles={r:set(meta['split_patient'][(meta['split_fold']==fold)&(meta['split_role']==r)]) for r in ['fit','validation','test']}
  assert all(roles[a].isdisjoint(roles[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')])
  assert set(d['patient'][d['train']])==roles['fit'] and set(d['patient'][d['val']])==roles['validation']
  assert set(d['patient'])==roles['fit']|roles['validation'] and not set(d['patient'])&roles['test']
  assert set(d['pre']['fit_patient_ids'])==roles['fit'] and np.array_equal(d['features'],meta['features'])
  lookup={tuple(k):i for i,k in enumerate(zip(meta['patient'],meta['channel']))};ix=np.array([lookup[tuple(k)] for k in zip(d['patient'].astype(str),d['channel'].astype(str))])
  assert len(ix)==len(set(ix));d=dict(d,archive_indices=ix,roles=roles)
  hroot=histroot/f'features/DELTA_ONLY/fold{fold}';hb=hroot/'BANK_PRIVATE.pt';cp=hroot/'TRAIN/BEST_PRIVATE.pt';r=next(q for q in fr['selections'] if q['arm']=='DELTA_ONLY' and q['fold']==fold)
  assert sha(hb)==r['bank_sha256'] and sha(cp)==r['checkpoint_sha256']
  hd=torch.load(hb,map_location='cpu',weights_only=False)
  for k in ['patient','channel','y','g','train','val']:assert np.array_equal(d[k],hd[k])
  for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(d['pre'][k],hd['pre'][k])
  hist.append((hd,cp,r));banks.append(d);hashes[f'legal_bank{fold}']=sha(bp);hashes[f'D0_checkpoint{fold}']=sha(cp);hashes[f'D0_bank{fold}']=sha(hb)
 torch.set_num_threads(2)
 return c,core,fac,trainer,records,rec,banks,hist,hashes
