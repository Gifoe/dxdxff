"""Single label-blind 60s bank extraction, reused spectra across five frozen CNNs.

Preserve historical batch16 CNN and float32 EDF reduction before float64
segment logit summation. No labels or other-fold features are fed to any head.
"""
import argparse,json,sys,time
from collections import defaultdict
import numpy as np
import torch
from common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    cfg=lock();inputs=json.loads((ROOT/'INPUTS_PRIVATE.json').read_text());assert inputs['protocol_sha256']==sha(PROTOCOL)
    sys.path.insert(0,str(BASE/'code'))
    from modeling import load_official_module,TimeConvMorphology,FixedSegmentSpectrum
    from train_fold_r3 import groups,state_hash
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    source=load_official_module(OFFICIAL);spectrum=FixedSegmentSpectrum(source);models={};hashes={}
    for fold in range(1,6):
        ck=OLD/f'checkpoints/fold_{fold}/baseline/last.pt';assert sha(ck)==cfg['backbone_checkpoints'][str(fold)]
        state=torch.load(ck,map_location='cpu',weights_only=False);assert state['epoch']==30 and state['fold']==fold and state['variant']=='baseline'
        raw=TimeConvMorphology(source).cuda();raw.load_state_dict(state['raw_state'],strict=True);raw.eval();raw.requires_grad_(False)
        assert all(not m.training for m in raw.modules()) and all(not p.requires_grad for p in raw.parameters())
        models[fold]=raw;hashes[fold]=state_hash(raw)
    mf=manifest();completed=0;started=time.perf_counter()
    for row in mf.itertuples():
        patient=str(row.patient_name);paths=[Path(p) for p in inputs['records'][patient]]
        binding={'protocol':sha(PROTOCOL),'checkpoints':cfg['backbone_checkpoints'],'records':{str(p):inputs['original_hashes'][str(p)] for p in paths},'extract_source':sha(Path(__file__)),'modeling':cfg['backbone_source_hashes']['modeling.py']}
        out=[ROOT/f'embeddings/fold_{fold}/{key(patient)}.npz' for fold in range(1,6)]
        valid=[]
        for p in out:
            seal=p.with_suffix('.json')
            if p.exists() or seal.exists():
                assert p.exists() and seal.exists(),'Partial extraction cell; inspect, never overwrite'
                s=json.loads(seal.read_text());assert s['binding']==binding and s['sha256']==sha(p)
                valid.append(p)
        if len(valid)==5:completed+=1;continue
        # Input features only: clinical label arrays are intentionally not accessed.
        records=[]
        for p in paths:
            assert sha(p)==binding['records'][str(p)]
            with np.load(p,allow_pickle=False) as d:
                records.append((d['channel_names'].astype(str).tolist(),torch.from_numpy(d['waveforms']).cuda()))
        names=sorted({c for ns,_ in records for c in ns});sumh={f:np.zeros((len(names),32),np.float64) for f in models};suml={f:np.zeros(len(names),np.float64) for f in models}
        with torch.inference_mode():
            for segment in range(30):
                byfold={f:defaultdict(list) for f in models}
                for ns,waves in records:
                    for ids in groups(torch.arange(len(ns),device='cuda')):
                        image=spectrum(waves[ids,segment*2000:(segment+1)*2000],1000)
                        positions=ids.cpu().tolist()
                        for fold,raw in models.items():
                            emb,logit=raw.embed_and_logit(image)
                            assert emb.shape==(len(ids),32) and torch.isfinite(emb).all() and torch.isfinite(logit).all()
                            for j,i in enumerate(positions):byfold[fold][ns[i]].append((emb[j],logit[j]))
                for fold in models:
                    h=torch.stack([torch.stack([a for a,b in byfold[fold][c]]).mean(0) for c in names]).cpu().numpy()
                    l=torch.stack([torch.stack([b for a,b in byfold[fold][c]]).mean(0) for c in names]).cpu().numpy()
                    sumh[fold]+=h.astype(np.float64);suml[fold]+=l.astype(np.float64)
        counts=np.array([sum(c in ns for ns,_ in records) for c in names],np.int32)
        for fold,path in zip(range(1,6),out):
            h=(sumh[fold]/30).astype(np.float32);l=suml[fold]/30
            if path in valid:continue
            save_npz(path,channels=np.array(names),embedding=h,logits=l,edf_count=counts,segment_count=counts*30,checkpoint_sha256=np.array(cfg['backbone_checkpoints'][str(fold)]),outer_held=np.array(row.fold==fold))
            write_json(path.with_suffix('.json'),{'binding':binding,'sha256':sha(path),'channels':len(names),'segments_per_edf':30,'fold':fold,'backbone_frozen':True,'CNN_in_sample':row.fold!=fold})
        completed+=1;del records
        write_json(ROOT/'public/RUN_STATUS.json',{'status':'EXTRACTING','completed_patient_banks':completed,'target':124,'completed_fold_patient_cells':completed*5,'seconds':time.perf_counter()-started,'new_CNN_training':False})
        print('EXTRACTION_PATIENT',completed,'OF124','SECONDS',round(time.perf_counter()-started,2),flush=True)
        if args.limit and completed>=args.limit:break
    assert all(state_hash(raw)==hashes[f] for f,raw in models.items())
    if completed==124:write_json(ROOT/'public/EMBEDDING_EXTRACTION_AUDIT.json',{'status':'PASS','patient_fold_cells':620,'records_per_fold':137,'all_backbones_frozen_unchanged':True,'BN_eval_dropout_disabled':True,'labels_accessed_in_extractor':False,'embedding_dim':32,'segments_per_edf':30,'shared_spectrum_exact_transform':True,'torch':torch.__version__,'numpy':np.__version__})

if __name__=='__main__':main()
