"""Resume-safe extraction of frozen representations for TRAIN or TEST."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, time
from pathlib import Path
import h5py, mne, numpy as np, torch
import torch.nn.functional as F

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_npz(path, **kw):
    tmp=path.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**kw); tmp.replace(path)

def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8'); tmp.replace(path)

def load_source(path):
    s=importlib.util.spec_from_file_location('locked_cnn',path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def load_model(src,ckpt,device):
    mod=load_source(src); model=mod.NeuralCNN(1,1).to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=False)['model_state_dict'],strict=True)
    model.eval(); prep=mod.NeuralCNNPreProcessing(224,1000,[10,300],60000,30000,[10,300],0)
    return model,prep

def hist_quantiles(img):
    out=[]
    for a in img:
        hist=torch.histc(a,bins=256,min=0,max=255); c=torch.cumsum(hist,0); n=float(a.numel())
        vals=[]
        for q in (.1,.5,.9):
            idx=torch.searchsorted(c,torch.tensor(q*n,device=c.device)).clamp(0,255)
            vals.append((idx.float()+.5)*(255/256))
        out.append(torch.stack(vals))
    return torch.stack(out)

def stage_vectors(image,model):
    mean=image.mean((1,2,3)); std=image.std((1,2,3),unbiased=False); qs=hist_quantiles(image)
    freq=image.mean((1,3)); temporal=F.adaptive_avg_pool1d(image.mean((1,2)),224)
    r0=torch.cat([mean[:,None],std[:,None],qs,freq,temporal],1)
    x=model.feature_extractor[1](model.feature_extractor[0](image))
    r1=torch.cat([x.mean((2,3)),x.std((2,3),unbiased=False)],1)
    x=model.feature_extractor[3](model.feature_extractor[2](x))
    r2=torch.cat([x.mean((2,3)),x.std((2,3),unbiased=False)],1)
    c=model.cnn; y=c.maxpool(c.relu(c.bn1(c.conv1(x)))); y=c.layer1(y); y=c.layer2(y); y=c.layer3(y); y=c.layer4(y)
    r3=torch.flatten(c.avgpool(y),1); r4=c.fc(r3)
    q=model.bn(model.relu(model.fc(r4))); p16=model.bn1(model.relu1(model.fc1(q))); r5=model.fc_out(p16)
    return {'R0':r0,'R1':r1,'R2':r2,'R3':r3,'R4':r4,'R5':r5,'P16':p16}

def physical_signal(h5,index,header):
    if str(header['dimension']).lower() not in ('uv','µv','μv'):raise ValueError(f"unexpected physical unit: {header['dimension']}")
    d=np.asarray(h5[f'digital/ch{index:04d}'][:],dtype=np.float64)
    dmin,dmax=float(header['digital_min']),float(header['digital_max']); pmin,pmax=float(header['physical_min']),float(header['physical_max'])
    x=(d-dmin)*((pmax-pmin)/(dmax-dmin))+pmin; rate=float(header['sample_frequency'])
    x=mne.filter.notch_filter(x,Fs=rate,freqs=[60],notch_widths=2,n_jobs=1,verbose=False)
    if rate!=1000: x=mne.filter.resample(x,up=1000,down=rate,npad='auto',n_jobs=1,verbose=False)
    return np.asarray(x,dtype=np.float32)

def starts_for(n):
    k=(int(n)-2000)//60000
    if k<1: raise ValueError(f'signal too short: {n}')
    return 1000+np.arange(k,dtype=np.int64)*60000

def summarize_batches(waves,model,prep,device,batch,w,selected_positions=None,base_position=0):
    acc={k:[] for k in ['R0','R1','R2','R3','R4','R5','P16']}; sampled={k:[] for k in acc}
    with torch.inference_mode():
        for i in range(0,len(waves),batch):
            image=prep(torch.from_numpy(np.asarray(waves[i:i+batch],dtype=np.float32)).to(device)); vec=stage_vectors(image,model)
            for k,v in vec.items():
                a=v.detach().cpu().numpy().astype(np.float32); acc[k].append(a)
                if selected_positions:
                    for j,row in enumerate(a):
                        if base_position+i+j in selected_positions: sampled[k].append(row)
    arr={k:np.concatenate(v) for k,v in acc.items()}
    summary={k:np.concatenate([v.mean(0),v.std(0)]).astype(np.float32) for k,v in arr.items()}
    u=w/np.linalg.norm(w); par=arr['P16']@u; perp=arr['P16']-par[:,None]*u[None,:]
    summary['TASK_PAR']=np.array([par.mean(),par.std()],dtype=np.float32)
    summary['TASK_PERP']=np.concatenate([perp.mean(0),perp.std(0)]).astype(np.float32)
    return summary,{k:np.asarray(v,dtype=np.float32) for k,v in sampled.items()}

def sample_positions(edf,total):
    seed=int.from_bytes(hashlib.sha256((edf+'|42').encode()).digest()[:8],'little')
    return set(np.random.default_rng(seed).choice(total,size=min(20,total),replace=False).tolist())

def key(edf): return hashlib.sha256(edf.encode()).hexdigest()[:24]

def run_train(path,args,model,prep,w,device):
    with np.load(path,allow_pickle=False) as z:
        patient,edf=str(z['patient']),str(z['edf']); channels=z['channel_names'].astype(str); labels=z['labels'].astype(np.int8)
    h5p=args.signal_cache/Path(edf).with_suffix('.edf.h5')
    rows={k:[] for k in ['R0','R1','R2','R3','R4','R5','P16','TASK_PAR','TASK_PERP']}
    sample_rows={k:[] for k in ['R0','R1','R2','R3','R4','R5','P16']}; sample_ci=[]; sample_si=[]
    with h5py.File(h5p,'r') as h5:
        if str(h5.attrs['dataset_revision'])!=args.dataset_revision:raise RuntimeError('dataset revision mismatch')
        meta=json.loads(h5['metadata_json'][()].decode()); heads=meta['signal_headers']; idx={str(x['label']):i for i,x in enumerate(heads)}
        duration=float(meta['file_duration_seconds'])
        # Determine the shared segment count from the first selected channel, then
        # stream channels. Keeping every reconstructed signal resident can exceed
        # RAM on long EDFs and has no analytical benefit.
        probe=physical_signal(h5,idx[channels[0]],heads[idx[channels[0]]]); nseg=len(starts_for(len(probe))); del probe
        selected=sample_positions(edf,len(channels)*nseg)
        for ci,ch in enumerate(channels):
            sig=physical_signal(h5,idx[ch],heads[idx[ch]]); starts=starts_for(len(sig))
            if len(starts)!=nseg:raise RuntimeError('channel segment counts differ')
            waves=np.stack([sig[int(s):int(s)+60000] for s in starts]); summ,samp=summarize_batches(waves,model,prep,device,args.batch_size,w,selected,ci*nseg)
            for k in rows: rows[k].append(summ[k])
            local_selected=[j for j in range(nseg) if ci*nseg+j in selected]
            sample_ci.extend([ci]*len(local_selected)); sample_si.extend(local_selected)
            for k in sample_rows: sample_rows[k].extend(list(samp[k]))
    out=dict(patient=patient,edf=edf,channel_names=channels,labels=labels,duration_seconds=duration,num_segments=np.full(len(channels),nseg,np.int32))
    out.update({k:np.stack(v) for k,v in rows.items()})
    out['sample_channel_index']=np.asarray(sample_ci,np.int32); out['sample_segment_index']=np.asarray(sample_si,np.int32)
    out.update({'sample_'+k:np.asarray(v,np.float32) for k,v in sample_rows.items()})
    return out

def run_test(path,args,model,prep,w,device):
    with np.load(path,allow_pickle=False) as z:
        data=np.asarray(z['data'],np.float32); names=z['name'].astype(str); labs=z['labels'].astype(np.int8); patient=str(z['patient']); edf=str(z['edf_name']); ends=np.asarray(z['end_indices'])
    channels=np.array(list(dict.fromkeys(names.tolist()))); rows={k:[] for k in ['R0','R1','R2','R3','R4','R5','P16','TASK_PAR','TASK_PERP']}; labels=[]; counts=[]
    selected=sample_positions(edf,len(data)); sample_rows={k:[] for k in ['R0','R1','R2','R3','R4','R5','P16']}; sample_ci=[]; sample_si=[]
    for ci,ch in enumerate(channels):
        ix=np.flatnonzero(names==ch)
        local_selected={j for j,original in enumerate(ix) if int(original) in selected}
        summ,samp=summarize_batches(data[ix],model,prep,device,args.batch_size,w,local_selected,0)
        for k in rows: rows[k].append(summ[k])
        ordered=sorted(local_selected); sample_ci.extend([ci]*len(ordered)); sample_si.extend(ordered)
        for k in sample_rows: sample_rows[k].extend(list(samp[k]))
        labels.append(labs[ix[0]]); counts.append(len(ix))
    out=dict(patient=patient,edf=edf,channel_names=channels,labels=np.asarray(labels,np.int8),duration_seconds=float(ends.max()/1000),num_segments=np.asarray(counts,np.int32))
    out.update({k:np.stack(v) for k,v in rows.items()}); out['sample_channel_index']=np.asarray(sample_ci,np.int32); out['sample_segment_index']=np.asarray(sample_si,np.int32)
    out.update({'sample_'+k:np.asarray(v,np.float32) for k,v in sample_rows.items()}); return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=['TRAIN','TEST'],required=True); p.add_argument('--input',type=Path,required=True); p.add_argument('--signal-cache',type=Path)
    p.add_argument('--official-cnn',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--num-shards',type=int,default=1); p.add_argument('--shard-index',type=int,default=0); p.add_argument('--max-new-edfs',type=int,default=0); a=p.parse_args()
    # The HDF5 TRAIN path streams channels and fits batch 4 safely with three
    # workers on the locked GPU. TEST accepts the caller's smaller batch.
    if a.split=='TRAIN' and a.batch_size<4: a.batch_size=4
    lock=json.loads(a.protocol.read_text());
    if sha256(a.checkpoint)!=lock['frozen_checkpoint_sha256'] or sha256(a.official_cnn)!=lock['official_cnn_source_sha256']: raise RuntimeError('source/checkpoint hash mismatch')
    a.dataset_revision=lock['dataset_revision']
    files=sorted(a.input.glob('*.npz') if a.split=='TRAIN' else a.input.rglob('*.npz')); expected=296 if a.split=='TRAIN' else 237
    if len(files)!=expected: raise RuntimeError(f'expected {expected}, got {len(files)}')
    files=files[a.shard_index::a.num_shards]; a.output.mkdir(parents=True,exist_ok=True); device=torch.device('cuda'); model,prep=load_model(a.official_cnn,a.checkpoint,device)
    w=model.fc_out.weight.detach().cpu().numpy().reshape(-1).astype(np.float32); new=0; started=time.time()
    for i,path in enumerate(files,1):
        with np.load(path,allow_pickle=False) as z: edf=str(z['edf'] if a.split=='TRAIN' else z['edf_name'])
        dest=a.output/(key(edf)+'.npz'); marker=a.output/(key(edf)+'.json')
        if dest.exists() and marker.exists(): print(json.dumps({'i':i,'edf':edf,'reused':True}),flush=True); continue
        lockfile=a.output/(key(edf)+'.lock')
        try: fd=os.open(lockfile,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
        except FileExistsError: continue
        try:
            if dest.exists() and marker.exists(): continue
            tic=time.time(); out=run_train(path,a,model,prep,w,device) if a.split=='TRAIN' else run_test(path,a,model,prep,w,device)
            atomic_npz(dest,**out); atomic_json(marker,{'split':a.split,'edf':edf,'sha256':sha256(dest),'channels':len(out['channel_names']),'seconds':time.time()-tic})
            new+=1; print(json.dumps({'i':i,'of':len(files),'edf':edf,'channels':len(out['channel_names']),'seconds':time.time()-tic}),flush=True)
        finally:
            try:lockfile.unlink()
            except FileNotFoundError:pass
        if a.max_new_edfs and new>=a.max_new_edfs: break
    print(json.dumps({'split':a.split,'shard':a.shard_index,'new':new,'total_outputs':len(list(a.output.glob('*.npz'))),'seconds':time.time()-started}),flush=True)
if __name__=='__main__': main()
