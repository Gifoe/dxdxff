"""Exactly15 formal cells; hash-bound fresh initialization and epoch-safe resume."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import sha,write_json,metric
from inspect_source import ROOT
from model import SourceA0
from runtime import seed_all,state_hash,torch_write
from train_core import train

def main():
    a=argparse.ArgumentParser(); a.add_argument('--protocol',type=Path,required=True); args=a.parse_args()
    lock=json.loads(args.protocol.read_text()); assert lock['formal_new_runs']==15 and not lock['selection']['outer_test']
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    audit=json.loads((ROOT/'public/ENGINEERING_TESTS.json').read_text()); assert audit['status']=='PASS' and audit['protocol_sha256']==sha(args.protocol)
    hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    for name,h in audit['code_sha256'].items(): assert hashes[name]==h,'Audited source changed'
    source_hash=sha(Path(__file__).parent.parent/'SOURCE_AUDIT.md')
    write_json(ROOT/'public/RUN_STATUS.json',{'status':'TRAINING','registered_runs':15,'outer_test_accessed':False})
    for fold in range(1,6):
        path=ROOT/f'fold{fold}/BANK_PRIVATE.pt'; obj=torch.load(path,map_location='cpu',weights_only=False); d=obj['original']; seed=42+1009*fold
        assert obj['binding']['protocol']==sha(args.protocol)
        initial=[]
        for arm in ['G1','G2','G3']:
            seed_all(seed); start=SourceA0(seed); initial.append(state_hash(start.state_dict()))
            cell=ROOT/f'fold{fold}'/arm
            binding={'protocol':sha(args.protocol),'code':hashes,'bank':sha(path),'source_audit':source_hash,'fold':fold,'arm':arm}
            model,best,history=train(obj['inputs'][arm],d['y'],d['patient'],d['g'],d['train'],d['val'],cell,seed,binding)
            complete=json.loads((cell/'COMPLETE_PRIVATE.json').read_text()); assert complete['binding']['initial_hash']==initial[-1]
            with torch.no_grad():
                x=torch.as_tensor(obj['inputs'][arm],device='cuda'); g=torch.as_tensor(d['g'],device='cuda'); scores=torch.sigmoid(model(x,g)).cpu().numpy()
                if arm=='G1':
                    neutral=x.clone(); neutral[:,88:]=0; zero=torch.sigmoid(model(neutral,g)).cpu().numpy()
                else: zero=None
            rows=[]
            for p in sorted(set(d['patient'][d['val']])):
                ix=d['val'][d['patient'][d['val']]==p]; ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
                rows.append({'fold':fold,'patient':p,'center':str(d['center'][ix[0]]),'method':arm,**metric(d['y'][ix],scores[ix],best['threshold'])})
                if arm=='G1': rows.append({'fold':fold,'patient':p,'center':str(d['center'][ix[0]]),'method':'G1_GRAPH_ZERO',**metric(d['y'][ix],zero[ix],best['threshold'])})
            W=model.network[1].weight.detach().cpu().numpy()
            artifact={'binding':binding,'scores':scores,'zero_scores':zero,'rows':rows,'history':[{'fold':fold,'method':arm,**h} for h in history],
                'threshold':best['threshold'],'selected_epoch':best['epoch'],
                'graph_weight_norm_per_input':float(np.linalg.norm(W[:,88:])/np.sqrt(16)),
                'original88_weight_norm_per_input':float(np.linalg.norm(W[:,:88])/np.sqrt(88))}
            out=cell/'VALIDATION_PRIVATE.pt'; seal=cell/'FINAL_COMPLETE.json'
            if out.exists() and seal.exists():
                old=json.loads(seal.read_text()); assert old['validation_sha256']==sha(out) and old['binding']==binding
                frozen=torch.load(out,weights_only=False,map_location='cpu'); assert np.array_equal(frozen['scores'],scores) and frozen['threshold']==best['threshold']
            else:
                torch_write(out,artifact); write_json(seal,{'status':'PASS','binding':binding,'best_sha256':sha(cell/'BEST_PRIVATE.pt'),
                    'validation_sha256':sha(out),'selected_epoch':best['epoch'],'threshold':best['threshold'],'completed_epochs':len(history)})
            print('FORMAL_CELL_COMPLETE',fold,arm,len(history),flush=True)
        assert len(set(initial))==1,'Arm scratch initial states differ'
    write_json(ROOT/'public/RUN_STATUS.json',{'status':'ALL15_TRAIN_VALIDATION_COMPLETE','registered_runs':15,'outer_test_accessed':False})
    print('ALL15_TRAIN_VALIDATION_COMPLETE',flush=True)

if __name__=='__main__': main()
