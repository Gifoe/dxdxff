"""Exactly15 sealed formal runs; no R4 retraining or interim outcome choice."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import sha,write_json,metric
from prepare import ROOT
from model import SourceA0
from runtime import seed_all,state_hash,torch_write
from train import train

def main():
    a=argparse.ArgumentParser(); a.add_argument('--protocol',type=Path,required=True); args=a.parse_args()
    lock=json.loads(args.protocol.read_text()); assert lock['formal_new_runs']==15
    audit=json.loads((ROOT/'public/ENGINEERING_TESTS.json').read_text()); assert audit['status']=='PASS' and audit['protocol_sha256']==sha(args.protocol)
    code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    assert code==audit['code_sha256'],'Pretraining audited source changed'
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    write_json(ROOT/'public/RUN_STATUS.json',{'status':'TRAINING','registered_runs':15,'R4_retrained':False,'outer_TEST_accessed':False})
    for fold in range(1,6):
        path=ROOT/f'fold{fold}/BANK_PRIVATE.pt'; obj=torch.load(path,map_location='cpu',weights_only=False); d=obj['original']; seed=42+1009*fold
        assert obj['binding']['protocol']==sha(args.protocol); shared=[]
        for arm,rank in lock['new_arms'].items():
            seed_all(seed); start=SourceA0(seed,rank=rank); initial=state_hash(start.network.state_dict()); shared.append(initial)
            cell=ROOT/f'fold{fold}'/arm; binding={'protocol':sha(args.protocol),'code':code,'bank':sha(path),'source_audit':sha(args.protocol.parent/'SOURCE_AUDIT.md'),'fold':fold,'arm':arm,'rank':rank,'shared_initial_hash':initial}
            model,best,history=train(d['x'],d['y'],d['patient'],d['g'],d['train'],d['val'],cell,seed,binding,rank=rank)
            arrays={k:np.full(len(d['x']),np.nan,dtype=np.float32) for k in ['scores','logits','base','direction','bias','direction_off','bias_off']}
            with torch.no_grad():
                for role in ['train','val']:
                    ix=d[role]; x=torch.as_tensor(d['x'][ix],device='cuda'); g=torch.as_tensor(d['g'][ix],device='cuda')
                    base,direction,bias=model.components(x,g); logits=base+(direction+bias)
                    for key,value in [('scores',torch.sigmoid(logits)),('logits',logits),('base',base),('direction',direction),('bias',bias),('direction_off',torch.sigmoid(base+bias)),('bias_off',torch.sigmoid(base+direction))]:arrays[key][ix]=value.cpu().numpy()
            assert all(np.isfinite(x).all() for x in arrays.values())
            rows=[]
            for p in sorted(set(d['patient'][d['val']])):
                ix=d['val'][d['patient'][d['val']]==p]; ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
                rows.append({'fold':fold,'patient':p,'center':str(d['center'][ix[0]]),'method':arm,**metric(d['y'][ix],arrays['scores'][ix],best['threshold'])})
            assert abs(np.mean([r['macro_f1'] for r in rows])-best['macro_f1'])<1e-12
            artifact={'binding':binding,**arrays,'rows':rows,'threshold':best['threshold'],'selected_epoch':best['epoch'],'history':[{'fold':fold,'method':arm,**h} for h in history]}
            out=cell/'PREDICTIONS_PRIVATE.pt'; seal=cell/'FINAL_COMPLETE.json'
            if seal.exists():
                prior=json.loads(seal.read_text()); assert prior['binding']==binding and prior['prediction_sha256']==sha(out)
                frozen=torch.load(out,map_location='cpu',weights_only=False)
                assert np.array_equal(frozen['scores'],arrays['scores']) and frozen['threshold']==best['threshold']
            else:
                torch_write(out,artifact); write_json(seal,{'status':'PASS','binding':binding,'best_sha256':sha(cell/'BEST_PRIVATE.pt'),'prediction_sha256':sha(out),'epochs':len(history),'selected_epoch':best['epoch'],'threshold':best['threshold']})
            print('FORMAL_CELL_COMPLETE',fold,arm,len(history),flush=True)
        assert len(set(shared))==1
    write_json(ROOT/'public/RUN_STATUS.json',{'status':'ALL15_TRAIN_VALIDATION_COMPLETE','registered_runs':15,'R4_retrained':False,'outer_TEST_accessed':False})
    print('ALL15_TRAIN_VALIDATION_COMPLETE',flush=True)

if __name__=='__main__':main()
