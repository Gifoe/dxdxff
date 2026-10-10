"""Required38 checks; synthetic tests and actual FIT-only smokes before training."""
import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common import write_json,sha,metric,METRICS
from inspect_source import ROOT
from graph_metrics import node_features,seizure_summary,patient_summary,NAMES
from repository_graph import compute_thresholded_abs_pearson_adjacency,compute_graph_node_features
from prepare import permutation,graph_preprocess,PRIOR
from model import SourceA0
from runtime import seed_all,state_hash,select_threshold
from train_core import train

def main():
    a=argparse.ArgumentParser(); a.add_argument('--protocol',type=Path,required=True); args=a.parse_args()
    lock=json.loads(args.protocol.read_text()); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    checks=[]
    def check(i,text,condition):
        assert condition,(i,text); checks.append({'id':i,'description':text,'status':'PASS'})
    source=json.loads((ROOT/'public/GRAPH_SOURCE_AUDIT.json').read_text()); alignment=json.loads((ROOT/'public/GRAPH_ALIGNMENT_AUDIT.json').read_text())
    check(1,'original80 patients',source['patients']==80)
    check(2,'7635 original identities',alignment['canonical_channels']==7635 and alignment['strict_one_to_one_identity_join'])
    prep=json.loads((ROOT/'public/GRAPH_PREPROCESSING_AUDIT.json').read_text()); check(3,'five original folds',len(prep['folds'])==5)
    banks=[torch.load(ROOT/f'fold{f}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False) for f in range(1,6)]
    check(4,'no patient leakage',all(set(b['original']['patient'][b['original']['train']]).isdisjoint(b['original']['patient'][b['original']['val']]) for b in banks))
    check(5,'run/channel identity alignment',alignment['runs']==256 and alignment['no_dropped_or_duplicated_channels'])
    check(6,'actual window time alignment',alignment['all_window_centers_match_feature_source'])
    check(7,'construction has no clinical fields',source['no_clinical_fields_used'])
    check(8,'cache values distinguished from placeholders',source['zero_placeholder_graphs']==0 and source['mode']=='RAW_REBUILT')
    rng=np.random.default_rng(42); wave=rng.normal(size=(18,500)).astype('float32'); wave+=rng.normal(size=(1,500)).astype('float32')*.3
    adj=compute_thresholded_abs_pearson_adjacency(wave)
    check(9,'symmetric undirected adjacency',np.array_equal(adj,adj.T))
    check(10,'diagonal excluded',not np.any(np.diag(adj)))
    try: node_features(np.full((3,3),np.nan)); rejected=False
    except AssertionError: rejected=True
    check(11,'nonfinite adjacency rejected',rejected)
    observed=np.array([True,False,True,True]); small=compute_thresholded_abs_pearson_adjacency(wave[:4][observed]); check(12,'missing electrodes excluded',small.shape==(3,3))
    check(13,'exact fixed16 feature order',len(NAMES)==16 and len(set(NAMES))==16)
    got=node_features(adj); connected_wave=np.tile(np.arange(500,dtype=np.float32),(8,1))
    complete_adj=compute_thresholded_abs_pearson_adjacency(connected_wave)
    check(14,'seven exact source metrics on unique connected graph',np.max(np.abs(node_features(complete_adj)-compute_graph_node_features(connected_wave)))<3e-6)
    m=np.zeros((3,2,7)); m[:,:,1]=np.array([[0,.1],[.2,.4],[.6,.8]])
    s,r=seizure_summary(m,[0,2,4]); check(15,'temporal strength change per elapsed second',np.allclose(s[:,14],[.15,.175]))
    summaries=[s,s]; ranks=np.array([[0.,1.],[.25,.75]]); output=patient_summary(summaries,ranks)
    check(16,'cross seizure rank consistency',np.allclose(output[:,15],.75))
    check(17,'single seizure stability missing',np.isnan(patient_summary([s],[r])[:,15]).all())
    disconnected=np.zeros((4,4),np.float32); disconnected[0,1]=disconnected[1,0]=.5; disconnected[2,3]=disconnected[3,2]=.5
    missing=node_features(disconnected); empty=node_features(np.zeros((3,3),np.float32))
    check(18,'disconnected tied eigenspace missing; defined empty graph',np.isnan(missing[:,3]).all() and np.isfinite(empty).all() and np.allclose(empty[:,4],1/3))
    check(19,'no fake clinical onset verification',source['independent_onsets_verified']==0 and not alignment['independent_onset_verified'])
    check(20,'deterministic extraction',np.array_equal(got,node_features(adj),equal_nan=True))
    check(21,'original88 inputs bitwise',all(np.array_equal(b['inputs'][arm][:,:88],b['original']['x']) for b in banks for arm in ['G1','G2','G3']))
    x=rng.normal(size=(12,16)); p=np.repeat(['a','b','c'],4); x[0,0]=np.nan; gx,px=graph_preprocess(x,p,np.arange(8)); modified=x.copy(); modified[8:]+=1000
    _,py=graph_preprocess(modified,p,np.arange(8)); check(22,'FIT-only graph mean imputation/scaling',all(np.array_equal(px[k],py[k]) for k in ['imputer_statistics','mean','scale','var']))
    check(23,'input104',all(b['inputs']['G1'].shape[1]==104 for b in banks))
    seed_all(1051); models=[]
    for arm in range(3): seed_all(1051); models.append(SourceA0(1051))
    check(24,'all10789 parameters and D2=9221',all(sum(v.numel() for v in m.parameters())==10789 for m in models) and sum(v.numel() for v in SourceA0(dimension=88).parameters())==9221)
    check(25,'identical scratch initial tensors',len(set(state_hash(m.state_dict()) for m in models))==1)
    yz=torch.tensor([0.,1.]); l=torch.tensor([-2.,2.]); check(26,'NEZ-positive weighted BCE',torch.nn.functional.binary_cross_entropy_with_logits(l,yz)<torch.nn.functional.binary_cross_entropy_with_logits(-l,yz))
    check(27,'patient-equal updates exact original trainer', 'for position in np.random.default_rng(seed+epoch).permutation(len(train_groups))' in Path(__file__).parents[1].joinpath('code/train_core.py').read_text())
    mm=models[0]; counts=torch.tensor([3.,2.,1.,0.]); expected=mm.V.square().mean()+((mm.u.square().mean(1)+mm.b.square())*((counts.sum()/4)/counts.clamp(min=1))).mean()
    check(28,'exact source regularizer',torch.equal(mm.regularizer(counts),expected))
    tx=torch.randn(7,104,requires_grad=True); g=torch.tensor([0,1,2,3,0,1,2]); mm(tx,g).sum().backward()
    check(29,'nonzero graph-input gradient',tx.grad[:,88:].abs().sum()>0 and mm.network[1].weight.grad[:,88:].abs().sum()>0)
    sig=np.array([[1,0],[1,0],[1,1],[1,1],[0,1]],bool); order=permutation(sig,'synthetic')
    check(30,'deterministic within-patient label-free shuffle',np.array_equal(sig,sig[order]) and np.array_equal(order,permutation(sig,'synthetic')) and set(order)==set(range(5)))
    vectors=rng.normal(size=(5,16)); check(31,'complete graph vectors preserved',np.array_equal(np.sort(vectors,axis=0),np.sort(vectors[order],axis=0)))
    check(32,'G3 exact neutral zeros',all(not np.any(b['inputs']['G3'][:,88:]) for b in banks))
    y=np.array([0,1,0,1,1,0]); score=np.array([.3,.9,.51,.4,.7,.6]); pts=np.array(['a']*3+['b']*3)
    candidates=[]
    for tau in np.round(np.arange(0,1.0001,.005),3):
        r0=[metric(y[pts==q],score[pts==q],tau) for q in ['a','b']]; pooled=metric(y,score,tau)
        candidates.append((np.mean([z['macro_f1'] for z in r0]),np.mean([z['ez_f1'] for z in r0]),pooled['balanced_accuracy'],-abs(tau-.5),-tau))
    selected=select_threshold(y,score,pts); best=max(candidates)
    check(33,'exact original threshold ties and earlier checkpoint epoch',selected['threshold']==-best[-1] and '(sel[\'macro_f1\'],sel[\'ez_f1\'],-epoch)' in Path(__file__).parents[1].joinpath('code/train_core.py').read_text())
    # Synthetic optimizer/RNG end-to-end resume, not another scientific arm.
    synthetic_x=rng.normal(size=(20,104)).astype('float32'); synthetic_y=np.tile([0,1],10); synthetic_pid=np.repeat(['a','b','c','d','e'],4); synthetic_g=np.repeat([0,1,2,3,0],4)
    testdir=Path(tempfile.mkdtemp(prefix='engineering_resume_',dir=ROOT)); tr=np.arange(12); va=np.arange(12,20)
    full,_,_=train(synthetic_x,synthetic_y,synthetic_pid,synthetic_g,tr,va,testdir/'full',1051,'SYNTHETIC_NOT_PREDICTIVE',max_epochs=6)
    train(synthetic_x,synthetic_y,synthetic_pid,synthetic_g,tr,va,testdir/'resumed',1051,'SYNTHETIC_NOT_PREDICTIVE',max_epochs=6,stop_after=2)
    resumed,_,_=train(synthetic_x,synthetic_y,synthetic_pid,synthetic_g,tr,va,testdir/'resumed',1051,'SYNTHETIC_NOT_PREDICTIVE',max_epochs=6)
    saved1=torch.load(testdir/'full/LAST_PRIVATE.pt',weights_only=False,map_location='cpu'); saved2=torch.load(testdir/'resumed/LAST_PRIVATE.pt',weights_only=False,map_location='cpu')
    check(34,'exact epoch-boundary model/Adam/RNG resume',state_hash(saved1['model'])==state_hash(saved2['model']) and all(torch.equal(saved1['optimizer']['state'][k]['exp_avg'],saved2['optimizer']['state'][k]['exp_avg']) for k in saved1['optimizer']['state']))
    check(35,'finite forward/backward/optimizer',all(torch.isfinite(v).all() for v in full.parameters()) and torch.isfinite(tx.grad).all())
    check(36,'no outer TEST scores used',not alignment['outer_labels_used'] and not lock['selection']['outer_test'])
    check(37,'all VAL identities shared',all(all(b['inputs'][arm].shape[0]==len(b['original']['patient']) for arm in ['G1','G2','G3']) for b in banks))
    hashes={q.name:sha(q) for q in Path(__file__).parents[1].joinpath('code').glob('*.py')}; check(38,'source/extraction/training hashes recorded',all(q in hashes for q in ['extract.py','graph_metrics.py','train_core.py']))
    smokes=[]
    for fold,b in enumerate(banks,1):
        d=b['original']; tr=d['train']; p=sorted(set(d['patient'][tr]))[0]; ix=tr[d['patient'][tr]==p]
        for arm in ['G1','G2','G3']:
            seed_all(42+1009*fold); m=SourceA0(42+1009*fold).cuda(); opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
            x=torch.as_tensor(b['inputs'][arm][ix],device='cuda'); y=torch.as_tensor(d['y'][ix],dtype=torch.float32,device='cuda'); g=torch.as_tensor(d['g'][ix],device='cuda')
            counts=torch.tensor([len(set(d['patient'][tr][d['g'][tr]==k])) for k in range(4)],device='cuda',dtype=torch.float32)
            weight=torch.tensor(max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1),device='cuda')
            loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x,g),y,pos_weight=weight)+.001*m.regularizer(counts)
            loss.backward(); assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
            gradient=float(m.network[1].weight.grad[:,88:].abs().sum())
            if arm!='G3': assert gradient>0
            # LN104 can center neutral input to nonzero hidden values; disclose it.
            torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step(); smokes.append({'fold':fold,'arm':arm,'FIT_channels':len(ix),'loss_finite':True,'graph_weight_gradient_sum':gradient})
    write_json(ROOT/'public/ENGINEERING_TESTS.json',{'status':'PASS','checks':checks,'FIT_smokes':smokes,'protocol_sha256':sha(args.protocol),'code_sha256':hashes,
        'formal_checkpoints_modified':False,'synthetic_tests_not_predictive_evidence':True})
    print('ALL38_ENGINEERING_AND15_FIT_SMOKES_PASS',flush=True)

if __name__=='__main__': main()
