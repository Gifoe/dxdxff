"""32 named synthetic/integrity checks followed by real FIT-only smoke."""
import argparse
import copy
import inspect
import json
import sys
from pathlib import Path
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True); a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).parents[1]/'code')); sys.path.insert(0,str(a.source/'pr_uncertainty_aware_supervision_seed42_v1/code'))
    from uas_core import PRMLP,seed_all,state_hash,train_model,select_threshold,oof_plan,validate_exclusion,torch_write
    from common import sha,write_json,metric,cluster_bootstrap
    from model import SourceA0,source_index
    from posterior import normalize,fit_density,infer,decode,calibration,log_densities
    from train import train
    torch.set_num_threads(2); rows=[]
    def check(number,name,condition):
        assert condition,name; rows.append({'number':number,'test':name,'pass':True})
    pub=a.runtime/'public'; rp=json.loads((pub/'A0_REPRODUCTION.json').read_text()); da=json.loads((pub/'DATA_AND_SPLIT_AUDIT.json').read_text())
    check(1,'A0 checkpoint replay',rp['status']=='PASS' and all(r['probability_drift']<=2e-7 for r in rp['folds']))
    lock=json.loads(a.protocol.read_text())
    check(2,'original feature/split hashes',da['feature_sha256']==lock['private_feature_sha256'] and da['split_sha256']==lock['split_sha256'])
    check(3,'canonical cohort and identities',da['patients']==80 and da['canonical_pairs']==7635 and da['validation_appearances']==65)
    check(4,'feature order',da['features']==88 and da['feature_order_sha256']==lock['feature_order_sha256'])
    check(5,'NEZ orientation',metric([0,1],[.1,.9],.5)['macro_f1']==1)
    seed_all(1051); base=PRMLP(); seed_all(1051); model=SourceA0(1051)
    check(6,'parameter count',sum(p.numel() for p in model.parameters())==9221)
    x=torch.randn(12,88); g=torch.tensor([0,1,2,3]*3); base.eval(); model.eval()
    check(7,'zero correction and shared init',all(torch.equal(base.state_dict()[k],model.state_dict()[k]) for k in base.state_dict()) and torch.equal(base(x),model(x,g)))
    model.train(); loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x,g),torch.arange(12)%2+torch.zeros(12)); loss.backward()
    check(8,'shared parameters gradient',all(p.grad is not None for p in model.network.parameters()))
    check(9,'associated source gradient',all(model.u.grad[i].abs().sum()>0 for i in range(4)))
    check(10,'source correction label blind',list(inspect.signature(model.forward).parameters)==['x','g'])
    model.eval(); perm=torch.tensor([2,3,0,1]); m2=copy.deepcopy(model)
    with torch.no_grad():
        model.u.normal_(); model.b.normal_(); m2.load_state_dict(model.state_dict()); m2.u[perm]=model.u; m2.b[perm]=model.b
    check(11,'source mapping permutation',torch.equal(model(x,g),m2(x,perm[g])) and np.array_equal(source_index(['HUP','LZU','Multicenter','Pediatric','unknown']),[0,1,2,3,-1]))
    yy=np.array([0,1]*6); pp=np.array(['a']*6+['b']*6)
    a0sel=select_threshold(yy,np.linspace(.05,.95,12),pp)
    candidates=[]
    for t in np.round(np.arange(0,1.0001,.005),3):
        mm=[metric(yy[pp==v],np.linspace(.05,.95,12)[pp==v],t) for v in ['a','b']]
        pooled=metric(yy,np.linspace(.05,.95,12),t)
        candidates.append((np.mean([r['macro_f1'] for r in mm]),np.mean([r['ez_f1'] for r in mm]),pooled['balanced_accuracy'],-abs(t-.5),-t,t))
    check(12,'original threshold/checkpoint tie semantics',a0sel['threshold']==max(candidates)[-1])
    oa=json.loads((pub/'A0_OOF_PROVENANCE.json').read_text())
    check(13,'reused A0 teacher provenance',len(oa['teachers'])==20 and all(r['query_train_overlap']==r['query_selection_overlap']==r['outer_overlap']==0 for r in oa['teachers']))
    bank=torch.load(a.runtime/'fold1/BANK_PRIVATE.pt',weights_only=False); fit=set(bank['patient'][bank['train']]); plans=oof_plan(fit,1)
    for plan in plans:validate_exclusion(plan['train'],plan['validation'],plan['query'],fit)
    check(14,'D2 OOF plan exclusion',plans==bank['plans'])
    check(15,'OOF once-only coverage',sorted(v for r in plans for v in r['query'])==sorted(fit) and np.isfinite(bank['a0_oof']).all())
    pid=np.repeat(np.array(['p'+str(i) for i in range(12)]),6); gg=np.repeat(np.arange(12)%4,6); y=np.tile([0,0,1,1,1,1],12)
    l=np.tile(np.array([-2.,-1.,0.,.4,.8,1.2]),12)
    den=fit_density(l,y,pid,gg,set(pid)); rejected=False
    try:fit_density(l,y,pid,gg,{'illegal'})
    except AssertionError:rejected=True
    check(16,'FIT-only density membership guard',rejected)
    check(17,'FIT-only patient equal prior',abs(den['global']['pi']-1/3)<1e-14)
    z,r=normalize([1.,2.,3.,4.]); z2,_=normalize([3.,5.,7.,9.])
    check(18,'unlabeled median IQR normalization',np.allclose(z,z2) and np.median(z)==0)
    le,ln=log_densities(np.array([-1e5,0,1e5]),den['global'])
    check(19,'stable Gaussian log domain',np.isfinite(le).all() and np.isfinite(ln).all())
    q,diag=infer(l[:6],0,den)
    check(20,'posterior normalization',np.isfinite(q).all() and ((q>=0)&(q<=1)).all())
    q2,d2=infer(l[:6],0,den)
    check(21,'deterministic converged MAP',np.array_equal(q,q2) and diag['pi']==d2['pi'] and diag['converged'])
    check(22,'no labels true count in inference',list(inspect.signature(infer).parameters)==['logits','source','density','fixed'] and list(inspect.signature(decode).parameters)==['q','channels','logits'])
    check(23,'posterior preserves NEZ ordering',np.all(np.diff(q)<=0) and np.array_equal(np.argsort(-q),np.argsort(l[:6])))
    pred,dec=decode(q,np.arange(6).astype(str),l[:6]); brute=[]
    for k in range(7):
        ez=np.arange(6)<k; tp=q[ez].sum(); fp=(1-q[ez]).sum(); fn=q[~ez].sum(); tn=(1-q[~ez]).sum()
        brute.append(.5*((2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)+(2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0)))
    check(24,'expected confusion ratios brute parity',np.allclose(brute,dec['all_k'],rtol=0,atol=1e-14))
    check(25,'enumerate all cardinalities',len(dec['all_k'])==7 and dec['k']==int(np.argmax(brute)))
    check(26,'smaller k tie',decode([.5],['a'])[1]['k']==0)
    check(27,'empty/single safe',decode([],[])[1]['k']==0 and np.isfinite(decode([1.],['a'])[1]['objective']))
    qc,dc=infer(np.ones(6),0,den); check(28,'constant patient',np.isfinite(qc).all() and dc['low_iqr'])
    sparse=fit_density(l[:18],y[:18],pid[:18],gg[:18],set(pid[:18])); qf,df=infer(l[:6],-1,sparse)
    check(29,'unknown and sparse source fallback',np.isfinite(qf).all() and sparse['sources'][3]['fallback'] and torch.equal(model.parts(x,torch.full((12,),-1))[1],torch.zeros(12)))
    check(30,'no outer labels/evaluation',not da['all_cohort_y_member_loaded'] and not da['outer_evaluation'] and set(bank['patient'][bank['train']]).isdisjoint(bank['patient'][bank['val']]))
    bs=cluster_bootstrap(['a','a','b'],[0.,2.,4.],100)
    check(31,'patient cluster keeps repeated appearances',set(np.round(bs[:,0],8))<={1.,2.,4.})
    smoke=a.runtime/'smoke'; xt=np.random.default_rng(42).normal(size=(18,88)).astype('float32'); yt=np.array([0,1]*9); pt=np.repeat(['a','b','c'],6); gt=np.repeat([0,1,2],6)
    tr=np.arange(12); va=np.arange(12,18)
    full,_,_=train(xt,yt,pt,gt,tr,va,smoke/'full',42,'synthetic','cpu',max_epochs=2)
    train(xt,yt,pt,gt,tr,va,smoke/'resume',42,'synthetic','cpu',max_epochs=2,stop_after=1)
    resumed,_,_=train(xt,yt,pt,gt,tr,va,smoke/'resume',42,'synthetic','cpu',max_epochs=2)
    aa=torch.load(smoke/'full/LAST_PRIVATE.pt',weights_only=False); bb=torch.load(smoke/'resume/LAST_PRIVATE.pt',weights_only=False)
    check(32,'exact interrupted resume',state_hash(aa['model'])==state_hash(bb['model']) and aa['history']==bb['history'])
    # Real FIT-only: original shared seed hash, zero-init, finite update; synthetic original optimizer parity.
    seed=1051; seed_all(seed); real=SourceA0(seed); sel=json.loads(Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\fold1\A0\SELECTION.json').read_text())
    shared={k:v for k,v in real.state_dict().items() if k.startswith('network.')}; assert state_hash(shared)==sel['initial_hash']
    pi=bank['train'][np.isin(bank['patient'][bank['train']],sorted(fit)[:3])]
    rx=bank['x'][pi]; ry=bank['y'][pi]; rp_=bank['patient'][pi]; rg=bank['g'][pi]
    real.train(); opt=torch.optim.AdamW(real.parameters(),lr=.001,weight_decay=.0001)
    for pat in sorted(set(rp_)):
        ii=np.flatnonzero(rp_==pat); opt.zero_grad(); loss=torch.nn.functional.binary_cross_entropy_with_logits(real(torch.from_numpy(rx[ii]),torch.from_numpy(rg[ii])),torch.as_tensor(ry[ii],dtype=torch.float32)); loss.backward(); opt.step(); assert torch.isfinite(loss)
    seed_all(42); init=copy.deepcopy(PRMLP().state_dict())
    old,ob,oh=train_model(xt,yt,pt,tr,va,smoke/'original',42,init,'parity','cpu',max_epochs=2)
    new,nb,nh=train(xt,yt,pt,np.full(18,-1),tr,va,smoke/'parity',42,'parity','cpu',max_epochs=2,penalty=0)
    for k in init:assert torch.equal(old.state_dict()[k],new.state_dict()[k]),'original optimizer parity'
    assert ob['threshold']==nb['threshold'] and ob['epoch']==nb['epoch']
    audit={'status':'PASS','tests':rows,'tests_passed':32,'real_FIT_smoke_patients':3,'shared_original_initial_hash_exact':True,
        'original_optimizer_parity_exact':True,'protocol_sha256':sha(a.protocol),'code_sha256':{p.name:sha(p) for p in (Path(__file__).parents[1]/'code').glob('*.py')},'test_source_sha256':sha(__file__)}
    write_json(pub/'DECODER_UNIT_TEST_AUDIT.json',audit); write_json(pub/'D2_PARAMETER_AUDIT.json',{'status':'PASS','A0':8817,'V':384,'u':16,'bias':4,'D2':9221,'source_initial_zero':True,'original_shared_initial_exact':True})
    print('ALL_32_AND_REAL_FIT_SMOKE_PASS',flush=True)

if __name__=='__main__':main()
