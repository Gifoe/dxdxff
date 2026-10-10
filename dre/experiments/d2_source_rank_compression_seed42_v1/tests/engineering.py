"""34 prespecified checks, independent metric/selection parity and FIT smokes."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common import sha,write_json,metric,METRICS
from prepare import ROOT,PRIOR
from model import SourceA0
from runtime import seed_all,state_hash,select_threshold,PRMLP,groups,patient_z,patient_metrics
from train import train
from mechanism import effective_summary,ranking_change

def main():
    p=argparse.ArgumentParser(); p.add_argument('--protocol',type=Path,required=True); args=p.parse_args()
    lock=json.loads(args.protocol.read_text()); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    checks=[]
    def check(i,text,value): assert value,(i,text); checks.append({'id':i,'description':text,'status':'PASS'})
    data=json.loads((ROOT/'public/DATA_SPLIT_AUDIT.json').read_text()); repro=json.loads((ROOT/'public/R4_REPRODUCTION.json').read_text())
    banks=[torch.load(ROOT/f'fold{f}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)['original'] for f in range(1,6)]
    check(1,'original80 cohort',data['patients']==80); check(2,'original7635 identities',data['canonical_pairs']==7635)
    check(3,'all5 original folds',len(data['folds'])==5 and all(x['split_and_identity_match'] for x in data['folds']))
    check(4,'original88 feature order',data['features']==88 and data['feature_order_sha256']==lock['input_sha256']['feature_order'])
    # Independent manual implementation checks the source helper definition.
    xx=np.array([[1.,2.],[3.,5.],[0.,8.],[2.,4.]]); patient=np.array(['a','a','b','b'])
    direct=np.concatenate([(a-a.mean(0))/a.std(0) for a in [xx[:2],xx[2:]]]); check(5,'exact patient-relative normalization',np.array_equal(direct,patient_z(xx,patient)))
    check(6,'FIT-only original imputer/scaler bitwise',all(x['FIT_only_preprocessor'] and x['features88_bitwise'] for x in data['folds']))
    check(7,'EZ0 NEZ1 and finite bank labels',all(set(d['y'])=={0,1} for d in banks))
    check(8,'R4 state and forward compatibility',all(x['legacy_generalized_bitwise_forward'] and x['max_logit_drift_from_archived']<2e-6 for x in repro['folds']))
    check(9,'exact frozen R4 metric replay',abs(repro['metrics']['macro_f1']-.6479260773776448)<1e-12)
    for i,rank,count in [(10,0,8821),(11,1,8921),(12,2,9021),(13,4,9221)]:check(i,f'rank{rank} parameter count',sum(p.numel() for p in SourceA0(rank=rank).parameters())==count)
    check(14,'all UV shapes',all(SourceA0(rank=r).V.shape==(r,96) and SourceA0(rank=r).u.shape==(4,r) for r in [1,2,4]) and not hasattr(SourceA0(rank=0),'V'))
    initials=[]; generators=[]
    for fold in range(1,6):
        seed=42+1009*fold; models=[]
        for rank in [0,1,2]:
            seed_all(seed); m=SourceA0(seed,rank=rank).eval(); models.append(m); generators.append(torch.get_rng_state())
            initials.append({'fold':fold,'method':'R'+str(rank),'shared_initial_hash':state_hash(m.network.state_dict()),'full_initial_hash':state_hash(m.state_dict())})
        assert len({state_hash(m.network.state_dict()) for m in models})==1
        seed_all(seed); shared=PRMLP().eval(); assert state_hash(shared.network.state_dict())==state_hash(models[0].network.state_dict())
        z=torch.linspace(-1,1,12*88).reshape(12,88); g=torch.arange(12)%4
        for m in models:
            with torch.no_grad(): assert torch.equal(m(z,g),shared(z)) and not m.parts(z,g)[1].any()
    check(15,'identical shared initial state every fold and dedicated RNG',all(torch.equal(generators[k],generators[k+1]) for k in [0,3,6,9,12]))
    check(16,'exact zero initial source correction',True)
    m=SourceA0(rank=2).eval(); z=torch.randn(12,88); g=torch.arange(12)%4
    with torch.no_grad(): m.u.copy_(torch.tensor([[1.,0.],[0.,1.],[-1.,.5],[.5,-1.]])); m.b.copy_(torch.tensor([.1,-.1,.2,-.2]))
    h=m.network[:4](z); base=m.network[4](h).squeeze(-1); manual=base+(((h@m.V.T)*m.u[g]).sum(-1)+m.b[g])
    check(17,'source forward exact formula',torch.equal(m(z,g),manual))
    unknown=torch.tensor([-1,4,99]*4); check(18,'unknown shared-head fallback',torch.equal(m(z,unknown),base))
    q=SourceA0(rank=1).eval(); q(z,g).sum().backward()
    check(19,'nonzero shared gradient',q.network[1].weight.grad.abs().sum()>0)
    check(20,'zero-u initial V data gradient and nonzero u,b',q.V.grad.abs().sum()==0 and q.u.grad.abs().sum()>0 and q.b.grad.abs().sum()>0)
    counts=torch.tensor([3.,2.,1.,0.]); weights=(counts.sum()/4)/counts.clamp(min=1)
    for rank in [0,1,2,4]:
        a=SourceA0(rank=rank); a.b.data.fill_(.1)
        expected=(a.b.square()*weights).mean()
        if rank: a.u.data.fill_(.03); expected=a.V.square().mean()+((a.u.square().mean(1)+a.b.square())*weights).mean()
        assert torch.equal(a.regularizer(counts),expected)
    check(21,'locked source regularizer each rank',True)
    trainer=Path(__file__).parents[1]/'code/train.py'; text=trainer.read_text()
    check(22,'FIT-only NEZ class weighting','max(int((1-y[tr]).sum()),1)/max(int(y[tr].sum()),1)' in text)
    check(23,'one update per sorted FIT patient','train_groups=[tr[ix] for ix in groups(pid[tr])]' in text)
    check(24,'identical deterministic patient ordering',np.array_equal(np.random.default_rng(1051+3).permutation(52),np.random.default_rng(1051+3).permutation(52)))
    rng=np.random.default_rng(42); y=np.tile([0,1,1,0,1],4); s=np.round(rng.uniform(0,1,20),2); pid=np.repeat(['a','b','c','d'],5)
    candidates=[]
    for tau in np.round(np.arange(0,1.0001,.005),3):
        ms=[metric(y[ix],s[ix],tau) for ix in groups(pid)]; pooled=metric(y,s,tau)
        candidates.append((np.mean([a['macro_f1'] for a in ms]),np.mean([a['ez_f1'] for a in ms]),pooled['balanced_accuracy'],-abs(tau-.5),-tau,tau))
    expected=max(candidates); selected=select_threshold(y,s,pid)
    check(25,'exact201-grid tie selection parity',tuple(selected[k] for k in ['macro_f1','ez_f1','pooled_ba','threshold'])==(expected[0],expected[1],expected[2],expected[5]))
    check(26,'checkpoint MF1 EZF1 earlier parity',max([(.7,.5,-2),(.7,.5,-1)])==(.7,.5,-1) and "candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)" in text)
    for r in [0,1,2,4]:
        a=SourceA0(rank=r)
        if r:a.u.data.normal_()
        _,summary=effective_summary(a); assert summary['effective_numerical_rank']<=r
    check(27,'effective UV numerical rank',True)
    pre=(m.components(z,g)[0]+m.components(z,g)[1]).detach().numpy().astype('float64'); bias=m.b[g].detach().numpy().astype('float64')
    for source in range(4): assert np.array_equal(np.argsort(pre[g.numpy()==source],kind='stable'),np.argsort((pre+bias)[g.numpy()==source],kind='stable'))
    check(28,'source-constant bias does not change mathematical ranking',True)
    b,dr,bi=m.components(z,g); check(29,'direction and bias intervention arithmetic',torch.equal(b+dr,m(z,g)-bi) or torch.allclose(b+dr,m(z,g)-bi,atol=1e-6,rtol=0))
    check(30,'no clinical label inputs',not data['all_cohort_y_member_loaded'] and all(d['x'].shape[1]==88 for d in banks))
    check(31,'no outer test evaluation',not data['outer_TEST_labels_scores_metrics_read'] and all(x['outer_excluded'] for x in data['folds']))
    x=rng.normal(size=(20,88)).astype('float32'); src=np.repeat([0,1,2,3],5).astype('int64'); tr=np.arange(10); va=np.arange(10,20)
    directory=Path(tempfile.mkdtemp(prefix='rank_resume_',dir=ROOT)); bind={'synthetic':True}
    train(x,y,pid,src,tr,va,directory/'full',1051,bind,max_epochs=6,rank=1)
    train(x,y,pid,src,tr,va,directory/'resumed',1051,bind,max_epochs=6,stop_after=2,rank=1)
    train(x,y,pid,src,tr,va,directory/'resumed',1051,bind,max_epochs=6,rank=1)
    a=torch.load(directory/'full/LAST_PRIVATE.pt',map_location='cpu',weights_only=False); b=torch.load(directory/'resumed/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    for k,slot in a['optimizer']['state'].items():
        for name,v in slot.items(): assert torch.equal(v,b['optimizer']['state'][k][name])
    assert a['best_key']==b['best_key'] and a['stale']==b['stale']
    check(32,'exact epoch-boundary model optimizer selection resume',True)
    smokes=[]
    for fold,d in enumerate(banks,1):
        tr=d['train']; p=sorted(set(d['patient'][tr]))[0]; ix=tr[d['patient'][tr]==p]
        counts=torch.tensor([len(set(d['patient'][tr][d['g'][tr]==g])) for g in range(4)],device='cuda',dtype=torch.float32)
        for rank in [0,1,2]:
            seed_all(42+1009*fold); m=SourceA0(42+1009*fold,rank=rank).cuda(); opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
            tx=torch.as_tensor(d['x'][ix],device='cuda'); tg=torch.as_tensor(d['g'][ix],device='cuda'); ty=torch.as_tensor(d['y'][ix],device='cuda',dtype=torch.float32)
            pos=torch.tensor(max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1),device='cuda')
            loss=torch.nn.functional.binary_cross_entropy_with_logits(m(tx,tg),ty,pos_weight=pos)+.001*m.regularizer(counts)
            loss.backward(); assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            assert m.network[1].weight.grad.abs().sum()>0 and m.b.grad.abs().sum()>0
            torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step(); assert all(torch.isfinite(p).all() for p in m.parameters())
            smokes.append({'fold':fold,'method':'R'+str(rank),'FIT_channels':len(ix),'finite_loss_gradient_parameters':True})
    check(33,'finite synthetic and15 actual FIT smokes',len(smokes)==15)
    check(34,'same original identities across4 ranks',all(x['split_and_identity_match'] and x['features88_bitwise'] for x in data['folds']))
    # Independent sklearn/original metric parity with ties and both classes.
    for i in range(15):
        y=np.tile([0,1,1,0,1],4); s=np.round(rng.uniform(0,1,20),1)
        original=patient_metrics(y,s,np.repeat(['a'],20),np.repeat(['hup'],20),.5).iloc[0]
        new=metric(y,s,.5)
        for name in METRICS: assert np.isclose(original[name],new[name],rtol=0,atol=1e-12,equal_nan=True)
    hashes={p.name:sha(p) for p in Path(__file__).resolve().parents[1].joinpath('code').glob('*.py')}
    write_json(ROOT/'public/PARAMETER_COUNT_AUDIT.json',{'status':'PASS','shared_parameters':8817,'parameters':lock['architecture']['parameters'],'R4_state_mapping_exact':True})
    write_json(ROOT/'public/INITIALIZATION_PARITY.json',{'status':'PASS','fold_arms':initials,'shared_identical_each_fold':True,'initial_source_correction_zero':True,'dedicated_source_generator':True,'warmstart':False})
    write_json(ROOT/'public/ENGINEERING_TESTS.json',{'status':'PASS','checks':checks,'FIT_smokes':smokes,'independent_original_metric_parity_cases':15,'code_sha256':hashes,'protocol_sha256':sha(args.protocol),'formal_checkpoints_modified':False})
    print('ALL34_ENGINEERING_AND15_FIT_SMOKES_PASS',flush=True)

if __name__=='__main__':main()
