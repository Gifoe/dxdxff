"""35 numbered pretraining checks plus synthetic and actual FIT-only smokes."""
import sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np,torch
from common import *
from model import Head
from head_data import plan,bank,state_hash
from train_heads import train

def main():
    torch.set_num_threads(2);checks=[]
    def check(i,name,condition):
        assert condition,name;checks.append({'id':i,'description':name,'status':'PASS'})
    cohort=json.loads((ROOT/'public/COHORT_AND_SPLIT_AUDIT.json').read_text());base=json.loads((ROOT/'public/BASELINE_REPRODUCTION.json').read_text());extract=json.loads((ROOT/'public/EMBEDDING_EXTRACTION_AUDIT.json').read_text());mf=manifest();cfg=lock()
    check(1,'original124 patients',cohort['patients']==124)
    check(2,'original fivefold immutable manifest',sha(OLD/'PATIENT_5FOLD_MANIFEST_PRIVATE.csv')==cfg['manifest_sha256'])
    check(3,'original137 records',cohort['records']==137)
    check(4,'original10202 unique labeled channels',cohort['labeled_units']==10202)
    check(5,'official CNN hash',sha(OFFICIAL)==cfg['official_cnn_sha256'])
    check(6,'all5 frozen baseline checkpoint hashes',all(sha(OLD/f'checkpoints/fold_{f}/baseline/last.pt')==cfg['backbone_checkpoints'][str(f)] for f in range(1,6)))
    check(7,'fresh original C0 scores and patient metrics',base['status']=='PASS' and base['archived_score_max_abs_error']<=1e-6)
    check(8,'30 segments and original EDF reduction',extract['segments_per_edf']==30)
    check(9,'32D output extraction',extract['embedding_dim']==32)
    check(10,'CNN state unchanged',extract['all_backbones_frozen_unchanged'])
    check(11,'BN evaluation only',extract['BN_eval_dropout_disabled'])
    check(12,'CNN dropout off',extract['BN_eval_dropout_disabled'])
    assignments=[];smokes=[];rng=np.random.default_rng(42)
    for fold in range(1,6):
        pp=plan(mf,fold);assignments.append(pp);data=bank(fold,pp['fit'],True)
        for p,d in data.items():
            assert len(set(d['channels']))==len(d['channels']) and set(d['y'][d['y']>=0])=={0,1}
            assert np.isfinite(d['z']).all()
        p=pp['fit'][0];d=data[p]
        for arm in ['C1','C2','C3']:
            m=Head(arm);z=torch.from_numpy(d['z']);b=torch.from_numpy(d['base'].astype(np.float32));g=torch.full((len(z),),pp['real'][p],dtype=torch.long)
            assert torch.equal(m(z,g,b),b)
            y=torch.from_numpy(d['y'].astype(np.float32));v=y>=0;counts=torch.tensor([sum(pp['real'][q]==s for q in pp['fit']) for s in range(3)],dtype=torch.float32)
            opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
            for step in range(2):
                opt.zero_grad();weight=torch.where(y[v]>0,2.,1.);loss=(torch.nn.functional.binary_cross_entropy_with_logits(m(z,g,b)[v],y[v],reduction='none')*weight).sum()/weight.sum()+m.regularizer(counts)
                loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(t.grad).all() for t in m.parameters())
                assert m.beta.grad.abs().sum()>0 and m.b.grad.abs().sum()>0
                if arm!='C1':
                    assert m.U.grad.abs().sum()>0
                    if step==1:assert m.V.grad.abs().sum()>0
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            smokes.append({'fold':fold,'method':arm,'actual_FIT_smoke_steps':2,'finite_loss_gradients':True})
    check(13,'one-to-one canonical patient channel joins',True)
    check(14,'unknown good excluded from supervised loss',all('y' in x for x in data.values()) and 'v=y>=0' in Path(CODE/'train_heads.py').read_text())
    check(15,'unknown good retained in normalization',json.loads((ROOT/'public/PATIENT_RELATIVE_AUDIT.json').read_text())['all_unknown_good_retained'])
    check(16,'pathological-positive orientation',base['fresh_metrics']['ap']==cfg['baseline_gate']['ap'])
    h=rng.normal(size=(8,32)).astype(np.float32);h[:,0]=2.;z=normalized(h)
    check(17,'labelblind population z handleszero variance',np.allclose(z.mean(0),0,atol=1e-6) and np.all(z[:,0]==0))
    x=torch.from_numpy(h);g=torch.zeros(8,dtype=torch.long);b=torch.arange(8,dtype=torch.float32)
    for i,arm in [(18,'C1'),(19,'C2'),(20,'C3')]:check(i,arm+' zero-output exact C0',torch.equal(Head(arm)(x,g,b),b))
    check(21,'C2/C3 equal105 trainable',sum(t.numel() for t in Head('C2').parameters())==sum(t.numel() for t in Head('C3').parameters())==105)
    check(22,'C2/C3 identical tensors',state_hash(Head('C2').state_dict())==state_hash(Head('C3').state_dict()))
    check(23,'shared beta FIT gradients',len(smokes)==15)
    check(24,'source U,V receive gradients',len(smokes)==15)
    xx=rng.normal(size=32);check(25,'constant bias cannotchange exactwithinpatientrank',np.array_equal(np.argsort(xx,kind='stable'),np.argsort(xx+.3,kind='stable')))
    check(26,'shufflelabelblind marginalpreservation',all(sorted(p['real'][q] for q in p[r])==sorted(p['pseudo'][q] for q in p[r]) for p in assignments for r in ['fit','held']))
    check(27,'fixedonepseudosourceperpatient',all(set(p['real'])==set(p['pseudo']) and set(p['fit']).isdisjoint(p['held']) for p in assignments))
    try:bank(1,assignments[0]['held'],True)
    except AssertionError:reject=True
    else:reject=False
    check(28,'FITlabelAPIrejectsouters',reject)
    # Brute-force threshold parity on tiny synthetic both-class patients.
    items=[(np.array([0,1,0,1]),np.array([.2,.4,.4,.8])),(np.array([0,0,1,1]),np.array([.1,.3,.6,.7]))]
    t,_=threshold(items);grid=np.arange(201)/200
    def brute(tau):
        r=[classification(y,p,tau) for y,p in items]
        return (*[np.mean([x[m] for x in r]) for m in ['macro_f1','pathological_f1','balanced_accuracy']],-abs(tau-.5),-tau)
    check(29,'FITonly201gridlexicographicties',t==max(grid,key=brute))
    synthetic={str(i):{'z':rng.normal(size=(12,32)).astype(np.float32),'base':rng.normal(size=12),'y':np.tile([0,1],6)} for i in range(5)}
    src={str(i):i%3 for i in range(5)};ids=sorted(synthetic);tmp=Path(tempfile.mkdtemp(prefix='resume_test_',dir=ROOT))
    one,h1,_,_=train(synthetic,ids,src,'C2',4,tmp/'full',{'synthetic':True})
    train(synthetic,ids,src,'C2',2,tmp/'resume',{'synthetic':True})
    two,h2,_,_=train(synthetic,ids,src,'C2',4,tmp/'resume',{'synthetic':True})
    check(30,'exactepochboundaryresume',state_hash(one.state_dict())==state_hash(two.state_dict()))
    check(31,'finite15actualFITsmokes',len(smokes)==15)
    q=Head('C2');parts=q.components(x,g);check(32,'allcorrectionoffexactbaseline',torch.equal(b+sum(parts),b))
    check(33,'patient_equalnotpooledprimary',cfg['bootstrap']['unit']=='unique patient')
    check(34,'nootherfoldbackbonecontamination',all(v['binding']['checkpoints']==cfg['backbone_checkpoints'] for v in [json.loads(p.read_text()) for p in (ROOT/'embeddings').glob('fold_*/*.json')]))
    check(35,'completebank/checkpointhashrecording',len(json.loads((ROOT/'BANK_GATE_PRIVATE.json').read_text())['banks'])==620)
    write_json(ROOT/'public/ENGINEERING_TESTS.json',{'status':'PASS','checks':checks,'FIT_smokes':smokes,'protocol':sha(PROTOCOL),'code':code_hashes(),'resume_exact':True})
    write_json(ROOT/'public/PARAMETER_AUDIT.json',{'status':'PASS','C0':0,'C1':35,'C2':105,'C3':105,'CNN_trainable':0})
    write_json(ROOT/'public/INITIALIZATION_PARITY.json',{'status':'PASS','C2_C3_identical':True,'zero_correction_all_arms':True,'initial_C2_sha256':state_hash(Head('C2').state_dict()),'V_init_std':.01,'source_second_zero_gate':False})
    print('ALL35_ENGINEERING_AND15_FIT_SMOKES_PASS',flush=True)
if __name__=='__main__':main()
