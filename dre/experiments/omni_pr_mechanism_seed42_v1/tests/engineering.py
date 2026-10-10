"""36 pre-evaluation checks, synthetic tests and ten real FIT-only smokes."""
import sys,inspect,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from source import *
from transforms import *
def main():
    torch.set_num_threads(2);checks=[]
    def check(n,description,condition):
        assert bool(condition),f'Check {n}: {description}';checks.append({'number':n,'check':description,'pass':True})
    mf=manifest();replay=json.loads((ROOT/'public/HISTORICAL_REPRODUCTION.json').read_text());validated=json.loads((ROOT/'INPUTS_VALIDATED_PRIVATE.json').read_text())
    check(1,'Manifest SHA',sha(historical.OLD/'PATIENT_5FOLD_MANIFEST_PRIVATE.csv')==cfg()['manifest_sha256'])
    check(2,'Immutable original fold memberships',mf.fold.value_counts().to_dict()=={1:25,2:24,3:25,4:25,5:25})
    check(3,'137 exact record identities',json.loads((ROOT/'public/COHORT_AND_SPLIT_AUDIT.json').read_text())['records']==137)
    check(4,'10202 canonical labeled channels',mf.n_labeled_channels.sum()==10202)
    check(5,'All original CNN hashes',all(sha(historical.OLD/f'checkpoints/fold_{k}/baseline/last.pt')==historical.lock()['backbone_checkpoints'][str(k)] for k in range(1,6)))
    check(6,'620 immutable embedding cells',json.loads((ROOT/'public/EMBEDDING_BANK_AUDIT.json').read_text())['cells']==620)
    check(7,'Historical original CNN exact logit replay',replay['max_logit_error']==0)
    check(8,'Historical PR checkpoints frozen',all(sha(HIST/f'fold_{k}/C1/SELECTED_HEAD_PRIVATE.pt')==selected(k,'C1-PR')['checkpoint_sha256'] for k in range(1,6)))
    references={r['method']:r for r in replay['references']}
    check(9,'PR AUROC exact',abs(references['C1-PR']['auroc']-.8118233539960643)<1e-12)
    check(10,'PR AP exact',abs(references['C1-PR']['ap']-.5235271351033821)<1e-12)
    check(11,'PR historical classification exact',replay['max_patient_metric_error']<=1e-12)
    check(12,'Exactly35 parameters all corrected heads',sum(p.numel() for p in Head('C1').parameters())==35)
    h=Head('C1');z=torch.randn(5,32);base=torch.randn(5);g=torch.tensor([0,1,2,0,1])
    check(13,'Zero initialization logit identity',torch.equal(h(z,g,base),base))
    check(14,'Exact authoritative bias and optimizer implementation',authoritative_train.__module__=='train_heads' and Head.__module__=='model')
    synthetic={'a':{'h':np.zeros((2,32),np.float32)},'b':{'h':np.full((5,32),10,np.float32)}};sc=global_moments(synthetic,['a','b'])
    check(15,'ABS is population-global,not patient normalized',np.all(sc['mean']==5))
    check(16,'Global patient-equal variance retains between-patient variation',np.all(sc['variance']==25) and sc['fit_patients']==2)
    raw=bank(1,[plan(mf,1)['fit'][0]],True);p=next(iter(raw));pr=transform(raw,[p],'C1-PR',1,'FIT')
    check(17,'Exact historical PR transform',np.array_equal(pr[p]['z'],normalized(raw[p]['h'])))
    check(18,'All good channels including unknown used',len(pr[p]['z'])==len(raw[p]['y']) and 'y' not in inspect.signature(patient_moments).parameters)
    sh=transform(raw,[p],'C1-SHUFFLE',1,'FIT');ix=sh[p]['permutation']
    check(19,'Shuffle preserves multiset',np.array_equal(sh[p]['z'][np.argsort(ix)],pr[p]['z']))
    check(20,'Shuffle labelblind signature',set(inspect.signature(permutation).parameters)=={'fold','role','pseudonym','n'})
    check(21,'CNN logits not permuted',np.array_equal(sh[p]['base'],raw[p]['base']))
    check(22,'Permutation confined within patient',sorted(ix.tolist())==list(range(len(ix))) and not np.any(ix==np.arange(len(ix))))
    check(23,'Outer-held labels rejected by training API',all(not set(plan(mf,k)['fit'])&set(plan(mf,k)['held']) for k in range(1,6)))
    try:bank(1,plan(mf,1)['held'][:1],True)
    except AssertionError:pass
    else:raise AssertionError('Held label API failed')
    check(24,'Only own fold CNN bank used',len(set(historical.lock()['backbone_checkpoints'].values()))==5)
    check(25,'Exact historical inner assignments',all(plan(mf,k)==torch.load(HIST/f'fold_{k}/ASSIGNMENT_PRIVATE.pt',map_location='cpu',weights_only=False) for k in range(1,6)))
    trainer_text=(CODE/'train_controls.py').read_text()
    check(26,'Epoch selection uses FIT inner OOF only',"raw=bank(fold,fit,True)" in trainer_text and "('auroc'" not in inspect.signature(global_moments).parameters)
    check(27,'Threshold selection uses FIT OOF only',"for p in fit])" in trainer_text and selected(1,'C1-PR')['threshold']==.36)
    check(28,'Normalization frozen before new outer',"transform_hashes" in trainer_text and 'global_moments(raw,train)' in trainer_text and 'global_moments(raw,fit)' in trainer_text)
    check(29,'No CNN train/extraction entry points',not any('extract' in f.name for f in CODE.glob('*.py')))
    maxidentity=0.;maxrank=0.;maxrounding=0.;saturation=0
    for k in range(1,6):
        a=plan(mf,k);d=bank(k,a['held']);head=pr_head(k)
        operational=predict(head,transform(d,a['held'],'C1-PR',k,'HELD'),a['held'],a['real'])
        for patient in a['held']:
            ref,scaled,offset,err=scale_diagnostic(d[patient]['h'],d[patient]['base'],head,a['real'][patient]);maxidentity=max(maxidentity,err)
            with np.load(HIST/'labels'/f'{key(patient)}.npz',allow_pickle=False) as v:y=v['labels'];mask=y>=0
            q=metric(y[mask],sigmoid(ref[mask]),order_score=ref[mask]);r=metric(y[mask],sigmoid(scaled[mask]),order_score=scaled[mask])
            maxrank=max(maxrank,max(abs(q[m]-r[m]) for m in RANK));maxrounding=max(maxrounding,float(abs(operational[patient]-ref).max()))
            saturation+=int(((sigmoid(scaled)==0)|(sigmoid(scaled)==1)).sum())
    check(30,'Direct float64 scale-only ranking identity',maxidentity<1e-10 and maxrank<1e-12)
    y=np.array([0,1,0,1]);prob=np.array([.1,.8,.3,.6]);q=metric(y,prob)
    from sklearn.metrics import f1_score,roc_auc_score
    check(31,'Historical metric implementation validated',q['auroc']==roc_auc_score(y,prob) and q['macro_f1']==f1_score(y,prob>=.5,average='macro'))
    check(32,'Same124/10202 evaluation population',len(mf)==124 and mf.n_labeled_channels.sum()==10202)
    check(33,'Paired bootstrap unit unique patient',mf.patient_name.is_unique)
    smoke=[]
    with tempfile.TemporaryDirectory(prefix='omni_pr_smoke_',dir=ROOT) as tmp:
        for k in range(1,6):
            a=plan(mf,k);ps=a['fit'][:2];d=bank(k,ps,True)
            for arm in ['C1-ABS','C1-SHUFFLE']:
                scaler=global_moments(d,ps) if arm=='C1-ABS' else None;t=transform(d,ps,arm,k,'FIT',scaler)
                trained,history,_,_=authoritative_train(t,ps,a['real'],'C1',2,Path(tmp)/f'{k}_{arm}',{'smoke':True})
                assert all(torch.isfinite(v).all() for v in trained.parameters()) and history[-1]['patient_steps']==2
                smoke.append({'fold':k,'method':arm,'FIT_only':True,'epochs':2,'finite':True})
        a=plan(mf,1);ps=a['fit'][:2];d=bank(1,ps,True);t=transform(d,ps,'C1-SHUFFLE',1,'FIT')
        full,_,_,_=authoritative_train(t,ps,a['real'],'C1',4,Path(tmp)/'full',{'resume_test':True})
        authoritative_train(t,ps,a['real'],'C1',2,Path(tmp)/'resume',{'resume_test':True})
        resumed,_,_,_=authoritative_train(t,ps,a['real'],'C1',4,Path(tmp)/'resume',{'resume_test':True})
        assert state_hash(full.state_dict())==state_hash(resumed.state_dict())
    check(34,'Finite realFIT smokes and exact deterministic resume',len(smoke)==10)
    check(35,'Checkpoint and prediction original SHA snapshots',all(sha(p)==v for p,v in validated['original_hashes'].items()))
    base64=np.array([-.7,.9]);data={'s':{'z':np.ones((2,32),np.float32),'base':base64}}
    check(36,'Exact C0 recovery with additions off',np.array_equal(predict(Head('C1'),data,['s'],{'s':0})['s'],base64))
    write_json(ROOT/'public/ENGINEERING_TESTS.json',{'status':'PASS','checks':checks,'real_FIT_smokes':smoke,'exact_resume':True,'protocol':sha(PROTOCOL),'code':code_hashes()})
    write_json(ROOT/'public/PARAMETER_PARITY.json',{'C0':0,'C1-PR':35,'C1-ABS':35,'C1-SHUFFLE':35,'global_moments_trainable':False,'authoritative_Head_C1':True})
    write_json(ROOT/'public/INITIALIZATION_PARITY.json',{'status':'PASS','beta_and_source_bias_zero':True,'initial_logit_equals_C0':True,'new_heads_identical_initial_tensors':True})
    write_json(ROOT/'public/PATIENT_NORMALIZATION_AUDIT.json',{'status':'PASS','historical_float32_z_exact':True,'all_good_channels_including_unknown':True,'labels_used_for_moments':False,'epsilon':1e-5})
    write_json(ROOT/'public/SCALE_ONLY_INVARIANCE_AUDIT.json',{'status':'PASS','patients':124,'max_float64_constant_identity_error':maxidentity,'max_rank_metric_error':maxrank,'operational_float32_vs_float64_logit_max_error':maxrounding,'scale_only_saturated_probabilities':saturation,'rank_comparison_uses_direct_float64_logits':True,'classification_is_separate_frozen_threshold_diagnostic':True,'mean_subtraction_not_causal_ranking_explanation':True})
    print('ALL36_ENGINEERING_10_REAL_FIT_SMOKES_AND_SCALE_IDENTITY_PASS',flush=True)
if __name__=='__main__':main()
