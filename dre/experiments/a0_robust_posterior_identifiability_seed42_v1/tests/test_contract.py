"""Synthetic C0–C9, exact predecessor parity and real FIT-only score smoke."""
import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit

def main():
    p=argparse.ArgumentParser()
    for name in ['runtime','previous','source','protocol']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); code=Path(__file__).parents[1]/'code'; sys.path.insert(0,str(code))
    from posterior import fit_floor,normalize,iqr,fit_density,infer,decode,DensityInvalid,log_densities
    from common import sha,write_json,digest,cluster_bootstrap
    checks=[]
    def passed(name):checks.append({'name':name,'status':'PASS'})
    rng=np.random.default_rng(42); P=16; C=40
    patients=np.repeat([f'synth{i:02}' for i in range(P)],C)
    groups=np.repeat(np.arange(P)%4,C)
    y=np.tile(np.r_[np.zeros(12,int),np.ones(28,int)],P)
    logits=np.where(y==0,-2.,2.)+rng.normal(0,.7,len(y)); channels=np.tile(np.array([f'ch{i:02}' for i in range(C)]),P)
    ids=set(patients); floor=fit_floor(logits,patients,ids)
    den=fit_density(logits,y,patients,groups,ids,'R0')
    # C0: constant target, density from separate ordered FIT; all-zero FIT R1
    # has no positive IQR and is explicitly invalid rather than fabricated.
    q,m=infer(np.zeros(C),0,den,'R1',floor); pe,dc=decode(q,channels[:C],np.zeros(C))
    assert np.isfinite(q).all() and m['converged']
    assert np.array_equal(pe,decode(q,channels[:C],np.zeros(C))[0])
    try:fit_floor(np.zeros(len(y)),patients,ids)
    except ValueError:pass
    else:raise AssertionError('constant FIT needs invalid R1')
    passed('C0_constant_scores_and_no_positive_FIT_IQR')
    # C1: arbitrarily extreme finite tails cannot cause unbounded R1 moments.
    adversarial=np.r_[np.full(98,1e-12),-1e8,1e8]
    z,r=normalize(adversarial,'R1',1e-3); assert np.abs(z).max()==8 and np.mean(z*z)<=64
    assert np.array_equal(normalize(adversarial,'R0')[0],adversarial)
    passed('C1_tiny_IQR_bounded_no_R0_clipping')
    try:fit_density(-logits,y,patients,groups,ids,'R0')
    except DensityInvalid:pass
    else:raise AssertionError('reversed density accepted')
    passed('C2_reversed_global_rejected_no_flip')
    # C3: well-separated known synthetic mixture and exact posterior equations.
    data=np.r_[rng.normal(-2,.7,600),rng.normal(2,.7,1400)]
    q,m=infer(data,0,den,'R0'); assert abs(m['pi']-.3)<.03 and m['converged']
    le,ln=log_densities(data,den['sources'][0]); expected=expit(np.log(m['pi'])-np.log1p(-m['pi'])+le-ln)
    assert np.array_equal(q,expected)
    step=1e-6; pi=m['pi']; prior=den['sources'][0]['pi']
    def objective(v):return np.logaddexp(np.log(v)+le,np.log1p(-v)+ln).sum()+20*prior*np.log(v)+20*(1-prior)*np.log1p(-v)
    assert abs((objective(pi+step)-objective(pi-step))/(2*step))<1e-4
    passed('C3_synthetic_MAP_Bayes_equations_and_stationary_point')
    unknown,mu=infer(data,-1,den,'R0'); mapped={**den,'sources':[den['global']]*4}
    assert np.array_equal(unknown,infer(data,0,mapped,'R0')[0]) and mu['prior']==den['global']['pi']
    passed('C4_unknown_source_global_prior_density')
    x=np.array([-1e6,-10,-1,0,1,10,1e6]); ch=np.array(['g','f','e','d','c','b','a'])
    q,m=infer(x,0,den,'R1',.2); order=np.lexsort((ch,x))
    assert np.all(np.diff(q[order])<=0)
    assert np.array_equal(np.flatnonzero(decode(q,ch,x)[0]),np.sort(order[:decode(q,ch,x)[1]['k']]))
    passed('C5_clipped_and_saturated_ranking_ties')
    target=patients=='synth00'; train=~target
    ld=fit_density(logits[train],y[train],patients[train],groups[train],set(patients[train]),'R1',fit_floor(logits[train],patients[train],set(patients[train])))
    def query(unused_target_y):
        fq,m=infer(logits[target],0,ld,'R1',fit_floor(logits[train],patients[train],set(patients[train])))
        return fq,decode(fq,channels[target],logits[target])[0]
    aa=query(y[target]); bb=query(1-y[target])
    assert all(np.array_equal(v,w) for v,w in zip(aa,bb))
    changed=y.copy(); changed[target]=1-changed[target]
    changed_floor=fit_floor(logits[train],patients[train],set(patients[train]))
    changed_density=fit_density(logits[train],changed[train],patients[train],groups[train],set(patients[train]),'R1',changed_floor)
    assert digest(changed_density)==digest(ld)
    q_changed,_=infer(logits[target],0,changed_density,'R1',changed_floor)
    assert np.array_equal(q_changed,aa[0]) and np.array_equal(decode(q_changed,channels[target],logits[target])[0],aa[1])
    assert 'y' not in inspect.signature(infer).parameters and 'y' not in inspect.signature(fit_floor).parameters
    passed('C6_target_label_and_true_count_isolation')
    try:fit_density(logits,y,patients,groups,ids-{'synth00'},'R0')
    except AssertionError:pass
    else:raise AssertionError('wrong FIT membership')
    try:fit_floor(logits,patients,ids-{'synth00'})
    except AssertionError:pass
    else:raise AssertionError('wrong floor membership')
    for file in code.glob('*.py'):
        text=file.read_text(encoding='utf-8'); assert 'optimizer.step(' not in text and 'loss.backward(' not in text
    passed('C7_membership_gate_and_no_training_path')
    for n in range(1,9):
        v=np.sort(rng.uniform(0,1,n))[::-1]; cc=np.array([str(i) for i in range(n)])
        pred,details=decode(v,cc,-v); brute=[]
        for k in range(n+1):
            tp=v[:k].sum(); fp=k-tp; fn=v[k:].sum(); tn=n-k-fn
            ez=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn>0 else 0.
            nez=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn>0 else 0.
            brute.append((ez+nez)/2)
        assert np.allclose(details['all_k'],brute,rtol=0,atol=1e-14)
        assert details['k']==int(np.argmax(brute))
    assert decode(np.array([]),np.array([]))[1]['k']==0
    assert decode(np.array([.5]),np.array(['c']))[1]['k']==0
    passed('C8_all_k_brute_force_zero_denominator_and_smaller_k_ties')
    repeated=fit_density(logits,y,patients,groups,ids,'R0'); assert digest(den)==digest(repeated)
    assert np.array_equal(infer(data,0,den,'R0')[0],infer(data,0,repeated,'R0')[0])
    s=cluster_bootstrap(np.array(['a','a','b']),np.array([1.,3.,8.]),10000)
    assert np.array_equal(s,cluster_bootstrap(np.array(['a','a','b']),np.array([1.,3.,8.]),10000))
    assert set(np.unique(s))=={2.,4.,8.} # Preserve both appearances of a.
    passed('C9_deterministic_parameters_predictions_masks_cluster_repeats')
    # Exact old density/MAP implementation, with ONLY the input transform
    # substituted. Test-only loaded module, never used for new outcomes.
    spec=importlib.util.spec_from_file_location('historical_density',a.source/'a0_source_posterior_decoder_seed42_v1/code/posterior.py')
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    for rep,floor_value in [('R0',None),('R1',floor)]:
        old.normalize=lambda v,rr=rep,ff=floor_value:normalize(v,rr,ff)
        new=fit_density(logits,y,patients,groups,ids,rep,floor_value)
        original=old.fit_density(logits,y,patients,groups,ids)
        assert digest(new)==digest(original)
        for fixed in [False,True]:
            q1,m1=infer(data,0,new,rep,floor_value,fixed=fixed); q2,m2=old.infer(data,0,original,fixed)
            assert np.array_equal(q1,q2) and digest(m1)==digest(m2)
    passed('exact_predecessor_moments_hierarchy_prior_MAP_parity_R0_R1')
    # Real FIT-only smoke, never VAL scores/labels.
    bank=a.previous/'fold1/BANK_PRIVATE.pt'; manifest=json.loads((a.protocol.parent/'audit/FROZEN_INPUT_MANIFEST.json').read_text())
    assert sha(bank)==manifest['fold1\\BANK_PRIVATE.pt']
    d=torch.load(bank,map_location='cpu',weights_only=False); logits=d['a0_oof']; patients=d['patient'][d['train']]
    floor=fit_floor(logits,patients,set(patients))
    for pid in sorted(set(patients)):
        z,_=normalize(logits[patients==pid],'R1',floor); assert np.isfinite(z).all() and np.abs(z).max()<=8
    passed('real_FIT_only_label_blind_floor_and_bounded_score_smoke')
    a.runtime.mkdir(parents=True,exist_ok=True)
    write_json(a.runtime/'ENGINEERING_TESTS.json',{'status':'PASS','checks':checks,'protocol_sha256':sha(a.protocol),
        'code':{q.name:sha(q) for q in code.glob('*.py')},'test_source_sha256':sha(__file__),'real_smoke_VAL_used':False})
    print('ENGINEERING_PASS',len(checks),flush=True)

if __name__=='__main__':main()
