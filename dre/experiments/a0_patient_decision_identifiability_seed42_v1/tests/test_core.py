"""Synthetic engineering tests; real checkpoint/data gates execute in run.py."""
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from core import metric,oracle,episode,support_indices,fit_probe,fit_pca,cluster_bootstrap,seed,write_json,sha

def fixture():
    rng=np.random.default_rng(42); z=rng.normal(size=(32,4)); score=expit(z[:,0]); y=(z[:,1]+z[:,0]>.2).astype(int)
    return y,score,z,np.asarray([f'C{i:03}' for i in range(32)])

def test_orientation_and_threshold():
    m=metric([0,1],[.2,.8],.5); assert m['TP']==m['TN']==1 and m['macro_f1']==1
    m=metric([0,1],[.5,.5],.5); assert m['FN']==1 and m['TN']==1

def test_metric_parity_sklearn_including_ties():
    from sklearn.metrics import average_precision_score,roc_auc_score,f1_score
    for y,s in [(np.array([0,1,0,1]),np.array([.2,.2,.8,.8])),(np.array([0,1,0,1]),np.array([.1,.3,.6,.9]))]:
        m=metric(y,s,.5)
        assert abs(m['ez_auprc']-average_precision_score(y==0,1-s))<1e-14
        assert abs(m['ez_auroc']-roc_auc_score(y==0,1-s))<1e-14
        assert abs(m['macro_f1']-f1_score(y,s>=.5,labels=[0,1],average='macro',zero_division=0))<1e-14

@pytest.mark.parametrize('scores',[[.2,.2,.8,.8],[0,0,1,1],[.5,.5,.5,.5]])
def test_exhaustive_ties_and_extremes(scores):
    y=np.array([0,1,0,1]); out=oracle(y,np.array(scores),.5)
    possibilities=[metric(y,scores,t)['macro_f1'] for t in np.r_[np.unique(scores),np.nextafter(max(scores),np.inf)]]
    assert out['oracle']['macro_f1']==max(possibilities)

def test_oracle_never_worse():
    y,s,_,_=fixture()
    for t in [0,.1,.37,.8,1]:assert oracle(y,s,t)['headroom']>=-1e-14

def test_topk_tie_separate():
    out=oracle(np.array([0,1,1]),np.array([.5,.5,.5]),.5); assert out['topk_cuts_tie']

def test_pca_fit_only():
    rng=np.random.default_rng(1); x=rng.normal(size=(30,88)); a,z=fit_pca(x,np.arange(20)); x[20:]+=1000; b,zz=fit_pca(x,np.arange(20))
    np.testing.assert_array_equal(a.components_,b.components_); np.testing.assert_array_equal(z[:20],zz[:20])

def test_support_label_blind_and_disjoint():
    y,s,z,c=fixture(); a,b=support_indices(abs(s-.5),c,8,'uncertainty',42)
    assert len(a)==8 and len(b)==24 and not set(a)&set(b)
    # Function accepts no labels, and episode partition remains equal under query-label changes.
    _,r=episode(y,s,z,c,.5,8,'uncertainty',42,(1,1)); changed=y.copy(); changed[b]=1-changed[b]
    _,r2=episode(changed,s,z,c,.5,8,'uncertainty',42,(1,1))
    np.testing.assert_array_equal(r['support'],r2['support'])
    for k in r['coeff']:np.testing.assert_array_equal(r['coeff'][k],r2['coeff'][k])

def test_all_methods_identical_queries_and_bias_rank():
    y,s,z,c=fixture(); out,r=episode(y,s,z,c,.5,8,'random',42,(.1,.1))
    for m in ['ez_auroc','ez_auprc','ez_mrr','top1_is_ez']:assert out['P0'][m]==out['P1'][m]
    assert all(o['channels']==len(r['query']) for o in out.values())

def test_finite_solver_and_gradient():
    y,s,z,_=fixture(); t,a=fit_probe(np.log(s/(1-s)),z,y,.1,True)
    assert np.isfinite(t).all() and a['converged'] and a['gradient_inf']<1e-9

def test_shrinkage_toward_baseline():
    y,s,z,_=fixture(); t,a=fit_probe(np.log(s/(1-s)),z,y,1e8,True); assert np.max(abs(t))<1e-7 and a['converged']

def test_constant_features():
    t,a=fit_probe(np.zeros(8),np.zeros((8,4)),np.arange(8)%2,.1,True); assert a['converged'] and np.isfinite(t).all()

@pytest.mark.parametrize('label',[0,1])
def test_one_class_regularized(label):
    t,a=fit_probe(np.zeros(8),np.ones((8,4)),np.full(8,label),.1,True)
    assert a['one_class'] and a['converged'] and np.isfinite(t).all()

def test_permutation_preserves_classes():
    y,s,z,c=fixture(); out,r=episode(y,s,z,c,.5,8,'random',42,(.1,.1)); support=r['support']
    perm=np.random.default_rng(seed(42,'permute')).permutation(y[support]); assert sorted(perm)==sorted(y[support])

def test_deterministic_repetitions_resume(tmp_path):
    y,s,z,c=fixture(); a,aa=episode(y,s,z,c,.5,8,'random',seed(1,'p','policy',2),(1,1)); b,bb=episode(y,s,z,c,.5,8,'random',seed(1,'p','policy',2),(1,1))
    assert a==b; path=tmp_path/'cell.json'; write_json(path,a); old=sha(path); write_json(path,b); assert sha(path)==old

def test_bootstrap_unique_patient_cluster():
    values=np.array([1.,3.,9.]); pid=np.array(['A','A','B']); boot=cluster_bootstrap(pid,values,200)[:,0]
    # With two unique patients, resampling AA gives mean2; AB gives13/3; BB gives9.
    assert set(np.round(boot,10))<={2.,round(13/3,10),9.}
    np.testing.assert_array_equal(boot,cluster_bootstrap(pid,values,200)[:,0])

def test_undefined_auc_not_half():
    assert np.isnan(metric([1,1],[.2,.7],.5)['ez_auroc'])
    assert np.isnan(metric([1,1],[.2,.7],.5)['ez_auprc'])

def test_predictive_input_exclusions():
    import inspect
    assert list(inspect.signature(fit_probe).parameters)==['offset','z','y','lam','direction']
    assert list(inspect.signature(support_indices).parameters)==['score','channels','budget','policy','rngseed']

def test_source_contract_and_no_outer_scoring():
    text=(Path(__file__).resolve().parents[1]/'code/run.py').read_text()
    assert "va=bank['val']" in text and "model(torch.from_numpy(xx[va]))" in text
    assert 'center_outcome_resection_used_as_predictors' in text
    clinical=(Path(__file__).resolve().parents[1]/'code/clinical.py').read_text()
    assert 'alternative_task2_function_used' in clinical and 'export_index_matches' in clinical
    assert 'read_raw' not in clinical and 'center_clinical_target(' not in clinical

def test_no_solver_query_api():
    import inspect
    assert 'query' not in inspect.signature(fit_probe).parameters

def test_threshold_grid_parity_full():
    y,s,_,_=fixture(); o=oracle(y,s,.465)
    assert o['oracle']['macro_f1']>=o['grid_oracle']['macro_f1']>=o['baseline']['macro_f1']-1e-14
