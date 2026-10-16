import numpy as np
from sklearn.linear_model import Ridge
from conditional_ridge import ConditionalRidge,prepare


def test_exact_block_solution_matches_full_and_sklearn_with_collinearity():
    rng=np.random.default_rng(42);x=rng.normal(size=(170,256));x[:,10]=x[:,3];x[:,195]=x[:,192];x[:,5]=0
    train=np.arange(110);x[112,221]=np.nan;xx,pre=prepare(x,train);target=rng.choice([-1.,1.],len(x));w=rng.uniform(.1,2,len(train))
    model=ConditionalRidge(xx,target,w,train);w=w*len(w)/w.sum()
    for groups in [[],[0],[0,1,2],[0,3,6,9,12,15,18,21,24],list(range(32))]:
        result=model.fit(groups);cols=result['columns'];ref=Ridge(alpha=1.,fit_intercept=True,solver='cholesky')
        ref.fit(xx[train][:,cols],target[train],sample_weight=w)
        np.testing.assert_allclose(result['score'],ref.predict(xx[:,cols]),rtol=0,atol=2e-10)
        a=xx[train][:,cols];mu=np.average(a,axis=0,weights=w);my=np.average(target[train],weights=w)
        coefficient=np.linalg.solve((a-mu).T@(w[:,None]*(a-mu))+np.eye(len(cols)),(a-mu).T@(w*(target[train]-my)))
        np.testing.assert_allclose(result['score'],xx[:,cols]@coefficient+my-mu@coefficient,rtol=0,atol=2e-10)


def test_held_values_cannot_fit_preprocessing_or_coefficients():
    rng=np.random.default_rng(9);a=rng.normal(size=(80,256));a[2,200]=np.nan;tr=np.arange(50);b=a.copy();b[50:]=10000
    x,p=prepare(a,tr);z,q=prepare(b,tr)
    for k in p:np.testing.assert_array_equal(p[k],q[k])
    y=rng.choice([-1.,1.],80);m=ConditionalRidge(x,y,np.ones(50),tr);n=ConditionalRidge(z,y,np.ones(50),tr)
    np.testing.assert_array_equal(m.fit([2,7])['coef'],n.fit([2,7])['coef'])
