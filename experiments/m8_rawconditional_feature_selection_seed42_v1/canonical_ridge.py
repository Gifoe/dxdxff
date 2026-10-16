"""Full predefined ridge reference, no approximate block shortcut or new tuning."""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge


class CanonicalRidge:
    def __init__(self,x,target,weight,train,raw_columns=192,alpha=1.):
        self.x=np.asarray(x,float);self.target=np.asarray(target,float);self.train=np.asarray(train,int)
        self.nraw=raw_columns;self.alpha=alpha;w=np.asarray(weight,float)
        assert len(w)==len(train) and (w>=0).all() and w.sum()>0
        self.weights=w*len(w)/w.sum()

    def fit(self,groups):
        cols=np.r_[np.arange(self.nraw),self.nraw+np.array([2*g+j for g in sorted(groups) for j in [0,1]],int)]
        model=Ridge(alpha=self.alpha,fit_intercept=True,solver='cholesky')
        model.fit(self.x[self.train][:,cols],self.target[self.train],sample_weight=self.weights)
        score=model.predict(self.x[:,cols]);assert np.isfinite(score).all()
        return dict(score=score,probability=expit(score),coef=model.coef_,intercept=float(model.intercept_),columns=cols)
