from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class ExpertPreprocessor:
    use_pca: bool=False
    variance: float=.85
    max_components: int=6
    imputer: object=None
    lower: np.ndarray=None
    upper: np.ndarray=None
    scaler: object=None
    pca: object=None
    fit_patient_keys: tuple[str,...]=()
    retained_indices: np.ndarray=None

    def fit(self,X,patient_keys=()):
        x=np.asarray(X,float); self.retained_indices=np.where(np.isfinite(x).any(axis=0))[0]
        if not self.retained_indices.size: raise ValueError("expert preprocessing has no observed training feature")
        x=x[:,self.retained_indices]; self.imputer=SimpleImputer(strategy="median").fit(x); x=self.imputer.transform(x)
        self.lower,self.upper=np.quantile(x,[.01,.99],axis=0); x=np.clip(x,self.lower,self.upper); self.scaler=StandardScaler().fit(x); x=self.scaler.transform(x)
        if self.use_pca:
            cap=max(1,min(self.max_components,x.shape[0],x.shape[1])); probe=PCA(n_components=cap).fit(x)
            n=max(1,min(cap,int(np.searchsorted(np.cumsum(probe.explained_variance_ratio_),self.variance)+1)))
            self.pca=PCA(n_components=n).fit(x)
        self.fit_patient_keys=tuple(map(str,patient_keys)); return self

    def transform(self,X):
        x=np.asarray(X,float)[:,self.retained_indices]; x=self.imputer.transform(x); x=self.scaler.transform(np.clip(x,self.lower,self.upper)); return self.pca.transform(x) if self.pca is not None else x


__all__=["ExpertPreprocessor"]
