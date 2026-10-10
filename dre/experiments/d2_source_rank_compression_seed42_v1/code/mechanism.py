"""Effective source matrix diagnostics; factors are not physiological objects."""
import numpy as np

def effective_summary(model):
    W=model.effective_matrix(); assert np.isfinite(W).all()
    singular=np.linalg.svd(W,compute_uv=False)
    tol=96*np.finfo(np.float64).eps*singular[0]
    rank=int((singular>tol).sum()) if singular[0]>0 else 0
    assert rank<=model.rank
    return W,{'declared_rank':model.rank,'effective_numerical_rank':rank,'sv1':float(singular[0]),'sv2':float(singular[1]),'sv3':float(singular[2]),'sv4':float(singular[3]),'rank_tolerance':float(tol),'frobenius_norm':float(np.linalg.norm(W))}

def ranking_change(reference,current):
    a=np.asarray(reference,dtype='float64'); b=np.asarray(current,dtype='float64'); n=len(a)
    upper=np.triu(np.ones((n,n),bool),1)
    one=np.sign(a[:,None]-a); two=np.sign(b[:,None]-b)
    r1=np.argsort(np.argsort(a,kind='stable'),kind='stable'); r2=np.argsort(np.argsort(b,kind='stable'),kind='stable')
    return {'pair_reversal_fraction':float((one*two<0)[upper].mean()) if n>1 else 0.,'changed_rank_fraction':float((r1!=r2).mean()),'absolute_rank_shift':float(np.abs(r1-r2).mean())}
