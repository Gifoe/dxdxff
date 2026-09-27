"""Synthetic implementation checks; contains no patient data."""
import numpy as np
import torch

import teacher_core as tc
from meta_train import pack, unrolled_scores, outer_loss
from prototype_select import prototype_score


def main():
    rng=np.random.default_rng(42)
    z=rng.normal(size=(24,64))
    m0=rng.normal(size=24)
    y=rng.integers(0,2,size=24)
    y[0]=0;y[1]=1
    b,w=tc.fit_head(m0[:8],z[:8],y[:8],.1,1.)
    b0,w0=tc.afc.fit_residual(m0[:8],z[:8],y[:8],.1)
    assert abs(b-b0)<1e-6 and np.max(np.abs(w-w0))<1e-6
    ep={"z":z,"m0":m0,"y":y,"support":np.arange(8),"query":np.arange(8,24),
        "lambda_by_dim":{4:{"lambda_w":.1,"lambda_b":1.}}}
    p=rng.normal(size=(4,64))
    score,fallback=prototype_score(ep,p,.3)
    assert not fallback and score.shape==(16,)
    ep["y"][:8]=0
    score,fallback=prototype_score(ep,p,.3)
    assert fallback and np.array_equal(score,m0[8:24])
    ep["y"][:8]=[0,1,0,1,0,1,0,1]
    batch=pack([ep],4,torch.device("cpu"))
    param=torch.nn.Parameter(torch.tensor(p,dtype=torch.float32))
    scores,qy,mask=unrolled_scores(param,batch)
    loss=outer_loss(scores,qy,mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(param.grad).all()
    print("SYNTHETIC_TEACHER_CHECKS_PASS")


if __name__=="__main__":main()
