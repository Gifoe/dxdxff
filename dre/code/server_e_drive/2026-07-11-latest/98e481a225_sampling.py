from __future__ import annotations
import math,numpy as np
def patient_batches(keys,labels,batch_size,seed):
    rng=np.random.default_rng(seed);positive=[k for k in keys if labels[k]==1];negative=[k for k in keys if labels[k]==0];rng.shuffle(positive);rng.shuffle(negative);result=[]
    while positive or negative:
        batch=[]
        while len(batch)<batch_size and (positive or negative):
            if positive and len(batch)<batch_size:batch.append(positive.pop())
            if negative and len(batch)<batch_size:batch.append(negative.pop())
        result.append(batch)
    flat=[k for b in result for k in b];assert len(flat)==len(set(flat))==len(keys);assert set(flat)==set(keys);assert len(result)==math.ceil(len(keys)/batch_size);return result
