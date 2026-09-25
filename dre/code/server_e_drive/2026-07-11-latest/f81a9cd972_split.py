import random,hashlib
def split(keys,meta,fraction,seed):
 r=random.Random(int(seed));g={}
 for k in keys:g.setdefault((meta[k]['center'],meta[k]['outcome_success']),[]).append(k)
 va=[]
 for x in g.values():r.shuffle(x);va+=x[:max(1,round(len(x)*fraction))] if len(x)>2 else x[:1]
 tr=[k for k in keys if k not in va]
 return sorted(tr),sorted(va)
def h(x):return hashlib.sha256('|'.join(sorted(x)).encode()).hexdigest()
