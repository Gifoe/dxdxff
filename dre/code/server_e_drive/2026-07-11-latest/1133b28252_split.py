from __future__ import annotations
import hashlib,random
def make_split(keys,meta,fraction,seed):
 rng=random.Random(int(seed));groups={}
 for k in keys:groups.setdefault((meta[k]['center'],meta[k]['outcome_success']),[]).append(k)
 valid=[]
 for g,rows in groups.items():rng.shuffle(rows);n=max(1,round(len(rows)*fraction)) if len(rows)>2 else 1;valid+=rows[:n]
 # Keep both classes in development train whenever possible.
 train=[k for k in keys if k not in valid]
 for label in (0,1):
  if not any(meta[k]['outcome_success']==label for k in train):
   move=next((k for k in valid if meta[k]['outcome_success']==label),None)
   if move:valid.remove(move);train.append(move)
 return sorted(train),sorted(valid)
def hash_keys(keys):return hashlib.sha256('|'.join(sorted(keys)).encode()).hexdigest()
