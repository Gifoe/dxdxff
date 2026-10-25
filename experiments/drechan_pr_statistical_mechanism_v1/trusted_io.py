"""Private locations configured only by environment; public evidence uses aliases."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import numpy as np

HERE=Path(__file__).resolve().parent
SCIENCE=['PROTOCOL_LOCK.json','PROTOCOL_LOCK.md','mechanism_math.py','trusted_io.py','run_mechanism.py','statistics_report.py','verify_mechanism.py','tests/test_mechanism.py']
KEEP=[0,1,2,3,4,5,6,7,10,11,12,13]
HASHES=dict(reconstruction='73095eefbb3035200894560de3c3cfdb6259a4f28c343ecd3d7195be5b94f609',
 labels='1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd',
 feature_pr='3b9823db47382cc3d4d66d226b4041f05efbc65041095fccaf298f459dc8148f',
 unit_pr='2429906e260f02b7be71ab23220b2953dede963aaf79bef2a6eac923afd65f7b')
BANK_HASHES=['0096095bb69cd63fc338389c43ccb3f5ea5bfd24498c28e9441c80122b453e2a',
 'a6bfd47e045e3d104fe0e925a950609fd1ef77f81001fafa882f968aae816cf9',
 '9452cfd64a122ff9191d18a006ed0b0b5e2c77a4e4ccf2e4f95d1727ab110c65',
 'd024f099e449b5143b6493ce9cd1baae67ecc5172c71f710583b9db052ef234f',
 'bf282533bb91826fb0dcff6f8343c351b6986ce629b164c397d250dd7a44fe89']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(q):return hashlib.sha256(json.dumps(clean(q),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def jr(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read(p):return pickle.loads(Path(p).read_bytes())


def clean(q):
    if isinstance(q,np.ndarray):return q.tolist()
    if isinstance(q,np.generic):return q.item()
    if isinstance(q,dict):return {str(k):clean(v) for k,v in q.items()}
    if isinstance(q,(list,tuple,set)):return [clean(v) for v in q]
    return q


def write(p,q):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),('Preserve existing artifact',p.name)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(clean(q),indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)


def save(p,q):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p.name
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_bytes(pickle.dumps(q,protocol=5));tmp.replace(p)


def sources():return {n:sha(HERE/n) for n in SCIENCE}
def path(key):return Path(os.environ['PRM_'+key.upper()])


def reference(key):
    p=path(key);assert sha(p)==HASHES[key]
    spec=importlib.util.spec_from_file_location('prm_'+key,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def ranked(values,salt):return sorted(values,key=lambda p:hashlib.sha256(f'{salt}|{p}'.encode()).digest())


def take(ids,centers,n,salt):
    groups={c:[p for p in ids if centers[p]==c] for c in sorted(set(centers[p] for p in ids))}
    desired={c:len(v)*n/len(ids) for c,v in groups.items()};counts={c:int(np.floor(v)) for c,v in desired.items()}
    order=sorted(groups,key=lambda c:(-(desired[c]-counts[c]),hashlib.sha256(f'{salt}|{c}'.encode()).digest()))
    for c in order[:n-sum(counts.values())]:counts[c]+=1
    return set(p for c in groups for p in ranked(groups[c],salt)[:counts[c]])


def full80_assignment(meta):
    from scipy.optimize import linear_sum_assignment
    ids=sorted(set(meta['patient']));assert len(ids)==80
    centers={p:str(meta['center'][np.flatnonzero(meta['patient']==p)[0]]) for p in ids}
    quotas={'hup':[8,7,7,7,7],'lzu':[4,5,4,4,4],'multicenter':[3]*5,'pediatric':[1,1,2,2,2]}
    fits={f:set(meta['split_patient'][(meta['split_fold']==f)&(meta['split_role']=='fit')]) for f in range(1,6)}
    # One real-cohort patient is never original FIT. The all80 requirement is
    # resolved BEFORE outcomes by one explicit original-validation-only path.
    never_fit=set(ids)-set.union(*fits.values())
    assert len(never_fit)<=1
    validation={f:set(meta['split_patient'][(meta['split_fold']==f)&(meta['split_role']=='validation')]) for f in range(1,6)}
    slots=[(f,s) for f in range(1,6) for s in sorted(quotas) for _ in range(quotas[s][f-1])]
    cost=np.full((80,80),1e6)
    for i,p in enumerate(ids):
        for j,(f,s) in enumerate(slots):
            eligible=p in fits[f] or (p in never_fit and p in validation[f])
            if eligible and centers[p]==s:
                cost[i,j]=int.from_bytes(hashlib.sha256(f'42|80ASSESS|{p}|{f}'.encode()).digest()[:8],'big')/2**64
    rows,cols=linear_sum_assignment(cost);assert (cost[rows,cols]<1).all()
    assigned={f:set() for f in range(1,6)}
    for i,j in zip(rows,cols):assigned[slots[j][0]].add(ids[i])
    assert all(len(a)==16 for a in assigned.values()) and set.union(*assigned.values())==set(ids)
    for f,a in assigned.items():
        assert a<=fits[f]|(never_fit&validation[f])
        assert all(sum(centers[p]==s for p in a)==quotas[s][f-1] for s in quotas)
    return assigned


def load_inputs():
    assert sha(path('reconstruction'))==HASHES['reconstruction'] and sha(path('labels'))==HASHES['labels']
    rec=read(path('reconstruction'));meta=rec['meta'];raw=rec['R14']
    with np.load(path('labels'),allow_pickle=True) as z:
        for k in ['patient','channel','center']:assert np.array_equal(z[k].astype(str),meta[k])
        y=(z['y']==0).astype(int)
    assert raw.shape==(7635,14) and np.isfinite(raw).all() and y.sum()==1743
    assert len(set(meta['patient']))==80 and sum(len(q['records']) for q in rec['ledgers'])==256
    assert sum(int(z['mask'].sum()) for q in rec['ledgers'] for z in q['records'])==1471965
    banks=[];assessed=[];original_assessed=[];assignments=full80_assignment(meta)
    for f in range(1,6):
        files=list(path('internal_banks').glob(f'*/fold{f}.pkl'));assert len(files)==1
        p=files[0];assert sha(p)==BANK_HASHES[f-1]==jr(p.with_suffix('.sha256.json'))['sha256']
        q=read(p);d=q['bank'];ix=d['archive_indices'];pid=meta['patient'][ix]
        for k,new in [('patient',pid),('channel',meta['channel'][ix]),('center',meta['center'][ix])]:assert np.array_equal(d[k],new)
        outer={k:set(meta['split_patient'][(meta['split_fold']==f)&(meta['split_role']==k)]) for k in ['fit','validation','test']}
        centers={p:str(meta['center'][np.flatnonzero(meta['patient']==p)[0]]) for p in outer['fit']}
        n=int(np.ceil(.15*len(outer['fit'])));a=take(outer['fit'],centers,n,f'42|{f}|ASSESS')
        cal=take(outer['fit']-a,centers,n,f'42|{f}|CAL');tr=outer['fit']-a-cal
        assert d['roles']==dict(train=tr,cal=cal,assessment=a,test=outer['test'])
        for key,values in [('train',tr),('val',cal),('assessment',a)]:
            assert np.array_equal(d[key],np.flatnonzero(np.isin(pid,list(values))))
        assert len(tr)==[35,35,34,36,35][f-1] and len(a)==len(cal)==8
        assert not(tr&cal or tr&a or cal&a) and set(pid)==outer['fit']
        original_assessed.extend(a)
        a=assignments[f];exceptions=a-outer['fit']
        assert len(exceptions)<=1 and exceptions<=outer['validation'] and not exceptions&outer['test']
        cal=take(outer['fit']-a,centers,8,f'42|{f}|CAL80V1');tr=outer['fit']-a-cal
        assert len(tr)==[27,27,26,28,27][f-1]+len(exceptions) and not(tr&a or tr&cal or a&cal)
        if exceptions:
            extra=np.flatnonzero(np.isin(meta['patient'],list(exceptions)))
            assert not set(extra)&set(ix)
            ix=np.concatenate([ix,extra]);pid=meta['patient'][ix]
        assessed.extend(a);banks.append(dict(fold=f,raw14=raw[ix],raw=raw[ix][:,KEEP].astype(float),
            patient=pid,source=meta['center'][ix],train=np.flatnonzero(np.isin(pid,list(tr))),
            cal=np.flatnonzero(np.isin(pid,list(cal))),assessment=np.flatnonzero(np.isin(pid,list(a))),
            y=y[ix],roles=dict(train=tr,cal=cal,assessment=a,test=outer['test']),archive_indices=ix,old=d,
            original_validation_only_assessment=len(exceptions)))
    assert len(original_assessed)==40 and len(set(original_assessed))==33
    assert len(assessed)==len(set(assessed))==80
    exception_count=sum(d['original_validation_only_assessment'] for d in banks)
    assert exception_count==1
    return banks,dict(input_hashes=HASHES,bank_hashes=BANK_HASHES,patients=80,channels=7635,EZ=1743,NEZ=5892,
        seizures=256,valid_windows=1471965,ASSESS_unique=80,ASSESS_appearances=80,
        legacy_internal_unique=33,legacy_internal_appearances=40,
        strict_original_FIT_assessed=80-exception_count,original_validation_only_assessed=exception_count,
        new_assignment_sha256=digest({f:sorted(v) for f,v in assignments.items()}),
        source_patients={s:len(set(meta['patient'][meta['center']==s])) for s in ['hup','lzu','multicenter','pediatric']})
