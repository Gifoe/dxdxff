"""FIT-only labels API; source permutations and inner splits use identities only."""
import torch
from common import *
def plan(mf,fold):
    fit=sorted(mf.loc[mf.fold!=fold,'patient_name'].astype(str));held=sorted(mf.loc[mf.fold==fold,'patient_name'].astype(str))
    real={str(r.patient_name):SOURCES.index(str(r.center)) for r in mf.itertuples()};pseudo={};inner={}
    for role,patients in [('FIT',fit),('HELD',held)]:
        values=np.array([real[p] for p in patients]);shuffled=np.random.default_rng(seeded(fold,role,'source-placebo')).permutation(values)
        assert sorted(shuffled.tolist())==sorted(values.tolist());pseudo.update(dict(zip(patients,map(int,shuffled))))
    for source in range(3):
        patients=[p for p in fit if real[p]==source];patients=np.random.default_rng(seeded(fold,source,'inner-split')).permutation(patients).tolist()
        for i,p in enumerate(patients):inner[p]=i%3
    assert set(inner)==set(fit) and set(inner.values())=={0,1,2} and not set(fit)&set(held)
    return {'fit':fit,'held':held,'real':real,'pseudo':pseudo,'inner':inner}
def bank(fold,patients,include_fit_labels=False):
    mf=manifest();allowed=set(mf.loc[mf.fold!=fold,'patient_name'].astype(str))
    if include_fit_labels:assert set(patients)<=allowed,'Outer-held clinical labels unavailable to training API'
    gate=json.loads((ROOT/'BANK_GATE_PRIVATE.json').read_text());assert gate['status']=='PASS' and gate['protocol']==sha(PROTOCOL)
    result={}
    for p in patients:
        path=ROOT/f'embeddings/fold_{fold}/{key(p)}.npz';assert sha(path)==gate['banks'][str(path)]
        with np.load(path,allow_pickle=False) as d:
            item={'channels':d['channels'].astype(str),'base':d['logits'].copy(),'z':normalized(d['embedding'])}
        if include_fit_labels:
            with np.load(ROOT/'labels'/f'{key(p)}.npz',allow_pickle=False) as d:
                assert np.array_equal(item['channels'],d['channels']);item['y']=d['labels'].copy()
        result[p]=item
    return result
def state_hash(state):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def save_torch(path,obj):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.partial');torch.save(obj,tmp);os.replace(tmp,path)
def predict(model,data,patients,source):
    result={}
    with torch.no_grad():
        for p in patients:
            x=data[p];z=torch.from_numpy(x['z']);g=torch.full((len(z),),source[p],dtype=torch.long)
            shared,direction,bias=model.components(z,g)
            # Corrections are float32, fixed baseline remains its exact historical
            # float64 segment mean. Zero correction exactly recovers C0.
            result[p]=x['base']+(shared+direction+bias).numpy().astype(np.float64)
    return result
