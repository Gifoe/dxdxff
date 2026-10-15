"""Label-blind, hash-bound slot mapping; preprocessing fits original FIT only."""
from audit_old7 import *
from raw_store import Store


def digest_array(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()


def recipes(selection):
    old=list(range(7));empty=[None]*3
    def slots(new):return list(new)+[None]*(3-len(new))
    extra=slots(selection['new_only_diagnostic_group_indices'])
    eligible=slots(selection['added_new_group_indices'])
    pruned=[k if k in selection['retained_old_group_indices'] else None for k in old]
    return {
        'SLOT10_OLD7':(old+empty,old+empty,False),
        'SLOT10_PRUNED':(pruned+empty,pruned+empty,False),
        'SLOT10_ELIGIBLE_ADD':(old+eligible,old+eligible,False),
        'SLOT10_TOP3_ADD':(old+extra,old+extra,False),
        'SLOT10_NEW_ONLY3':([None]*7+extra,[None]*7+extra,False),
        'SLOT10_NUMERIC_POWER7':([18,19,20,3,4,5,6]+empty,[18,19,20,3,4,5,6]+empty,False),
        'SLOT10_D1_ONLY_NEW':(old+extra,old+empty,False),
        'SLOT10_GATE_ONLY_NEW':(old+empty,old+extra,False),
        'SLOT10_WRONG_PAIR_TOP3':(old+extra,old+extra,True)}


class FeaturePool:
    def __init__(self,rec):
        self.rec=rec;self.raw=Store(BASELINE)
        self.registries=[read(Path(r'D:\dgm_net_seed42_v1_runtime\CONDITIONING_REGISTRY_PRIVATE.pkl')),
            read(OUT/'CANDIDATE_REGISTRY_PRIVATE.pkl'),read(OUT/'NUMERICAL_CONTROL_REGISTRY_PRIVATE.pkl')]
        first=read(OUT/'CANDIDATE_FEATURES_PRIVATE.pkl');num=read(OUT/'NUMERICAL_CONTROL_FEATURES_PRIVATE.pkl')
        self.summary=np.concatenate([rec['R14'],first['candidate22'],num['summary6']],1)
        assert self.summary.shape==(7635,42) and np.isfinite(self.summary).all()
        self.cache={};self.mask={};self.permutation={};self.files={};self.coverage=[]

    def get(self,pid):
        pid=str(pid)
        if pid not in self.cache:
            plan=self.raw.registry['plans'][pid];items=[]
            for reg in self.registries:
                row=reg['files'][pid];assert sha(row['path'])==row['sha256'];q=read(row['path'])
                assert q['channels']==plan['channels'] and q['raw_bank_sha256']==plan['sha256']
                items.append(q)
            ix=items[0]['archive_indices']
            assert all(np.array_equal(ix,q['archive_indices']) for q in items)
            values=np.concatenate([q['delta'] for q in items],-1)
            raw,mask,times,_=self.raw.get(pid,plan['channels'],'cpu');mask=mask.numpy()
            assert values.shape==(*mask.shape,21) and np.isfinite(values).all()
            assert hashlib.sha256(mask.tobytes()).hexdigest()==items[1]['mask_sha256']
            groups={}
            for c in range(mask.shape[1]):groups.setdefault(mask[:,c].tobytes(),[]).append(c)
            perm=np.arange(mask.shape[1]);seed=42+int.from_bytes(hashlib.sha256(pid.encode()).digest()[:8],'little')
            rng=np.random.default_rng(seed)
            for members in groups.values():
                if len(members)>=2:
                    order=rng.permutation(members);perm[order]=np.roll(order,1)
            assert np.array_equal(mask,mask[:,perm])
            movable=sum(len(g) for g in groups.values() if len(g)>=2)
            assert int((perm!=np.arange(len(perm))).sum())==movable
            self.cache[pid]=values;self.mask[pid]=mask;self.permutation[pid]=perm
            self.files[pid]=dict(source_files=[dict(path=r['files'][pid]['path'],sha256=r['files'][pid]['sha256']) for r in self.registries],
                delta_sha256=digest_array(values),mask_sha256=digest_array(mask),permutation_sha256=digest_array(perm),raw_sha256=plan['sha256'])
            self.coverage.append(dict(center=str(np.unique(self.rec['meta']['center'][ix])[0]),channels=len(ix),
                moved_channels=movable,unchanged_singleton_channels=len(ix)-movable,mask_classes=len(groups)))
            del raw;self.raw.cache.clear()
        return self.cache[pid]

    def populate(self):
        for number,pid in enumerate(sorted(self.raw.registry['plans']),1):
            self.get(pid);print('INTEGRATION_FEATURE_IDENTITY_CHECK',number,'of80',flush=True)
        assert len(self.files)==80

    def mapped_summary(self,slotmap,wrong=False):
        assert len(slotmap)==10
        result=np.zeros((7635,20),np.float32)
        for pid,plan in self.raw.registry['plans'].items():
            ix=np.flatnonzero(self.rec['meta']['patient']==pid)
            # Reconstruction order need not equal canonical raw-bank channel order.
            lookup={str(self.rec['meta']['channel'][i]):i for i in ix}
            canonical=np.asarray([lookup[c] for c in plan['channels']]);perm=self.permutation[pid]
            for j,k in enumerate(slotmap):
                if k is None:continue
                source=canonical[perm] if wrong and k>=7 else canonical
                result[canonical,2*j:2*j+2]=self.summary[source,2*k:2*k+2]
        assert np.isfinite(result).all();return result

    def bank(self,original,slotmap,wrong,core):
        rr=self.mapped_summary(slotmap,wrong)[original['archive_indices']]
        # Separate 14+6 transforms retain legacy14 column arithmetic exactly.
        parts=[];preparts=[]
        for sl in [slice(0,14),slice(14,20)]:
            x,pre=core.prepare(rr[:,sl],original['patient'],original['train']);parts.append(x);preparts.append(pre)
        x=np.concatenate(parts,1)
        pre={k:np.concatenate([p[k] for p in preparts]) for k in ['imputer_statistics','mean','scale','var']}
        pre['fit_patient_ids']=preparts[0]['fit_patient_ids']
        assert set(pre['fit_patient_ids'])==original['roles']['fit']
        # Missing slots are physical zeros, including after the fitted transform.
        for j,k in enumerate(slotmap):
            if k is None:assert np.count_nonzero(x[:,2*j:2*j+2])==0
        return dict(original,x=x,pre=pre,slotmap=slotmap,wrong_pair=wrong)

    def scale(self,original,gate_map):
        ids=sorted(original['roles']['fit']);magnitudes=[]
        for pid in ids:
            v=self.get(pid);mask=self.mask[pid]
            magnitudes.append(np.median(np.abs(v[mask]),axis=0))
        typical=np.median(np.stack(magnitudes),axis=0)
        original_scale=read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f"scale_fold{original['fold']}_PRIVATE.pkl")
        assert set(original_scale['fit_patients'])==set(ids)
        assert np.array_equal(np.maximum(typical[:7],1e-30).astype(np.float32),original_scale['scale'])
        scale=np.asarray([1. if k is None else max(float(typical[k]),1e-30) for k in gate_map],np.float32)
        return scale


class MappedInputs:
    def __init__(self,pool,gate_map,wrong=False):
        self.pool=pool;self.gate_map=gate_map;self.wrong=wrong;self.cache={}

    def get(self,patient,channels,device):
        pid=str(patient);plan=self.pool.raw.registry['plans'][pid]
        assert len(channels)==len(plan['channels']) and set(channels)==set(plan['channels'])
        raw,mask,times,perms=self.pool.raw.get(pid,channels,device)
        if pid not in self.cache:
            v=self.pool.get(pid);p=self.pool.permutation[pid];out=np.zeros((*v.shape[:-1],10),np.float32)
            for j,k in enumerate(self.gate_map):
                if k is not None:out[...,j]=v[:,p,:,k] if self.wrong and k>=7 else v[...,k]
            self.cache[pid]=out
        ix=[plan['channels'].index(c) for c in channels]
        delta=torch.from_numpy(self.cache[pid][:,ix]).to(device)
        assert torch.isfinite(delta).all();return raw,mask,times,delta,perms


def transform_frozen(raw,patient,pre,core):
    parts=[]
    for sl in [slice(0,14),slice(14,20)]:
        z=core.patient_z(raw[:,sl],patient)
        parts.append(((np.where(np.isnan(z),pre['imputer_statistics'][sl],z)-pre['mean'][sl])/pre['scale'][sl]).astype(np.float32))
    result=np.concatenate(parts,1);assert np.isfinite(result).all();return result
