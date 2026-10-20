"""Immutable completed-parent roles/math; fixed-no-Late seed42 execution."""
import csv
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
OWN=sys.modules[__name__]
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\drechan_f0_no_lategate_ablation_seed42_v1_runtime')
PUB=ROOT/'public'
PARENT=Path(r'D:\drechan_f0_complete_ablation_3seed_v1')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value

parent=module('no_late_parent_common',PARENT/'common.py')
current=parent
sys.path.insert(0,str(HERE))
MODEL=module('no_late_current_model',HERE/'model.py')
io,legacy,EV=parent.io,parent.legacy,parent.EV
feature_math=parent.feature_math
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
probability,METRICS=parent.probability,parent.METRICS
ARMS,SEEDS=MODEL.ARMS,[42]
FILES=['PROTOCOL_LOCK.json','model.py','common.py','test_math.py','test_model.py','test_reporting.py',
       'admit.py','train.py','verify_internal.py','evaluate_outer.py','report_outer.py','verify_outer.py',
       'run_all.ps1','affinity_entry.py','package_public.py']
CONTEXT={}
sys.modules['common']=OWN;sys.modules['model']=MODEL

def sources():return {n:sha(HERE/n) for n in FILES}

def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:previous,core,fac,records,rec,banks=parent.setup(False)
    finally:
        sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json')
    assert lock['arms']==ARMS and lock['seeds']==SEEDS and lock['parameters']==MODEL.PARAMETERS
    assert lock['aliases']==MODEL.ALIASES and lock['reuses']==MODEL.REUSE
    assert sha(parent.ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')==lock['parent_freeze_sha256']
    assert sha(parent.ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')==lock['parent_score_seal_sha256']
    assert jr(parent.PUB/'INDEPENDENT_OUTER_VERIFICATION.json')['status']=='PASS'
    previous_fr=parent.frozen()
    assert rec['R14'].shape==(7635,14) and len(set(rec['meta']['patient']))==80
    assert sum(len(r['records']) for r in rec['ledgers'])==256
    binding=dict(source=sources(),parent_source=parent.sources(),parent_input=digest(previous),
        parent_freeze=lock['parent_freeze_sha256'],parent_seal=lock['parent_score_seal_sha256'],
        reconstruction=io.EXPECTED_REC,raw_registry=io.EXPECTED_REG,torch=torch.__version__,numpy=np.__version__,
        scientific_status='EXPLORATORY_REPEATED_TEST')
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    CONTEXT.update(binding=binding,rec=rec,parent_freeze=previous_fr,maps={})
    return binding,core,fac,records,rec,banks

def input_arm(arm):return MODEL.REUSE.get(arm,arm)

def bank(fold,banks,arm):
    assert arm in ARMS and CONTEXT
    return parent.bank(fold,banks,input_arm(arm))

def query_features(raw,patients,pre,arm,core):
    return parent.query_features(raw,patients,pre,input_arm(arm),core)

def scale(fold):return parent.scale(fold)
def preprocessor(d,ss):return parent.preprocessor(d,ss)
def source_for(core,fac,records,fold,seed):
    assert seed==42
    return parent.source_for(core,fac,records,fold,seed)

def make(core,fac,records,fold,arm,seed):
    source=source_for(core,fac,records,fold,seed);m=MODEL.NoLateF0(source,arm).cuda();del source
    return m

def reference(arm,seed,fold,optional=False):
    assert seed==42
    if arm not in MODEL.REUSE:return None
    original=MODEL.REUSE[arm]
    r=next(r for r in CONTEXT['parent_freeze']['cells'] if r['arm']==original and r['seed']==seed and r['fold']==fold)
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['validation'])
    assert state['binding']==v['binding']
    return dict(r,arm=arm,seed=seed,epochs=len(v['history']),reused=True,reuse_kind='COMPLETE_PARENT42',
                source_arm=original,parameters=MODEL.PARAMETERS[arm]),state,v

def reference_score(r):
    key=parent.key_for(r['source_arm'],42,r['fold']);path=parent.ROOT/'scores_private'/(key+'.pkl')
    assert sha(path)==jr(parent.ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')['files'][key]
    q=read(path)
    assert q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
    return q,sha(path),parent.ROOT

def store():
    result = legacy.OLD.P3.Inputs(legacy.OLD.BASELINE, Path(r'D:\dgm_net_seed42_v1_runtime'), ROOT)
    result.arm = 'M1_SCALED_DGM'
    return result


def mismatch(mask, channels, pid, fold):
    key = (fold, pid)
    cb = dict(input=digest(CONTEXT['binding']), channels=digest(channels),
              mask=legacy.array_sha(mask.cpu().numpy()), mapping_salt=42)
    path = ROOT/'mismatch_maps_private'/digest(CONTEXT['binding'])[:16]/f'fold{fold}'/(hashlib.sha256(pid.encode()).hexdigest()+'.pkl')
    if key in CONTEXT['maps']:
        q, ph = CONTEXT['maps'][key]
        assert q['binding'] == cb
    else:
        if path.exists():
            q = read(path)
            assert q['binding'] == cb
        else:
            mapping, info = MODEL.channel_mapping(mask, channels, pid, fold)
            q = dict(binding=cb, permutation=mapping.cpu(), info=info)
            save(path, q)
        ph = sha(path)
        CONTEXT['maps'][key] = (q, ph)
    assert q['info']['total_valid_windows'] == int(mask.sum())
    audit = {('mismatch_'+k): q['info'][k] for k in ['changed_valid_windows', 'eligible_valid_windows', 'singleton_valid_windows']}
    return q['permutation'].to(mask.device), dict(audit, mapping_sha256=ph)


def forward(model, d, ix, inputs, ss, fold, binding):
    assert len(set(d['patient'][ix])) == 1
    x = torch.as_tensor(d['x'][ix], device='cuda')
    if model.branch == 'phys':
        return model(x)['logit'], dict(valid_windows=0)
    pid = str(d['patient'][ix[0]])
    channels = d['channel'][ix].astype(str).tolist()
    raw, mask, times, delta, _ = inputs.get(pid, channels, 'cuda')
    permutation, audit = None, {}
    if model.arm == 'EARLY_FILM_MISMATCH':
        permutation, audit = mismatch(mask, channels, pid, fold)
    output = model(x, raw, mask, times, delta, ss, permutation)['logit']
    info = dict(model.last_guidance_audit)
    if 'singleton_valid_windows' in info:
        info['relative_singleton_valid_windows'] = info.pop('singleton_valid_windows')
    if 'valid_windows' in info:
        assert info['valid_windows'] == int(mask.sum())
    info.update(audit)
    info['valid_windows'] = int(mask.sum())
    return output, info


def predict(model, d, ix, inputs, ss, fold, binding):
    model.eval()
    result = np.empty(len(ix), np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            jj = np.flatnonzero(d['patient'][ix] == pid)
            logits, _ = forward(model, d, ix[jj], inputs, ss, fold, binding)
            result[jj] = logits.cpu().numpy()
    assert np.isfinite(result).all()
    return result


def expected_cells():
    return {(a, s, f) for a in ARMS for s in SEEDS for f in range(1, 6)}


def key_for(arm, seed, fold):
    return f'{arm}_seed{seed}_fold{fold}'


def csv_out(name, rows):
    assert rows and all(set(r) == set(rows[0]) for r in rows)
    path = PUB/(name+'.csv')
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def frozen():
    value = jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert value['source'] == sources() and value['input'] == jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert {(r['arm'], r['seed'], r['fold']) for r in value['cells']} == expected_cells()
    assert jr(PUB/'INTERNAL_VERIFICATION.json')['status'] == 'PASS'
    assert sha(PUB/'INTERNAL_VERIFICATION.json') == value['internal_verification']
    for r in value['cells']:
        assert sha(r['checkpoint']) == r['checkpoint_sha256'] and sha(r['validation']) == r['validation_sha256']
    return value


def vector(rows):
    return np.asarray([[r[m] for m in METRICS] for r in rows], np.float64)


def interval(values, draw, levels=(.025, .975)):
    assert values.shape == (80, 9) and np.isfinite(values).all()
    samples = np.concatenate([values[draw[k:k+250]].mean(1) for k in range(0, 10000, 250)])
    return values.mean(0), np.quantile(samples, levels, axis=0)


PAIRS={'FULL_MINUS_'+a:{'FULL':1,a:-1} for a in ARMS if a!='FULL'}
INTERACTIONS={'PR_INTERACTION':{'FULL':1,'WO_FEATURE_PR':-1,'WO_RAW_PR':-1,'WO_BOTH_PR':1}}
