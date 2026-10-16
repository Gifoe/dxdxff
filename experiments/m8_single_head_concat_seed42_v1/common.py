"""Trusted original dependencies, roles, deterministic initializers and reference hashes."""
import faulthandler
import sys
from pathlib import Path
faulthandler.enable()
HERE=Path(__file__).resolve().parent
sys.path.insert(0,r'D:\m8_physiology_feature_discovery_seed42_v1')
import train_old7_m8 as OLD
from audit_old7 import *
sys.path.insert(0,str(HERE))
Single=load('single_head_new_model',HERE/'model.py').SingleHeadConcat
ROOT=Path(r'D:\m8_single_head_concat_seed42_v1_runtime');PUB=ROOT/'public'
JOINT=Path(r'D:\m8_joint_physiology_ablation_seed42_v1_runtime')
MECH=Path(r'D:\m8_mechanism_ablations_seed42_v1_runtime')
RAW=Path(r'D:\m8_raw_only_seed42_v1_runtime')

def jr(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def array_sha(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def code_hashes():return {p.name:sha(p) for p in HERE.iterdir() if p.suffix in ['.py','.json']}

def setup():
    _,c,core,fac,trainer,records,rec,banks,prior=OLD.setup_new()
    prot=jr(HERE/'PROTOCOL_LOCK.json')
    assert jr(JOINT/'public/INDEPENDENT_COMPLETION_AUDIT.json')['status']=='PASS'
    assert jr(MECH/'public/INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    jointseal=jr(JOINT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');mechseal=jr(MECH/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    protected={str(p):sha(p) for root in [JOINT,MECH] for p in [root/'FROZEN_BEFORE_OUTER_PRIVATE.json',root/'OUTER_PREDICTIONS_SEAL_PRIVATE.json']}
    assert jointseal['freeze_sha256']==sha(JOINT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert mechseal['freeze_sha256']==sha(MECH/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    refs=[]
    for fold in range(1,6):
        for name,code,head in [('REF_D1','000','D1'),('REF_M8','000','M8'),('REF_M8_NO_VAR','100','M8')]:
            key=code+'_'+head+'_'+str(fold);p=JOINT/'outer_scores_private'/(key+'.pkl');assert sha(p)==jointseal['files'][key]
            refs.append(dict(model=name,fold=fold,path=str(p),sha256=sha(p),freeze_sha256=jointseal['freeze_sha256']))
        for name,arm in [('REF_RAW_ONLY','M8_RAW_ONLY'),('REF_LATE_FUSION','A6_LATE_FUSION')]:
            key=arm+'_'+str(fold);p=MECH/'outer_scores_private'/(arm+f'_fold{fold}.pkl');assert sha(p)==mechseal['files'][key]
            refs.append(dict(model=name,fold=fold,path=str(p),sha256=sha(p),freeze_sha256=mechseal['freeze_sha256']))
    # Read public aggregate baseline only; never use Outer endpoints to select candidate.
    ref=pd.read_csv(JOINT/'public/outer/MAIN_RESULTS.csv',dtype={'config':str})
    rr=ref[(ref.config=='000')&(ref['head']=='M8')].iloc[0]
    expected=dict(accuracy=.760166,auroc=.762397,auprc=.554067,macro_f1=.634097,f1=.434087,nez_f1=.834107,mrr=.749310,hit_at_1=.637500)
    assert all(round(float(rr[k]),6)==v for k,v in expected.items())
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    binding=dict(source=code_hashes(),prior=prior,protected=protected,references=refs,metric_engine_sha256=sha(OLD.EV.__file__),
        reconstruction_sha256=sha(OLD.BASELINE/'RECONSTRUCTION_PRIVATE.pkl'),
        raw_source_sha256=sha(Path(r'D:\dgm_net_seed42_v1\code\dgm_net.py')),
        gate_source_sha256=sha(Path(r'D:\dgm_net_seed42_v1\code\physiology_gate.py')),
        raw_only_model_sha256=sha(Path(r'D:\m8_raw_only_seed42_v1\model.py')),
        fullprecision_baseline={k:float(rr[k]) for k in expected})
    fp=ROOT/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert jr(fp)==binding
    else:write(fp,binding)
    return prot,core,fac,records,rec,banks,binding

def bank(fold,banks):
    d=torch.load(OLD.BASELINE/f'D1/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
    original=banks[fold-1]
    for key in ['patient','channel','y','train','val']:assert np.array_equal(d[key],original[key])
    assert set(d['patient'][d['train']])==original['roles']['fit'] and set(d['patient'][d['val']])==original['roles']['validation']
    assert set(d['patient']).isdisjoint(original['roles']['test'])
    return dict(d,archive_indices=original['archive_indices'],roles=original['roles'])

def template(core,fac,records,fold):
    seed=42+1009*fold;core.seed_all(seed)
    # Exact original RNG consumption, but NEVER load trained D1 weights here.
    base=OLD.CompactDelta(fac(seed).network,records)
    return OLD.P3.MorphCorrection(base).cuda()

def make(core,fac,records,fold,arm):
    original=template(core,fac,records,fold)
    m=Single(arm,original.cnn,original.gate,type(original).features,42+1009*fold+170003).cuda()
    assert core.state_hash(m.cnn.state_dict())==core.state_hash(original.cnn.state_dict())
    if arm in ['S3','S4']:assert core.state_hash(m.gate.state_dict())==core.state_hash(original.gate.state_dict())
    assert not any(k.startswith(('base.','alpha','margin')) for k in m.state_dict())
    del original
    return m

def scale(fold):
    q=read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f'scale_fold{fold}_PRIVATE.pkl')
    return torch.tensor(q['scale'],device='cuda')

def store():
    q=OLD.P3.Inputs(OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);q.arm='M1_SCALED_DGM';return q

def predict(model,d,ix,inputs,ss):
    model.eval();out=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            jj=np.flatnonzero(d['patient'][ix]==pid);ii=ix[jj]
            raw,mask,times,delta,perms=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
            out[jj]=model(torch.as_tensor(d['x'][ii],device='cuda'),raw,mask,times,delta,ss)['logit'].cpu().numpy()
    assert np.isfinite(out).all();return out

def protected_unchanged(binding):
    for path,h in {**binding['protected'],**binding['prior']['protected']}.items():assert sha(path)==h
    for name,h in binding['source'].items():assert sha(HERE/name)==h
