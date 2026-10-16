"""Only pinned Raw representations and own-FIT role/label metadata are used."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\raw_macro_f1_head_screen_seed42_v1_runtime')
PUB=ROOT/'public'
ENC=Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_runtime')
FACTOR=Path(r'D:\raw_latent_contrast_optimization_seed42_v1_runtime')
HISTORY=Path(r'D:\m8_physiology_discovery_selection_seed42_v2_runtime')
REFERENCE=Path(r'D:\raw_macro_f1_optimization_seed42_v1')
REFERENCE_RUNTIME=Path(r'D:\raw_macro_f1_optimization_seed42_v1_runtime')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


io=load('macro_head_raw_identity',Path(r'D:\raw_latent_contrast_optimization_seed42_v1\common.py'))
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash=io.atomic_torch,io.state_hash
assert sha(Path(io.__file__))=='c5dd9fb82fab45dd3b1e4ae0911f642a221fbf5bd039a5a642f838a3a5ffa499'
corepath=Path(r'E:\DRE-nips\new-pipeline\7-11\pr_uncertainty_aware_supervision_seed42_v1\code\uas_core.py')
assert sha(corepath)=='1b1ccb0a361e6245cb2e6089347e147036aac90437c0992194f6db7bebf1503a'
core=load('macro_head_pinned_threshold',corepath)
metricpath=Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py')
assert sha(metricpath)=='6c8974e097ff6cb85bbe6f8fa59b246d5e4100c91c50bbb4b082a7dbcf702688'
ev=load('macro_head_pinned_metric',metricpath)
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
CRITICAL=['PROTOCOL_LOCK.json','common.py','head.py','train_internal.py','summarize_internal.py']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    assert sha(REFERENCE/'TARGET_DEFINITION.json')=='e069078dcda4352e7026d5de2fbcf0e8e12df93da8869ff088c8cff7f1b3b9e6'
    assert jr(REFERENCE_RUNTIME/'public/BASELINE_REFERENCE_PARITY.json')['status']=='PASS'
    assert jr(REFERENCE_RUNTIME/'public/BASELINE_ROLE_VERIFICATION.json')['status']=='PASS'
    assert sha(io.BASE/'RECONSTRUCTION_PRIVATE.pkl')==io.EXPECTED_REC
    assert sha(io.BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')==io.EXPECTED_REG
    assert sha(ENC/'ENCODER_REGISTRY_PRIVATE.pkl')==io.EXPECTED_ENCODERS
    enc=read(ENC/'ENCODER_REGISTRY_PRIVATE.pkl')
    assert sha(HISTORY/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==enc['binding']['parent_Outer_seal_protection']
    gate=jr(FACTOR/'public/EXTRACTION_ADMISSION.json');assert gate['status']=='PASS'
    assert gate['seal_sha256']==sha(FACTOR/'SCREEN_FEATURES_SEAL_PRIVATE.json')
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},target=sha(REFERENCE/'TARGET_DEFINITION.json'),
        encoder_registry=io.EXPECTED_ENCODERS,raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,
        factor_seal=sha(FACTOR/'SCREEN_FEATURES_SEAL_PRIVATE.json'),original_Outer_seal=sha(HISTORY/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),
        threshold_source=sha(corepath),metric_source=sha(metricpath),torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    return binding


def contexts():
    meta,original=io.identity_and_banks(labels=True);out=[]
    registry=read(ENC/'ENCODER_REGISTRY_PRIVATE.pkl');parts=jr(FACTOR/'SCREEN_FEATURES_SEAL_PRIVATE.json')['parts']
    for e in registry['entries']:
        fold,screen=int(e['fold']),int(e['screen'])
        if not screen:continue
        assert sha(e['path'])==e['sha256'];stored=read(e['path']);old=original[fold-1];fit=old['train']
        assert np.array_equal(stored['archive_indices'],old['archive_indices'][fit])
        d={k:old[k][fit].copy() for k in ['patient','channel','center','archive_indices','y']}
        rr=stored['roles'];sets=[set(d['patient'][rr[k]]) for k in ['train','cal','choice','held']]
        assert all(sets[i].isdisjoint(sets[j]) for i in range(4) for j in range(i))
        assert set.union(*sets)==old['roles']['fit'] and set.union(*sets).isdisjoint(old['roles']['validation']|old['roles']['test'])
        role=HISTORY/'screens_private'/f'fold{fold}_screen{screen}.pkl'
        assert sha(role)==registry['binding']['original_role_files'][f'{fold}_{screen}']
        source_roles=read(role)
        for k in ['train','cal','choice','held']:
            assert np.array_equal(d['archive_indices'][rr[k]],old['archive_indices'][source_roles[k]])
        cp=ENC/'encoders_private'/f'fold{fold}_screen{screen}'/'BEST_PRIVATE.pt'
        assert sha(cp)==stored['checkpoint_sha256']
        out.append(dict(fold=fold,screen=screen,data=d,stored=stored,roles=rr,checkpoint=cp,
                        representation_sha256=e['sha256'],factor_parts=parts))
    assert len(out)==15
    return sorted(out,key=lambda q:(q['fold'],q['screen']))


def inputs(ctx):
    d,stored=ctx['data'],ctx['stored'];n=len(d['patient'])
    assert stored['raw'].shape==(n,192) and stored['original_validation_and_Outer_excluded']
    z={'1111':np.asarray(stored['raw'][:,:96],np.float32).copy()}
    for recipe in ['0111','1011','1101','1110']:z[recipe]=np.empty((n,96),np.float32)
    for pid in sorted(set(d['patient'])):
        ix=np.flatnonzero(d['patient']==pid)
        p=FACTOR/'screen_features_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/(io.private_key(pid)+'.pkl')
        assert sha(p)==ctx['factor_parts'][str(p.relative_to(FACTOR))]
        q=read(p);b=q['binding']
        assert b['archive_indices']==list(map(int,d['archive_indices'][ix]))
        assert b['encoder']['representation_sha256']==ctx['representation_sha256']
        assert b['encoder']['checkpoint_sha256']==stored['checkpoint_sha256']
        assert q['all_complete_patient_channels'] and q['no_labels_or_manual_features_used']
        assert np.max(np.abs(q['features'][15]-z['1111'][ix]))<=1e-6
        for recipe in list(z)[1:]:z[recipe][ix]=q['features'][int(recipe,2)]
    assert all(np.isfinite(a).all() for a in z.values())
    checkpoint=torch.load(ctx['checkpoint'],map_location='cpu',weights_only=False)
    assert torch.count_nonzero(checkpoint['model']['keep32'])==0
    assert checkpoint['binding']['input']['role_sha256']==read(ENC/'ENCODER_REGISTRY_PRIVATE.pkl')['binding']['original_role_files'][f"{ctx['fold']}_{ctx['screen']}"]
    return z,checkpoint


def probability(logit):return torch.sigmoid(torch.as_tensor(logit,dtype=torch.float32)).numpy()


def patient_rows(d,ix,logit,threshold):
    pn=probability(logit);out=[]
    for pid in sorted(set(d['patient'][ix])):
        loc=d['patient'][ix]==pid;y=1-d['y'][ix][loc];p=1-pn[loc].astype(float);pred=pn[loc]<threshold
        out.append(dict(patient=str(pid),center=str(d['center'][ix][loc][0]),
            **ev.classification(y,p,pred),**ev.ranking(y,p,d['channel'][ix][loc])))
    return out


def cluster_interval(rows):
    frame=pd.DataFrame(rows);ids=sorted(frame.patient.unique())
    sums=np.stack([frame[frame.patient==p][METRICS].sum().to_numpy(float) for p in ids])
    counts=np.asarray([int((frame.patient==p).sum()) for p in ids])
    draw=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
    values=sums[draw].sum(1)/counts[draw].sum(1)[:,None];ci=np.quantile(values,[.025,.975],axis=0)
    return [dict(metric=m,mean=float(frame[m].mean()),low=float(ci[0,j]),high=float(ci[1,j]),
                 appearances=len(frame),unique_patients=len(ids),draws=10000) for j,m in enumerate(METRICS)]
