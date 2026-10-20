"""Unlabeled arithmetic extension; never alter the completed diagnostic or model."""
import json
import math
import os
import re
import zipfile
from audit_support import *
from audit_math import legal_scope

OUT=PUB/'feature_numeric_extension'
MASK=np.array([1,1,1,1,1,1,1,1,0,0,1,1,1,1],dtype=bool)


def reconstruct(before,pre):
    x=np.asarray(before,np.float64)
    imputed=np.where(np.isnan(x),np.asarray(pre['imputer_statistics']),x)
    return ((imputed-np.asarray(pre['mean']))/np.asarray(pre['scale'])).astype(np.float32)


def physiology_projection(x,w):
    return (np.where(MASK,np.asarray(x,np.float64),0)*math.sqrt(110/28))@np.asarray(w,np.float64).T


def selftests():
    pre=dict(imputer_statistics=np.zeros(14),mean=np.zeros(14),scale=np.full(14,1e-9))
    assert np.all(reconstruct(np.ones((2,14)),pre)==np.float32(1e9))
    x=np.ones((2,14));w=np.arange(3*14).reshape(3,14);before=physiology_projection(x,w)
    x[:,8:10]=1e99;assert np.array_equal(before,physiology_projection(x,w))


def out_csv(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    temp=OUT/(name+'.csv.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(OUT/(name+'.csv'))


def main():
    selftests();binding,core,fac,records,rec,banks,fr=setup();store=inputs()
    commit=os.environ.get('DIAG_EXTENSION_SOURCE_COMMIT','');assert re.fullmatch('[a-f0-9]{40}',commit)
    seal=jr(ROOT/'AUDIT_CACHE_SEAL_PRIVATE.json');assert seal['input']==digest(binding) and seal['all45_before_aggregate_reporting']
    ownsource={n:sha(HERE/n) for n in ['FEATURE_NUMERIC_AUDIT_LOCK.json','feature_numeric_audit.py']}
    extra=dict(source=ownsource,source_commit=commit,diagnostic_binding=digest(binding),diagnostic_cache_seal_sha256=sha(ROOT/'AUDIT_CACHE_SEAL_PRIVATE.json'))
    freeze(ROOT/'FEATURE_NUMERIC_BINDING_PRIVATE.json',extra);OUT.mkdir(exist_ok=True)
    coordinate_rows=[];path_rows=[];reconstructed=0;projection_checks=0;max_relative_error=0.
    cells=[r for r in fr['cells'] if r['arm'] in LATENT]
    for fold in range(1,6):
        d,scale=rdv.data(fold,core,rec,banks,store,jr(rdv.ROOT/'INPUT_FREEZE_PRIVATE.json'))
        for phase,ix in [('TRAIN',d['train']),('CAL',d['val'])]:
            legal_scope(d['patient'],ix,d['roles'],phase)
            raw=np.asarray(rec['R14'][d['archive_indices'][ix]],np.float64)
            for prep,pre,target in [('FeaturePR',d['pre'],d['x'][ix]),('NoFeaturePR',d['pre_no_pr'],d['x_no_pr'][ix])]:
                assert set(pre['fit_patient_ids'])==d['roles']['train']
                before=core.patient_z(raw,d['patient'][ix]) if prep=='FeaturePR' else raw.copy()
                result=reconstruct(before,pre);assert np.array_equal(result,target);reconstructed+=1
                for j in range(14):
                    valid=before[:,j][np.isfinite(before[:,j])];values=target[:,j].astype(np.float64)
                    assert np.isfinite(values).all()
                    coordinate_rows.append(dict(fold=fold,phase=phase,preprocessor=prep,coordinate=f'coord_{j:02d}',active=bool(MASK[j]),
                                                channels=len(ix),patients=len(set(d['patient'][ix])),before_finite_fraction=float(np.isfinite(before[:,j]).mean()),
                                                before_abs_q99=float(np.quantile(abs(valid),.99)) if len(valid) else float('nan'),before_abs_max=float(abs(valid).max()) if len(valid) else float('nan'),
                                                TRAIN_scale=float(pre['scale'][j]),TRAIN_variance=float(pre['var'][j]),TRAIN_scale_below_1e_minus8=bool(pre['scale'][j]<1e-8),
                                                after_abs_q50=float(np.quantile(abs(values),.5)),after_abs_q99=float(np.quantile(abs(values),.99)),after_abs_max=float(abs(values).max()),
                                                fraction_after_abs_above10=float(np.mean(abs(values)>10)),fraction_after_abs_above1000=float(np.mean(abs(values)>1000))))
            x=np.asarray(d['x'][ix],np.float64)
            for r in [r for r in cells if r['fold']==fold]:
                key=f"{r['arm']}_seed{r['seed']}_fold{fold}";path=cell_path(r['arm'],r['seed'],fold)
                assert sha(path)==seal['files'][key];q=read(path);assert q['binding']==cell_binding(binding,r,d)
                values=q['phases'][phase];assert np.array_equal(values['indices'],d['archive_indices'][ix]) and np.array_equal(values['patient'],d['patient'][ix])
                v=np.asarray(values['v'],np.float64);relative=np.empty_like(v)
                for pid in set(values['patient']):
                    keep=values['patient']==pid;a=v[keep];sd=a.std(0,ddof=0);relative[keep]=(a-a.mean(0))/np.where(sd>1e-8,sd,1.)
                g=(v-q['anchor_mean'])/q['anchor_scale'];z=np.concatenate([g,relative],1) if r['arm']=='FULL' else relative
                assert sha(r['checkpoint'])==r['checkpoint_sha256'];state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
                w=state['model']['head.0.weight'].numpy().astype(np.float64);assert w.shape[1]==z.shape[1]+14
                a=(z*math.sqrt(110/192))@w[:,:z.shape[1]].T;b=physiology_projection(x,w[:,-14:])
                t_raw=torch.nn.functional.linear(torch.from_numpy(z*math.sqrt(110/192)),torch.from_numpy(w[:,:z.shape[1]])).numpy()
                t_phys=torch.nn.functional.linear(torch.from_numpy(np.where(MASK,x,0)*math.sqrt(110/28)),torch.from_numpy(w[:,-14:])).numpy()
                for expected,actual in [(a,t_raw),(b,t_phys)]:
                    assert np.allclose(expected,actual,rtol=1e-10,atol=1e-12)
                    max_relative_error=max(max_relative_error,float(np.max(abs(expected-actual)/(1+abs(actual)))));projection_checks+=1
                ar=float(np.sqrt(np.mean(a*a)));br=float(np.sqrt(np.mean(b*b)))
                path_rows.append(dict(arm=r['arm'],seed=r['seed'],fold=fold,phase=phase,channels=len(ix),
                                      raw_preactivation_rms=ar,physiology_preactivation_rms=br,physiology_to_raw_rms=br/max(ar,1e-12),
                                      raw_rms_floor_active=ar<1e-12,raw_preactivation_abs_max=float(abs(a).max()),physiology_preactivation_abs_max=float(abs(b).max()),
                                      float64_descriptive_projection_not_FP32_logit=True,no_prediction_or_model_change=True))
    assert reconstructed==20 and len(coordinate_rows)==280 and len(path_rows)==60 and projection_checks==120
    out_csv('FEATURE_COORDINATE_NUMERICS',coordinate_rows);out_csv('HEAD_INPUT_PATH_SCALE',path_rows)
    write(OUT/'INPUT_NUMERIC_VERIFICATION.json',dict(status='PASS',source=extra,unique_selftests=2,exact_input_bank_reconstructions=20,
                                                    coordinate_rows=280,head_path_rows=60,independent_projection_checks=120,maximum_relative_projection_error=max_relative_error,
                                                    outcome_labels_used=False,neural_fits=0,new_deployment_thresholds=0,assessment_metrics=0,new_Outer=False,active_goal_achieved=False))
    allowed=['FEATURE_COORDINATE_NUMERICS.csv','HEAD_INPUT_PATH_SCALE.csv','INPUT_NUMERIC_VERIFICATION.json']
    manifest=dict(status='EXPLICIT_ANONYMOUS_ALLOWLIST',files={n:sha(OUT/n) for n in allowed},private_artifacts_exported=False)
    target=ROOT/'public_feature_numeric_extension_v1.zip'
    if target.exists():
        with zipfile.ZipFile(target) as z:assert json.loads(z.read('PACKAGE_MANIFEST.json'))==manifest
    else:
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for n in allowed:z.write(OUT/n,n)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print('FEATURE_NUMERIC_20_EXACT_280_COORDS_60_PATHS_120_CHECKS_PASS',sha(target),target.stat().st_size,flush=True)


if __name__=='__main__':main()
