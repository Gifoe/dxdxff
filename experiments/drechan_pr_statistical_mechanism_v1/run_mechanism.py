"""Admission -> TRAIN directions freeze -> ASSESS endpoint seal -> reports -> verify."""
import argparse
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import mechanism_math as m
import trusted_io as io


def make_bank(d):
    pid=d['patient'];pre=m.fit_global(d['raw'][d['train']]);x=m.global_query(d['raw'],pre)
    ids=sorted(set(pid));train=sorted(d['roles']['train'])
    xdict={p:x[pid==p] for p in ids};stats={p:m.patient_stats(xdict[p]) for p in ids}
    sources={p:str(d['source'][np.flatnonzero(pid==p)[0]]) for p in ids};counts={p:int((pid==p).sum()) for p in ids}
    donors=m.donor_map(ids,train,sources,counts,d['fold'])
    op,_=io.reference('feature_pr').prepare(d['raw14'],pid,d['train'])
    unit,_=io.reference('unit_pr').prepare(d['raw14'],pid,d['train'])
    assert np.array_equal(op,unit)
    old_op,_=io.reference('feature_pr').prepare(d['raw14'],pid,d['old']['train'])
    assert np.array_equal(old_op[:len(d['old']['patient']),io.KEEP],d['old']['x'][:,io.KEEP])
    base=x.astype(np.float32).astype(float)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    # Historical global preprocessing uses float64 arithmetic before its final
    # FP32 bank rounding, as does feature_pr's TRAIN-unit fit.
    raw64=np.asarray(d['raw14'],np.float64)
    imputer=SimpleImputer().fit(raw64[d['train']]);z=imputer.transform(raw64)
    expected=StandardScaler().fit(z[d['train']]).transform(z).astype(np.float32)
    assert np.array_equal(base.astype(np.float32),expected[:,io.KEEP])
    reps={p:dict(m.represent(xdict[p],stats[p],stats[donors[p]]),OP_PR=op[pid==p][:,io.KEEP].astype(float),OP_GLOBAL=base[pid==p]) for p in ids}
    return dict(pre=pre,x=x,reps=reps,stats=stats,donors=donors,sources=sources,counts=counts)


def ys(d,y):return {p:y[d['patient']==p] for p in sorted(set(d['patient']))}


def train_y(d,b):
    out=d['y'].copy();ii=d['train']
    if b>=0:out[ii]=m.permute(out[ii],d['patient'][ii],d['fold'],b)
    return ys(d,out)


def train_moments(d,bank,y,arm):
    return [m.moments(bank['reps'][p][arm],y[p]) for p in sorted(d['roles']['train'])]


def real_checks(d,bank):
    from sklearn.metrics import roc_auc_score
    rank_error=0.;identity=0.;affine_error=0.;support_crossings=0
    v=np.arange(1,13,dtype=float);v/=np.linalg.norm(v)
    rng=np.random.default_rng(42+d['fold']);affine={}
    for p,x in bank['reps'].items():
        base=x['G0_GLOBAL'];a=np.exp(rng.uniform(-.6,.6,12));b=rng.normal(size=12)
        affine[p]=base*a+b
    newpre=m.fit_global(np.concatenate([affine[p] for p in sorted(d['roles']['train'])]))
    for p in sorted(bank['reps']):
        new=m.global_query(affine[p],newpre);st=m.patient_stats(new);old=bank['stats'][p]
        good=st['support']&old['support'];support_crossings+=int(np.count_nonzero(st['support']!=old['support']))
        if good.any():affine_error=max(affine_error,float(np.max(abs(((new-st['mean'])/st['safe'])[:,good]-bank['reps'][p]['G3_FULL'][:,good]))))
        if p not in d['roles']['train']:continue # no ASSESS endpoint access at admission
        y=d['y'][d['patient']==p];rep=bank['reps'][p]
        for j in range(12):
            auc=roc_auc_score(y,rep['G0_GLOBAL'][:,j])
            for arm in m.FACTOR[1:]:rank_error=max(rank_error,abs(roc_auc_score(y,rep[arm][:,j])-auc))
        for a,b in [('G0_GLOBAL','G1_CENTER'),('G2_SCALE','G3_FULL')]:
            qa=m.moments(rep[a],y);qb=m.moments(rep[b],y)
            identity=max(identity,float(np.max(abs(qa['delta']-qb['delta']))),float(np.max(abs(qa['covariance']-qb['covariance']))))
            model=dict(v=v,noise=1.,reference=0.,degenerate=False)
            ea=m.endpoints(qa,model);eb=m.endpoints(qb,model)
            identity=max(identity,abs(ea['T']-eb['T']),abs(ea['A']-eb['A']))
    assert rank_error<1e-12 and identity<1e-8 and affine_error<1e-7
    # Query-only perturbations cannot change TRAIN preprocessing/donor TRAIN paths.
    altered=d['raw'].copy();held=np.setdiff1d(np.arange(len(altered)),d['train']);altered[held]=altered[held]*137+4.
    assert io.digest(m.fit_global(altered[d['train']]))==io.digest(bank['pre'])
    assert all(q in d['roles']['train'] and q!=p for p,q in bank['donors'].items())
    return dict(fold=d['fold'],TRAIN_patients=len(d['roles']['train']),CAL_patients=8,ASSESS_patients=16,
        original_validation_only_assessment=d['original_validation_only_assessment'],
        original_feature_and_operational_PR_bit_parity=True,TRAIN_rank_AUC_invariance_max_error=rank_error,
        TRAIN_delta_fixed_projection_identity_max_error=identity,unlabeled_affine_invariance_max_error=affine_error,
        affine_support_crossings=support_crossings,held_query_cannot_change_TRAIN=True,
        ASSESS_endpoint_labels_used=False)


def admission(root):
    assert not (root/'private/INPUT_FREEZE.json').exists(),'Preserve failed inputs/admission before a new attempt'
    data,audit=io.load_inputs();checks=[];prepared=[]
    for d in data:
        bank=make_bank(d);checks.append(real_checks(d,bank));prepared.append(bank)
    ids=sorted(set(p for d in data for p in d['roles']['assessment']))
    sc={p:bank['sources'][p] for d,bank in zip(data,prepared) for p in d['roles']['assessment']}
    audit.update(status='PASS',source=io.sources(),checks=checks,
        ASSESS_source_patients={s:sum(sc[p]==s for p in ids) for s in sorted(set(sc.values()))},
        python=__import__('platform').python_version(),numpy=np.__version__,scipy=__import__('scipy').__version__,
        sklearn=__import__('sklearn').__version__,neural_imports=False,raw_extraction=False,CAL_used=False)
    io.write(root/'private/INPUT_FREEZE.json',dict(source=io.sources(),input_hashes=io.HASHES,bank_hashes=io.BANK_HASHES))
    io.save(root/'private/PREPARED.pkl',dict(banks=prepared))
    io.write(root/'public/INPUT_AUDIT.json',audit)
    print('ADMISSION_PASS exact_internal_roles,currentPR,TRAIN_invariances,0ASSESS_endpoints',flush=True)


def setup(root):
    binding=io.jr(root/'private/INPUT_FREEZE.json');assert binding['source']==io.sources()
    data,_=io.load_inputs();prepared=io.read(root/'private/PREPARED.pkl')['banks']
    return data,prepared,binding


def freeze(root):
    data,banks,binding=setup(root);commit=os.environ.get('PRM_SOURCE_COMMIT','');assert re.fullmatch('[a-f0-9]{40}',commit)
    states=[];stats=[]
    for d,bank in zip(data,banks):
        by=[]
        for b in range(-1,100):
            y=train_y(d,b)
            for arm in m.ARMS:
                ms=train_moments(d,bank,y,arm)
                for estimator in m.ESTIMATORS:
                    model=m.direction(ms,estimator)
                    states.append(dict(fold=d['fold'],arm=arm,estimator=estimator,permutation=b,model=model))
                    by.append((arm,estimator,b,model))
        # The TRAIN estimator inherits the algebraic centering invariance.
        indexed={(a,e,b):q for a,e,b,q in by}
        for b in range(-1,100):
            for e in m.ESTIMATORS:
                for a,z in [('G0_GLOBAL','G1_CENTER'),('G2_SCALE','G3_FULL')]:
                    np.testing.assert_allclose(indexed[a,e,b]['v'],indexed[z,e,b]['v'],atol=1e-8,rtol=1e-8)
        stats.append(dict(fold=d['fold'],directions=len(by),degenerate=sum(q['degenerate'] for *_,q in by),
            donor_same_source=sum(bank['sources'][p]==bank['sources'][q] for p,q in bank['donors'].items()),
            donor_pairs=len(bank['donors']),donor_self_pairs=0))
        print('DIRECTIONS_FROZEN_IN_MEMORY',d['fold'],len(by),flush=True)
    assert len(states)==7070
    io.save(root/'private/DIRECTIONS.pkl',dict(binding=binding,states=states))
    io.write(root/'private/FROZEN_BEFORE_ASSESS.json',dict(source=io.sources(),input=io.sha(root/'private/INPUT_FREEZE.json'),
        prepared=io.sha(root/'private/PREPARED.pkl'),directions=io.sha(root/'private/DIRECTIONS.pkl'),source_commit=commit))
    io.write(root/'public/FREEZE_STATUS.json',dict(status='ALL7070_DIRECTIONS_FROZEN_BEFORE_ASSESS',source_commit=commit,
        freeze_sha256=io.sha(root/'private/FROZEN_BEFORE_ASSESS.json'),states=7070,folds=stats,source=io.sources(),CAL_used=False))


def evaluate(root):
    data,banks,binding=setup(root);fr=io.jr(root/'private/FROZEN_BEFORE_ASSESS.json')
    assert fr['source']==io.sources() and fr['directions']==io.sha(root/'private/DIRECTIONS.pkl') and fr['prepared']==io.sha(root/'private/PREPARED.pkl')
    states=io.read(root/'private/DIRECTIONS.pkl')['states'];rows=[];support=[]
    for d,bank in zip(data,banks):
        ytrue=ys(d,d['y']);patients=sorted(d['roles']['assessment'])
        for p in patients:
            donor=bank['donors'][p];own=bank['stats'][p];dr=bank['stats'][donor]
            support.append(dict(fold=d['fold'],patient=p,source=bank['sources'][p],
                supported_coordinates=int(own['support'].sum()),donor_supported_coordinates=int(dr['support'].sum()),
                donor_same_source=bank['sources'][p]==bank['sources'][donor],
                donor_same_count_bin=int(np.floor(np.log2(bank['counts'][p])))==int(np.floor(np.log2(bank['counts'][donor]))),
                donor_log2_count_distance=abs(np.log2(bank['counts'][donor]/bank['counts'][p])),
                donor_effective_scale_ratio_median=float(np.median(own['safe']/dr['safe'])),
                variance_correct=float(np.mean(np.var(bank['reps'][p]['G3_FULL'],axis=0))),
                variance_donor=float(np.mean(np.var(bank['reps'][p]['DONOR_FULL'],axis=0)))))
        for state in [q for q in states if q['fold']==d['fold']]:
            b=state['permutation'];arm=state['arm'];model=state['model'];estimator=state['estimator']
            yp=None
            if b>=0 and arm in m.FACTOR:
                yy=d['y'].copy();ii=d['assessment'];yy[ii]=m.permute(yy[ii],d['patient'][ii],d['fold'],b,True);yp=ys(d,yy)
            for p in patients:
                moment=m.moments(bank['reps'][p][arm],ytrue[p])
                for factor in m.FACTORS if b==-1 else [1e-8]:
                    row=dict(patient=p,source=bank['sources'][p],fold=d['fold'],estimator=estimator,arm=arm,
                        factor=factor,permutation=b,diagnostic='TRUE_ASSESS',
                        strict_original_FIT=p in set(d['old']['patient']),**m.endpoints(moment,model,factor))
                    rows.append(row)
                if yp is not None:
                    rows.append(dict(patient=p,source=bank['sources'][p],fold=d['fold'],estimator=estimator,arm=arm,
                        factor=1e-8,permutation=b,diagnostic='MATCHED_H3_LABEL_NULL',
                        strict_original_FIT=p in set(d['old']['patient']),
                        **m.endpoints(m.moments(bank['reps'][p][arm],yp[p]),model)))
        print('ASSESS_ENDPOINTS_COMPUTED',d['fold'],len(rows),flush=True)
    assert len(rows)==179360
    io.save(root/'private/ASSESS_STATISTICS.pkl',dict(rows=rows,support=support))
    io.write(root/'private/ASSESS_SEAL.json',dict(freeze=io.sha(root/'private/FROZEN_BEFORE_ASSESS.json'),
        endpoint_sha256=io.sha(root/'private/ASSESS_STATISTICS.pkl'),rows=len(rows)))
    io.write(root/'public/ASSESS_STATUS.json',dict(status='ALL_ASSESS_STATISTICS_SEALED_BEFORE_AGGREGATES',rows=len(rows),
        freeze_sha256=io.sha(root/'private/FROZEN_BEFORE_ASSESS.json'),seal_sha256=io.sha(root/'private/ASSESS_SEAL.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['admission','freeze','evaluate','report','verify','package'])
    parser.add_argument('--runtime',type=Path,required=True);a=parser.parse_args()
    import psutil
    psutil.Process().cpu_affinity([2])
    assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    a.runtime.mkdir(parents=True,exist_ok=True)
    if a.stage in ['admission','freeze','evaluate']:globals()[a.stage](a.runtime)
    elif a.stage=='verify':
        import verify_mechanism;verify_mechanism.main(a.runtime)
    else:
        import statistics_report;getattr(statistics_report,a.stage)(a.runtime)
