"""Read sealed Outer scores only. No training, checkpoint loading or inference."""
import os,sys,argparse,csv,json,pickle,hashlib,math,zipfile
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True
MASTER=Path(r'D:\Codex_Research\experiments')
NAME='drechan_pr_outer_representative_cases_v1'
SOURCE=MASTER/NAME;RUNTIME=MASTER/(NAME+'_runtime');PUBLIC=RUNTIME/'public';PRIVATE=RUNTIME/'private'
sys.path.insert(0,str(MASTER/'drechan_pr_representation_vs_score_v1'))
from path_adapter import install
install(r'D:\Codex_Research\_maintenance\cleanup_20261028\LEGACY_PATHS.json',[RUNTIME])
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
DRE=MASTER/'drechan_f0_complete_ablation_3seed_v1_runtime'
SVM=MASTER/'m8_6d_no_variance_ablation_baselines_seed42_v1_runtime'
METRICS=['auroc','macro_f1','top1','recall_at_10pct','recall_at_20pct'];EPS=1e-12

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def jr(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def read(path):
    with Path(path).open('rb') as f:return pickle.load(f)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')
def csvout(path,rows):
    assert rows
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def protect(store,path,expected=None):
    path=Path(path);h=sha(path)
    if expected:assert h==expected,'Parent hash mismatch'
    store[str(path)]=h;return h

def endpoint(y,score,pred,channels):
    assert y.sum()>0 and y.sum()<len(y) and np.isfinite(score).all()
    order=np.lexsort((channels.astype(str),-score));sorted_y=y[order];n=len(y);p=int(y.sum())
    primary=dict(auroc=float(roc_auc_score(y,score)),macro_f1=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),top1=float(sorted_y[0]))
    primary.update({f'recall_at_{int(q*100)}pct':float(sorted_y[:int(math.ceil(q*n))].sum()/p) for q in [.1,.2]})
    # Separate pairwise AUROC, confusion-matrix F1 and Python ordering formulas.
    auc=float(np.mean((score[y==1,None]>score[None,y==0])+.5*(score[y==1,None]==score[None,y==0])))
    tp=int(np.sum((y==1)&pred));tn=int(np.sum((y==0)&~pred));fp=int(np.sum((y==0)&pred));fn=p-tp
    f1=.5*(2*tp/max(1,2*tp+fp+fn)+2*tn/max(1,2*tn+fp+fn))
    second_order=sorted(range(n),key=lambda i:(-float(score[i]),str(channels[i])))
    assert np.array_equal(order,np.array(second_order)) and abs(auc-primary['auroc'])<EPS and abs(f1-primary['macro_f1'])<EPS
    for q in [.1,.2]:
        k=int(math.ceil(q*n));hit=sum(int(y[i]) for i in second_order[:k]);assert abs(hit/p-primary[f'recall_at_{int(q*100)}pct'])<EPS
    return primary,order

def admission():
    protected={};protocol=jr(SOURCE/'PROTOCOL_LOCK.json')
    df=jr(DRE/'FROZEN_BEFORE_OUTER_PRIVATE.json');ds=jr(DRE/'OUTER_SCORE_SEAL_PRIVATE.json')
    sf=jr(SVM/'FROZEN_BEFORE_OUTER_PRIVATE.json');ss=jr(SVM/'OUTER_SCORE_SEAL_PRIVATE.json');tab=read(SVM/'TABULAR_FREEZE_PRIVATE.pkl')
    for root,fr,seal in [(DRE,df,ds),(SVM,sf,ss)]:
        for name in ['FROZEN_BEFORE_OUTER_PRIVATE.json','OUTER_SCORE_SEAL_PRIVATE.json','INPUT_FREEZE_PRIVATE.json','PATIENT_METRICS_PRIVATE.pkl','public/INDEPENDENT_OUTER_VERIFICATION.json']:
            protect(protected,root/name)
        assert jr(root/'public/INDEPENDENT_OUTER_VERIFICATION.json')['status']=='PASS'
        assert fr['input']==jr(root/'INPUT_FREEZE_PRIVATE.json')
        assert seal['freeze']==sha(root/'FROZEN_BEFORE_OUTER_PRIVATE.json')
        source_root=MASTER/root.name.removesuffix('_runtime')
        for name,h in fr['source'].items():protect(protected,source_root/name,h)
    assert ds['all285_cells_sealed_before_metric_reporting'] and len(ds['files'])==285
    assert tab['input']==sf['input'];protect(protected,SVM/'TABULAR_FREEZE_PRIVATE.pkl',sf['tabular_freeze_sha256'])
    context=read(DRE/'METADATA_PRIVATE.pkl');meta=context['meta'];protect(protected,DRE/'METADATA_PRIVATE.pkl')
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    protect(protected,archive,context['archive_sha256'])
    with np.load(archive,allow_pickle=True) as z:
        for k in ['patient','channel','center','split_patient','split_fold','split_role']:assert np.array_equal(meta[k].astype(str),z[k].astype(str))
        truth=(1-z['y']).astype(int)
    assert len(truth)==7635 and truth.sum()==1743 and len(set(meta['patient']))==80
    assert len(set(zip(meta['patient'],meta['channel'])))==7635
    meta={k:np.asarray(v) for k,v in meta.items()}
    roles={}
    for fold in range(1,6):
        roles[fold]={r:set(meta['split_patient'][(meta['split_fold']==fold)&(meta['split_role']==r)]) for r in ['fit','validation','test']}
        assert all(roles[fold][a].isdisjoint(roles[fold][b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')])
        assert len(roles[fold]['test'])==[16,16,17,15,16][fold-1]
    assert len(set.union(*(roles[f]['test'] for f in roles)))==sum(len(roles[f]['test']) for f in roles)==80
    config=[];scores={};audit=[]
    for model,root,freeze,seal,cells in [('SVM',SVM,sf,ss,[r for r in tab['cells'] if r['arm'] in ['Linear_SVM_NO_PR','Linear_SVM_PR']]),('DRECHANNET',DRE,df,ds,[r for r in df['cells'] if r['arm'] in ['WO_BOTH_PR','FULL']])]:
        for r in cells:
            arm,fold=r['arm'],r['fold'];seed=42 if model=='SVM' else int(r['seed'])
            assert model=='SVM' or (seed in [42,43,44] and int(r['parameters'])==11425)
            key=f'{arm}_{fold}' if model=='SVM' else f'{arm}_seed{seed}_fold{fold}'
            path=root/'scores_private'/(key+'.pkl');protect(protected,path,seal['files'][key]);q=read(path)
            protect(protected,r['checkpoint'],r['checkpoint_sha256']);protect(protected,r['validation'],r['validation_sha256'])
            assert q['freeze']==sha(root/'FROZEN_BEFORE_OUTER_PRIVATE.json') and q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
            expected=np.flatnonzero(np.isin(meta['patient'],list(roles[fold]['test'])))
            assert np.array_equal(q['indices'],expected)
            cond='NO_PR' if arm in ['Linear_SVM_NO_PR','WO_BOTH_PR'] else 'PR'
            prob=q['probability_ez'].astype(float) if model=='SVM' else 1-q['score_nez'].astype(float)
            pred=q['prediction_ez'].astype(bool) if model=='SVM' else q['score_nez']<r['threshold']
            if model=='SVM':assert np.array_equal(pred,prob>=r['threshold'])
            assert np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1))
            scores[model,seed,cond,fold]=dict(indices=expected,score=prob,pred=pred)
            config.append(dict(model=model,seed=seed,condition=cond,fold=fold,arm=arm))
            audit.append(dict(model=model,seed=seed,condition=cond,fold=fold,checkpoint_sha256=r['checkpoint_sha256'],validation_sha256=r['validation_sha256'],score_sha256=sha(path),threshold=float(r['threshold']),orientation='EZ_GE' if model=='SVM' else 'NEZ_LT'))
    assert len(config)==len(scores)==40
    assert len([r for r in config if r['model']=='DRECHANNET'])==30
    for k in scores:
        if k[0]=='DRECHANNET':assert np.array_equal(scores[k]['indices'],scores['SVM',42,'NO_PR',k[3]]['indices'])
    return protocol,meta,truth,roles,scores,config,protected,audit

def compute(meta,truth,roles,scores,config):
    patients=sorted(set(meta['patient']));size_to_source={36:'HUP',21:'LZU',15:'Multicenter',8:'Pediatric'};source={}
    for center in set(meta['center']):
        ids=set(meta['patient'][meta['center']==center]);assert len(ids) in size_to_source
        for pid in ids:source[pid]=size_to_source[len(ids)]
    rows=[];detail={}
    for r in config:
        q=scores[r['model'],r['seed'],r['condition'],r['fold']];ii=q['indices']
        for pid in sorted(set(meta['patient'][ii])):
            j=np.flatnonzero(meta['patient'][ii]==pid);ix=ii[j];y=truth[ix];p=q['score'][j];pred=q['pred'][j];ch=meta['channel'][ix].astype(str)
            v,order=endpoint(y,p,pred,ch)
            rows.append(dict(patient=str(pid),source=source[pid],fold=r['fold'],channels=len(ix),ez=int(y.sum()),model=r['model'],seed=r['seed'],condition=r['condition'],**v))
            rank=np.empty(len(ix),int);rank[order]=np.arange(1,len(ix)+1)
            detail[str(pid),r['model'],r['seed'],r['condition']]=dict(indices=ix,score=p,pred=pred,y=y,channel=ch,rank=rank)
    assert len(rows)==640 and len(detail)==640
    for model,seeds in [('SVM',[42]),('DRECHANNET',[42,43,44])]:
        for seed in seeds:
            for cond in ['NO_PR','PR']:
                z=[r for r in rows if (r['model'],r['seed'],r['condition'])==(model,seed,cond)]
                assert len(z)==80 and sum(r['channels'] for r in z)==7635
    # Replay the parent patient endpoints rather than copying their summaries.
    replay=0
    for model,root in [('SVM',SVM),('DRECHANNET',DRE)]:
        original=read(root/'PATIENT_METRICS_PRIVATE.pkl')
        if hasattr(original,'to_dict'):original=original.to_dict('records')
        bykey={(r['arm'],int(r.get('seed',42)),int(r['fold']),str(r['patient'])):r for r in original}
        for r in [x for x in rows if x['model']==model]:
            arm=('Linear_SVM_NO_PR' if r['condition']=='NO_PR' else 'Linear_SVM_PR') if model=='SVM' else ('WO_BOTH_PR' if r['condition']=='NO_PR' else 'FULL')
            old=bykey[arm,r['seed'],r['fold'],r['patient']]
            for a,b in [('auroc','auroc'),('macro_f1','macro_f1'),('top1','hit_at_1')]:assert abs(r[a]-float(old[b]))<EPS
            replay+=1
    assert replay==640
    lookup={(r['patient'],r['model'],str(r['seed']),r['condition']):r for r in rows}
    for pid in patients:
        for cond in ['NO_PR','PR']:
            rr=[lookup[str(pid),'DRECHANNET',str(seed),cond] for seed in [42,43,44]]
            mean=dict(rr[0]);mean.update(seed='MEAN_42_43_44',**{k:float(np.mean([r[k] for r in rr])) for k in METRICS})
            lookup[str(pid),'DRECHANNET','MEAN_42_43_44',cond]=mean
    delta={}
    for pid in patients:
        for model,seed in [('SVM','42')]+[('DRECHANNET',s) for s in ['42','43','44','MEAN_42_43_44']]:
            no,pr=[lookup[str(pid),model,seed,c] for c in ['NO_PR','PR']]
            delta[str(pid),model,seed]={k:pr[k]-no[k] for k in METRICS}
    return rows,detail,lookup,delta

def pct(population,value):
    a=np.array(population,float);return float(100*(np.sum(a<value-EPS)+.5*np.sum(np.abs(a-value)<=EPS))/len(a))
def coords(pid,delta):return [delta[pid,'SVM','42'][k] for k in ['auroc','recall_at_10pct']]+[delta[pid,'DRECHANNET','MEAN_42_43_44'][k] for k in ['auroc','recall_at_10pct']]
def stability(pid,delta,sign):
    return sum(all(sign*delta[pid,'DRECHANNET',str(s)][k]>EPS for k in ['auroc','recall_at_10pct']) for s in [42,43,44])
def select(pool,delta,sign):
    assert pool,'No patient satisfies the locked case rule'
    a=np.array([coords(p,delta) for p in pool])
    ranked=[]
    for i,pid in enumerate(pool):
        pr=[pct(a[:,j],a[i,j]) for j in range(4)];distance=float(np.mean(np.abs(np.array(pr)-50)))
        ranked.append((distance,-stability(pid,delta,sign),hashlib.sha256(('case_selection_v1|'+pid).encode()).hexdigest(),pid,pr))
    ranked.sort();chosen=ranked[0]
    # Independent sorted-group counting and candidate-order reconstruction.
    independent=[]
    for pid in pool:
        v=coords(pid,delta);ranks=[]
        for j in range(4):
            column=sorted(float(x[j]) for x in a);below=sum(x<v[j]-EPS for x in column);equal=sum(abs(x-v[j])<=EPS for x in column)
            ranks.append(100*(below+equal/2)/len(column))
        independent.append((sum(abs(x-50) for x in ranks)/4,-stability(pid,delta,sign),hashlib.sha256(('case_selection_v1|'+pid).encode()).hexdigest(),pid))
    independent.sort();assert independent[0][3]==chosen[3] and abs(independent[0][0]-chosen[0])<EPS
    return chosen[3],dict(pool_size=len(pool),median_distance_percentile_points=chosen[0],within_pool_selection_percentiles=chosen[4],pool_medians=np.median(a,0).tolist(),chosen_changes=coords(chosen[3],delta),maximum_median_distance_in_pool=max(x[0] for x in ranked),independent_selector_agreement=True)

def choose(lookup,delta):
    ids=sorted({k[0] for k in lookup});basic=[]
    for p in ids:
        r=lookup[p,'SVM','42','NO_PR']
        if r['channels']>=20 and r['ez']>=3 and r['ez']<r['channels']:basic.append(p)
    joint_positive=[p for p in basic if all(v>EPS for v in coords(p,delta))]
    positive=[p for p in joint_positive if stability(p,delta,1)>=2]
    negative_tiers=[
        [p for p in basic if all(v< -EPS for v in coords(p,delta)) and stability(p,delta,-1)>=2],
        [p for p in basic if all(delta[p,'DRECHANNET','MEAN_42_43_44'][k]< -EPS for k in ['auroc','recall_at_10pct']) and stability(p,delta,-1)>=2],
        [p for p in basic if delta[p,'DRECHANNET','MEAN_42_43_44']['auroc']< -EPS and delta[p,'DRECHANNET','MEAN_42_43_44']['recall_at_10pct']<=EPS and sum(delta[p,'DRECHANNET',str(s)]['auroc']< -EPS for s in [42,43,44])>=2]]
    tier=next(i for i,z in enumerate(negative_tiers) if z)
    pos,pa=select(positive,delta,1);neg,na=select(negative_tiers[tier],delta,-1);assert pos!=neg
    pools={'CASE-P01':positive,'CASE-N01':negative_tiers[tier]}
    audit=dict(cohort_patients=80,basic_eligible=len(basic),positive_both_models_before_seed_stability=len(joint_positive),positive_stable_candidates=len(positive),negative_tier_counts=[len(z) for z in negative_tiers],negative_selected_tier=tier+1,selection_coordinate_order=['SVM_delta_AUROC','SVM_delta_Recall10','DRE_mean_delta_AUROC','DRE_mean_delta_Recall10'],positive=pa,negative=na)
    return {'CASE-P01':pos,'CASE-N01':neg},pools,audit

def export_cases(cases,pools,audit,lookup,delta,detail):
    ids=sorted({k[0] for k in lookup});metadata=[];metrics=[];channels=[];named=[];curves=[];rationale=[]
    for case,pid in cases.items():
        ref=lookup[pid,'SVM','42','NO_PR'];sign=1 if case=='CASE-P01' else -1
        metadata.append(dict(case_id=case,center=ref['source'],outer_fold=ref['fold'],total_channels=ref['channels'],ez_channels=ref['ez'],nez_channels=ref['channels']-ref['ez'],dre_joint_consistent_seeds=stability(pid,delta,sign),dre_auroc_consistent_seeds=sum(sign*delta[pid,'DRECHANNET',str(s)]['auroc']>EPS for s in [42,43,44]),dre_recall10_consistent_seeds=sum(sign*delta[pid,'DRECHANNET',str(s)]['recall_at_10pct']>EPS for s in [42,43,44])))
        for model,seed in [('SVM','42')]+[('DRECHANNET',s) for s in ['42','43','44','MEAN_42_43_44']]:
            for metric in METRICS:
                no,pr=[lookup[pid,model,seed,c][metric] for c in ['NO_PR','PR']];d=pr-no
                metrics.append(dict(case_id=case,model=model,seed=seed,metric=metric,no_pr=no,pr=pr,change=d,percentile_among_all80=pct([delta[p,model,seed][metric] for p in ids],d),percentile_among_case_pool=pct([delta[p,model,seed][metric] for p in pools[case]],d)))
        original_channels=sorted(detail[pid,'SVM',42,'NO_PR']['channel'].tolist());alias={name:f'CH{i:04d}' for i,name in enumerate(original_channels,1)}
        for model,seeds in [('SVM',[42]),('DRECHANNET',[42,43,44])]:
            for seed in seeds:
                for condition in ['NO_PR','PR']:
                    q=detail[pid,model,seed,condition];n=len(q['y']);assert n==ref['channels']
                    anon=np.array([alias[str(x)] for x in q['channel']]);o=np.lexsort((anon,-q['score']))
                    rank=np.empty(n,int);rank[o]=np.arange(1,n+1);assert np.array_equal(rank,q['rank'])
                    for j in np.argsort(anon):
                        row=dict(case_id=case,channel_id=anon[j],truth_label='EZ' if q['y'][j] else 'NEZ',truth_ez=int(q['y'][j]),model=model,seed=seed,condition=condition,score_ez=float(q['score'][j]),rank_ez=int(q['rank'][j]),prediction_ez=int(q['pred'][j]))
                        channels.append(row);named.append(dict(row,private_patient_id=pid,original_channel_name=str(q['channel'][j])))
                    yy=q['y'][o];cumulative=np.r_[0,np.cumsum(yy)];p=int(q['y'].sum())
                    for k in range(n+1):
                        independent_hits=int(q['y'][q['rank']<=k].sum());assert independent_hits==cumulative[k]
                        curves.append(dict(case_id=case,model=model,seed=seed,condition=condition,k=k,total_channels=n,review_fraction=k/n,ez_recovered=int(cumulative[k]),ez_total=p,recall=float(cumulative[k]/p),precision=float(cumulative[k]/k) if k else ''))
                    for fraction,metric in [(.1,'recall_at_10pct'),(.2,'recall_at_20pct')]:assert abs(cumulative[math.ceil(fraction*n)]/p-lookup[pid,model,str(seed),condition][metric])<EPS
        rationale.append(dict(case_id=case,selection_kind='representative_improvement' if case=='CASE-P01' else 'declining_control',**audit['positive' if case=='CASE-P01' else 'negative']))
    csvout(PUBLIC/'CASE_METADATA.csv',metadata);csvout(PUBLIC/'CASE_METRICS.csv',metrics);csvout(PUBLIC/'CHANNEL_SCORES_ANONYMOUS.csv',channels);csvout(PUBLIC/'RECALL_AT_K.csv',curves)
    csvout(PRIVATE/'SELECTED_CHANNELS_ORIGINAL_NAMES.csv',named)
    write(PRIVATE/'IDENTITY_MAP.json',cases);write(PUBLIC/'SELECTION_AUDIT.json',audit);write(PUBLIC/'CASE_SELECTION_REASON.json',rationale)
    return dict(selected_case_count=2,case_metric_rows=len(metrics),anonymous_channel_score_rows=len(channels),recall_at_k_rows=len(curves),private_original_channel_score_rows=len(named),selected_channels=sum(r['total_channels'] for r in metadata))

def main():
    args=argparse.ArgumentParser();args.add_argument('--source-commit',required=True);arg=args.parse_args();assert len(arg.source_commit)==40
    assert not (RUNTIME/'STATUS_PRIVATE.json').exists(),'Unique run required; inspect state before resuming'
    RUNTIME.mkdir(exist_ok=True);PUBLIC.mkdir(exist_ok=True);PRIVATE.mkdir(exist_ok=True)
    source={p.name:sha(p) for p in SOURCE.iterdir() if p.suffix in ['.py','.json','.md']}
    write(PRIVATE/'PREANALYSIS_SOURCE_LOCK.json',dict(source_commit=arg.source_commit,source=source,protocol_sha256=sha(SOURCE/'PROTOCOL_LOCK.json')))
    write(RUNTIME/'STATUS_PRIVATE.json',dict(status='ADMISSION'))
    protocol,meta,truth,roles,scores,config,protected,cell_audit=admission()
    write(PRIVATE/'INPUT_BINDING.json',dict(protected=protected,cell_audit=cell_audit))
    write(PUBLIC/'ADMISSION.json',dict(status='PASS',patients=80,channels=7635,ez_channels=1743,score_cells=40,all_roles_patient_disjoint=True,original_frozen_source_hashes=True,original_checkpoint_validation_hashes=True,new_model_fits=0,new_model_forward_passes=0))
    print('CASE_ADMISSION_PASS_40_SEALED_CELLS',flush=True)
    rows,detail,lookup,delta=compute(meta,truth,roles,scores,config)
    csvout(PRIVATE/'ALL80_PATIENT_ENDPOINTS.csv',rows)
    changes=[dict(patient=p,model=model,seed=seed,**v) for (p,model,seed),v in delta.items()]
    csvout(PRIVATE/'ALL80_PATIENT_CHANGES.csv',changes)
    cases,pools,selection=choose(lookup,delta)
    write(PRIVATE/'SELECTED_CASES_FROZEN.json',dict(cases=cases,selection=selection,changes_sha256=sha(PRIVATE/'ALL80_PATIENT_CHANGES.csv')))
    counts=export_cases(cases,pools,selection,lookup,delta,detail)
    distributions=[]
    ids=sorted({k[0] for k in lookup})
    for model,seed in [('SVM','42'),('DRECHANNET','MEAN_42_43_44')]:
        for metric in METRICS:
            a=np.array([delta[p,model,seed][metric] for p in ids])
            for q in [0,.1,.25,.5,.75,.9,1]:distributions.append(dict(model=model,seed=seed,metric=metric,quantile=q,change=float(np.quantile(a,q)),patients=80))
    csvout(PUBLIC/'ALL80_CHANGE_DISTRIBUTION.csv',distributions)
    aggregate=[]
    for model,seed in [('SVM','42')]+[('DRECHANNET',s) for s in ['42','43','44','MEAN_42_43_44']]:
        for cond in ['NO_PR','PR']:
            aggregate.append(dict(model=model,seed=seed,condition=cond,patients=80,channels=7635,**{m:float(np.mean([lookup[p,model,seed,cond][m] for p in ids])) for m in METRICS}))
    csvout(PUBLIC/'COHORT_AGGREGATES.csv',aggregate)
    for path,h in protected.items():assert sha(path)==h,'Parent artifact changed'
    verification=dict(status='PASS',patients=80,score_cells=40,patient_condition_rows=len(rows),patient_change_rows=len(changes),endpoint_values=len(rows)*5,original_parent_patient_metric_replays=640,independent_metric_formulas=True,independent_selector_agreement=True,all_selected_channel_ranks_and_curves_reconstructed=True,parent_hashes_unchanged=True,score_ensembling=False,**counts)
    write(PUBLIC/'INDEPENDENT_VERIFICATION.json',verification)
    write(PUBLIC/'PROVENANCE.json',dict(source_commit=arg.source_commit,source=source,selection_protocol_sha256=sha(SOURCE/'PROTOCOL_LOCK.json'),cell_audit=cell_audit,svm='12D frozen Linear SVM NO PR vs Legacy PR, original seed42 run',drechan='Original complete FULL vs WO_BOTH_PR, seeds42/43/44, original five-fold Ictal Outer; 11425 parameters with early FiLM/late Gate',privacy='Only CASE/CH aliases outside trusted server; real IDs/contact names in private export; user-authorized deidentified selected-case scores',private_export_location=str(PRIVATE),selection_after_prior_cohort_results=True,interpretation='Retrospective descriptive case illustration, not independent validation'))
    write(PUBLIC/'STATUS.json',dict(status='COMPLETE_VERIFIED_FROZEN_OUTER_REPRESENTATIVE_CASES',new_training=0,new_inference=0))
    files=sorted(PUBLIC.iterdir());manifest={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    archive=RUNTIME/'public_deidentified_case_export_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'public/'+p.name)
        z.writestr('PUBLIC_MANIFEST.json',json.dumps(dict(files=manifest,real_patient_identifiers_included=False,original_contact_names_included=False,user_authorized_deidentified_channel_scores=True),sort_keys=True))
    write(RUNTIME/'STATUS_PRIVATE.json',dict(status='COMPLETE_VERIFIED_FROZEN_OUTER_REPRESENTATIVE_CASES',archive_sha256=sha(archive)))
    print('FROZEN_OUTER_CASES_COMPLETE_VERIFIED',sha(archive),flush=True)

if __name__=='__main__':main()
