"""Engineering recovery: same frozen rule; independent Python endpoint formulas.

The original V2 run failed with a native access violation in the parent's
endpoint routine. Preserve it; use a separate runtime and replay all endpoints
against the previously completed private tables. No model/threshold change.
NumPy is still used by the unchanged input reader, not by these endpoint or
confusion formulas. The native failure's underlying cause is not established.
"""
import csv,hashlib,json,math,os,runpy,sys,types
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True
BASE=Path(r'D:\Codex_Research\experiments')
SOURCE=BASE/'drechan_pr_outer_controlled_fp_case_v2'
OLD=BASE/'drechan_pr_outer_controlled_fp_case_v2_runtime'
NEW=BASE/'drechan_pr_outer_controlled_fp_case_v2_runtime_pure01'
PARENT=BASE/'drechan_pr_outer_representative_cases_v1_runtime_run02'
METRICS=['auroc','macro_f1','top1','recall_at_10pct','recall_at_20pct']

class Array(list):
    def astype(self,t):return Array(map(t,self))
    def tolist(self):return list(self)
    def sum(self):return sum(self)
    def __getitem__(self,key):
        if isinstance(key,list):return Array(v for v,k in zip(self,key) if k)
        r=super().__getitem__(key)
        return Array(r) if isinstance(key,slice) else r
    def __and__(self,other):return Array(bool(a) and bool(b) for a,b in zip(self,other))
    def __invert__(self):return Array(not a for a in self)
    def __le__(self,other):return Array(a<=other for a in self)
    def __rmul__(self,other):return Array(other*a for a in self)
    def __add__(self,other):return Array(a+b for a,b in zip(self,other))

class MathShim:
    sum=staticmethod(sum)
    @staticmethod
    def mean(v):return sum(v)/len(v)
    @staticmethod
    def bincount(v,minlength):
        out=Array([0]*minlength)
        for x in v:out[x]+=1
        return out

def pct(pop,value):
    return 100*(sum(x<value-1e-12 for x in pop)+.5*sum(abs(x-value)<=1e-12 for x in pop))/len(pop)

def csvread(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def compute(meta,truth,roles,scores,config):
    patients=[str(x) for x in meta['patient'].tolist()]
    centers=[str(x) for x in meta['center'].tolist()]
    names=[str(x) for x in meta['channel'].tolist()];labels=truth.tolist()
    size_source={36:'HUP',21:'LZU',15:'Multicenter',8:'Pediatric'};source={}
    for center in set(centers):
        pp={p for p,c in zip(patients,centers) if c==center}
        for p in pp:source[p]=size_source[len(pp)]
    prior=csvread(PARENT/'private/ALL80_PATIENT_ENDPOINTS.csv')
    prior={(r['patient'],r['model'],int(r['seed']),r['condition']):r for r in prior}
    rows=[];detail={}
    for r in config:
        model,seed,cond,fold=r['model'],r['seed'],r['condition'],r['fold']
        q=scores[model,seed,cond,fold];ii=q['indices'].tolist();ss=q['score'].tolist();pred=q['pred'].tolist();groups={}
        for j,i in enumerate(ii):groups.setdefault(patients[i],[]).append(j)
        for pid,positions in sorted(groups.items()):
            ix=[ii[j] for j in positions];y=Array(int(labels[i]) for i in ix);p=Array(float(ss[j]) for j in positions);pr=Array(bool(pred[j]) for j in positions);ch=Array(names[i] for i in ix)
            n=len(y);ez=sum(y);order=sorted(range(n),key=lambda i:(-p[i],ch[i]))
            pos=[p[i] for i in range(n) if y[i]];neg=[p[i] for i in range(n) if not y[i]]
            auc=sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
            tp=sum(a and b for a,b in zip(y,pr));fp=sum(not a and b for a,b in zip(y,pr));fn=ez-tp;tn=n-ez-fp
            vals=dict(auroc=auc,macro_f1=.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn)),top1=float(y[order[0]]))
            for f in [.1,.2]:vals[f'recall_at_{round(100*f)}pct']=sum(y[i] for i in order[:math.ceil(f*n)])/ez
            old=prior[pid,model,seed,cond]
            assert all(abs(vals[m]-float(old[m]))<1e-12 for m in METRICS),'Previous endpoint mismatch'
            assert n==int(old['channels']) and ez==int(old['ez'])
            rows.append(dict(patient=pid,source=source[pid],fold=fold,channels=n,ez=ez,model=model,seed=seed,condition=cond,**vals))
            rank=Array([0]*n)
            for k,j in enumerate(order,1):rank[j]=k
            detail[pid,model,seed,cond]=dict(indices=Array(ix),score=p,pred=pr,y=y,channel=ch,rank=rank)
    assert len(rows)==len(detail)==len(prior)==640
    lookup={(r['patient'],r['model'],str(r['seed']),r['condition']):r for r in rows}
    ids=sorted(set(patients));delta={}
    for pid in ids:
        for cond in ['NO_PR','PR']:
            rr=[lookup[pid,'DRECHANNET',str(s),cond] for s in [42,43,44]]
            r=dict(rr[0]);r.update(seed='MEAN_42_43_44',**{m:sum(x[m] for x in rr)/3 for m in METRICS});lookup[pid,'DRECHANNET','MEAN_42_43_44',cond]=r
        for model,seed in [('SVM','42')]+[('DRECHANNET',s) for s in ['42','43','44','MEAN_42_43_44']]:
            delta[pid,model,seed]={m:lookup[pid,model,seed,'PR'][m]-lookup[pid,model,seed,'NO_PR'][m] for m in METRICS}
    changed=PARENT/'private/ALL80_PATIENT_CHANGES.csv'
    lock=json.loads((PARENT/'private/SELECTED_CASES_FROZEN.json').read_text())
    assert hashlib.sha256(changed.read_bytes()).hexdigest()==lock['changes_sha256']
    old=csvread(changed);assert len(old)==400
    for r in old:assert all(abs(delta[r['patient'],r['model'],r['seed']][m]-float(r[m]))<1e-12 for m in METRICS)
    print('PURE_PYTHON_ALL640_ENDPOINTS_ALL400_CHANGES_REPLAY_PASS',flush=True)
    return rows,detail,lookup,delta

def main():
    assert not NEW.exists() and (OLD/'private/PREANALYSIS_LOCK.json').exists()
    old_hash=hashlib.sha256((OLD/'private/PREANALYSIS_LOCK.json').read_bytes()).hexdigest()
    sys.path.insert(0,str(BASE/'drechan_pr_representation_vs_score_v1'))
    from path_adapter import install
    install(r'D:\Codex_Research\_maintenance\cleanup_20261028\LEGACY_PATHS.json',[NEW])
    run=runpy.run_path
    module=run(str(SOURCE/'run.py'),run_name='frozen_v2_recovery');ns=module['main'].__globals__
    def bridge(path,*args,**kwargs):
        g=run(path,*args,**kwargs)
        assert Path(path)==BASE/'drechan_pr_outer_representative_cases_v1/run.py'
        g['compute']=compute;g['pct']=pct
        return g
    ns.update(RUNTIME=NEW,PUBLIC=NEW/'public',PRIVATE=NEW/'private',np=MathShim,runpy=types.SimpleNamespace(run_path=bridge))
    os.environ['CASE_SOURCE_COMMIT']='49dba1624e887ced72358a26621cd706cfe9c938'
    ns['main']()
    assert hashlib.sha256((OLD/'private/PREANALYSIS_LOCK.json').read_bytes()).hexdigest()==old_hash
    binding=dict(original_source_commit=os.environ['CASE_SOURCE_COMMIT'],implementation='Independent Python endpoint/confusion formulas with full frozen-table replays; original admission retained',selection_threshold_model_changes=False,old_failed_lock_sha256=old_hash,new_runtime=str(NEW),native_root_cause='Not established',engineering_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (NEW/'private/ENGINEERING_BINDING.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    print('OLD_FAILED_LOCK_PRESERVED_PURE_ENGINEERING_RECOVERY_PASS',flush=True)

if __name__=='__main__':main()
