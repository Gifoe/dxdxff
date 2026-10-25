"""CPU-only checks of the real source AST, no private data/server imports."""
import ast
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
SOURCES=['hup','lzu','multicenter','pediatric']
ARMS=['OURS_RAW_FEATURE','OURS_RAW','OURS_FEATURE','LINEAR_SVM_FEATURE_PR','TIMECONVCNN_RAW']

def functions(file,names,scope):
    tree=ast.parse((HERE/file).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(HERE/file),'exec'),scope)
    return scope

class Tests(unittest.TestCase):
    def test_fixed_original_roles_all_seeds(self):
        pid=[];centers=[]
        for source,n in zip(SOURCES,[36,21,15,8]):
            for j in range(n):pid.extend([source+str(j)]*3);centers.extend([source]*3)
        p=np.array(pid);c=np.array(centers)
        fn=functions('common.py',['split_roles'],dict(np=np,math=math,SOURCES=SOURCES))['split_roles']
        original=ast.parse((HERE.parent/'drechan_loco_five_models_seed42_v1/common.py').read_text())
        node=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='split_roles')
        scope=dict(np=np,math=math,SOURCES=SOURCES)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'parent_split','exec'),scope)
        for seed in [42,43,44,45,46,47,48,49]:
            for k in range(1,5):
                r=fn(p,c,k);self.assertEqual(r,scope['split_roles'](p,c,k))
                self.assertEqual(r,fn(p[::-1],c[::-1],k));self.assertEqual(len(r['test']),[36,21,15,8][k-1])
                self.assertEqual(len(r['fit'])+len(r['validation'])+len(r['test']),80)

    def test_seed_parameterization_and_phase_barriers(self):
        q=json.loads((HERE/'PROTOCOL_LOCK.json').read_text())
        self.assertEqual(q['seeds'],[42,43,44,45,46,47,48,49]);self.assertEqual(q['fresh_seeds'],[45,46,47,48,49])
        self.assertEqual(q['statistics']['rows'],3200);self.assertEqual(q['statistics']['metric_values'],28800)
        self.assertEqual(q['svm']['C'],1.)
        c=(HERE/'common.py').read_text();p=(HERE/'pipeline.py').read_text();o=(HERE/'orchestrate.py').read_text()
        self.assertIn("('seed' in p.parts[1] or p.parts[1]=='drechan_no_guidance_ablation_42_44_62_v1')",c)
        self.assertNotIn('seed_all(42+1009*k)',c+p);self.assertIn('seed=TRAIN_SEED+1009*k',(HERE/'svm.py').read_text())
        self.assertIn('ALL160_SELECTIONS_FROZEN_PRIVATE.json',c);self.assertIn('ALL160_TEST_SCORES_SEALED_PRIVATE.json',o)
        for name in ['train','verify_internal','score_test']:
            tree=ast.parse(p);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertNotIn('all_scores(',ast.unparse(node));self.assertNotIn('patient_metrics(',ast.unparse(node))
        self.assertIn('assert TRAIN_SEED in [45,46,47,48,49]',p)
        self.assertIn('old_predictions_reused_not_recomputed=True',o)
        for file in HERE.glob('*.py'):compile(file.read_bytes(),str(file),'exec')

    def test_unchanged_parent_math_and_timeconv(self):
        parent=HERE.parent/'drechan_loco_five_models_3seed_v1'
        checks=0
        for filename,names in {
            'common.py':['split_roles','labels','feature_fit','feature_query','make_source','make','rng_state','restore_rng','forward','predict','pn','select'],
            'pipeline.py':['cellpaths','fit','patient_metrics','bootstrap_draws','boot','tables'],
            'svm.py':['weights','fit','predict']
        }.items():
            def nodes(path):
                return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
            original,current=nodes(parent/filename),nodes(HERE/filename)
            for name in names:
                self.assertEqual(ast.dump(original[name],include_attributes=False),ast.dump(current[name],include_attributes=False),(filename,name))
                checks+=1
        self.assertEqual(checks,21)
        self.assertEqual((parent/'timeconv_engine.py').read_text(),(HERE/'timeconv_engine.py').read_text())
        q=json.loads((HERE/'PROTOCOL_LOCK.json').read_text())
        self.assertEqual(q['seed_policy']['additional'],[45,46,47,48,49])
        self.assertTrue(q['seed_policy']['no_outcome_dependent_stopping'])
        self.assertTrue(q['seed_policy']['no_seed_exclusion'])

    def test_csv_destination_common_namespace(self):
        c=SimpleNamespace(PUB='old_common');p=SimpleNamespace(PUB='old_pipeline')
        written=[]
        p.csv_out=lambda name,rows:written.append((name,rows,c.PUB,p.PUB))
        fn=functions('orchestrate.py',['export_csv'],dict(P=p,C=c,GLOBAL_PUB='global'))['export_csv']
        fn('example',[{'value':1}])
        self.assertEqual(written,[('example',[{'value':1}],'global','global')])
        self.assertEqual((c.PUB,p.PUB),('old_common','old_pipeline'))

    def test_csv_destination_restored_after_failure(self):
        c=SimpleNamespace(PUB='old_common');p=SimpleNamespace(PUB='old_pipeline')
        def failed(*args):raise RuntimeError('synthetic I/O failure')
        p.csv_out=failed
        fn=functions('orchestrate.py',['export_csv'],dict(P=p,C=c,GLOBAL_PUB='global'))['export_csv']
        with self.assertRaises(RuntimeError):fn('example',[{'value':1}])
        self.assertEqual((c.PUB,p.PUB),('old_common','old_pipeline'))

    def test_eight_seed_metric_average_not_ensemble_and_independent_bootstrap(self):
        evpath=HERE.parent/'m8_raw_backbones_pr_seed42_v1/evaluate.py'
        spec=importlib.util.spec_from_file_location('loco3_test_ev',evpath);ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
        scope=dict(np=np,pd=pd,EV=ev,SOURCES=SOURCES,ARMS=ARMS,METRICS=METRICS)
        functions('pipeline.py',['bootstrap_draws','boot','tables'],scope)
        p=SimpleNamespace(tables=scope['tables'])
        multi=dict(scope,P=p,SEEDS=[42,43,44,45,46,47,48,49])
        functions('orchestrate.py',['endpoint_rows','combined_tables'],multi)
        rows=[];rng=np.random.default_rng(20261023)
        for seed in [42,43,44,45,46,47,48,49]:
            for source,n in zip(SOURCES,[36,21,15,8]):
                for i in range(n):
                    y=np.array([1,0,1,0]);prob=rng.uniform(.02,.98,4)
                    if seed==42:prob=.7*y+.1+rng.uniform(0,.09,4)
                    for arm in ARMS:
                        for j in range(4):
                            rows.append(dict(train_seed=seed,arm=arm,held_source=source,patient=source+str(i),channel='c'+str(j),
                                             y=int(y[j]),probability=float(prob[j]),prediction=bool(prob[j]>=.5)))
        frame=pd.DataFrame(rows);points=multi['endpoint_rows'](frame)
        independent=multi['endpoint_rows'](frame,True)
        self.assertEqual(len(points),3200)
        self.assertLess(np.abs(points[METRICS].to_numpy()-independent[METRICS].to_numpy()).max(),1e-12)
        tables=multi['combined_tables'](frame,points)
        for name,count in {'MAIN_RESULTS':5,'SOURCE_RESULTS':20,'SEED_RESULTS':40,'SOURCE_SEED_RESULTS':160,'SEED_MEAN_SD':5,'SOURCE_SEED_SD':20,'POOLED_SEED_RESULTS':200,'INTERVALS':405,'PRIMARY_FAMILY_INTERVALS':8}.items():
            self.assertEqual(len(tables[name]),count)
        seedresults=pd.DataFrame(tables['SEED_RESULTS'])
        mean=seedresults.groupby('arm')[METRICS].mean()
        actual=pd.DataFrame(tables['MAIN_RESULTS']).set_index('arm')
        self.assertLess(np.abs(mean-actual[METRICS]).to_numpy().max(),1e-12)
        self.assertGreater(tables['SEED_MEAN_SD'][0]['macro_f1_sample_sd'],0.)
        for row in tables['INTERVALS']:
            if row['comparison'].startswith('FULL_MINUS_'):
                self.assertEqual(row['estimate'],0.);self.assertEqual(row['ci_low'],0.);self.assertEqual(row['ci_high'],0.)
        avg=points.groupby(['arm','held_source','patient'],as_index=False)[METRICS].mean()
        ref=avg[avg.arm==ARMS[0]].sort_values('patient');draw=scope['bootstrap_draws'](ref.held_source.to_numpy())
        counts=np.zeros((10000,80));np.add.at(counts,(np.arange(10000)[:,None],draw),1.)
        dist=counts@ref[METRICS].to_numpy()/80;ci=np.quantile(dist,[.025,.975],axis=0)
        nominal=pd.DataFrame(tables['INTERVALS'])
        for j,m in enumerate(METRICS):
            row=nominal[(nominal.scope=='ALL')&(nominal.comparison==ARMS[0])&(nominal.metric==m)]
            self.assertLess(np.abs(row[['ci_low','ci_high']].to_numpy()[0]-ci[:,j]).max(),1e-12)

if __name__=='__main__':unittest.main()
