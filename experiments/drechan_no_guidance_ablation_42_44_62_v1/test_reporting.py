"""Synthetic endpoint resampling and three-seed averaging tests, not performance."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']


def functions(path,names,ns):
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)


class ReportingTests(unittest.TestCase):
    def test_matrix_shapes_and_independent_metric_count(self):
        lock=json.loads((HERE/'PROTOCOL_LOCK.json').read_text())
        counts=lock['counts'];groups=len(lock['arms']);seeds=len(lock['seeds'])
        self.assertEqual(groups,17);self.assertEqual(lock['seeds'],[42,44,62])
        self.assertEqual(counts['cells'],groups*seeds*5)
        self.assertEqual(counts['patient_seed_rows'],groups*seeds*80)
        self.assertEqual(counts['metric_values'],groups*seeds*80*9)
        self.assertEqual(counts['new_fits'],counts['cells']-counts['reused'])
        self.assertEqual(counts['nominal_intervals'],(groups+groups-1+1)*9)
        self.assertEqual(counts['family_intervals'],2*(groups-1))
        self.assertIn('numeric==36720',(HERE/'verify_outer.py').read_text())
    def test_chunked_vs_scalar_80patient_intervals(self):
        ns=dict(np=np);functions(HERE/'common.py',['interval'],ns)
        v=np.random.default_rng(1).normal(size=(80,9));draw=np.random.default_rng(42).integers(0,80,(10000,80))
        for levels in [(.025,.975),(.05/64,1-.05/64)]:
            point,lim=ns['interval'](v,draw,levels)
            for j in range(9):
                self.assertLess(float(np.max(np.abs(lim[:,j]-np.quantile(v[:,j][draw].mean(1),levels)))),1e-14)
                self.assertAlmostEqual(point[j],v[:,j].mean(),places=14)

    def test_seed_metrics_are_averaged_not_predictions(self):
        ns=dict(np=np,ARMS=['FULL'],SEEDS=[42,44,62]);functions(HERE/'report_outer.py',['collapse_seeds'],ns)
        values={('FULL',s):np.full((80,9),s/100) for s in [42,44,62]}
        self.assertTrue(np.allclose(ns['collapse_seeds'](values)['FULL'],(42+44+62)/300))
        self.assertEqual(ns['collapse_seeds'](values)['FULL'].shape,(80,9))

    def test_f1_only_improvement_is_not_dual_metric_evidence(self):
        ns=dict(np=np,ARMS=['FULL','OTHER'],SEEDS=[42,44,62],METRICS=METRICS,PAIRS={'TEST':{'FULL':1,'OTHER':-1}})
        functions(HERE/'common.py',['vector'],ns);functions(HERE/'report_outer.py',['direction_evidence'],ns)
        matrices={};rows=[]
        for arm in ns['ARMS']:
            for seed in ns['SEEDS']:
                z=np.full((80,9),.6);z[:,0]+=.02 if arm=='FULL' else 0;z[:,1]-=.01 if arm=='FULL' else 0
                matrices[(arm,seed)]=z
                rows.extend(dict(arm=arm,seed=seed,fold=i//16+1,**dict(zip(METRICS,z[i]))) for i in range(80))
        avg={a:matrices[(a,42)] for a in ns['ARMS']}
        family=[dict(contrast='TEST',metric=m,low=v) for m,v in [('macro_f1',.01),('auroc',-.02)]]
        summary,detail=ns['direction_evidence'](rows,matrices,avg,family)
        self.assertFalse(summary[0]['both_point_estimates_positive']);self.assertEqual(summary[0]['seeds_both_positive'],0)
        self.assertEqual(len(detail),15);self.assertFalse(summary[0]['both_family_lower_bounds_positive'])


if __name__=='__main__': unittest.main()
