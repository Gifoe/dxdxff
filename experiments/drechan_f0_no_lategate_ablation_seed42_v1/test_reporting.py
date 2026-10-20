"""Synthetic endpoint resampling and three-seed averaging tests, not performance."""
import ast
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
    def test_chunked_vs_scalar_80patient_intervals(self):
        ns=dict(np=np);functions(HERE/'common.py',['interval'],ns)
        v=np.random.default_rng(1).normal(size=(80,9));draw=np.random.default_rng(42).integers(0,80,(10000,80))
        for levels in [(.025,.975),(.05/68,1-.05/68)]:
            point,lim=ns['interval'](v,draw,levels)
            for j in range(9):
                self.assertLess(float(np.max(np.abs(lim[:,j]-np.quantile(v[:,j][draw].mean(1),levels)))),1e-14)
                self.assertAlmostEqual(point[j],v[:,j].mean(),places=14)


    def test_aliases_are_not_extra_groups_and_no_guide_interaction(self):
        import json
        lock=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'))
        self.assertEqual(lock['counts']['unique_groups'],18)
        self.assertEqual(lock['counts']['fresh_fits'],60)
        self.assertNotIn('WO_BOTH_GUIDANCE',lock['arms'])
        self.assertEqual(lock['aliases']['WO_BOTH_GUIDANCE'],'WO_EARLY_FILM')
        self.assertEqual((18+17+1)*9,324)

    def test_f1_only_improvement_is_not_dual_metric_evidence(self):
        ns=dict(np=np,METRICS=METRICS,PAIRS={'TEST':{'FULL':1,'OTHER':-1}})
        functions(HERE/'common.py',['vector'],ns);functions(HERE/'report_outer.py',['direction_evidence'],ns)
        matrices={};rows=[]
        for arm in ['FULL','OTHER']:
            z=np.full((80,9),.6);z[:,0]+=.02 if arm=='FULL' else 0;z[:,1]-=.01 if arm=='FULL' else 0
            matrices[arm]=z
            rows.extend(dict(arm=arm,fold=i//16+1,**dict(zip(METRICS,z[i]))) for i in range(80))
        family=[dict(contrast='TEST',metric=m,low=v) for m,v in [('macro_f1',.01),('auroc',-.02)]]
        result,details=ns['direction_evidence'](rows,matrices,family)
        self.assertFalse(result[0]['both_point_estimates_positive']);self.assertEqual(result[0]['folds_both_positive'],0)
        self.assertEqual(len(details),5);self.assertFalse(result[0]['both_family_lower_bounds_positive'])

if __name__=='__main__':unittest.main()
