"""CPU tests load pure functions from the real source AST, no server/data imports."""
import ast
import math
from pathlib import Path
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent

def function(file,name,globals_):
    t=ast.parse((HERE/file).read_text());node=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope=dict(globals_);exec(compile(ast.Module(body=[node],type_ignores=[]),str(HERE/file),'exec'),scope);return scope[name]

class ContractTests(unittest.TestCase):
    def fixture(self):
        sources=['hup','lzu','multicenter','pediatric'];patient=[];center=[]
        for z,n in zip(sources,[36,21,15,8]):
            for j in range(n):patient.extend([z+str(j)]*3);center.extend([z]*3)
        return np.asarray(patient),np.asarray(center),sources
    def test_source_roles_counts_patient_disjoint_and_repeatable(self):
        pid,center,sources=self.fixture();split=function('common.py','split_roles',dict(np=np,math=math,SOURCES=sources));tests=set()
        for k in range(1,5):
            r=split(pid,center,k);self.assertEqual(r,split(pid[::-1],center[::-1],k))
            self.assertEqual(set(pid[center==sources[k-1]]),r['test']);tests.update(r['test'])
            self.assertEqual(r['fit']|r['validation']|r['test'],set(pid))
        self.assertEqual(len(tests),80)
    def test_bootstrap_preserves_source_sizes_and_pairing(self):
        pid,center,sources=self.fixture();unique=np.unique(pid);c=np.array([center[np.flatnonzero(pid==p)[0]] for p in unique])
        draw=function('pipeline.py','bootstrap_draws',dict(np=np,SOURCES=sources))(c)
        self.assertEqual(draw.shape,(10000,80))
        for s in sources:self.assertTrue(np.all((c[draw]==s).sum(1)==(c==s).sum()))
        boot=function('pipeline.py','boot',dict(np=np));x=np.arange(80.)[:,None]
        self.assertTrue(np.array_equal(boot(x+1,draw)-boot(x,draw),np.ones((10000,1))))
    def test_frozen_matrix_and_phase_barriers(self):
        import json
        q=json.loads((HERE/'PROTOCOL_LOCK.json').read_text());self.assertEqual(len(q['arms'])*len(q['sources']),20)
        self.assertEqual(q['statistics']['rows'],400);self.assertEqual(q['svm']['C'],1.)
        src=(HERE/'pipeline.py').read_text();t=ast.parse(src)
        for name in ['train','verify_internal','score_test']:
            n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertNotIn('all_scores',ast.unparse(n));self.assertNotIn('patient_metrics',ast.unparse(n))
        self.assertNotIn('LEG.setup(',src);self.assertNotIn('LEG.setup(', (HERE/'common.py').read_text())

if __name__=='__main__':unittest.main()
