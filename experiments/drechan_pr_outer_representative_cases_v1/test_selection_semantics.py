"""Synthetic statistical checks, without server access or patient information."""
import ast,hashlib,math,unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score

tree=ast.parse(Path(__file__).with_name('run.py').read_text(encoding='utf-8'))
names={'endpoint','pct','coords','stability','select','choose'}
ns=dict(np=np,math=math,hashlib=hashlib,roc_auc_score=roc_auc_score,f1_score=f1_score,EPS=1e-12,
    METRICS=['auroc','macro_f1','top1','recall_at_10pct','recall_at_20pct'])
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'frozen_case_math','exec'),ns)

class SelectionSemantics(unittest.TestCase):
    def test_tied_scores_use_actual_canonical_names(self):
        y=np.array([1,0,1,0]);s=np.array([.9,.9,.2,.2]);ch=np.array(['b','a','d','c'])
        v,o=ns['endpoint'](y,s,s>.5,ch)
        self.assertEqual(list(o),[1,0,3,2]);self.assertEqual(v['auroc'],.5);self.assertEqual(v['top1'],0)
    def test_recall_is_fraction_of_ez_not_a_hit(self):
        y=np.zeros(20,int);y[[0,1,19]]=1;s=np.arange(20,0,-1,dtype=float)
        v,o=ns['endpoint'](y,s,np.zeros(20,bool),np.array([f'CH{i:04}' for i in range(20)]))
        self.assertEqual(v['top1'],1);self.assertEqual(v['recall_at_10pct'],2/3);self.assertEqual(v['recall_at_20pct'],2/3)
    def test_median_selection_does_not_choose_maximum_gain(self):
        pool=['low','middle1','middle2','maximum'];delta={}
        for p,x in zip(pool,[.01,.03,.04,.9]):
            for model,seed in [('SVM','42')]+[('DRECHANNET',s) for s in ['42','43','44','MEAN_42_43_44']]:
                delta[p,model,seed]=dict(auroc=x,recall_at_10pct=x)
        chosen,audit=ns['select'](pool,delta,1)
        self.assertIn(chosen,['middle1','middle2']);self.assertTrue(audit['independent_selector_agreement'])
    def test_stability_requires_joint_improvement_in_same_seeds(self):
        delta={('x','DRECHANNET',s):dict(auroc=a,recall_at_10pct=b) for s,a,b in [('42',.1,-.1),('43',.1,.1),('44',-.1,.1)]}
        self.assertEqual(ns['stability']('x',delta,1),1)
    def test_percentile_uses_midrank_for_ties(self):
        self.assertEqual(ns['pct']([0,0,1,2],0),25)
        self.assertEqual(ns['pct']([0,0,1,2],1),62.5)
        self.assertEqual(ns['pct']([0,0,1,2],2),87.5)

if __name__=='__main__':unittest.main()
