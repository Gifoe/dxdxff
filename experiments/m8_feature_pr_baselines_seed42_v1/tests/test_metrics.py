"""Synthetic only: metric definitions, undefined cases and leakage guards."""
import importlib.util
import sys
from pathlib import Path
import unittest
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,log_loss
ROOT=Path(__file__).parents[1]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
E=module('metric_test',ROOT/'evaluate.py')
M=module('models_test',ROOT/'models.py')

class Contract(unittest.TestCase):
    def test_binary(self):
        y=np.array([1,0,1,0,0,1]);p=np.array([.9,.7,.3,.1,.5,.6]);pred=p>=.5;r=E.classification(y,p,pred)
        self.assertAlmostEqual(r['macro_f1'],f1_score(y,pred,average='macro'))
        self.assertAlmostEqual(r['auroc'],roc_auc_score(y,p));self.assertAlmostEqual(r['auprc'],average_precision_score(y,p))
        self.assertAlmostEqual(r['log_loss'],log_loss(y,p));self.assertEqual((r['TP'],r['TN'],r['FP'],r['FN']),(2,1,2,1))
        self.assertFalse(set(r)&{'balanced_accuracy','BA'})
    def test_undefined_truth(self):
        r=E.classification([0,0],[.1,.9],[False,True]);self.assertTrue(np.isnan(r['auroc']) and np.isnan(r['auprc']) and np.isnan(r['f1']))
        self.assertTrue(all(np.isnan(v) for v in E.ranking([0,0],[.1,.9],['x','y']).values()))
    def test_undefined_prediction(self):
        r=E.classification([0,1],[.1,.2],[False,False]);self.assertTrue(np.isnan(r['precision']) and np.isnan(r['mcc']));self.assertEqual(r['f1'],0.)
    def test_ranking_ties(self):
        r=E.ranking([0,1,0,1],[.5,.5,.1,.9],['b','a','d','c'])
        self.assertEqual(r['mrr'],1.);self.assertEqual(r['hit_at_1'],1.);self.assertAlmostEqual(r['precision_at_3'],2/3);self.assertEqual(r['recall_at_3'],1.)
    def test_ece_edges(self):
        r=E.classification([0,1,0,1],[0,.1,.9,1],[False,False,True,True]);self.assertAlmostEqual(r['ece'],.45)
    def test_threshold_is_patient_equal(self):
        y=np.array([0,1,0,1,1,0]);p=np.array([.2,.8,.4,.3,.9,.7]);pid=np.array(['a','a','b','b','b','b'])
        r=E.threshold(y,p,pid);self.assertAlmostEqual(r['threshold']/.005,round(r['threshold']/.005))
        repeat=np.array([0,1,0,1,0,1,2,3,4,5]);rr=E.threshold(y[repeat],p[repeat],pid[repeat]);self.assertEqual(r,rr)
    def test_train_only_scale(self):
        x=np.arange(70,dtype=float).reshape(5,14);tr=np.array([0,1,2]);z,pre=M.preprocessing(x,tr,True);xx=x.copy();xx[3:]+=100000
        zz,pp=M.preprocessing(xx,tr,True);np.testing.assert_array_equal(pre['scaler'].mean_,pp['scaler'].mean_);np.testing.assert_array_equal(z[tr],zz[tr])
    def test_patient_equal_weights(self):
        w=M.patient_weights(np.array([0,1,0,0,1]),np.array(['a','a','b','b','b']),False)
        self.assertAlmostEqual(w[:2].sum(),w[2:].sum())

if __name__=='__main__':unittest.main()
