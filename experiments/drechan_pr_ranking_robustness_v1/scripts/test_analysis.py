import unittest
import numpy as np
import analysis_math as m

class Semantics(unittest.TestCase):
    def test_ties_are_ap_groups_but_hits_use_frozen_channel_order(self):
        y=np.array([1,0,1,0]);s=np.array([.8,.8,.2,.2]);channels=np.array(['b','a','d','c'])
        a=m.metrics(y,s,s>.5,channels);b=m.metrics(y,s,s>.5,channels,True)
        self.assertEqual(a['auroc'],.5);self.assertEqual(a['auprc'],.5)
        self.assertEqual(a['hit_at_1'],0);self.assertEqual(a['hit_at_3'],1)
        for k in m.METRICS:self.assertAlmostEqual(a[k],b[k])
    def test_hit_is_not_recall_and_short_budget_is_clamped(self):
        y=np.array([1,0,1]);s=np.array([3.,2.,1.]);ch=['a','b','c']
        a=m.budget_values(y,s,ch,'fixed',1);b=m.budget_values(y,s,ch,'fixed',5)
        self.assertEqual(a['hit'],1);self.assertEqual(a['recall'],.5)
        self.assertEqual(b['review_channels'],3);self.assertEqual(b['precision'],2/3)
        self.assertEqual(m.budget_values(y,s,ch,'fraction',.05)['review_channels'],1)
    def test_missing_single_class_endpoints_are_explicit(self):
        a=m.metrics(np.zeros(3,int),np.arange(3),np.zeros(3,bool),['a','b','c'])
        self.assertTrue(np.isnan(a['auroc']));self.assertTrue(np.isnan(a['auprc']));self.assertTrue(np.isnan(a['mrr']))
        self.assertTrue(np.isfinite(a['macro_f1']))
    def test_no_ensemble_or_best_subset_in_sampling(self):
        self.assertEqual(len(m.seizure_subsets('hidden',8,2)),28)
        a=m.seizure_subsets('hidden',20,2);self.assertEqual(len(a),50);self.assertEqual(len(set(a)),50)
        self.assertEqual(a,m.seizure_subsets('hidden',20,2));self.assertEqual(m.seizure_subsets('hidden',8,8),[tuple(range(8))])
    def test_source_bootstrap_retains_counts(self):
        src=np.array(['hup']*4+['lzu']*3+['multicenter']*2+['pediatric'])
        d=m.draw(src)
        for s,n in [('hup',4),('lzu',3),('multicenter',2),('pediatric',1)]:self.assertTrue(np.all((src[d]==s).sum(1)==n))
        c=m.interval(np.arange(10)/10,src,6);self.assertLessEqual(c['adjusted_low'],c['low']);self.assertGreaterEqual(c['adjusted_high'],c['high'])

if __name__=='__main__':unittest.main()
