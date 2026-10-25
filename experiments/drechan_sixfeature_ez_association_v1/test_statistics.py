import unittest
import numpy as np
from scipy.stats import false_discovery_control
from statistics_core import effect,summarize,bh,permuted_effects


class Tests(unittest.TestCase):
    def test_protocol_utf8_and_fixed_coordinates(self):
        import json
        from pathlib import Path
        q=json.loads(Path(__file__).with_name('PROTOCOL.json').read_text(encoding='utf-8'))
        self.assertEqual(len(q['coordinates']),12)
        self.assertEqual([r['r14_index'] for r in q['coordinates']],[0,1,2,3,4,5,6,7,10,11,12,13])
        self.assertEqual(q['multiplicity']['family_tests'],12)

    def test_effect_sign_ties_missing_constants(self):
        self.assertEqual(effect([3,4,1,2],[1,1,0,0]),1.)
        self.assertEqual(effect([1,2,3,4],[1,1,0,0]),-1.)
        self.assertEqual(effect([1,1,1,1],[1,1,0,0]),0.)
        self.assertEqual(effect([1,2,2,3],[1,1,0,0]),-.75)
        self.assertEqual(effect([np.nan,3,1],[0,1,0]),1.)
        self.assertTrue(np.isnan(effect([np.nan,3],[0,1])))

    def test_zero_effect_and_zero_aware_proportions(self):
        r=summarize([0,0,np.nan]);self.assertEqual(r['pvalue'],1.)
        self.assertEqual(r['ci_low'],0.);self.assertEqual(r['zero_percent'],100.)
        r=summarize([-1,0,1,np.nan]);self.assertEqual(r['valid_patients'],3)
        self.assertAlmostEqual(r['positive_percent'],100/3)
        self.assertAlmostEqual(r['positive_percent']+r['negative_percent']+r['zero_percent'],100.)
        self.assertTrue(np.isnan(summarize([np.nan])['pvalue']))

    def test_BH_fixed_twelve_family(self):
        p=np.linspace(.001,.8,12)
        np.testing.assert_allclose(bh(p),false_discovery_control(p))
        self.assertTrue(np.array_equal(bh(np.ones(12)),np.ones(12)))

    def test_independent_tied_signed_rank(self):
        from verify_analysis import signed_rank_p
        from scipy.stats import wilcoxon
        for v in [[0,0,0],[1,2,-1,0],[1,1,-2,-2,0,0],list(np.random.default_rng(42).normal(size=80))]:
            p,stat=signed_rank_p(v)
            if np.any(np.asarray(v)!=0):
                q=wilcoxon(v,zero_method='wilcox',correction=True,method='approx')
                self.assertAlmostEqual(p,q.pvalue,places=13);self.assertAlmostEqual(stat,q.statistic)
            else:self.assertEqual((p,stat),(1.,0.))

    def test_permutation_preserves_counts_and_joint_coordinates(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(17,12));y=np.arange(17)<4
        labels=rng.permuted(np.broadcast_to(y,(30,len(y))),axis=1)
        r=permuted_effects(x,y,labels)
        for b in range(30):
            for j in range(12):self.assertAlmostEqual(r[b,j],effect(x[:,j],labels[b]))
        x[:2,2]=np.nan;x[:,3]=0
        r=permuted_effects(x,y,labels)
        for b in range(30):self.assertAlmostEqual(r[b,2],effect(x[:,2],labels[b]));self.assertEqual(r[b,3],0.)


if __name__=='__main__':unittest.main()
