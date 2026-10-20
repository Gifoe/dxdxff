import unittest
import numpy as np
from sklearn.metrics import f1_score,balanced_accuracy_score
from audit_math import GRID,choose,leave_one,raw_stats,score_stats,legal_scope


class Tests(unittest.TestCase):
    def sample(self):
        rng=np.random.default_rng(42);p=np.repeat(np.arange(8),20);y=np.tile([0,1]*10,8)
        score=rng.choice([0.,.01,.23,.5,.775,.99,1.],len(p));return y,score,p

    def test_original_201_grid_and_ties_against_brute_sklearn(self):
        y,s,p=self.sample();rows=[]
        for t in GRID:
            v=s>=t;macro=np.mean([f1_score(y[p==q],v[p==q],labels=[0,1],average='macro',zero_division=0) for q in np.unique(p)])
            ez=np.mean([f1_score(1-y[p==q],1-v[p==q],zero_division=0) for q in np.unique(p)])
            rows.append((macro,ez,balanced_accuracy_score(y,v),-abs(float(t)-.5),-float(t),float(t)))
        winner=max(rows);actual=choose(y,s,p)
        self.assertEqual(winner[-1],actual['threshold']);self.assertAlmostEqual(winner[0],actual['macro_f1'],places=14)

    def test_held_CAL_labels_do_not_choose_their_threshold(self):
        y,s,p=self.sample();before=leave_one(y,s,p,.5)
        changed=y.copy();changed[p==3]=1-changed[p==3];after=leave_one(changed,s,p,.5)
        self.assertEqual(next(r['threshold'] for r in before if r['patient']==3),next(r['threshold'] for r in after if r['patient']==3))

    def test_label_free_raw_statistics_and_anchor_units(self):
        v=np.tile(np.array([-1.,1.])[:,None],(1,96));mu=np.zeros(96);scale=np.ones(96)
        q=raw_stats(v,mu,scale);self.assertEqual(q['mean_coordinate_rms'],0);self.assertEqual(q['population_sd_coordinate_rms'],1)
        self.assertEqual(q['abs_max'],1);self.assertEqual(q['fraction_abs_above5'],0)
        with self.assertRaises(AssertionError):raw_stats(v,mu,np.zeros(96))

    def test_constant_scores_finite_and_saturation_reported(self):
        q=score_stats(np.zeros(10),np.full(10,.5,np.float32),np.repeat([0,1],5))
        self.assertEqual(q['logit_population_sd'],0);self.assertEqual(q['probability_saturation_fraction'],0)
        self.assertEqual(q['unique_logit_fraction'],.1)
        q=score_stats(np.array([-100.,100.]),np.array([0.,1.],np.float32),np.array([0,1]));self.assertEqual(q['probability_saturation_fraction'],1)

    def test_phase_and_complete_patient_scope(self):
        p=np.repeat(np.arange(6),2);roles=dict(train={0,1},cal={2,3},assessment={4},test={5})
        legal_scope(p,np.arange(4),roles,'TRAIN');legal_scope(p,np.arange(4,8),roles,'CAL')
        for indices,phase in [(np.arange(8,10),'ASSESS'),(np.array([0,0,2,3]),'TRAIN'),(np.arange(6,10),'CAL')]:
            with self.assertRaises(AssertionError):legal_scope(p,indices,roles,phase)


if __name__=='__main__':unittest.main(verbosity=2)
