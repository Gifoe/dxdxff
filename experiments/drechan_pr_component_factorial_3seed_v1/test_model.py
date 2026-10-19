"""Changed PR paths, TRAIN-only preprocessing and original initialization."""
import unittest
from common import *

stub=module('prf_synthetic_helpers',PREVIOUS/'test_model.py')
sys.modules['common']=OWN;sys.modules['model']=MODEL

class Tests(unittest.TestCase):
    def test_shared_initialization_and_variance_invariance(self):
        src=stub.source();args=list(stub.batch());state=None
        for arm in ARMS:
            rng=torch.random.get_rng_state().clone();m=MODEL.PRFactorial(src,arm)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            if state is None:state=m.state_dict()
            self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in state.items()))
            y=m(*args)['logit'];changed=list(args);changed[0]=args[0].clone();changed[4]=args[4].clone()
            changed[0][:,8:10]=float('nan');changed[4][...,4]=float('nan')
            self.assertTrue(torch.equal(y,m(*changed)['logit']))
            self.assertFalse(m.interaction);self.assertTrue(m.pre_reference and m.early and m.late)

    def test_only_raw_pr_operation_differs(self):
        src=stub.source();args=stub.batch()
        full=MODEL.PRFactorial(src,'FULL');off=MODEL.PRFactorial(src,'WO_RAW_PR')
        zr,gr=off.features(*args[1:]);zz,gz=full.features(*args[1:]);x=zr.double()
        sd=x.std(0,unbiased=False);wanted=((x-x.mean(0))/torch.where(sd>1e-8,sd,torch.ones_like(sd))).float()
        self.assertTrue(torch.equal(gr,gz));self.assertTrue(torch.equal(zz,wanted))
        self.assertTrue(torch.equal(full(*args)['logit'],MODEL.BASE.Ablation(src,'FULL_F0')(*args)['logit']))

    def test_full_gradients_in_all_arms(self):
        args=stub.batch()
        for arm in ARMS:
            m=MODEL.PRFactorial(stub.source(),arm);m(*args)['logit'].square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            for n in ['cnn.stem.0.weight','head.0.weight']:
                self.assertGreater(float(dict(m.named_parameters())[n].grad.norm()),0)

    def test_no_feature_pr_fits_train_only(self):
        rng=np.random.default_rng(1);raw=rng.normal(size=(30,14));patients=np.repeat(np.arange(6),5);train=np.arange(20)
        raw[0,1]=np.nan;x,p=fit_no_feature_pr(raw,patients,train)
        changed=raw.copy();changed[20:]*=1000;y,q=fit_no_feature_pr(changed,patients,train)
        for k in p:self.assertTrue(np.array_equal(p[k],q[k]))
        self.assertTrue(np.array_equal(x[train],y[train]));self.assertEqual(set(p['fit_patient_ids']),{0,1,2,3})
        manual=StandardScaler();imp=SimpleImputer();imp.fit(raw[train]);manual.fit(imp.transform(raw[train]))
        self.assertTrue(np.array_equal(x,manual.transform(imp.transform(raw)).astype(np.float32)))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
