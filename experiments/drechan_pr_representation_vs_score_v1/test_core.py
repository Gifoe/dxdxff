import unittest
import numpy as np
from math_core import StreamingMoments,fit_feature,query_feature,score_transform,preserved_eval,normalize_tensor

class CoreTests(unittest.TestCase):
    def test_held_labels_and_global_input_separation(self):
        x=np.arange(60,dtype=float).reshape(20,3);p=np.repeat(['a','b','c','d'],5);fit=np.arange(10)
        z,pre=fit_feature(x,p,fit,'CENTER');changed=x.copy();changed[10:]+=99
        _,pre2=fit_feature(changed,p,fit,'CENTER')
        np.testing.assert_array_equal(pre['mu'],pre2['mu']);np.testing.assert_array_equal(pre['unit'],pre2['unit'])
        np.testing.assert_array_equal(z,query_feature(x,p,pre))
    def test_zero_and_output_order(self):
        x=np.array([-3.,0.,2.,2.,6.,7.,7.,7.]);p=np.array(['a']*5+['b']*3)
        for c in ['SCORE_CENTER','SCORE_Z']:
            z=score_transform(x,p,c);self.assertTrue(np.isfinite(z).all());np.testing.assert_array_equal(z[5:],0.)
            self.assertEqual(np.argsort(x[:5],kind='stable').tolist(),np.argsort(z[:5],kind='stable').tolist())
    def test_location_scale_definitions(self):
        from math_core import feature_view
        x=np.array([[1.,3.],[5.,3.],[2.,7.],[6.,7.]])
        pid=np.array(['a','a','b','b']);mu=np.array([2.,1.]);unit=np.array([2.,4.])
        center=feature_view(x,pid,'CENTER',mu,unit,True)
        np.testing.assert_array_equal(center,np.array([[-1.,0.],[1.,0.],[-1.,0.],[1.,0.]]))
        scale=feature_view(x,pid,'SCALE',mu,unit,True)
        np.testing.assert_array_equal(scale,np.array([[-.5,0.],[1.5,0.],[0.,0.],[2.,0.]]))
    def test_streaming_population(self):
        x=np.random.default_rng(1).normal(size=(91,7));x[:,0]=2;s=StreamingMoments()
        for a in np.array_split(x,9):s.add(a)
        q=s.result();self.assertEqual(q['n'],91);np.testing.assert_allclose(q['mean'],x.mean(0),atol=1e-15);np.testing.assert_allclose(q['sd'],x.std(0),atol=1e-15)
    def test_eval_state_and_gradients(self):
        import torch
        torch.manual_seed(3);m=torch.nn.Sequential(torch.nn.BatchNorm1d(4),torch.nn.Dropout(.3),torch.nn.Linear(4,2));m.train()
        before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
        with preserved_eval(m):m(torch.randn(6,4));np.random.normal();torch.randn(4)
        self.assertTrue(m.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(before[k],v))
        x=torch.randn(8,4,requires_grad=True);stats={'mean':np.arange(4,dtype=float),'sd':np.array([0.,1.,2.,3.])}
        for c in ['GLOBAL','CENTER','SCALE']:
            z=normalize_tensor(x,c,stats);self.assertTrue(torch.isfinite(z).all());g=torch.autograd.grad(z.square().mean(),x,retain_graph=True)[0];self.assertTrue(torch.isfinite(g).all())

if __name__=='__main__':unittest.main()
