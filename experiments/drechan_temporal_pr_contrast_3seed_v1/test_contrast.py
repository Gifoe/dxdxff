"""Independent brute-force robust references and mask/gradient boundary tests."""
import unittest
import numpy as np
import torch
from contrast import *

class Tests(unittest.TestCase):
    def test_exact_median_MAD_odd_even_missing_ties(self):
        rng=np.random.default_rng(42)
        for C in [1,2,3,4,5,6,17]:
            a=rng.normal(size=(C,7,5)).astype(np.float32);a[...,0]=np.round(a[...,0]);a[...,1]=2
            support=rng.random((C,7))>.2;support[:,0]=False
            mu,md,ok=loo_reference(torch.tensor(a),torch.tensor(support))
            for c in range(C):
                for w in range(7):
                    others=[j for j in range(C) if j!=c and support[j,w]]
                    if not support[c,w] or len(others)<MIN_OTHERS:
                        self.assertFalse(ok[c,w].any());continue
                    center=np.median(a[others,w],axis=0);disp=np.median(np.abs(a[others,w]-center),axis=0)
                    np.testing.assert_allclose(mu[c,w],center,rtol=0,atol=1e-7)
                    np.testing.assert_allclose(md[c,w],disp,rtol=0,atol=1e-7)
                    np.testing.assert_array_equal(ok[c,w],disp>MAD_FLOOR)

    def test_self_exclusion_and_invalid_values(self):
        x=torch.arange(7*4*3,dtype=torch.float32).reshape(7,4,3);mask=torch.ones(7,4,dtype=torch.bool)
        a,b,o=loo_reference(x,mask);changed=x.clone();changed[2]+=1000
        c,d,p=loo_reference(changed,mask)
        self.assertTrue(torch.equal(a[2],c[2]));self.assertTrue(torch.equal(b[2],d[2]))
        mask[3,1]=False;a,b,o=loo_reference(x,mask);changed=x.clone();changed[3,1]=float('nan')
        c,d,p=loo_reference(changed,mask)
        self.assertTrue(torch.equal(a,c));self.assertTrue(torch.equal(b,d));self.assertTrue(torch.equal(o,p))

    def test_relative_fallback_and_recipient_gradient(self):
        r=torch.randn(8,4,3,requires_grad=True);mask=torch.ones(8,4,dtype=torch.bool)
        m,d,o=loo_reference(r,mask);self.assertFalse(m.requires_grad);self.assertFalse(d.requires_grad)
        q=relative(r,m,d,o);q.square().sum().backward();self.assertTrue(torch.isfinite(r.grad).all());self.assertGreater(float(r.grad.norm()),0)
        zero=torch.zeros_like(r);a,b,o=loo_reference(zero,mask);self.assertTrue(torch.equal(relative(zero,a,b,o),zero))

    def test_shift_full_support_and_seed_independence(self):
        x=torch.randn(8,7,24);mask=torch.ones(8,7,dtype=torch.bool);mask[0,2]=False
        t=torch.tensor([-4.,-2.,0.,2.,4.,6.,8.]);m,d,o=loo_reference(x,mask)
        a,b,c,counts=shifted_reference(m,d,o,mask,t,1,0)
        self.assertEqual(counts['time_shifted_valid_windows']+counts['time_singleton_valid_windows'],counts['post_valid_windows'])
        self.assertGreater(counts['time_shifted_valid_windows'],0);self.assertTrue(torch.equal(a[:,2],m[:,2]))
        q=shifted_reference(m,d,o,mask,t,1,0);self.assertTrue(torch.equal(a,q[0]))

    def test_fusion_identity_matching_and_live_path(self):
        rng=torch.random.get_rng_state().clone();a=Fusion(71);b=Fusion(71)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        r=torch.randn(8,5,24,requires_grad=True);q=torch.randn_like(r)
        self.assertTrue(torch.equal(a(r,q),r));opt=torch.optim.AdamW(a.parameters(),lr=.0003)
        for step in range(2):
            opt.zero_grad(set_to_none=True);loss=a(r,q).square().mean();loss.backward()
            self.assertGreater(float(a.network[2].weight.grad.norm()),0)
            if step:self.assertGreater(float(a.network[0].weight.grad.norm()),0)
            opt.step()
        self.assertFalse(torch.equal(a(r,q),a(r,torch.zeros_like(q))))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
