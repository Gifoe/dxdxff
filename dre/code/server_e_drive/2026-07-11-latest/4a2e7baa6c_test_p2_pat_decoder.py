import unittest
import numpy as np
from neuroez_c.p2_pat_decoder import P2PATDecoder,probability_to_logit,logit_to_probability


class PATDecoderTests(unittest.TestCase):
    def test_probability_logit_roundtrip(self): self.assertAlmostEqual(logit_to_probability(probability_to_logit(.425)),.425)
    def test_requires_fit(self):
        with self.assertRaises(RuntimeError): P2PATDecoder(1,.5,.5).predict_residual([[0.]])
    def test_scaler_fit_on_given_rows(self):
        decoder=P2PATDecoder(1,.5,.5).fit(np.array([[0.],[2.]]),np.array([0.,1.])); self.assertAlmostEqual(decoder.scaler.mean_[0],1.)
    def test_shrinkage(self):
        decoder=P2PATDecoder(.1,.25,.75).fit(np.array([[0.],[1.],[2.]]),np.array([0.,1.,2.])); self.assertLessEqual(abs(decoder.predict_residual([[10.]])[0]),.25*.75+1e-12)
    def test_clip(self):
        decoder=P2PATDecoder(.1,1.,.25).fit(np.array([[0.],[1.],[2.]]),np.array([0.,10.,20.])); self.assertLessEqual(abs(decoder.predict_residual([[100.]])[0]),.25+1e-12)
    def test_logit_space_addition(self):
        decoder=P2PATDecoder(1,.5,.5).fit(np.array([[0.],[1.],[2.]]),np.array([0.,0.,0.])); threshold,residual=decoder.predict_threshold_logit([[1.]],np.array([-.3])); self.assertAlmostEqual(threshold[0],-.3+residual[0])
