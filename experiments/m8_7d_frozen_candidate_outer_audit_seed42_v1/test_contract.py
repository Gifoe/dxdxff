"""Only new critical paths: complete arm contract, cache identity and resampling."""
import tempfile
import unittest
from common import *

class Contract(unittest.TestCase):
    def test_complete_fixed_contract(self):
        p=jr(HERE/'PROTOCOL_LOCK.json')
        self.assertEqual(p['arms'],ARMS);self.assertEqual(len(NEW),10)
        self.assertEqual(len(PAIRS),15);self.assertEqual(len(set(PAIRS)),15)
        self.assertEqual(p['metrics'],METRICS);self.assertEqual(sum(p['cohort']['fold_patients']),80)
        self.assertEqual(len(FILES),len(set(FILES)))

    def test_cache_identity_and_tamper(self):
        ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='contract_',dir=ROOT) as temp:
            path=Path(temp)/'test.pkl';q=dict(binding={'freeze':'a','checkpoint':'b'},score_nez=np.array([.2,.4],np.float32))
            save(path,q);self.assertEqual(read(path)['binding'],q['binding'])
            with path.open('ab') as f:f.write(b'tamper')
            with self.assertRaises(AssertionError):read(path)
        a=cache_binding('f',{'checkpoint_sha256':'c'},np.arange(2),np.ones((2,14),np.float32),torch.ones(7))
        b=cache_binding('g',{'checkpoint_sha256':'c'},np.arange(2),np.ones((2,14),np.float32),torch.ones(7))
        self.assertNotEqual(a,b)

    def test_paired_interval_order(self):
        a=np.arange(720,dtype=float).reshape(80,9)/720;b=a+.03
        draw=np.random.default_rng(42).integers(0,80,(10000,80))
        mean,ci=intervals(b-a,draw)
        self.assertLess(np.max(np.abs(mean-.03)),1e-12);self.assertLess(np.max(np.abs(ci-.03)),1e-12)
        mean,ci=intervals(a,draw)
        independent=np.stack([a[:,j][draw].mean(1) for j in range(9)],axis=1)
        self.assertLess(np.max(np.abs(ci-np.quantile(independent,[.025,.975],axis=0))),1e-12)

if __name__=='__main__':unittest.main()
