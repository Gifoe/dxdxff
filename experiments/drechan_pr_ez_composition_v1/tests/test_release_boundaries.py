import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import io_contract
from release_check import check

class ReleaseBoundaryTests(unittest.TestCase):
    def test_reject_private_payload_and_wrong_whole_hash(self):
        with tempfile.TemporaryDirectory(prefix='PRCOMP_SYNTHETIC_RELEASE_') as tmp:
            p=Path(tmp)/'synthetic_invalid.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('patient_predictions.csv','SYNTHETIC_INVALID')
            with self.assertRaises(AssertionError):check(p,'0'*64)
            with self.assertRaises(AssertionError):check(p,io_contract.sha(p))
    def test_parent_fingerprint_contract(self):
        p=io_contract.protocol();self.assertEqual(len(p['parent_reference_sha256']),64);self.assertEqual(len(p['parent_endpoint_sha256']),64)
        self.assertNotIn('REFERENCES.pkl',io_contract.PAYLOADS);self.assertNotIn('ENDPOINTS.pkl',io_contract.PAYLOADS)

if __name__=='__main__':unittest.main()
