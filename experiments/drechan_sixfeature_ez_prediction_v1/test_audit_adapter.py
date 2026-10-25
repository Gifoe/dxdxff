import unittest
from pathlib import Path
import hashlib
from audit_adapter import adapt,OLD,NEW,RUN_SHA


class AuditAdapterTests(unittest.TestCase):
    def test_exact_three_metadata_sites_and_reverse_equivalence(self):
        path=Path(__file__).with_name('run.py')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),RUN_SHA)
        source=path.read_text(encoding='utf-8');changed=adapt(source)
        self.assertEqual(changed.replace(NEW,OLD),source)
        meta=dict(C=.01,fold=1);pack=dict(fit=dict(C=.01,warnings=['ExampleWarning'],iterations=7),seconds=1.25)
        with self.assertRaises(TypeError):eval(OLD,{},dict(meta=meta,pack=pack))
        result=eval(NEW,{},dict(meta=meta,pack=pack))
        self.assertEqual(result,dict(C=.01,fold=1,iterations=7,seconds=1.25,warnings='ExampleWarning'))


if __name__=='__main__':unittest.main()
