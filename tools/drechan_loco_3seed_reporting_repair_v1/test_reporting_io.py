"""Release-only I/O regression test with no patient data."""
import ast
import csv
from pathlib import Path
import tempfile
import types
import unittest
import reporting_entry as R
REPO=Path(__file__).resolve().parents[2]

class Tests(unittest.TestCase):
    def test_real_writer_resolves_common_not_pipeline_and_is_restored(self):
        tree=ast.parse((REPO/'experiments/drechan_loco_five_models_3seed_v1/common.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='csv_out')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=types.ModuleType('fake_common');c.PUB=root/'seed43';c.csv=csv
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_csv_writer','exec'),c.__dict__)
            m=types.SimpleNamespace(GLOBAL_PUB=root/'public')
            p=types.SimpleNamespace(PUB=c.PUB,csv_out=c.csv_out)
            p.PUB=m.GLOBAL_PUB;p.csv_out('LEGACY',[{'n':1}])
            self.assertTrue((root/'seed43/LEGACY.csv').is_file())
            self.assertFalse((root/'public/LEGACY.csv').exists())
            R.install_route(c,m);m.export_csv('CORRECT',[{'n':2}])
            self.assertTrue((root/'public/CORRECT.csv').is_file())
            self.assertFalse((root/'seed43/CORRECT.csv').exists())
            self.assertEqual(c.PUB,root/'seed43')
            with self.assertRaises(AssertionError):m.export_csv('BAD',[{'a':1},{'b':2}])
            self.assertEqual(c.PUB,root/'seed43')
    def test_recover_only_actual_seed43_rows(self):
        rows=[dict(train_seed=str(s),arm='arm'+str(i//4),held_source=str(i%4),value=str(s+i)) for s in [42,43,44] for i in range(20)]
        selected=R.seed43_rows(rows)
        self.assertEqual(len(selected),20)
        self.assertTrue(all(int(r['value'])==43+i for i,r in enumerate(selected)))
        self.assertTrue(all('train_seed' not in r for r in selected))

if __name__=='__main__':unittest.main()
