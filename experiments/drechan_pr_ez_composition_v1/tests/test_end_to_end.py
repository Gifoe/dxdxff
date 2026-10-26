"""Synthetic-only integration; no real inputs or published results are used."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_composition import fixture
import core
import stats_report
import independent

class EndToEndTests(unittest.TestCase):
    def test_all_tables_and_independent_intervals_synthetic_only(self):
        rows=[];verified=[]
        for i,source in enumerate(np.repeat(core.SOURCES,2)):
            p=fixture(seed=100+i);y=np.array([1,1,0,0,0,0,0]);f=core.freeze_patient(p)
            a=core.evaluate_patient(p,f,y);b=independent.replay(p,y,f)
            rows.append(dict(a,source=source,patient=f'SYNTHETIC_ONLY_{i}',channels=7,class_counts=np.array([2,5])))
            verified.append(dict(b,source=source,patient=f'SYNTHETIC_ONLY_{i}',channels=7,class_counts=np.array([2,5])))
        tables=stats_report.summarize(rows);check=independent.independent_table_check(verified,tables)
        self.assertEqual(check['primary_family_intervals'],2);self.assertEqual(check['secondary_family_intervals'],11);self.assertEqual(check['permutation_values'],20000)
        expected={'E1_CLASS_LOSS':650,'E1_GAIN':650,'E1_H':325,'E2_LOSS':3900,'E2_GAIN':300,'E2_H':1300,'E2_ANOMALY':1040,'E2_SUPPORT':520,'REFERENCE_COMPOSITION':80,'COMPOSITION_SLOPE':20,'PRIMARY':2,'SECONDARY':11,'H_PERMUTATION':20000,'H_PERMUTATION_SUMMARY':20}
        self.assertEqual({k:len(v) for k,v in tables.items()},expected)
        for row in tables['COMPOSITION_SLOPE']:
            if row['fraction']==1.:self.assertEqual(row['status'],'not_estimable');self.assertEqual(row['mean'],'')
        self.assertTrue(all(r['family_low']!='' for r in tables['PRIMARY']))

if __name__=='__main__':unittest.main()
