import math
import unittest
import pandas as pd
from neuroez_c.p2_q10_lzu_patient_batch import PatientAdapterDataset,PatientBatchSampler,collate_complete_patients
from tests.test_p2_q10_lzu_adapter_protocol import frame


def patients(count=5):
    rows=[]
    for index in range(count):
        current=frame().copy(); current["subject_id"]=f"lzu:s{index}"; current["channel_name"]=[f"A{index}",f"B{index}"]
        if index%2: current=pd.concat([current,current.iloc[[0]].assign(channel_name=f"C{index}")],ignore_index=True)
        rows.append(current)
    return pd.concat(rows,ignore_index=True)


class PatientBatchTests(unittest.TestCase):
    def test_complete_patient_stays_together(self):
        dataset=PatientAdapterDataset(patients()); batch=collate_complete_patients([dataset[0],dataset[1]])
        self.assertEqual(set(batch["patient_index"].tolist()),{0,1}); self.assertEqual(batch["n_batch_channels"],len(dataset[0].channel_names)+len(dataset[1].channel_names))
    def test_each_patient_once_per_epoch(self):
        dataset=PatientAdapterDataset(patients()); indices=[index for batch in PatientBatchSampler(dataset,2,global_seed=42,outer_fold=1,epoch=1) for index in batch]
        self.assertEqual(sorted(indices),list(range(len(dataset))))
    def test_patient_not_split_across_batches(self):
        dataset=PatientAdapterDataset(patients()); batches=list(PatientBatchSampler(dataset,2,global_seed=42,outer_fold=1,epoch=1)); self.assertEqual(len({index for batch in batches for index in batch}),len(dataset))
    def test_batch_size_two_is_effective(self):
        batches=list(PatientBatchSampler(PatientAdapterDataset(patients()),2,global_seed=42,outer_fold=1,epoch=1)); self.assertTrue(all(len(batch)<=2 for batch in batches))
    def test_optimizer_step_count(self):
        dataset=PatientAdapterDataset(patients()); self.assertEqual(len(PatientBatchSampler(dataset,2,global_seed=42,outer_fold=1,epoch=1)),math.ceil(len(dataset)/2))
    def test_last_batch_not_dropped(self):
        batches=list(PatientBatchSampler(PatientAdapterDataset(patients()),2,global_seed=42,outer_fold=1,epoch=1)); self.assertEqual(len(batches[-1]),1)
    def test_same_seed_same_batches(self):
        dataset=PatientAdapterDataset(patients()); a=list(PatientBatchSampler(dataset,2,global_seed=42,outer_fold=1,epoch=1)); b=list(PatientBatchSampler(dataset,2,global_seed=42,outer_fold=1,epoch=1)); self.assertEqual(a,b)
    def test_different_epoch_changes_order(self):
        dataset=PatientAdapterDataset(patients(8)); a=list(PatientBatchSampler(dataset,2,global_seed=42,outer_fold=1,epoch=1)); b=list(PatientBatchSampler(dataset,2,global_seed=42,outer_fold=1,epoch=2)); self.assertNotEqual(a,b)
