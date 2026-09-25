from neuroez_c.task2.trace_rawmil.trainer import batches
from neuroez_c.task2.trace_rawmil.schema import PatientSample
def test_patient_batches_keep_equal_units():
 p=[PatientSample(str(i),'x',i%2,[]) for i in range(6)];result=batches(p,4,2);keys=[x.patient_key for batch in result for x in batch];assert all(len(x)<=4 for x in result) and len(keys)==len(set(keys))==6 and set(keys)=={str(i) for i in range(6)}
def test_88_patients_make_exactly_22_batches_without_replacement():
 p=[PatientSample(str(i),'x',i%2,[]) for i in range(88)];result=batches(p,4,42);keys=[x.patient_key for batch in result for x in batch];assert len(result)==22 and len(keys)==len(set(keys))==88
def test_imbalanced_classes_still_fill_batches_without_oversampling():
 p=[PatientSample(str(i),'x',int(i<12),[]) for i in range(88)];result=batches(p,4,42);keys=[x.patient_key for batch in result for x in batch];assert len(result)==22 and len(keys)==len(set(keys))==88
