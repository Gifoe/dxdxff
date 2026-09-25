from neuroez_c.task2.trace_rawmil.trainer import batches
from neuroez_c.task2.trace_rawmil.schema import PatientSample
def test_patient_batches_keep_equal_units():
 p=[PatientSample(str(i),'x',i%2,[]) for i in range(6)];assert all(len(x)<=4 for x in batches(p,4,2))
