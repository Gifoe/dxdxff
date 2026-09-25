import numpy as np
import pytest

from neuroez_c.task2.ngbr.schema import NGBRSchemaError,RawRunRecord,validate_target


def test_target_requires_inside_and_outside_with_identifiers():
    record=RawRunRecord("patient-x","c","seizure-y",np.ones((4,20)),250,10,["A1","A2","A3","A4"],np.ones(4,bool),np.ones(4,bool),0,.04)
    with pytest.raises(NGBRSchemaError,match="patient-x.*seizure-y"):validate_target(record)
