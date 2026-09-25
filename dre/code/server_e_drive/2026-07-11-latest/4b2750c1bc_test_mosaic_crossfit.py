import pandas as pd
from neuroez_c.task2.mosaic.crossfit import fit_expert

def test_expert_fit_rejects_held_out_patient():
    frame=pd.DataFrame({"patient_key":["a","b","c","d"],"center":["x"]*4,"outcome_true":[0,1,0,1],"f":[0.,1.,.1,.9]})
    fitted=fit_expert("recruitment",frame,forbidden_patients=["z"]); assert fitted.fit_patient_keys==("a","b","c","d")
    try: fit_expert("recruitment",frame,forbidden_patients=["a"])
    except ValueError: pass
    else: raise AssertionError("leakage was not rejected")
