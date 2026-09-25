import pandas as pd
def test_outer_test_not_in_train():
 f=pd.DataFrame({'patient_key':['a','b','c'],'outer_fold':[1,1,2]});test=set(f[f.outer_fold==1].patient_key);train=set(f.patient_key)-test;assert not test&train
