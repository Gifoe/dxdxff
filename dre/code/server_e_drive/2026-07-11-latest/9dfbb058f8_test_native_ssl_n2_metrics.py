import pandas as pd
from neuroez_c.task2.native_ssl_n2.evaluation import metric_row
def test_failure_direction():
 m=metric_row(pd.DataFrame({'outcome_true':[1,0],'probability_success':[.9,.1]})); assert m['auroc']==1 and m['failure_auprc']==1
