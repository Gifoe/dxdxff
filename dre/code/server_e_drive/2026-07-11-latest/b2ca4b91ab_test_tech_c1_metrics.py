import pandas as pd
from neuroez_c.task2.tech_outcome_c1.evaluation import metrics
def test_metric_directions():
 m=metrics(pd.DataFrame({'outcome_true':[0,1],'probability_success':[.1,.9]}));assert m['auroc']==1 and m['failure_auprc']==1 and m['specificity']==1
