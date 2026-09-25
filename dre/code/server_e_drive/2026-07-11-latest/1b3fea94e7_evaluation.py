from ..mosaic.evaluation import metric_row,bootstrap_ci
def selection_score(metrics):return float(metrics['auroc']-.10*metrics['brier'])
__all__=['metric_row','bootstrap_ci','selection_score']
