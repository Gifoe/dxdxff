"""Static and synthetic contract for the added B1-only authorization path."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path(__file__).parents[1];p=json.loads((root/'PROTOCOL_AMENDMENT.json').read_text())
    assert p['new_outer_models']==['B1'] and not p['B2_B3_outer_authorized']
    assert not any(p[k] for k in ['training','checkpoint_modification','threshold_reselection','feature_reextraction'])
    assert p['PMA_gate_unchanged'] and sum(p['test_channels_by_fold'])==7635 and sum(p['test_patients_by_fold'])==80
    text=(root/'code/run.py').read_text();assert 'trainer.train(' not in text and 'select_threshold(' not in text
    assert text.index('all_B1_scores_before_full_y=True')<text.index("y=f['y']")
    score=np.array([.2,.5,.9]);tau=.5
    assert (score<tau).tolist()==[True,False,False]
    assert 'canonical=np.lexsort' in text and 'stable' in text
    print('B1_OUTER_CONTRACT_PASS',flush=True)


if __name__=='__main__':main()
