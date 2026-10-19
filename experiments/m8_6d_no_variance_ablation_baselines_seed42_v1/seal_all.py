"""Join both score ledgers only after all150 frozen prediction cells are sealed."""
from common import *
def main():
    fr=frozen();fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');files={}
    for name,n in [('NEURAL_SCORE_SEAL_PRIVATE.json',80),('TABULAR_SCORE_SEAL_PRIVATE.json',70)]:
        q=jr(ROOT/name);assert q['freeze']==fh and len(q['files'])==n
        for key,h in q['files'].items():
            assert key not in files and sha(ROOT/'scores_private'/(key+'.pkl'))==h;files[key]=h
    assert len(files)==150 and len(all_cells())==150
    freeze(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json',dict(freeze=fh,files=files,all150_cells_sealed_before_metric_reporting=True))
    write(PUB/'PREDICTION_STATUS.json',dict(status='ALL150_OUTER_CELLS_SEALED',groups=30,cells=150,reused_cells=10,new_neural_cells=70,new_tabular_cells=70,
        freeze_sha256=fh,score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')))
    print('ALL150_SIXD_OUTER_CELLS_SEALED_BEFORE_REPORTING',flush=True)
if __name__=='__main__':main()

