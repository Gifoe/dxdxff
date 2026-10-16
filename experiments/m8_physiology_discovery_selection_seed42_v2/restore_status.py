"""Restore a shared public progress field from immutable main-model evidence."""
from common import *


def main():
    assert jr(PUB/'SUPPORT30_STATUS.json')['status']=='ALL5_SUPPORT_MATCHED_RAW_MODELS_FROZEN'
    frozen=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');rows=frozen['selections'];assert len(rows)==80
    for r in rows:assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    hist=pd.read_csv(PUB/'NEURAL_HISTORY.csv');assert len(hist)==sum(r['epochs_run'] for r in rows if r['alias_of'] is None)
    for name,value in frozen['binding']['source'].items():assert sha(HERE/name)==value
    write(PUB/'NEURAL_STATUS.json',dict(status='ALL80_LOGICAL_MODELS_FROZEN',logical_cells=80,
        actual_models=sum(r['alias_of'] is None for r in rows),epochs=len(hist),
        model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl'),before_Outer=True,source=frozen['binding']['source']))
    print('SHARED_PROGRESS_FIELD_RESTORED_FROM_ALL80_FROZEN_MODELS',flush=True)


if __name__=='__main__':main()
