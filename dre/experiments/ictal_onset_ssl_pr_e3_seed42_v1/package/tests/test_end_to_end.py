import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from run_experiment import PatientFiles,run_ssl,train_one,score_patient,metric_one


def test_tiny_training_ssl_pr_four_arms(tmp_path):
    pdir=tmp_path/'patients';pdir.mkdir()
    cohort={}
    rng=np.random.default_rng(42)
    for i in range(3):
        # Onset increases power for the two labelled EZ contacts.
        pair=rng.standard_normal((1,4,2,2500)).astype('float32')
        pair[:,:,1]*=np.array([2.0,2.0,1.0,1.0],dtype='float32')[None,:,None]
        present=np.ones((1,4),dtype=bool)
        labels=np.array([1,1,0,0],dtype='float32')
        np.savez_compressed(pdir/f'{i}.npz',pair=pair,present=present,labels_ez=labels)
        cohort[str(i)]={'file':f'patients/{i}.npz','center':'testcenter'}
    mf={'cohort':cohort};store=PatientFiles(tmp_path,mf,'cpu')
    file,hist=run_ssl(1,store,['0','1'],tmp_path,1,42)
    assert file.is_file() and len(hist)==1
    for name in ('E0','E1','E2','E3'):
        chosen,trace=train_one(1,name,store,['0','1'],['2'],tmp_path,1,52,
                               ssl_file=file)
        assert Path(chosen['checkpoint']).exists() and len(trace)==1
        assert 0.05<=chosen['threshold']<=0.95
        assert 0<=chosen['val_metrics']['macro_f1']<=1
        raw=score_patient(torch_model(chosen,store,name),store,'2')
        assert len(raw['labels_ez'])==4


def torch_model(chosen,store,name):
    from model import IctalLocalization
    from run_experiment import VARIANTS
    _,pr,att=VARIANTS[name]
    model=IctalLocalization(pr,att)
    model.load_state_dict(torch.load(chosen['checkpoint'],weights_only=True,map_location='cpu')['state_dict'])
    return model
