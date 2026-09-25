import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
from neuroez_c.evidence_views import WINDOW_NODE_FEATURE_NAMES
from task1_clean_nez.dataset import PHYSICS_FEATURE_NAMES

def test_synthetic_cpu_smoke(tmp_path: Path):
    names=[*WINDOW_NODE_FEATURE_NAMES,*PHYSICS_FEATURE_NAMES]; records=[]; index={}
    for pi in range(4):
        sid=f"test:p{pi}"; channels=["A1","A2","A3"]
        index[sid]={"canonical_channels":channels,"labels":np.array([0,1,0],np.float32),"label_mask":np.ones(3,bool),"outcome_group":"success","surgery_success":True,"source_center":"multicenter"}
        for si in range(2):
            x=np.random.default_rng(pi*10+si).normal(size=(6,3,len(names))).astype(np.float32)
            records.append({"subject_id":sid,"run_id":f"r{si}","task":"ictal","phase_group":"ictal","channel_names_norm":channels,"labels":index[sid]["labels"],"sample":{"sample_id":f"s{si}","window_features":x,"window_adjacency":np.zeros((6,3,3),np.float32),"window_relative_centers_sec":np.array([-2,-1,0,1,2,3],np.float32),"window_feature_names":names,"outcome_group":"success","surgery_success":True}})
    cache=tmp_path/"cache.pkl"; out=tmp_path/"out"
    with cache.open("wb") as f:pickle.dump({"run_records":records,"patient_index":index},f)
    cmd=[sys.executable,"run_task1_clean_nez.py","--window-cache-path",str(cache),"--output-dir",str(out),"--require-n-patients","4","--n-splits","2","--epochs","2","--patience","2","--min-epochs-before-early-stop","2","--batch-size","2","--device","cpu","--max-folds","1"]
    subprocess.run(cmd,cwd=Path(__file__).parents[1],check=True)
    for name in ("feature_audit.json","protocol_audit.json","oof_channel_predictions.csv","fold_metrics.csv","heldout_summary_task1_clean_nez.json","fold_1/best_checkpoint.pt","fold_1/training_history.csv","fold_1/validation_predictions.csv","fold_1/test_predictions.csv"):assert (out/name).exists()
