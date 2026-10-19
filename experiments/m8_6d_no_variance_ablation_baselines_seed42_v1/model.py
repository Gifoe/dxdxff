"""Compose each approved switch with an irreversible explicit Variance mask."""
import importlib.util
from pathlib import Path
import sys
PARENT=Path(r'D:\m8_7d_f0_complete_ablation_seed42_v1')
spec=importlib.util.spec_from_file_location('sixd_original_math',PARENT/'model.py')
BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=BASE;spec.loader.exec_module(BASE)
ARMS=["FULL_F0","WO_EARLY_FILM","WO_LATE_GATE","WO_BOTH_GUIDANCE","WO_BOTH_PR","WO_RAW_PRE_REFERENCE","WO_HEAD_INTERACTION","EARLY_FILM_MISMATCH","RAW_ONLY","PHYSIOLOGY_ONLY","DROP_BETA","DROP_LOW_GAMMA","DROP_PARTIAL_HIGH_GAMMA","DROP_RMS","DROP_LINE_LENGTH","DROP_SPECTRAL_ENTROPY"]
CLASSIFIERS=["LogisticRegression","Linear_SVM","RBF_SVM","RandomForest","XGBoost","MLP","AdaBoost"]
TABULAR_ARMS=["LogisticRegression_NO_PR","LogisticRegression_PR","Linear_SVM_NO_PR","Linear_SVM_PR","RBF_SVM_NO_PR","RBF_SVM_PR","RandomForest_NO_PR","RandomForest_PR","XGBoost_NO_PR","XGBoost_PR","MLP_NO_PR","MLP_PR","AdaBoost_NO_PR","AdaBoost_PR"]
MAP={a:('DROP_VARIANCE' if a=='FULL_F0' else 'WO_RAW_PR' if a=='WO_BOTH_PR' else a) for a in ARMS}
class Ablation(BASE.Ablation):
    def __init__(self,source,arm):
        assert arm in ARMS
        super().__init__(source,MAP[arm])
        # Raw-only never reads physiology; retain its exact historical state/buffer for verified reuse.
        if arm!='RAW_ONLY':self.keep7[4]=0
        self.arm=arm
channel_mapping=BASE.channel_mapping
objective=BASE.objective

