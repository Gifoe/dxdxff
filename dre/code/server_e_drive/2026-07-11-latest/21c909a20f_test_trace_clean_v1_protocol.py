from pathlib import Path
from neuroez_c.task2.trace_rawmil.losses import rank_weight

RUNNER=(Path(__file__).parents[2]/'scripts/task2/run_trace_rawmil_clean_v1.py').read_text()

def test_exact_clean_v1_rank_schedule():
 expected=[0,0,0,0,0,.01,.02,.03,.04,.05,.06,.07,.08,.09,.10]
 assert all(abs(rank_weight(epoch)-value)<1e-12 for epoch,value in enumerate(expected,1))

def test_no_scheduler_and_auroc_checkpoint_is_loaded():
 assert 'torch.optim.lr_scheduler' not in RUNNER and 'ReduceLROnPlateau' not in RUNNER
 assert "best_val_auroc_ema.pt" in RUNNER and "early_stopping_monitor':'validation_auroc'" in RUNNER

def test_autotune_probe_is_discarded_before_formal_model():
 assert 'probe=TRACERawMIL().to(device)' in RUNNER
 assert 'del probe' in RUNNER and 'seed_all(fold_seed);val_hash' in RUNNER
