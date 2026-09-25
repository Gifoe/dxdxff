import unittest
from scripts.run_p2_q10_lzu_adapter import _checkpoint_eligible,_selection_key


def metrics(loss=.5,f1=.6,auprc=.5,truek=.6,ezf1=.4,delta=.01):
    return {"val_total_loss":loss,"val_lzu_formal_macro_f1":f1,"val_lzu_ez_auprc":auprc,"val_lzu_truek_macro_f1":truek,"val_lzu_ez_f1":ezf1,"val_mean_abs_delta":delta}


class CheckpointTests(unittest.TestCase):
    def test_min_steps_blocks_checkpoint(self): self.assertFalse(_checkpoint_eligible(20,99,10,100))
    def test_min_epochs_blocks_checkpoint(self): self.assertFalse(_checkpoint_eligible(9,120,10,100))
    def test_both_requirements_enable_checkpoint(self): self.assertTrue(_checkpoint_eligible(10,100,10,100))
    def test_validation_loss_is_primary(self): self.assertLess(_selection_key(metrics(loss=.4,f1=.1),10),_selection_key(metrics(loss=.5,f1=.9),10))
    def test_f1_is_tie_break(self): self.assertLess(_selection_key(metrics(f1=.7),10),_selection_key(metrics(f1=.6),10))
    def test_earlier_epoch_final_tie_break(self): self.assertLess(_selection_key(metrics(),10),_selection_key(metrics(),11))
