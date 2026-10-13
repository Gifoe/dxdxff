"""Exactly five raw-only folds through original A0 trainer."""
from audit_sources import arguments
from training import train_stage
if __name__=='__main__':train_stage(arguments(),raw=True)
