"""Server-only launcher: preserve the OOF code while using its 16-channel eval path.

The original server module predates the finalized batch-16 trainer.  Rebinding
only ``patient_forward`` changes no checkpoint, data, score, aggregation, or
model mode: BatchNorm is in inference mode and channel batching is
mathematically independent.  It only avoids needless kernel-launch overhead.
"""
from __future__ import annotations

import evaluate_oof
from train_fold_r3 import patient_forward


evaluate_oof.patient_forward = patient_forward


if __name__ == "__main__":
    evaluate_oof.main()
