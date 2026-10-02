# Frozen representation source

R4 is the 32-D output of the frozen official CNN `cnn` block. P16 is the true 16-D tensor after frozen `fc1 → relu1 → bn1` and immediately before `fc_out`. Both were formed by the predeclared plain mean over segments within `(EDF, channel)`.

The extraction read pre-existing 60-s feature NPZs, not EDF files. It did not train, fine-tune, mutate, or replace the frozen checkpoint. Private cache entries, channel identities, and patient identities are excluded from this repository.
