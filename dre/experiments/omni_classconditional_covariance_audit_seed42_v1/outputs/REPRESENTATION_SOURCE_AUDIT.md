# Frozen representation source

R4 is the existing 32-D output of the frozen official CNN `cnn` block. P16 is the true 16-D tensor after the frozen `fc1 → relu1 → bn1` path and immediately before `fc_out`. Both are plain segment means within `(EDF, channel)`. TRAIN representations were regenerated only from the validated `omni_bag_mismatch_audit_seed42_v1` full-record artifact and native HDF5, not historical five-clip TRAIN NPZs. Private caches and identities are excluded from Git.
