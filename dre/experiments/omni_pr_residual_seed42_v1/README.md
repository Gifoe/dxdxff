# Omni PR-Residual seed42

Omni-iEEG Task 2 only. The frozen official-style CNN is reused at checkpoint
SHA-256 `442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852`.
Only a 1,761-parameter residual head is trained on frozen 32-D embeddings.
Private waveforms, embeddings, checkpoints, channel/patient predictions, and
runtime logs are deliberately excluded from Git.

The exact commands and private paths are recorded on the execution host. Code
in `code/` is resume-safe for embedding extraction and hard-gates the baseline,
checkpoint hashes, train/validation split, and pre-test model freeze.

Final primary result: PR-Residual AUROC `0.798489`, below the exact frozen CNN
replay `0.798767` and ABS-only `0.798727`; terminal `FAIL`. Per the user's
stop instruction, unfinished secondary bootstrap/center/shuffle diagnostics
were not continued after the primary no-gain result.
