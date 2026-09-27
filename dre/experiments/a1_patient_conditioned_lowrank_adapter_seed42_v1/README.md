# Frozen A1 patient-conditioned rank-four adapter

Run `prepare.py`, `train_adapters.py`, `diagnostics.py`, and `finalize.py` on the original server with private runtime variables. Source A1 remains frozen. Public outputs contain aggregate rows only; no patient-level records, embeddings, logits, teachers or checkpoints are committed.
