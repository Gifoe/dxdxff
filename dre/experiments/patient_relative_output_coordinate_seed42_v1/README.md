# Patient-relative output coordinate (seed 42)

Validation-only, frozen-A1 analysis. The experiment compares raw A1 logits (C0), patient-median-centered logits (C1), and patient-median/MAD robust-z logits (C2). It does not retrain A1 or inspect current outer-test data.

The public `PROTOCOL_LOCK.json` fixes the transformations, source hashes, VLOO selection, rank audit, and development gate before validation analysis. `code/reproduce_source.py` first rebuilds 150 frozen-checkpoint validation predictions in a private runtime and verifies published A1 VLOO. Only after that passes may `code/analyze_coordinates.py` compute C1/C2 results. Patient-level logits, labels, predictions, checkpoints and CSVs remain in the private server runtime; this directory contains aggregates only.

The rank audit failed the pre-registered metric-invariance tolerance, so the experiment stopped before development candidate selection. The terminal status is in `FINAL_REPORT.md`; no `development/` candidate files or frozen outer-test manifest were published. The private validation logits and intermediate patient-level diagnostics remain on the server.
