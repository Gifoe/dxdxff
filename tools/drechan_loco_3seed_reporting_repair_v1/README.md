# Reporting-only I/O bridge

The frozen scientific `orchestrate.export_csv` changed `pipeline.PUB`, but the
imported `csv_out` writer resolves its globals in `common`. Actual results were
therefore written to seed43/public instead of global public. Three combined60
metadata CSVs also replaced the corresponding20row public seed43 summaries.
No checkpoints, optimizer/RNG states, validation records or sealed scores were
changed. The original independent verifier correctly stopped at the missing
global MAIN_RESULTS.csv; it was not bypassed or given a relaxed tolerance.

This external bridge leaves ALL12 frozen scientific files unchanged. It first
preserves the misrouted aggregate files, copies the already-computed numeric
tables byte-for-byte, restores actual20row seed43 metadata from the immutable
selection ledger and existing replay/coverage rows, then redirects only the
writer's common.PUB destination while invoking the ORIGINAL independent
verification and packaging functions. All60 checkpoint, validation and score
bindings and the global freezes/seals are hashed before and after.

On the trusted server deploy the two Python files to D:\\loco_reporting_release_v1
(outside scientific seed-named roots). Run with the ORIGINAL experiment Python:

    python reporting_entry.py repair

For a fresh replay whose60 scores are already sealed but which has not yet
generated a report, use `report_verify_package`: it installs the I/O bridge
before the original report/verify/package functions. No fitting or inference is
invoked by either mode. The JSON supplement contains only aggregate proof and
hashes; all original failure artifacts and runtime logs stay on the server.
