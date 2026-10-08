# Ictal onset SSL + PR E3, seed42

User scope: execute E3 on the trusted private server using the attached package. E0/E1/E2/E4/E5 were not requested and have not been trained. The `package/` directory preserves the supplied source and documents; its historical status statements are not current execution results.

Initial audit: **BLOCKED_RAW_WINDOW_PADDING**. The live frozen caches and exact five-fold assignments were recovered and audited. One of 256 required preictal crops contains four seconds of padding.

The user subsequently approved excluding only this record. [PROTOCOL_AMENDMENT.json](PROTOCOL_AMENDMENT.json) freezes the revised scope: 255 records, all 80 patients and 7,635 channels, original folds and time windows; E3 training/validation only. The earlier blocker report and JSONs are retained as audit history, not the amended run's current state.

Initial evidence: [INITIAL_BLOCKER_REPORT.md](INITIAL_BLOCKER_REPORT.md), [DATA_BLOCKER_AUDIT.json](audit/DATA_BLOCKER_AUDIT.json), and [ONSET_PROVENANCE_AUDIT.json](audit/ONSET_PROVENANCE_AUDIT.json). The approved export passes [AMENDED_DATA_AUDIT.json](audit/AMENDED_DATA_AUDIT.json).

Source branch: `codex/rawtiny-patient-relative-seed42-v1`, commit `e8d50cb29a3c80ed9f9688008a5ba30d54da271d`.

Working branch: `codex/ictal-onset-ssl-pr-e3-seed42-v1`.

Server source: `E:\DRE-nips\new-pipeline\7-11\ictal_onset_ssl_pr_e3_seed42_v1`.

Private runtime: `C:\ictal_onset_ssl_pr_e3_seed42_runtime`.

`legacy_folds.py` reconstructed the original roles from the frozen A1 source, not a fresh random split. The private `frozen_folds.json` and `PADDING_RECORDS_PRIVATE.json` stay on the server. No raw arrays, individual patients/channels, runtime logs, checkpoints, or prediction files are included here.

Execution helpers: `code/prepare_runtime.py` is the exhaustive initial audit source, deployed as `code/prepare_runtime_final.py` on the server. It fails closed on the original model's data contract. Alternative intervals were not applied. The subsequent one-record exclusion was applied only after explicit user approval.

Amended execution: `code/run_amended.cmd` runs the patched exporter with the exact private exclusion ledger, then `code/run_e3_validation.py`. The exporter rechecks the original source hashes, full source footprint and the excluded record's exact invalid interval before removing its model input. Patient files are individually hashed. The runner verifies these hashes and gates each fold to FIT+VAL patients, preloads unchanged tensor values to avoid repeated compressed-file I/O, and saves complete optimizer/model/random states atomically after each epoch. Source and protocol bindings reject changed resumes. It neither scores outer test patients nor trains extra arms.

Server numerical tests: all 13 package/amendment tests passed. [CPU synthetic interrupted/resumed SSL and E3 training](audit/RESUME_PARITY_AUDIT.json) match the supplied implementation exactly (maximum parameter differences zero), including selected epoch, threshold and metrics. These checks verify engineering behavior, not scientific performance. GPU training uses the same unmodified architecture, FP32 operations, chunking, data order and seeds. Execution amendments and recovered native failure are recorded in [ENGINEERING_LEDGER.md](ENGINEERING_LEDGER.md).

Completed: **VALIDATION_COMPLETE_TEST_NOT_ACCESSED**. All five folds finished 25 SSL + 30 supervised epochs. Five-fold validation means: Macro-F1 **0.616545**, EZ-F1 **0.397900**, EZ-AUPRC **0.494767**, EZ-AUROC **0.691482**, MRR **0.666778**, Top1 **0.553846**. These are post-selection development estimates, not an outer benchmark or independent test.

See [FINAL_REPORT.md](FINAL_REPORT.md), [results](results), and [completion audit](audit/COMPLETION_AUDIT.json). The 65 validation cells contain 47 unique patients; the report separates the fold mean from the repeated-patient-adjusted descriptive bootstrap summary. Repeated NumPy native crashes were resolved for finalization through an isolated, parity-tested NumPy 1.26.4 environment; all training completed in the unchanged original environment.

Private amended outputs: `C:\ictal_onset_ssl_pr_e3_seed42_runtime\validation_amended`. Source data, exports, individual records, predictions, checkpoint and runtime logs remain private. Only source and compact aggregate results/audits are included here.

Follow-up: the [matched E1 control and E1/E3 comparison](../ictal_onset_ssl_pr_e1_seed42_v1/FINAL_REPORT.md) are now complete. E3's frozen checkpoint and private validation artifacts were not modified. The predeclared E3-minus-E1 development gain gate was not met; E4/E5 were not run.
