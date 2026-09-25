# Task 1 upper-bound audit

This audit reads existing old-90 patient-channel outer-test OOF ledgers. It does not train models and it never modifies an input ledger.

## Statistical boundary

- Label direction remains NEZ=1 and EZ=0.
- `strict_count_free_*` uses the stored outer-test `predicted_nez`, or a reconstruction from a stored probability and an inner-OOF-selected threshold.
- True-K uses the observed number of EZ channels and is diagnostic only.
- Patient oracle threshold chooses the best monotone score cut for each test patient and is diagnostic only. Equal-score blocks cannot be split.
- Oracle prefix is ultra-optimistic because it may split equal-score blocks.
- Patient-wise model selection, full-OOF ensemble weights, and a full-OOF global threshold are oracle diagnostics and are not deployable estimates.
- Without adjudicated clean labels, the result is an observed-label ceiling rather than a clinical-truth ceiling.

## Input contract

The V3 ledger freezes the 90 subjects, centers, outer folds, canonical normalized channel keys, and labels. Automatic discovery is limited to:

```text
<task1-output-dir>/oof_ledgers/*/seed_*_channel_oof.csv
```

Additional patient-channel ledgers can be supplied explicitly with `--extra-ledger` or `--extra-ledger-glob`. Summary CSVs are not discovered. Channel names are stripped, upper-cased, and stripped of ordinary spaces; identity-bearing letters, digits, and connectors are retained.

Probability fields must be finite and in `[0,1]`. Generic scores need only be finite. A hard-prediction-only ledger can contribute strict metrics but cannot contribute ranking, oracle, or ensemble metrics. Candidate ledgers with duplicate keys, label/fold/center conflicts, forbidden threshold provenance, incompatible protocol hashes, or noncanonical channel keys are rejected from comparisons. Explicitly requested invalid models fail under `--strict`.

## Command

```bash
python scripts/task1_baselines/run_task1_upper_bound_audit.py \
  --task1-output-dir "$DRE_TASK1_OUTPUT_DIR" \
  --v3-ledger "$DRE_TASK1_V3_LEDGER_PATH" \
  --output-dir "$DRE_TASK1_UPPER_BOUND_OUTPUT_DIR" \
  --bootstrap-samples 2000 \
  --ensemble-random-candidates 5000 \
  --random-seed 42 \
  --strict
```

The output path falls back to `DRE_TASK1_UPPER_BOUND_OUTPUT_DIR`, then `<task1-output-dir>/upper_bound_audit`. The audit writes input inventories, alignment and rejection tables, the canonical manifest, normalized score matrices, model/center/fold/patient metrics, patient-bootstrap intervals, ensemble candidates and weights, a blank adjudication template, and `reports/TASK1_UPPER_BOUND_REPORT.md`.

## Clean-label subset

Pass `--clean-labels-csv` with `subject_id`, `channel_name`, `adjudicated_label_nez`, and `include_clean_subset`. Only explicitly included rows with an adjudicated 0/1 label are evaluated. Missing adjudications are never filled from observed labels.

