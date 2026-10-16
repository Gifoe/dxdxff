# Physiological evidence atlas:32 window groups and six record groups

This descriptive extension audits all800 original fold/feature/phase effect
rows and3200 source rows from fixed private patient effects. It adds all12
qualified record-coordinate associations in the same original fold1 FIT51
patients. No new classifier, feature selection or Outer evaluation is run.

The original160-primary family is preserved. New12 tests receive their own
BH correction; an explicitly exploratory172-test combined atlas is also
published. Eight new coordinates have record-level analogues; four seizure-SD
coordinates do not. Zero seizure effects remain in directed agreement rather
than being silently discarded. Conditional observed crossing times are not
clinical onset estimates. Every result remains exploratory and must not be
used to reopen failed predictive continuation rules.

Run `python -m pytest test_statistics.py -q`, then `python run_atlas.py` on the
trusted server with the pinned inputs. The accepted entry applies the published
guard-only per-fold-count amendment; original `atlas.py` preserves the failed
equal-count assumption and must not be called the accepted result. Verify with
`verify_atlas.py` before `export_public.py`. Publish anonymous aggregates only.
Source, qualification and existing model/prediction artifacts remain unchanged.

## Actual result

All800 original estimates and3200 source rows were reaggregated. The new612
patient effects and1296 per-record effects passed independent verification.
Ten of12 coordinates meet the descriptive association rule, but only3 of8
applicable coordinates meet directed repetition support. These associations
do not overturn the failed conditional predictive screen. See [REPORT.md](REPORT.md)
and the38-group/28-definition public evidence ledgers in `results/`.

Public-only reproduction runs `build_evidence_ledger.py`, then `figures.py`.
Re-rendering resets the manual visual-review field to PENDING; inspect the
exports and record that review in `FIGURE_QA.json` before `verify_release.py`.
This manual rendering check is not physiological or predictive evidence.
