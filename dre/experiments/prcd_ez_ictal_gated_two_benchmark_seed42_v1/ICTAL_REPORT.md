# PRCD-EZ Ictal report

See `FINAL_REPORT.md` for the complete Ictal-only analysis. The locked result is:

| Model | AUROC | AP | Macro-F1 | MRR | Top1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical A1 | 0.746382 | 0.576743 | 0.620810 | 0.740038 | 0.654771 |
| CD-EZ | 0.680595 | 0.484880 | 0.545221 | 0.663231 | 0.557797 |
| PRCD-EZ | 0.730063 | 0.550435 | 0.610955 | 0.721537 | 0.627618 |

PRCD-EZ failed AUROC, AP, fold-consistency and MRR gates; only Macro-F1 safety passed. No outer test or Omni data was accessed.

**Terminal: `STOP_ICTAL_GATE_FAILED`.**
