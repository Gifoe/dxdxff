# Final report mirror

The complete public interpretation is in [`../FINAL_REPORT.md`](../FINAL_REPORT.md).  This mirror exists so the required compact-result directory retains a `FINAL_REPORT.md` entry.

Terminal decision: `A1_FEATURES_WEAKLY_COMPLEMENTARY_BUT_PLUGIN_GATE_NOT_MET`.

- Frozen TimeConv OOF replay: 0.7621631 patient-equal AUROC.
- F9/F36 feature-only AUROC: 0.5976709 / 0.7199595.
- F36 patient-equal rescue/destroy: 0.526149 / 0.234774.
- Fixed F36 beta diagnostics: +0.015736 (0.10), +0.020943 (0.25), +0.012015 (0.50) AUROC versus TimeConv.

No beta was selected through a nested train-only process, so these are descriptive fixed-grid diagnostics rather than a deployable fusion result.  No feature plugin was trained.
