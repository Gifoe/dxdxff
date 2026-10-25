# Figure B contract

Core question: quantify patient-held-out AUROC for every one of the eight fixed
main inputs without assuming combined features or PR must win.
Archetype: quantitative single-panel forest plot. All80patient means and nominal
95%patient-bootstrap CIs; chance AUROC0.5 reference. Fixed user-specified row order.
Backend: saved Python/matplotlib. Style-only inheritance from Figure A (restrained
blue/orange, white background, editable vector text), not reuse of its effect-axis
statistics. Source fields: arm->input label; AUROC->dot; AUROC_ci_low/high->interval.
Export:89mm single-column width,7pt minimum text,PDF/SVG/600dpiPNG.
No fabricated/mock results or patient exclusions. No individual rows exported.
Risk: selected historical cohort/features; patient-level internal holdout does not
make external clinical validation; fixed-model bootstrap omits retraining variance.
The single-column booktabs table contains all8 rows and four primary endpoints,
with true unrounded maxima only bolded; full CIs remain in the report/CSV.
