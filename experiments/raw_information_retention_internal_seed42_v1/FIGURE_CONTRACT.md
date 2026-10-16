# Internal information-retention figures

## Evidence contract

The figures describe the actual internal patient-held-out effects of normalization,
encoder family, patient context, and four end-to-end pooling deletions. They do
not establish final EZ-F1>0.65 or independent test generalization.

1. Hero panel: all nine arms, patient EZ-F1 with patient-ID cluster intervals.
2. Supporting panel: all nine arms, patient AUROC and AP from the same executed
   cells. This distinguishes decision improvement from improved ranking.
3. Component panel: baseline minus each separately trained component deletion,
   EZ-F1 and AUROC with the correctly reversed paired interval endpoints.

Archetype: quantitative grid. Python/matplotlib only; no simulated observations,
no selective arms or removed outcome rows. The scripts require the independently
verified terminal. Source tables are the anonymous FIRST_PATH tables, with all
nine arms retained in a fixed protocol order. Additional raced paths remain in
the report and are not mixed into the balanced first-path figure.

## Export and statistics

183mm-wide figures, sans-serif editable text, SVG/PDF plus 600dpi PNG/TIFF.
Panel labels are bold lowercase. Patient is the biological unit; repeated
patient appearances are carried together in 10000 patient-ID bootstrap draws.
Intervals are descriptive, selection-conditioned 95% percentile intervals, not
independent confirmation. No p values or multiplicity-adjusted significance
claims. Epochs/windows/channels are not independent replicates. The report
supplies exact logical and unique patient counts from the independent verifier.

Long versus short also changes parameter count/width/depth/kernel sizes;
causality is not assigned solely to receptive field. Missing/unfinished output
is a stop condition, never drawn as zero. Readability, interval direction,
axis limits and artifact source consistency must be visually checked before
delivery. No microscopy/image manipulation applies.
