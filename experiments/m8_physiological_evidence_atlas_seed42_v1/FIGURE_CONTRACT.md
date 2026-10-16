# Figure contract

Conclusion: Several qualified record measurements show patient-level EZ/NEZ
associations across sources, but source precision and repeated-seizure support
vary; these associations are not evidence of conditional predictive gain.

Archetype: quantitative two-panel grid. The forest plot is the primary panel:
all12 means and patient-bootstrap95% intervals, with finite-patient n. The
aligned heatmap shows all48 feature-coordinate/source means and their n,
including the pediatric stratum with4 patients. Nothing is dropped because
an association is weak or a source is small. Crossing-position rows are marked
as conditional on observed eligible crossing; SD rows measure across-seizure
dispersion and have no single-seizure analogue.

Python/matplotlib only, established saved backend.183mm by150mm, editable
SVG/PDF,600dpi PNG/LZW-TIFF. Quantitative sources are the12-row patient-effect
and48-row source tables. No individual signals, IDs or cases are plotted.

n is patients, not channels, seizures or windows. Intervals use10000 seed42
patient resamples and are unadjusted marginal95% intervals. All12 p-values and
their BH family are supplied in the table, with an exploratory joint172-test
atlas separately; no significance stars are shown. This is the original
fold1 FIT cohort, not a new independent cohort or source-held-out test.
