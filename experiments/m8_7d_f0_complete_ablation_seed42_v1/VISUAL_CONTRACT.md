# Evidence-bound visual contract

Destination: technical experiment report, with reusable editable scientific plots; not an architecture graphic or final manuscript claim.

- Feature contribution: seven rows in original physiological-coordinate order; panels for patient Macro-F1, AUROC and AP. Points show Full-minus-drop and horizontal nominal95% paired patient-bootstrap intervals. Positive means the complete model scores higher; zero-crossing intervals remain visible. No feature subset is selected from these plots.
- Core paired uncertainty: ten mechanism/branch comparisons plus the predefined early-by-late interaction, same three metrics. Fixed protocol order, not ranking by the most favorable result. Interaction is a metric-scale contrast, not a causal physiology interaction.
- Source data: independently verified `MAIN_RESULTS.csv`, `INTERVALS.csv`, `MACRO_FAMILY_INTERVALS.csv`; figures keep those source hashes. All data are anonymous aggregates of the original80patients.
- Grammar: reproducible horizontal interval plots, adapting the bundled plotting recipe's direct-label/axis/semantic-color principles to paired uncertainty. White background, Okabe-Ito blue points, neutral gray uncertainty and dashed zero lines, shape distinction for the factorial interaction. No red/green significance coding, invented p values or winner badges.
- Labels: natural sentence case; canonical AUROC/AP/Macro-F1 acronyms unchanged. Metric-unit differences, patient pairing,10000draws and exploratory repeated-test status stated in captions.
- Export: editable SVG, vector PDF and PNG preview. Text stays live in SVG. Inspect actual previews for clipped labels, overlapping intervals, unreadable type and inconsistent zero/units; record visible QA separately, never infer visual PASS from file generation.

Reporting distinguishes nominal intervals from approximate Bonferroni-level family intervals. The figures show nominal intervals; the additional family intervals are reported in tables. No private patient/channel IDs, individual scores or waveforms enter figure generation.
