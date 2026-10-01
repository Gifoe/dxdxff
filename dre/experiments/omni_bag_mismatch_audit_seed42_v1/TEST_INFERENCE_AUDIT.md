# TEST Inference Audit

Frozen TEST inference uses deterministic full-record coverage: discard the first and last second, then take non-overlapping 60-second windows with 60-second stride. Every selected channel in an EDF uses the same starts. Segment normal probabilities are converted to pathological probabilities before arithmetic mean aggregation.

- EDFs: 174
- Patients: 96
- Segment rows across all selected channels: 240074
- All selected EDF-channel units: 16543; mean segments/unit: 14.512120
- Labeled segment rows: 90930; labeled EDF-channel units: 8104
- Mean segments/channel: 11.220385
- Median segments/channel: 5.0
- EDF segment-count range: 1 to 119
- Overlap: 0 seconds. Full-record tail shorter than 60 seconds is not used.
