# Patient-reference audit

For each genuine same-patient/seizure/channel/feature, b is the median of at least one pre[-10,0) observation; d=x-b. Missing baseline invalidates its derived series. At each same-time point, m is the median of finite, baseline-qualified d values from **other** genuine channels; the query is excluded and at least4 other channels are required. u=d-m. No clinical labels participate and the median is not interpreted as normal tissue.

The implementation sorts finite values once per time/feature and computes exact leave-one-out medians by ranks, including ties. Synthetic direct-brute-force, query exclusion, channel permutation and time/seizure independence tests passed. References may include any genuine source-cache channels within the selected record; this execution found0 noncanonical reference-channel instances, so no extra unlabeled reference channels contributed.

Each early/later stage uses mean/population SD/actual-time OLS slope. Equal-seizure masked mean and population SD aggregate only observed channel incidences. The final matched width is182:14 baseline coordinates,84 self coordinates,84 context coordinates. C0 context coordinates are universal0, standardized0, LR coefficient0. Counts/masks/IDs/center are not predictors.

In the80-person construction,7 patients have one seizure,16 two,57 three-plus. All80 have genuine simultaneous context; every valid baseline-qualified query-feature window has enough other observations, including100% early/later coverage conditional on valid self windows in every center. There are272 absent record-channel incidences, not contextual failures. Baseline-reference fractions including these absent slots: HUP1.0000,LZU0.97303,Multicenter0.97540,Pediatric1.0000. Do not interpret conditional100% coverage as observing every channel in every seizure.

Supervised subset is79 legal FIT identities (Pediatric7,other centers36/21/15); the excluded identity is not chosen by feature quality. Full construction Pediatric count8 is retained in data availability statistics. Private ledgers hold patient/seizure/stage-level coverage, while public files contain nonidentifying summaries only.
