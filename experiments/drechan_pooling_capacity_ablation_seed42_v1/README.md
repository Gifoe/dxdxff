# Fixed no-Late DREChannNet: temporal pooling and capacity controls

Implementation stage; no new performance outcomes claimed. Three new alternatives are fixed before fitting: P1 unsegmented temporal Mean/SD then cross-seizure Mean/SD96D; P2 masked time-aware stage attention then original cross-seizure Mean/SD96D; A2 noEarlyFiLM with a73-wide ordinary head. Full and original noEarly are strict existing references. Five groups ×original5folds ×seed42 =25cells,10reused/15fresh.

Parameters: Full10377; P1 10377; P2 10403(+26); original noEarly9329; A2 10337(-40 versusFull,-0.39%). No inactive padding parameters. Main still excludes Variance, Head Interaction and Late Gate. The same six physiological measurements/directPR, original raw data/roles/normalization/pre-reference/loss/budget remain unless explicitly changed. [Frozen operation definitions](PROTOCOL_LOCK.json).

All25selections freeze before newOuter; all25scores seal before metrics. Each group must finish Outer without a performance gate. Report all9patient metrics,pooled/fold/source/coverage separately,400rows/3600metric values,10000patient draws,90nominal/10additional-family intervals. No outcome-driven architecture/threshold or main change; historical Outer viewing makes this exploratory repeated-test, not independent confirmation.

P1 jointly changes segmentation and temporal statistics; P2 adds26learned coefficients; A2 matches parameter count approximately rather than exact functional capacity. Interpret those boundaries explicitly. Source is pushed before formal fitting. Only source and anonymous aggregates are uploaded; no data/identities/private predictions/weights/preprocessors/caches/logs/secrets.
