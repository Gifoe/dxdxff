# Frozen C2 + patient-context hidden affine adaptation

Prospective third bounded study after two actual Raw architecture studies
failed their unchanged internal Macro gate. Original7D/14processedcoordinates
and the full80patient contract stay unchanged. The previously paused
displacement-only proposal is preserved separately and was never run.

One assembled model has exactly one scalar classifier. Its ownfold C2 Raw
encoder,Gate and110→64GELU body/readout are frozen. G1 learns hidden gains
and shifts from the mean/populationSD of the64learned hidden coordinates
over all channels of the same patient. G0 is a global affine control.
Both start as exact C2 functions. Unlike a common output bias,the gains can
change channel ranking. No clinical measurement,source,ID or label is a
contextinput. Full models:10506/12562parameters;trainable128/2184.

The control is not capacity-matched and additional head optimization differs
from original C2 training. A gain therefore cannot prove pure architecture
causality or novelty. Body selection and all new selections reuse original
Development;estimates remain selection-conditioned,not independentCV.

All5ownFIT|Development Raw banks/320patient appearances are actually inferred
and hash-sealed. Ten new head fits use completepatient updates,original loss,
seed42,AdamW and201threshold rules,max100/min15/patience15. No body/readout
updates. G1 qualifies onlyif Macro≥max(C2,G0)+.003 andAUROC/AP≥theirmax−.01.
Otherwise no newOuter. Any final stage requires a separate immutable freeze
and a single full80exploratory repeated-test,not independent confirmation.

All retained cells are independently replayed from actual Raw through the
assembled model,not merely verified against cached scores. Release includes
all9metrics,54cluster intervals,fold/source/coverage,actual epochs and reuse.
Only code,hashes and anonymous real aggregates are uploaded. Weights,Raw,
identities,hidden/prediction caches,scalers and logs stay on the server.
