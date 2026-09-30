# Reference operator lock

The reference unit is one synchronized EDF. Membership is every extracted
signal-valid/good channel and never depends on its pathological/normal label.
For three or more channels, each channel receives the coordinate-wise median
of all *other* channels. For fewer than three channels, the all-channel median
is used and the fallback is counted. The difference is the channel embedding
minus this reference. Coordinate-wise empirical ranks use average ranks for
ties and map the first/last ranks to -1/+1. No MAD or standard-deviation
division is performed.

The historical official scorer averages segment sigmoid probabilities. The
prompt's later claim that sigmoid(mean segment logit) is the official scorer
is false on the frozen predictions: it changes AUROC from 0.798767 to
0.790142. Primary evaluation therefore adds the one channel residual to each
of that channel's frozen segment logits and then applies the unchanged
official sigmoid-then-mean aggregation. The contradictory mean-logit result is
reported only as a diagnostic.
