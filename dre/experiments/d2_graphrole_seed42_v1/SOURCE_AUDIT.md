# Source and input audit — completed before new training

## Frozen D2 and feature provenance

Read the actual `model.py`, `train.py`, `prepare.py`, protocol, validation summary,
five-fold metrics and source-correction audit in `a0_source_posterior_decoder_seed42_v1`.
The model at commit `394dbff2085114d5f2d162cc7802384c95f0ef8d` is the same 9,221-parameter
LayerNorm88/Linear96/GELU/Dropout.15/Linear1 plus rank4 V/u/b model. Its correction
is jointly learned, not a calibrated post-hoc source offset. Freeze this model;
G1/G2/G3 are scratch 10,789-parameter enlarged-input controls, not warm starts.
The original trainer uses one update per FIT patient, NEZ-positive weighted BCE,
source-count-adjusted L2, and the stated threshold/checkpoint lexicographic rules.
The original `uas_core.py` is SHA-locked and reused without changing metric semantics.

Read `8a534ffa24_feature_aggregation.py` and `d3debece8f_evidence_views.py`.
The actual private exported 88 names were inspected without loading the `y` member.
They contain nine spectral/classical variables in four self-reference views,
each aggregated to seizure mean/std (72D), plus four physics variables in delta
and zdelta, also mean/std (16D). None of these names is a graph-node descriptor.
This is verified from the export, not inferred merely from older source.
Name inventory and the original order hash are published in aggregate schema audits.
Correlation overlap is still possible and is measured on FIT channels only.

## Graph source and definitions

Read `2aacd6c11e_graph_channel.py` completely. `repository_graph.py` retains its
absolute Pearson, positive-edge .70 quantile, .10 floor, diagonal-zero,
symmetric construction and single-edge fallback. Degree/strength divide by C-1;
weighted clustering normalizes by the graph's maximum edge. K-core divides by
maximum graph core, not C-1. Local efficiency averages reciprocal shortest paths
only over reachable ordered pairs of each node's neighbor-induced graph.
This last convention differs from libraries dividing by all possible pairs.

The old centrality implementation silently substitutes zeros or uniform PageRank
after exceptions. Those are not valid solver outputs. Our connected-graph metrics
match its float32 outputs in the representative parity audit. Eigenvectors are
computed with deterministic symmetric eigensolvers; tied dominant eigenspaces
are explicitly missing. On disconnected graphs with a unique dominant component,
the mathematical principal vector is retained rather than the old exception-to-zero
fallback. This is an explicit numerical integrity repair, not an approximation.
No-edge graph conventions (zero degree/strength/clustering/eigen/core/efficiency,
uniform PageRank) are defined mathematical conventions, not caught exceptions.

Actual cache inspection: 80 patients, 256 records, 7,635 canonical channels,
24,995 run-channel incidences and 1,471,965 valid channel-window observations.
Cached adjacency is finite, symmetric, diagonal-zero and nonzero; it is **not**
reported as a zero placeholder. However, full-window reconstruction from the
hash-verified measured waveform does not match (maximum adjacency difference
0.999971). Consistent measured-window construction provenance cannot be confirmed.
The source rule therefore selects `RAW_REBUILT` uniformly for all four sources:
15,074 measured two-second graph windows at actual 250Hz. No cached/rebuilt mixing.
Three padded records are retained; all selected windows are inside measured intervals.
Onset is independently verified in 0/256 records. No causal propagation or clinical
onset interpretation is licensed by undirected cache-relative correlation graphs.

## Research inspiration and prior attempts

Read [EvoBrain graph learner](https://github.com/Kotoge/EvoBrain/blob/f48dbbd65e6b471311531e03418518c8418abe1d/model/graph_learner.py)
and [EvoBrain model](https://github.com/Kotoge/EvoBrain/blob/f48dbbd65e6b471311531e03418518c8418abe1d/model/EvoBrain.py),
pinned to `f48dbbd65e6b471311531e03418518c8418abe1d` before this experiment.
The learner implements weighted cosine, cosine, self-attention and adaptive graph
scores. EvoBrain reduces node/edge sequences with temporal modules, constructs
nonzero edges and applies graph convolutions with Laplacian eigenvector features.
It is inspiration for varying network roles, not a reproduced architecture:
no Mamba, learned graph, Laplacian positional feature, message passing or graph loss
is introduced here. We compute fixed descriptors, then feed the original D2 MLP.

Read the actual historical `03c7c8c2b8_graph_spectral_encoder.py` and dataset adapter.
Despite its name, `WindowGraphSpectralEncoder` explicitly discards adjacency and
uses spectral MLP/channel attention; a previous graph-named model is therefore not
evidence of a tested connectivity benefit. The dataset adapter can zero-fill missing
adjacency; this is why the actual cache was audited. The matched R1-HLV experiment
reported only +0.000985 validation Macro-F1 and failed its continuation gate, but
its new feature was HLV, not GraphRole16. A1-NET's graph/router was never trained
because its CNN reproduction hard gate failed. Neither run establishes success
or failure of the present channel-specific graph descriptors. A0-SSS-MIL found
no supported raw correspondence gain; it trained a raw encoder, unlike this fixed
network-role inventory with shuffled and null input controls.

No outer clinical label, score or metric is used in this experiment. Raw trusted
pickles contain clinical fields but only waveform/identity/time metadata are used.
