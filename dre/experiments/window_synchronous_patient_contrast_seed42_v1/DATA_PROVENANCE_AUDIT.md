# Data provenance audit

## Verified this execution

All three supplied source SHA256 hashes passed before construction; see `results/SOURCE_HASHES.json` and `PROTOCOL_LOCK.json`. Metadata-only NPZ access read canonical patient/channel/center and original fold roles, not its x/y members. The original 80-patient, 256-record, 7,635-channel feature contract and fixed split were not regenerated. The clinical totals 1,743 EZ / 5,892 NEZ are **inherited historical contract values**, not a new scan of the prohibited global label member.

Window cache inputs are exactly the seven named features. Actual zero-based source columns, discovered by names: beta3, low-gamma4, high-gamma5, RMS7, variance8, line length/sec9, entropy10. Column positions are reported, not guessed. No delta/theta/alpha, graph, center, channel index, patient ID or validity/count predictor was added.

The supplied pickle is monolithic. Deserialization may incidentally materialize its clinical/raw fields; code never indexes or uses those fields. This is a **no-use boundary**, not a claim that such fields never enter memory. Only window features, feature names, record/channel identity and genuine-window/time metadata drive construction. No Raw file or EDF was opened and no features were re-extracted.

Probe labels are read from each original fold's independent FIT-only PR-UAS OOF artifact: patient/channel/y_nez only. All teacher scores, uncertainty, model weights and prior predictions are ignored. Their source hashes remain in the private probe freeze. For each fold, IDs equal the original FIT set exactly, are disjoint from its validation/test roles and retain complete canonical channel identity. No monolithic FIT/VAL model bank, global label member or original Development/Outer prediction was used.

## Role accounting

The full label-free construction covers80 patients. Legal supervised evaluation covers **255 FIT-fold appearances /79 unique patients**. One cohort identity never appears in FIT in the frozen five-fold split. Evaluated unique-patient appearance counts are: 2 patients once,11 twice,33 three times,33 four times. This is a discovered split property, not data-dependent filtering. Original FIT counts remain51/51/50/52/51. No forbidden labels were admitted to make the supervised count80.

Roles are interpreted relative to the corresponding original fold. A patient can be FIT in one fold and validation/test in another; this is the original repeated-FIT design. Its held-out prediction within any cell never uses its own supervised label for fitting or threshold selection. Globally excluding the union of every fold's validation/test patients would change the specified FIT cohorts and was not done.

## Unresolved provenance inherited unchanged

The FCPA-LF audit failed low-frequency fidelity/lineage gates; unknown extraction filters, raw-generation CLI and94 unresolved EDF+D records are not repaired here. Cache timing is annotation-relative, not independently verified EDF physical onset. The seven stored window descriptors may contain acquisition/preprocessing effects. No causal propagation, normal-tissue-reference, complete250-Hz raw80-150-Hz gamma or biological irrelevance of low frequencies is claimed.

Public outputs are aggregate counts/statistics/hashes only. Waveforms, cache matrices, labels, IDs, model objects, scores and execution logs remain on the trusted server.

## Subsequent explicit Outer authorization

The preceding no-global-label statements describe the original FIT-only phase. The user then requested Outer and chose fixed FIT-internal final TRAIN/threshold splits. OUTER_PROTOCOL_LOCK records this separate authorization; no inner seal/protocol changes. All15 final models/thresholds seal before Outer inference and all15 predictions seal before first access to the NPZ clinical y member. Its sole authorized purpose is scoring80 original TEST identities exactlyonce in their corresponding folds; no Development labels/scores are used in fitting, threshold selection or evaluation. This phase actually confirms1743 EZ/5892 NEZ/7635 channels, instead of merely inheriting those totals.
