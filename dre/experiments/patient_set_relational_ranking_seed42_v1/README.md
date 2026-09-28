# Patient-set Relational Ranking Zero-shot Study

Development-only, seed 42. Frozen exact-A1 selected-epoch R4 and EZ margin are the only model inputs. R1 uses DeepSets context; R2 uses explicit all-pairs relations; R3 is FIT-gated. Every readout is zero-residual at step 0. Inference uses no patient labels or adaptation.

The `PROTOCOL_LOCK.json` is committed before any new relational target outcomes. Private caches, scores, checkpoints, per-patient/channel records and runtime logs remain on the Windows server data disk, never in Git. Public output contains aggregate CSV/JSON and the final report only.

The A1 VLOO source protocol has cross-patient label dependencies. The legacy loader also materializes 80 labels. Both limitations must be disclosed; neither permits outer predictions or target-result-guided method selection.
