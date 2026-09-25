# HiFOS-PACT Final Protocol Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining representation, calibration, cohort, trace, LOCO, and resume inconsistencies before formal server experiments.

**Architecture:** Keep the existing pipeline and add fail-closed contracts at branch fusion, reporting, replay, cache loading, and artifact reuse boundaries.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pandas, scikit-learn, PyYAML, pytest.

## Global Constraints

- Preserve actual singular local cache paths and repository-local outputs.
- Never use outer-test information for fitting or selection.
- Do not change Task 1 semantics or import its labels into Task 2.
- Test first; do not push.

### Task 1: Raw-logit H10 contract

- [ ] Add failing representation-consistency tests.
- [ ] Require raw logits and matching representation metadata in inner and outer branch tables.
- [ ] Update automatic H10 assembly and protocol audit.
- [ ] Run synthetic H10.

### Task 2: Shortcut and calibrated replay

- [ ] Add failing shortcut-risk and calibrator tests.
- [ ] Separate metadata shortcut metrics from robustness metrics.
- [ ] Apply fold Platt calibrator to replay logits and persist provenance.

### Task 3: Pairwise bootstrap and cohort provenance

- [ ] Add failing pair/config and cohort-mismatch tests.
- [ ] Persist cohort and subject hashes on predictions.
- [ ] Restrict comparisons to matching provenance and emit unavailable comparison rows.

### Task 4: Fusion alignment, FM trace, time origin, LOCO, resume

- [ ] Add failing alignment, duplicate trace, three-center LOCO, and stale-resume tests.
- [ ] Enforce full run alignment or explicit exclusion.
- [ ] Validate chunked cache trace integrity and builder source hash.
- [ ] Preserve `loco_test` semantics and reject stale resume files.

### Task 5: Documentation and verification

- [ ] Update configs and README.
- [ ] Run dry-run, real audit, H1-H6 smoke, synthetic H10/LOCO, and full pytest.
- [ ] Save pytest output and commit locally without pushing.
