# FCPA-LF: sequential-gated investigation, seed42

The first mandatory stage is an actual label-blind EDF/Raw fidelity audit.
No model, LF24 bank, development inference or outer evaluation is admitted
unless this stage passes. The task's95% canonical-channel/all-four-center
contract is evaluated separately from the earlier stricter data-only audit's
100%/historical27-mask contract. A failure of the latter is not automatically
used as the reason to fail FCPA-LF.

Inputs are the exact frozen80-patient/256-record/7,635-channel contract and
the independently hash-bound companion signal audit:
`../a0_fcpa_data_feasibility_seed42_v1/`.
Individual identities, EDF paths/hashes, waveforms and aligned per-window
values stay in its trusted runtime. Public results are anonymous aggregates.

```text
python tests/gate_contract.py
python code/audit_source.py --audit-runtime PRIVATE_SIGNAL_AUDIT --output PRIVATE_PUBLIC_OUTPUT
python code/audit_spectral_fidelity.py --audit-runtime PRIVATE_SIGNAL_AUDIT --output PRIVATE_PUBLIC_OUTPUT
python code/audit_spectral_fidelity_qualified.py --audit-runtime PRIVATE_SIGNAL_AUDIT --output PRIVATE_PUBLIC_OUTPUT
python code/qualify_native_bins.py --output PRIVATE_PUBLIC_OUTPUT
python code/audit_header_summary.py --audit-runtime PRIVATE_SIGNAL_AUDIT --output PRIVATE_PUBLIC_OUTPUT
python code/verify.py --output PRIVATE_PUBLIC_OUTPUT
```

See `FINAL_REPORT.md` and `results/SCIENTIFIC_TERMINAL.json` for the actual
stop condition. Conditional model counts in `PROTOCOL_LOCK.json` are design
targets, not measured trained-model counts. If Stage A fails, model/trainer/
LF-feature-extraction modules are intentionally not implemented or run;
later result files explicitly record NOT_RUN rather than invented metrics.
Use `LF_PSD_COMPARISON_QUALIFIED_V2.csv` for timeline/native-bin interpretation;
the original index diagnostics remain only as transparent execution evidence.
Runtime artifact hashes refer to the original trusted-server bytes. Aggregate
delivery copies may have Git-normalized LF line endings; these copies are not
substituted for hash-bound private runtime inputs. Protocol/source files use LF.

No installed skills are used. No new Outer TEST is authorized or evaluated.
Any later development on this historically viewed cohort would be exploratory,
not an independent blind evaluation. No rescue bands or architecture search.
