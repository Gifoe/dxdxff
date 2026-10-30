# Representative and declining PR cases from frozen Ictal Outer predictions

Read-only reanalysis of the original 80-patient, 7,635-channel cohort. Compare
the frozen 12D Linear SVM No PR/Legacy PR and the original complete DREChanNet
WO_BOTH_PR/FULL at seeds 42/43/44. The latter is the 11,425-parameter original
three-seed FULL with early FiLM and late Gate, not the later no-guidance model.

`PROTOCOL_LOCK.json` fixes eligibility, median-oriented selection, stability,
negative-control priorities and privacy before candidate outcomes are computed.
The script never trains a classifier or performs a new model forward pass.

True identifiers and original contact names remain in server-private exports.
Only two selected, pseudonymized cases have channel-level exports; all80
patient-wise changes remain private. Anonymous contact aliases preserve the
original lexicographic tie order for exact rank reconstruction.

Completed and independently verified on 2026-10-30. See
[Chinese case report](results/REPORT_ZH.md),
[execution audit](RUN_AUDIT.md), and
[server verification](results/public/INDEPENDENT_VERIFICATION.json).

`results/public` preserves the 12 authenticated payloads and their manifest.
`results/readable` provides one-channel-per-row tables for CASE-P01 and
CASE-N01. Original contact names and patient identity mappings stay in the
trusted server's private directory. The GitHub release contains these anonymous
case tables, the fixed protocol, verification scripts and the Chinese report.

Local verification and report regeneration:

```powershell
python verify_anonymous.py results/public
python test_selection_semantics.py
python build_case_report.py
```

The completed archive passed `release_check.py --package ... --extract ...`
before any extraction. No historical parent files were modified.
