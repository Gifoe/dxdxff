# Actual clinical label provenance

This audit follows the **actual original Task1 export**, not the similarly named Task2 target function. Current adapter hashes and cache/export/label-alignment evidence are in `audit/CLINICAL_SOURCE_INVENTORY.json` and `audit/LABEL_ALIGNMENT_AUDIT.json`; the source chain is documented in SOURCE_AUDIT.md.

| Center | Patients | Channels | EZ channels | Operational target source |
|---|---:|---:|---:|---|
| HUP | 36 | 3,667 | 616 | BIDS channel SOZ OR resection OR SOZ/seizure-onset/resection status-description cue |
| LZU | 21 | 2,141 | 817 | Spreadsheet annotated channel-number EZ designation; bad channels removed |
| Multicenter | 15 | 925 | 164 | BIDS `soz`; actual source mode is SOZ, not optional SOZ-or-resection |
| Pediatric | 8 | 902 | 146 | Workbook-backed clinical channel target; sparse source subcategories suppressed/pooled |

All 7,635 frozen export labels exactly match canonical patient-index labels, with unambiguous canonical identities and zero missing/exported-invalid channels. Reader labels are NEZ-positive, converted to EZ-positive in the index, and back to NEZ-positive by Task1 export. The relevant patient-record builders take the minimum reader NEZ label across seizures, equivalently an EZ union; metadata retain the first channel record.

**22 LZU channels differ between the index and the retained first-record `final_label`/`is_ez_or_soz`. All 7,635 index labels match the EZ union over actual cached run labels; there are zero union disagreements.** These differences are explained by the documented aggregation rule, not classified as annotation errors. Substituting first-record metadata would silently change the benchmark target and is not authorized.

HUP's cached independent `soz` and `resection` fields are missing for its channels, although its operational composite and final labels are present. The source rule plus available status-description values reproduce the cached composite. It is therefore not possible to independently quantify SOZ-vs-resection disagreement from this cache. Multicenter `soz` is available for all 925 channels and agrees with the index. LZU/pediatric do not expose independent SOZ/resection binary fields here.

The inspected alternative `center_clinical_target()` is not called by the A0 export or Task1 loader. It can use HUP SOZ-or-resection, composite metadata, final-label fallback, or index fallback; this is an **alternative**, not the executed A0 label policy. Its retained-metadata branches must not be used to relabel the 22 union-positive LZU channels.

No mixed-source patients were found at the observed source-category level. Center target definitions differ operationally; this is `HETEROGENEOUS_SOURCE_DEFINITIONS_CONFIRMED`. It does not demonstrate different biological EZ truth or prove that any label is incorrect. Current source hashes are not execution-time attestations: the cache lacks original source hashes/complete creation arguments. Clinical intent behind authored workbook targets and original outcome/quality inclusion requires independent source-document validation.

Label-source error associations use patient-level aggregation only (65 development appearances/47 IDs), with source groups confounded with centers. Pediatric has lower A0 performance than multicenter, but neither center nor source causality can be inferred. No outcome-conditioned relabeling, model input, threshold, target selection, or new cohort filter was used.
