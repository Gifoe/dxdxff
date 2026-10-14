# Patient-relative preprocessing audit

**A0/A1 VERIFIED no PR:** Raw ABS14/DELTA14 pass directly to float64 FIT-only SimpleImputer and StandardScaler, then float32 classifier input. There is no patient-local normalization, score centering, unlabeled TEST fitting or shared global all-cohort scaler. Imputer/scaler are independently fitted for each arm/fold on the original FIT patients only, saved privately and frozen before Outer. Synthetic perturbation of non-FIT values leaves FIT statistics/outputs unchanged.

**A2 VERIFIED original PR:** exact same DELTA14 measurements, original float64 cross-channel population mean/SD and original denominator protection(<=1e-8→1), followed by original frozen FIT-fitted imputer/scaler/checkpoint. No refit/recalibration. Frozen legal-bank preprocessing is checked by original source contracts and reuse provenance.

**A3 VERIFIED unchanged:** original D1 and Raw-branch PatientZ/physiology scale remain frozen. This experiment does not independently isolate Raw-branch PR.

PROTOCOL_LOCK.json inherits the original D1 StageB contract as a source specification; its inherited PatientZ entry describes frozen A2. The effective A0/A1 override is explicitly fixed in the top-level matrix/patient_z fields and executable no-PR preprocessing. No normalization was switched during execution.

The historical88D engineered no-PR/source-conditioned ablations were inspected but not substituted: their features/topology differ from these new Raw14D models.
