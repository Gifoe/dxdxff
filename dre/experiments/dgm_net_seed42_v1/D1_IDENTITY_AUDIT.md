# Frozen D1 identity

The exact1537-parameter CompactDelta uses duplicate/zero-aware historical88-position normalization, not LayerNorm14. Own-fold original Raw R14, PatientZ, FIT-only imputer/scaler and selected D1 checkpoint are reused. Labels are NEZ-positive. No new D1 fitting, dropout-state drift or engineered-E14 inference is allowed.

The executed replay must match stored legal Development scores within1e-6, all decisions and selected threshold exactly, and rounded AUROC/AP/MacroF1/EZF1 within5.1e-7. Selected checkpoint and bank SHA256 must equal the original StageB freeze. `results/D1_PARITY.json` records actual checks; absence is not a pass.

At A2 initialization alpha0 produces exact D1 logits. Residual-disabled intervention must equal frozen base logits bitwise after training, but retains A2's own threshold, not A0's threshold. A difference in F1 there does not mean D1 checkpoint changed.
