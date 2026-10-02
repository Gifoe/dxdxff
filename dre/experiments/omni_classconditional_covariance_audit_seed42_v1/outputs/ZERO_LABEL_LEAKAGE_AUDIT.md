# Zero-label construction audit

The UTC estimator signature is `(X, patient_ids)`. It receives only a representation matrix and patient IDs; its function body contains none of: label, soz, resection, outcome. UTC, diagonal UTC, CORAL, and UTC-Mahalanobis were constructed before evaluation metrics, with target pathology labels excluded from their construction. TEST labels were used only for explicitly marked diagnostic geometry and evaluation.
