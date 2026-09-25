"""RCC fixed-subject/fold protocol uses the existing validated V3 contract."""
from .v3_qbc_protocol import file_sha256, read_allowed_subjects, read_outer_fold_manifest, validate_v3_qbc_protocol

def validate_v3_rcc_protocol(**kwargs):
    audit = validate_v3_qbc_protocol(**kwargs)
    audit = dict(audit)
    audit["protocol_name"] = "v3_rcc_fixed_outer_validation_only_raw_threshold"
    audit["threshold_protocol"] = "outer_train_fixed_validation_raw_probability_threshold"
    return audit

__all__=["file_sha256","read_allowed_subjects","read_outer_fold_manifest","validate_v3_rcc_protocol"]
