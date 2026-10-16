"""Canonical API conformance, explicitly not an independent solver replication."""
import conditional_io as io
from canonical_ridge import CanonicalRidge


if __name__=='__main__':
    destination=io.Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_dense_reference_runtime')
    b=io.jr(destination/'INPUT_FREEZE_PRIVATE.json')
    for name,value in b['numerical_engine'].items():assert io.sha(io.HERE/name)==value
    module=io.load('conditional_dense_canonical_api_admission',io.HERE/'verify_ridge_admission.py')
    module.ConditionalRidge=CanonicalRidge;module.PUB=destination/'public'
    module.main()
    io.write(destination/'public/CANONICAL_ADMISSION_CONTEXT.json',dict(status='PASS',
        meaning='Exact conformance to predefined sklearn canonical API; not independent numerical or clinical replication.',
        original_failed_Schur_gate_not_waived=True,unchanged_absolute_score_tolerance=2e-10,
        no_new_encoder_training_or_Outer=True,source_sha256=io.sha(io.HERE/'verify_dense_reference.py')))
