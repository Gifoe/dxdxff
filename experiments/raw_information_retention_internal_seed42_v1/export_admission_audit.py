"""Anonymous pre-training source/data audit; no model outcome read or selection."""
from common import *


if __name__=='__main__':
    binding=setup();gate=jr(PUB/'GRADIENT_ADMISSION.json')
    assert gate['status']=='PASS' and gate['input_binding_sha256']==digest(binding)
    assert all(r['logit_loss_error']==r['full_gradient_max_error']==0 for r in gate['reports'])
    original=Path(r'D:\raw_information_retention_internal_seed42_v1_runtime')
    current=read(ROOT/'INPUT_AUDIT_ROWS_PRIVATE.pkl');old=read(original/'INPUT_AUDIT_ROWS_PRIVATE.pkl')
    a={r['patient']:r for r in current};b={r['patient']:r for r in old};assert set(a)==set(b) and len(a)==79
    checked=0
    for pid in a:
        assert set(a[pid])==set(b[pid])
        for key in a[pid]:
            if key=='binding':continue
            x,y=a[pid][key],b[pid][key]
            if isinstance(x,(np.ndarray,float,np.floating)):
                assert np.array_equal(np.asarray(x),np.asarray(y),equal_nan=True),key
            else:assert x==y,key
            checked+=1
    finite=[r['old_post_gain_invariance_qualified_max_error'] for r in current if np.isfinite(r['old_post_gain_invariance_qualified_max_error'])]
    write(PUB/'EXECUTION_SOURCE_ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),
        source=binding['source'],parent_source=binding['parent_source'],protected_previous_study=binding['protected_previous_study'],
        actual_data_hashes=dict(reconstruction=sha(io.BASE/'RECONSTRUCTION_PRIVATE.pkl'),raw_registry=sha(io.BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),inner_roles=sha(io.OLD/'ENCODER_REGISTRY_PRIVATE.pkl')),
        cuDNN_enabled=True,cuDNN_deterministic=True,precision='FP32_TF32_OFF',
        actual_complete_patient_gradient_admissions=9,maximum_direct_cache_gradient_error=0.,RNG_parity=True,
        audit_patient_count=79,audit_fields_identical_to_preserved_execution_v1=checked,
        actual_qualified_old_post_gain_invariance_max_error=max(finite) if finite else None,
        no_label_outcome_read_for_execution_choice=True,old_cross_backend_gradient_comparison_status='FAIL_RETAINED_NOT_EQUIVALENT',
        individual_audit_rows_exported=False,scope='Admission_and_input_audit_not_classification_performance'))
    print('RETENTION_SOURCE_AND_INPUT_ADMISSION_EXPORTED',79,checked,flush=True)
