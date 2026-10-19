"""Freeze an execution-only environment repair, never change science binding."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(r'D:\drechan_temporal_pr_contrast_3seed_v1_runtime');HERE=Path(__file__).resolve().parent
def jr(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=jr(ROOT/'public/ADMISSION.json');b=jr(ROOT/'public/ISOLATED_RUNTIME_PARITY_01.json');v=jr(ROOT/'public/RUNTIME_INTEGRITY_isolated02.json')
    assert a['status']==b['status']=='PASS' and v['all_import_trials_pass']
    assert b['source']==a['source'] and all(sha(HERE/n)==h for n,h in a['source'].items())
    assert b['maximum_gradient_difference']==0 and b['one_complete_update_logits_loss_all_gradients_model_Adam_RNG_bit_exact']
    assert len(b['all30_completed_cells_exact'])==30 and len(b['original_b0_bridge'])==len(b['all5_TRAIN_preprocessing_exact'])==5
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
    k.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    own=ctypes.c_size_t();system=ctypes.c_size_t();assert k.GetProcessAffinityMask(k.GetCurrentProcess(),ctypes.byref(own),ctypes.byref(system))
    assert own.value==0xffff0000
    sources={n:sha(HERE/n) for n in ['admit_execution.py','affinity_entry_v2.py','run_all_isolated.ps1','runtime_parity.py','runtime_integrity_tagged.py']}
    result=dict(status='PASS',scientific_source=a['source'],execution_source=sources,logical_cpu_affinity_hex='ffff0000',scope='Owned worker and children only; no BIOS/clocks/reboot/security-service change',
        python=sys.version.split()[0],python_environment='D-drive isolated CPython3.13.9, no system site packages',torch=v['torch'],numpy='2.3.5 pip OpenBLAS',
        exact_critical_Torch_binaries=v['files'],numeric_admission_sha256=sha(ROOT/'public/ISOLATED_RUNTIME_PARITY_01.json'),
        import_admission_sha256=sha(ROOT/'public/RUNTIME_INTEGRITY_isolated02.json'),all30_completed_cells_retained=True,
        model_optimizer_RNG_reference_selection_thresholds_unchanged=True,no_new_assessment_or_Outer=True,
        root_cause_not_proven=True,system_wide_native_faults_not_claimed_fixed=True,no_KMP_DUPLICATE_LIB_OK=True)
    path=ROOT/'public/EXECUTION_RUNTIME_AUDIT.json'
    if path.exists():assert jr(path)==result
    else:path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print('EXECUTION_ONLY_EXACT_RUNTIME_RECOVERY_ADMITTED',flush=True)
if __name__=='__main__':main()
